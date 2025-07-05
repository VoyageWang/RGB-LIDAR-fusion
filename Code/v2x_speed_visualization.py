#!/usr/bin/env python3
"""
V2X速度可视化脚本
可视化GT标签和预测结果中的速度信息，生成对比视频
"""

import json
import cv2
import numpy as np
import open3d as o3d
import os
import sys
import gc
import traceback
import argparse
from tqdm import tqdm
from collections import defaultdict

# 导入必要的模块
from ultralytics import YOLO
from fusion import *
from utils import *
from visualization import *


class V2XSpeedCalibration:
    """V2X速度可视化标定类"""
    
    def __init__(self, camera_intrinsic_path, lidar_to_camera_path):
        try:
            # 读取相机内参
            with open(camera_intrinsic_path, 'r') as f:
                cam_data = json.load(f)
            
            # 读取LiDAR到相机的外参
            with open(lidar_to_camera_path, 'r') as f:
                lidar_cam_data = json.load(f)
            
            # 相机内参矩阵 (3x3)
            self.K = np.array(cam_data['cam_K'], dtype=np.float64).reshape(3, 3)
            
            # 畸变参数
            # self.D = np.array(cam_data['cam_D'], dtype=np.float64)
            
            # LiDAR到相机的旋转矩阵 (3x3)
            self.R = np.array(lidar_cam_data['rotation'], dtype=np.float64)
            
            # LiDAR到相机的平移向量 (3x1)
            self.t = np.array(lidar_cam_data['translation'], dtype=np.float64).reshape(3, 1)
            
            # 构建变换矩阵 (4x4)
            self.T = np.eye(4, dtype=np.float64)
            self.T[:3, :3] = self.R
            self.T[:3, 3:4] = self.t
            
        except Exception as e:
            print(f"标定初始化失败: {e}")
            raise
    
    def convert_3D_to_2D(self, points_3D):
        """将3D点云投影到2D图像平面"""
        try:
            if points_3D is None or len(points_3D) == 0:
                return np.array([]), np.array([])
            
            points_3D = np.asarray(points_3D, dtype=np.float64)
            
            if points_3D.ndim != 2 or points_3D.shape[1] != 3:
                return np.array([]), np.array([])
            
            # 过滤无效值
            valid_mask = np.isfinite(points_3D).all(axis=1)
            if not np.any(valid_mask):
                return np.array([]), np.array([])
            
            points_3D_valid = points_3D[valid_mask]
            
            # 转换为齐次坐标
            points_3D_homo = np.hstack([points_3D_valid, np.ones((points_3D_valid.shape[0], 1), dtype=np.float64)])
            
            # 应用LiDAR到相机的变换
            points_cam = (self.T @ points_3D_homo.T).T[:, :3]
            
            # 过滤掉相机后方的点
            front_mask = points_cam[:, 2] > 0.1
            points_cam_valid = points_cam[front_mask]
            
            if len(points_cam_valid) == 0:
                return np.array([]), valid_mask
            
            # 投影到图像平面
            points_2D_homo = (self.K @ points_cam_valid.T).T
            
            # 避免除零
            z_coords = points_2D_homo[:, 2:3]
            z_coords = np.where(np.abs(z_coords) < 1e-6, 1e-6, z_coords)
            
            points_2D = points_2D_homo[:, :2] / z_coords
            
            # 更新有效掩码
            final_valid_mask = np.zeros(len(points_3D), dtype=bool)
            valid_indices = np.where(valid_mask)[0]
            front_indices = valid_indices[front_mask]
            final_valid_mask[front_indices] = True
            
            return points_2D, final_valid_mask
            
        except Exception as e:
            print(f"3D到2D投影失败: {e}")
            return np.array([]), np.array([])


class SpeedTrackingDetector:
    """支持速度跟踪的检测器"""
    
    def __init__(self, model_path, tracking=True, device=None):
        self.model = YOLO(model_path)
        self.tracking = tracking
        self.device = device
        self.last_ground_center_of_id = {}
        self.last_timestamp_of_id = {}
        self.velocity_history = defaultdict(list)  # 存储速度历史
        
        # 如果指定了设备，设置模型设备
        if device is not None:
            try:
                import torch
                if torch.cuda.is_available() and isinstance(device, int):
                    self.model.to(f'cuda:{device}')
                    print(f"YOLO模型已设置到GPU {device}")
                elif device == 'cpu':
                    self.model.to('cpu')
                    print("YOLO模型已设置到CPU")
            except Exception as e:
                print(f"设置模型设备时出错: {e}")
        
    def process_frame(self, frame, points, calibration, timestamp, erosion_factor=25, depth_factor=20):
        """处理单帧数据，计算速度"""
        try:
            # YOLO推理 - 指定设备
            inference_kwargs = {
                'source': frame,
                'classes': [2, 5, 6, 7] if self.tracking else [0, 1, 2, 3, 5, 6, 7],
                'verbose': False,
                'show': False,
            }
            
            # 如果指定了设备，添加设备参数
            if self.device is not None:
                if isinstance(self.device, int):
                    inference_kwargs['device'] = f'cuda:{self.device}'
                else:
                    inference_kwargs['device'] = self.device
            
            if self.tracking:
                results = self.model.track(
                    **inference_kwargs,
                    persist=True,
                    tracker='bytetrack.yaml'
                )
            else:
                results = self.model.predict(**inference_kwargs)
            
            r = results[0]
            boxes = r.boxes
            masks = r.masks
            
            # 点云投影
            pts_2D, valid_mask = calibration.convert_3D_to_2D(points)
            if len(pts_2D) > 0:
                img_width, img_height = frame.shape[1], frame.shape[0]
                valid_2d_mask = (
                    (pts_2D[:, 0] >= 0) & (pts_2D[:, 0] < img_width) &
                    (pts_2D[:, 1] >= 0) & (pts_2D[:, 1] < img_height)
                )
                valid_indices = np.where(valid_mask)[0][valid_2d_mask]
                pts_3D = points[valid_indices]
                pts_2D = pts_2D[valid_2d_mask]
            else:
                pts_3D, pts_2D = np.array([]), np.array([])
            
            # 处理检测结果
            all_corners_3D = []
            all_filtered_points_of_object = []
            all_object_IDs = []
            objects_with_speed = []
            
            if boxes is None or len(boxes) == 0:
                return objects_with_speed, all_corners_3D, pts_3D, pts_2D, all_filtered_points_of_object, all_object_IDs
            
            for j, cls in enumerate(boxes.cls.tolist()):
                try:
                    conf = boxes.conf.tolist()[j] if boxes.conf is not None else None
                    box_id = int(boxes.id.tolist()[j]) if boxes.id is not None else j
                    
                    all_object_IDs.append(box_id)
                    
                    # 获取2D边界框
                    xyxy = boxes.xyxy[j].cpu().numpy()
                    
                    # 检查掩码
                    if masks is None or j >= len(masks.xy) or masks.xy[j].size == 0:
                        continue
                    
                    # 融合处理
                    fusion_result = lidar_camera_fusion(
                        pts_3D, pts_2D, frame, masks.xy[j], int(cls), 
                        calibration, erosion_factor=erosion_factor, 
                        depth_factor=depth_factor, PCA=False
                    )
                    
                    if fusion_result is not None:
                        filtered_points_of_object, corners_3D, yaw = fusion_result
                        all_corners_3D.append(corners_3D)
                        all_filtered_points_of_object.append(filtered_points_of_object)
                        
                        # 计算目标信息
                        ROS_type = int(np.int32(cls))
                        bottom_indices = np.argsort(corners_3D[:, 2])[:4]
                        ROS_ground_center = np.mean(corners_3D[bottom_indices], axis=0)
                        ROS_dimensions = np.ptp(corners_3D, axis=0)
                        
                        # 计算速度
                        velocity = np.array([0.0, 0.0, 0.0])
                        speed_magnitude = 0.0
                        
                        if (box_id in self.last_ground_center_of_id and 
                            box_id in self.last_timestamp_of_id):
                            
                            last_center = self.last_ground_center_of_id[box_id]
                            last_time = self.last_timestamp_of_id[box_id]
                            time_diff = timestamp - last_time
                            
                            if time_diff > 0:
                                displacement = ROS_ground_center - last_center
                                velocity = displacement / time_diff
                                speed_magnitude = np.linalg.norm(velocity)
                                
                                # 存储速度历史（用于平滑）
                                self.velocity_history[box_id].append(speed_magnitude)
                                if len(self.velocity_history[box_id]) > 5:  # 保持最近5帧
                                    self.velocity_history[box_id].pop(0)
                                
                                # 使用平滑后的速度
                                speed_magnitude = np.mean(self.velocity_history[box_id])
                        
                        # 更新历史信息
                        self.last_ground_center_of_id[box_id] = ROS_ground_center.copy()
                        self.last_timestamp_of_id[box_id] = timestamp
                        
                        objects_with_speed.append({
                            'id': box_id,
                            'class': ROS_type,
                            'confidence': conf,
                            'bbox_2d': xyxy,
                            'center_3d': ROS_ground_center,
                            'dimensions_3d': ROS_dimensions,
                            'corners_3d': corners_3D,
                            'velocity': velocity,
                            'speed': speed_magnitude,
                            'yaw': yaw
                        })
                        
                except Exception as e:
                    continue
            
            return objects_with_speed, all_corners_3D, pts_3D, pts_2D, all_filtered_points_of_object, all_object_IDs
            
        except Exception as e:
            print(f"处理帧时出错: {e}")
            return [], [], np.array([]), np.array([]), [], []


def load_gt_labels_with_speed(data_root, frame_info):
    """加载GT标签并计算速度"""
    try:
        frame_id = frame_info['frame_id']
        camera_label_path = os.path.join(data_root, f"label/camera/{frame_id}.json")
        lidar_label_path = os.path.join(data_root, f"label/virtuallidar/{frame_id}.json")
        
        if not os.path.exists(camera_label_path) or not os.path.exists(lidar_label_path):
            return []
        
        with open(camera_label_path, 'r') as f:
            camera_labels = json.load(f)
        
        with open(lidar_label_path, 'r') as f:
            lidar_labels = json.load(f)
        
        gt_objects = []
        for cam_label, lidar_label in zip(camera_labels, lidar_labels):
            # 计算GT速度（如果有track_id的话）
            gt_speed = 0.0
            if 'track_id' in cam_label:
                # 这里可以根据track_id计算速度，暂时设为0
                gt_speed = 0.0
            
            gt_objects.append({
                'track_id': cam_label.get('track_id', 'unknown'),
                'type': cam_label['type'],
                'bbox_2d': [
                    cam_label['2d_box']['xmin'],
                    cam_label['2d_box']['ymin'],
                    cam_label['2d_box']['xmax'],
                    cam_label['2d_box']['ymax']
                ],
                'center_3d': [
                    lidar_label['3d_location']['x'],
                    lidar_label['3d_location']['y'],
                    lidar_label['3d_location']['z']
                ],
                'dimensions_3d': [
                    lidar_label['3d_dimensions']['l'],
                    lidar_label['3d_dimensions']['w'],
                    lidar_label['3d_dimensions']['h']
                ],
                'speed': gt_speed,
                'rotation': lidar_label.get('rotation', 0.0)
            })
        
        return gt_objects
        
    except Exception as e:
        print(f"加载GT标签失败: {e}")
        return []


def calculate_gt_speeds(data_root, data_info, frame_window=5):
    """计算GT标签的速度"""
    print("计算GT标签速度...")
    
    # 按track_id组织数据
    track_data = defaultdict(list)
    
    for i, frame_info in enumerate(tqdm(data_info, desc="收集轨迹数据")):
        try:
            frame_id = frame_info['frame_id']
            camera_label_path = os.path.join(data_root, f"label/camera/{frame_id}.json")
            lidar_label_path = os.path.join(data_root, f"label/virtuallidar/{frame_id}.json")
            
            if not os.path.exists(camera_label_path) or not os.path.exists(lidar_label_path):
                continue
            
            with open(camera_label_path, 'r') as f:
                camera_labels = json.load(f)
            
            with open(lidar_label_path, 'r') as f:
                lidar_labels = json.load(f)
            
            for cam_label, lidar_label in zip(camera_labels, lidar_labels):
                track_id = cam_label.get('track_id', None)
                if track_id:
                    track_data[track_id].append({
                        'frame_index': i,
                        'frame_id': frame_id,
                        'center_3d': np.array([
                            lidar_label['3d_location']['x'],
                            lidar_label['3d_location']['y'],
                            lidar_label['3d_location']['z']
                        ]),
                        'type': cam_label['type']
                    })
        except Exception as e:
            continue
    
    # 计算每个轨迹的速度
    gt_speeds = {}
    for track_id, trajectory in track_data.items():
        if len(trajectory) < 2:
            continue
        
        # 按帧索引排序
        trajectory.sort(key=lambda x: x['frame_index'])
        
        speeds = {}
        for i in range(1, len(trajectory)):
            prev_point = trajectory[i-1]
            curr_point = trajectory[i]
            
            # 计算位移和时间差
            displacement = curr_point['center_3d'] - prev_point['center_3d']
            frame_diff = curr_point['frame_index'] - prev_point['frame_index']
            time_diff = frame_diff * 0.1  # 假设10fps
            
            if time_diff > 0:
                velocity = displacement / time_diff
                speed = np.linalg.norm(velocity)
                speeds[curr_point['frame_id']] = speed
        
        gt_speeds[track_id] = speeds
    
    return gt_speeds


def convert_gt_to_3d_corners(center_3d, dimensions_3d, rotation_y=0):
    """将GT的中心点和尺寸转换为8个角点"""
    try:
        x, y, z = center_3d
        l, w, h = dimensions_3d
        
        # 创建标准边界框的8个角点（以原点为中心）
        corners = np.array([
            [-l/2, -w/2, -h/2],  # 0: 左下后
            [+l/2, -w/2, -h/2],  # 1: 右下后
            [+l/2, +w/2, -h/2],  # 2: 右上后
            [-l/2, +w/2, -h/2],  # 3: 左上后
            [-l/2, -w/2, +h/2],  # 4: 左下前
            [+l/2, -w/2, +h/2],  # 5: 右下前
            [+l/2, +w/2, +h/2],  # 6: 右上前
            [-l/2, +w/2, +h/2],  # 7: 左上前
        ])
        
        # 应用旋转（如果有）
        if rotation_y != 0:
            cos_y = np.cos(rotation_y)
            sin_y = np.sin(rotation_y)
            rotation_matrix = np.array([
                [cos_y, 0, sin_y],
                [0, 1, 0],
                [-sin_y, 0, cos_y]
            ])
            corners = corners @ rotation_matrix.T
        
        # 平移到实际位置
        corners += np.array([x, y, z])
        
        return corners
        
    except Exception as e:
        print(f"转换GT角点时出错: {e}")
        return np.zeros((8, 3))


def draw_simple_3d_bbox(image, corners_3d, calibration, color, obj_info=None, line_thickness=2):
    """绘制简洁的3D边界框"""
    try:
        # 投影3D角点到2D
        corners_2d, _ = calibration.convert_3D_to_2D(corners_3d)
        if len(corners_2d) < 8:
            return image
        
        # 只绘制主要的边界框轮廓，不绘制所有12条边
        # 绘制底面矩形 (0,1,2,3)
        bottom_indices = [0, 1, 2, 3, 0]  # 闭合矩形
        for i in range(len(bottom_indices) - 1):
            if bottom_indices[i] < len(corners_2d) and bottom_indices[i+1] < len(corners_2d):
                pt1 = tuple(np.int32(corners_2d[bottom_indices[i]]))
                pt2 = tuple(np.int32(corners_2d[bottom_indices[i+1]]))
                
                # 检查点是否在图像范围内
                if (0 <= pt1[0] < image.shape[1] and 0 <= pt1[1] < image.shape[0] and
                    0 <= pt2[0] < image.shape[1] and 0 <= pt2[1] < image.shape[0]):
                    cv2.line(image, pt1, pt2, color, line_thickness)
        
        # 绘制顶面矩形 (4,5,6,7)
        top_indices = [4, 5, 6, 7, 4]  # 闭合矩形
        for i in range(len(top_indices) - 1):
            if top_indices[i] < len(corners_2d) and top_indices[i+1] < len(corners_2d):
                pt1 = tuple(np.int32(corners_2d[top_indices[i]]))
                pt2 = tuple(np.int32(corners_2d[top_indices[i+1]]))
                
                # 检查点是否在图像范围内
                if (0 <= pt1[0] < image.shape[1] and 0 <= pt1[1] < image.shape[0] and
                    0 <= pt2[0] < image.shape[1] and 0 <= pt2[1] < image.shape[0]):
                    cv2.line(image, pt1, pt2, color, line_thickness)
        
        # 绘制4条垂直边连接底面和顶面
        vertical_pairs = [(0, 4), (1, 5), (2, 6), (3, 7)]
        for bottom_idx, top_idx in vertical_pairs:
            if bottom_idx < len(corners_2d) and top_idx < len(corners_2d):
                pt1 = tuple(np.int32(corners_2d[bottom_idx]))
                pt2 = tuple(np.int32(corners_2d[top_idx]))
                
                # 检查点是否在图像范围内
                if (0 <= pt1[0] < image.shape[1] and 0 <= pt1[1] < image.shape[0] and
                    0 <= pt2[0] < image.shape[1] and 0 <= pt2[1] < image.shape[0]):
                    cv2.line(image, pt1, pt2, color, line_thickness)
        
        # 只添加简单的ID标签（如果需要）
        if obj_info and 'id' in obj_info and len(corners_2d) > 0:
            # 使用边界框的左上角作为标签位置
            min_x = int(np.min(corners_2d[:, 0]))
            min_y = int(np.min(corners_2d[:, 1]))
            label_pt = (max(0, min_x), max(20, min_y - 5))
            
            if (0 <= label_pt[0] < image.shape[1] and 0 <= label_pt[1] < image.shape[0]):
                # 只显示ID和速度
                label_text = f"ID:{obj_info['id']}"
                if 'speed' in obj_info:
                    label_text += f" {obj_info['speed']:.1f}m/s"
                
                # 简单的文本标签，不要复杂的背景框
                cv2.putText(image, label_text, label_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return image
        
    except Exception as e:
        print(f"绘制简洁3D边界框时出错: {e}")
        return image


def draw_enhanced_3d_bbox(image, corners_3d, calibration, color, obj_info=None, line_thickness=2):
    """绘制增强的3D边界框"""
    try:
        # 投影3D角点到2D
        corners_2d, _ = calibration.convert_3D_to_2D(corners_3d)
        if len(corners_2d) < 8:
            return image
        
        # 获取边界框的边
        edges = get_pred_bbox_edges(corners_2d)
        
        # 绘制边界框边缘
        for edge in edges:
            pt1 = tuple(np.int32(edge[0]))
            pt2 = tuple(np.int32(edge[1]))
            
            # 检查点是否在图像范围内
            if (0 <= pt1[0] < image.shape[1] and 0 <= pt1[1] < image.shape[0] and
                0 <= pt2[0] < image.shape[1] and 0 <= pt2[1] < image.shape[0]):
                cv2.line(image, pt1, pt2, color, line_thickness)
        
        # 绘制中心点
        center_3d = np.mean(corners_3d, axis=0)
        center_2d, _ = calibration.convert_3D_to_2D([center_3d])
        if len(center_2d) > 0:
            center_pt = tuple(np.int32(center_2d[0]))
            if (0 <= center_pt[0] < image.shape[1] and 0 <= center_pt[1] < image.shape[0]):
                cv2.circle(image, center_pt, 4, color, -1)
        
        # 添加3D信息标签
        if obj_info and len(corners_2d) > 7:
            # 使用顶部前左角作为标签位置
            label_pt = tuple(np.int32(corners_2d[7]))
            if (0 <= label_pt[0] < image.shape[1] and 0 <= label_pt[1] < image.shape[0]):
                # 准备标签文本
                labels = []
                if 'id' in obj_info:
                    labels.append(f"ID: {obj_info['id']}")
                if 'speed' in obj_info:
                    labels.append(f"Speed: {obj_info['speed']:.1f}m/s")
                # 移除类型和尺寸信息，只保留ID和速度
                
                # 绘制标签背景
                label_height = len(labels) * 15 + 10
                cv2.rectangle(image, 
                            (label_pt[0] - 5, label_pt[1] - label_height - 5),
                            (label_pt[0] + 200, label_pt[1] + 5),
                            (0, 0, 0), -1)
                cv2.rectangle(image, 
                            (label_pt[0] - 5, label_pt[1] - label_height - 5),
                            (label_pt[0] + 200, label_pt[1] + 5),
                            color, 2)
                
                # 绘制标签文本
                for i, label in enumerate(labels):
                    text_pt = (label_pt[0], label_pt[1] - label_height + i * 15 + 12)
                    cv2.putText(image, label, text_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return image
        
    except Exception as e:
        print(f"绘制3D边界框时出错: {e}")
        return image


def draw_speed_visualization(image, objects, calibration, title, is_gt=False):
    """绘制速度可视化"""
    result_image = image.copy()
    
    # 添加标题
    cv2.putText(result_image, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(result_image, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
    
    for obj in objects:
        try:
            if is_gt:
                # GT对象 - 完全参考video_v2x_3d_detection.py的绘制方式
                bbox = [int(x) for x in obj['bbox_2d']]
                speed = obj['speed']
                track_id = obj.get('track_id', 'unknown')
                
                # 绘制2D边界框（绿色）
                cv2.rectangle(result_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                
                # 绘制GT的3D边界框（完全参考video_v2x_3d_detection.py）
                if 'center_3d' in obj and 'dimensions_3d' in obj:
                    rotation = obj.get('rotation', 0.0)
                    gt_corners_3d = convert_gt_to_3d_corners(
                        obj['center_3d'], obj['dimensions_3d'], rotation
                    )
                    
                    try:
                        gt_corners_2d, _ = calibration.convert_3D_to_2D(gt_corners_3d)
                        if len(gt_corners_2d) >= 8:
                            # 使用utils.py中的get_pred_bbox_edges函数
                            gt_edges_2d = get_pred_bbox_edges(gt_corners_2d)
                            
                            # 绘制3D边界框的所有边（绿色）
                            for gt_edge in gt_edges_2d:
                                pt1 = tuple(np.int32(gt_edge[0]))
                                pt2 = tuple(np.int32(gt_edge[1]))
                                cv2.line(result_image, pt1, pt2, (0, 255, 0), 2)
                            
                            # 添加标签（参考video_v2x_3d_detection.py的标签位置）
                            if len(gt_corners_2d) > 7:
                                top_left_front_corner = gt_corners_2d[7]
                                top_left_front_pt = (int(np.round(top_left_front_corner[0])), int(np.round(top_left_front_corner[1])) - 10)
                                
                                label_text = f'ID: {track_id} Speed: {speed:.1f}m/s'
                                
                                # 绘制带阴影的文本（完全参考video_v2x_3d_detection.py）
                                cv2.putText(result_image, label_text, top_left_front_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 3, cv2.LINE_AA)
                                cv2.putText(result_image, label_text, top_left_front_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 1, cv2.LINE_AA)
                    except Exception as e:
                        pass
                
            else:
                # 预测对象 - 完全参考video_v2x_3d_detection.py的方式绘制3D边界框
                speed = obj['speed']
                obj_id = obj['id']
                
                # 根据速度选择颜色
                if speed < 1.0:
                    color = (0, 255, 255)  # 黄色 - 静止
                elif speed < 5.0:
                    color = (0, 165, 255)  # 橙色 - 慢速
                elif speed < 10.0:
                    color = (0, 0, 255)    # 红色 - 中速
                else:
                    color = (255, 0, 255)  # 紫色 - 高速
                
                # 完全参考video_v2x_3d_detection.py的绘制方式
                if 'corners_3d' in obj:
                    pred_corner_3D = obj['corners_3d']
                    try:
                        pred_corners_2D, _ = calibration.convert_3D_to_2D(pred_corner_3D)
                        if len(pred_corners_2D) >= 8:
                            # 使用utils.py中的get_pred_bbox_edges函数
                            pred_edges_2D = get_pred_bbox_edges(pred_corners_2D)
                            
                            # 绘制3D边界框的所有边
                            for pred_edge in pred_edges_2D:
                                pt1 = tuple(np.int32(pred_edge[0]))
                                pt2 = tuple(np.int32(pred_edge[1]))
                                cv2.line(result_image, pt1, pt2, color, 2)
                            
                            # 添加ID和速度标签（参考video_v2x_3d_detection.py的标签位置）
                            if len(pred_corners_2D) > 7:
                                top_left_front_corner = pred_corners_2D[7]
                                top_left_front_pt = (int(np.round(top_left_front_corner[0])), int(np.round(top_left_front_corner[1])) - 10)
                                
                                label_text = f'ID: {obj_id} Speed: {speed:.1f}m/s'
                                
                                # 绘制带阴影的文本（完全参考video_v2x_3d_detection.py）
                                cv2.putText(result_image, label_text, top_left_front_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 3, cv2.LINE_AA)
                                cv2.putText(result_image, label_text, top_left_front_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 1, cv2.LINE_AA)
                    except Exception as e:
                        continue
                
        except Exception as e:
            continue
    
    # 添加速度图例
    legend_y = 70
    cv2.putText(result_image, "Speed Legend:", (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(result_image, "< 1 m/s", (10, legend_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.putText(result_image, "1-5 m/s", (10, legend_y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
    cv2.putText(result_image, "5-10 m/s", (10, legend_y + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(result_image, "> 10 m/s", (10, legend_y + 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
    
    # 3D边界框图例
    legend_3d_y = legend_y + 120
    cv2.putText(result_image, "3D Box Legend:", (10, legend_3d_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(result_image, "Green: GT 3D Box", (10, legend_3d_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(result_image, "Colored: Pred 3D Box", (10, legend_3d_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return result_image


def create_bev_with_enhanced_3d_boxes(points, gt_objects, pred_objects, max_points=5000):
    """创建包含增强3D边界框的鸟瞰图"""
    try:
        # 限制点数
        if len(points) > max_points:
            indices = np.random.choice(len(points), max_points, replace=False)
            sample_points = points[indices]
        else:
            sample_points = points
        
        # 设置BEV参数
        x_range = (-50, 50)
        y_range = (-50, 50)
        resolution = 0.2
        
        width = int((x_range[1] - x_range[0]) / resolution)
        height = int((y_range[1] - y_range[0]) / resolution)
        
        bev_image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # 绘制点云
        valid_mask = (
            (sample_points[:, 0] >= x_range[0]) & (sample_points[:, 0] <= x_range[1]) &
            (sample_points[:, 1] >= y_range[0]) & (sample_points[:, 1] <= y_range[1])
        )
        
        valid_points = sample_points[valid_mask]
        
        if len(valid_points) > 0:
            x_img = ((valid_points[:, 0] - x_range[0]) / resolution).astype(int)
            y_img = ((valid_points[:, 1] - y_range[0]) / resolution).astype(int)
            
            heights = valid_points[:, 2]
            height_norm = (heights - np.min(heights)) / (np.max(heights) - np.min(heights) + 1e-6)
            
            for i in range(len(valid_points)):
                if 0 <= x_img[i] < width and 0 <= y_img[i] < height:
                    color = (
                        int(255 * (1 - height_norm[i])),
                        int(255 * (1 - abs(height_norm[i] - 0.5) * 2)),
                        int(255 * height_norm[i])
                    )
                    cv2.circle(bev_image, (x_img[i], height - 1 - y_img[i]), 1, color, -1)
        
        # 绘制GT 3D边界框（绿色）
        for gt_obj in gt_objects:
            if 'center_3d' in gt_obj and 'dimensions_3d' in gt_obj:
                rotation = gt_obj.get('rotation', 0.0)
                corners_3d = convert_gt_to_3d_corners(
                    gt_obj['center_3d'], gt_obj['dimensions_3d'], rotation
                )
                
                # 只使用底面的4个点
                bottom_indices = np.argsort(corners_3d[:, 2])[:4]
                bottom_corners = corners_3d[bottom_indices]
                
                bev_corners = []
                for corner in bottom_corners:
                    if x_range[0] <= corner[0] <= x_range[1] and y_range[0] <= corner[1] <= y_range[1]:
                        x_bev = int((corner[0] - x_range[0]) / resolution)
                        y_bev = int((corner[1] - y_range[0]) / resolution)
                        bev_corners.append([x_bev, height - 1 - y_bev])
                
                if len(bev_corners) >= 3:
                    bev_corners = np.array(bev_corners, dtype=np.int32)
                    cv2.polylines(bev_image, [bev_corners], True, (0, 255, 0), 2)
                    
                    # 添加中心点
                    center = np.mean(bev_corners, axis=0).astype(int)
                    cv2.circle(bev_image, tuple(center), 3, (0, 255, 0), -1)
        
        # 绘制预测3D边界框（根据速度着色）
        for pred_obj in pred_objects:
            if 'corners_3d' in pred_obj:
                corners_3d = pred_obj['corners_3d']
                speed = pred_obj.get('speed', 0)
                
                # 根据速度选择颜色
                if speed < 1.0:
                    color = (0, 255, 255)  # 黄色
                elif speed < 5.0:
                    color = (0, 165, 255)  # 橙色
                elif speed < 10.0:
                    color = (0, 0, 255)    # 红色
                else:
                    color = (255, 0, 255)  # 紫色
                
                # 只使用底面的4个点
                bottom_indices = np.argsort(corners_3d[:, 2])[:4]
                bottom_corners = corners_3d[bottom_indices]
                
                bev_corners = []
                for corner in bottom_corners:
                    if x_range[0] <= corner[0] <= x_range[1] and y_range[0] <= corner[1] <= y_range[1]:
                        x_bev = int((corner[0] - x_range[0]) / resolution)
                        y_bev = int((corner[1] - y_range[0]) / resolution)
                        bev_corners.append([x_bev, height - 1 - y_bev])
                
                if len(bev_corners) >= 3:
                    bev_corners = np.array(bev_corners, dtype=np.int32)
                    cv2.polylines(bev_image, [bev_corners], True, color, 2)
                    
                    # 添加中心点
                    center = np.mean(bev_corners, axis=0).astype(int)
                    cv2.circle(bev_image, tuple(center), 3, color, -1)
                    
                    # 添加ID标签
                    if 'id' in pred_obj:
                        cv2.putText(bev_image, str(pred_obj['id']), 
                                  tuple(center + np.array([5, -5])), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # 添加网格线
        grid_spacing = int(10 / resolution)
        for i in range(0, width, grid_spacing):
            cv2.line(bev_image, (i, 0), (i, height-1), (50, 50, 50), 1)
        for i in range(0, height, grid_spacing):
            cv2.line(bev_image, (0, i), (width-1, i), (50, 50, 50), 1)
        
        # 添加中心点（车辆位置）
        center_x, center_y = width // 2, height // 2
        cv2.circle(bev_image, (center_x, center_y), 5, (255, 255, 255), -1)
        cv2.putText(bev_image, "EGO", (center_x + 8, center_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return bev_image
        
    except Exception as e:
        print(f"BEV可视化失败: {e}")
        return np.zeros((500, 500, 3), dtype=np.uint8)


def create_combined_video(gt_frames, pred_frames, output_path, fps=10):
    """创建左右对比视频"""
    if len(gt_frames) == 0 or len(pred_frames) == 0:
        print("没有足够的帧来创建视频")
        return
    
    # 确保两个视频帧数相同
    min_frames = min(len(gt_frames), len(pred_frames))
    gt_frames = gt_frames[:min_frames]
    pred_frames = pred_frames[:min_frames]
    
    # 获取图像尺寸
    h, w = gt_frames[0].shape[:2]
    
    # 创建组合视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    combined_writer = cv2.VideoWriter(output_path, fourcc, fps, (w * 2, h))
    
    print(f"创建组合视频: {output_path}")
    for i in tqdm(range(min_frames), desc="合并视频帧"):
        # 水平拼接两个帧
        combined_frame = np.hstack([gt_frames[i], pred_frames[i]])
        combined_writer.write(combined_frame)
    
    combined_writer.release()
    print(f"组合视频已保存: {output_path}")


def process_v2x_speed_visualization(data_root, data_info_path, output_dir, start_frame=None, end_frame=None, max_frames=None):
    """处理V2X速度可视化 - 原始版本"""
    print("=== 开始V2X速度可视化处理 ===")
    
    try:
        # 加载数据信息
        with open(data_info_path, 'r') as f:
            data_info = json.load(f)
        
        print(f"数据集总帧数: {len(data_info)}")
        
        # 确定处理范围
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = len(data_info)
        if max_frames is not None:
            end_frame = min(start_frame + max_frames, end_frame)
        
        frames_to_process = data_info[start_frame:end_frame]
        print(f"将处理帧 {start_frame} 到 {end_frame-1}，共 {len(frames_to_process)} 帧")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 计算GT速度
        gt_speeds = calculate_gt_speeds(data_root, data_info)
        
        # 初始化检测器和标定
        first_frame = frames_to_process[0]
        camera_intrinsic_path = os.path.join(data_root, first_frame['calib_camera_intrinsic_path'])
        lidar_to_camera_path = os.path.join(data_root, first_frame['calib_virtuallidar_to_camera_path'])
        
        calibration = V2XSpeedCalibration(camera_intrinsic_path, lidar_to_camera_path)
        
        model_path = "yolov8m-seg.pt"
        detector = SpeedTrackingDetector(model_path, tracking=True)
        
        print(f"初始化完成，开始处理...")
        
        # 存储处理后的帧
        gt_frames = []
        pred_frames = []
        bev_frames = []
        speed_stats = []
        
        # 处理每一帧
        for i, frame_info in enumerate(tqdm(frames_to_process, desc="处理帧")):
            try:
                frame_id = frame_info['frame_id']
                
                # 构建文件路径
                image_path = os.path.join(data_root, frame_info['image_path'])
                pointcloud_path = os.path.join(data_root, frame_info['pointcloud_path'])
                
                # 检查文件存在性
                if not os.path.exists(image_path) or not os.path.exists(pointcloud_path):
                    print(f"跳过帧 {i}: 文件不存在")
                    continue
                
                # 加载图像
                image = cv2.imread(image_path)
                if image is None:
                    print(f"跳过帧 {i}: 无法读取图像")
                    continue
                
                # 加载点云
                pcd = o3d.io.read_point_cloud(pointcloud_path)
                points = np.asarray(pcd.points, dtype=np.float64)
                
                if len(points) == 0:
                    print(f"跳过帧 {i}: 点云为空")
                    continue
                
                # 过滤无效点
                if np.any(np.isnan(points)) or np.any(np.isinf(points)):
                    valid_mask = np.isfinite(points).all(axis=1)
                    points = points[valid_mask]
                
                # 加载GT标签并添加速度信息
                gt_objects = load_gt_labels_with_speed(data_root, frame_info)
                
                # 为GT对象添加计算出的速度
                for gt_obj in gt_objects:
                    track_id = gt_obj['track_id']
                    if track_id in gt_speeds and frame_id in gt_speeds[track_id]:
                        gt_obj['speed'] = gt_speeds[track_id][frame_id]
                
                # 运行预测检测
                timestamp = i * 0.1  # 假设10fps
                pred_objects, all_corners_3D, pts_3D, pts_2D, all_filtered_points_of_object, all_object_IDs = detector.process_frame(
                    image, points, calibration, timestamp, erosion_factor=25, depth_factor=20
                )
                
                # 绘制GT速度可视化
                gt_frame = draw_speed_visualization(
                    image, gt_objects, calibration, 
                    f"Ground Truth Speed - Frame {start_frame + i}", is_gt=True
                )
                
                # 绘制预测速度可视化
                pred_frame = draw_speed_visualization(
                    image, pred_objects, calibration, 
                    f"Predicted Speed - Frame {start_frame + i}", is_gt=False
                )
                
                # 创建鸟瞰图
                bev_frame = create_bev_with_enhanced_3d_boxes(points, gt_objects, pred_objects)
                
                # 添加BEV标题
                cv2.putText(bev_frame, f"Bird's Eye View - Frame {start_frame + i}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                gt_frames.append(gt_frame)
                pred_frames.append(pred_frame)
                bev_frames.append(bev_frame)
                
                # 记录速度统计
                gt_speeds_frame = [obj['speed'] for obj in gt_objects if obj['speed'] > 0]
                pred_speeds_frame = [obj['speed'] for obj in pred_objects if obj['speed'] > 0]
                
                speed_stats.append({
                    'frame_id': frame_id,
                    'gt_objects': len(gt_objects),
                    'pred_objects': len(pred_objects),
                    'gt_avg_speed': np.mean(gt_speeds_frame) if gt_speeds_frame else 0,
                    'pred_avg_speed': np.mean(pred_speeds_frame) if pred_speeds_frame else 0,
                    'gt_max_speed': np.max(gt_speeds_frame) if gt_speeds_frame else 0,
                    'pred_max_speed': np.max(pred_speeds_frame) if pred_speeds_frame else 0
                })
                
                # 定期清理内存
                if i % 10 == 0:
                    gc.collect()
                
            except Exception as e:
                print(f"处理帧 {i} 时出错: {e}")
                continue
        
        print(f"成功处理 {len(gt_frames)} 帧")
        
        if len(gt_frames) > 0 and len(pred_frames) > 0:
            # 创建单独的视频
            print("创建GT速度视频...")
            gt_video_path = os.path.join(output_dir, "gt_speed_visualization.mp4")
            create_video(gt_frames, gt_video_path)
            
            print("创建预测速度视频...")
            pred_video_path = os.path.join(output_dir, "pred_speed_visualization.mp4")
            create_video(pred_frames, pred_video_path)
            
            # 创建鸟瞰图视频
            if len(bev_frames) > 0:
                print("创建鸟瞰图视频...")
                bev_video_path = os.path.join(output_dir, "bev_3d_boxes_visualization.mp4")
                create_video(bev_frames, bev_video_path)
            
            # 创建对比视频
            print("创建速度对比视频...")
            combined_video_path = os.path.join(output_dir, "speed_comparison.mp4")
            create_combined_video(gt_frames, pred_frames, combined_video_path)
            
            # 创建三合一视频（GT + 预测 + BEV）
            if len(bev_frames) > 0:
                print("创建三合一对比视频...")
                triple_video_path = os.path.join(output_dir, "triple_view_comparison.mp4")
                create_triple_combined_video(gt_frames, pred_frames, bev_frames, triple_video_path)
            
            # 保存速度统计
            stats_path = os.path.join(output_dir, "speed_statistics.json")
            with open(stats_path, 'w') as f:
                json.dump(speed_stats, f, indent=2, default=str)
            
            # 打印统计信息
            print(f"\n=== 速度统计分析 ===")
            all_gt_speeds = [stat['gt_avg_speed'] for stat in speed_stats if stat['gt_avg_speed'] > 0]
            all_pred_speeds = [stat['pred_avg_speed'] for stat in speed_stats if stat['pred_avg_speed'] > 0]
            
            if all_gt_speeds:
                print(f"GT平均速度: {np.mean(all_gt_speeds):.2f} m/s")
                print(f"GT最大速度: {np.max(all_gt_speeds):.2f} m/s")
            
            if all_pred_speeds:
                print(f"预测平均速度: {np.mean(all_pred_speeds):.2f} m/s")
                print(f"预测最大速度: {np.max(all_pred_speeds):.2f} m/s")
            
            print(f"结果保存在: {output_dir}")
            return True
        else:
            print("没有成功处理任何帧")
            return False
    
    except Exception as e:
        print(f"处理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def process_v2x_speed_visualization_with_device(data_root, data_info_path, output_dir, start_frame=None, end_frame=None, max_frames=None, device=None, process_id=None):
    """处理V2X速度可视化 - 支持设备参数的版本"""
    process_prefix = f"[进程 {process_id}] " if process_id else ""
    print(f"{process_prefix}=== 开始V2X速度可视化处理 ===")
    
    try:
        # 加载数据信息
        with open(data_info_path, 'r') as f:
            data_info = json.load(f)
        
        print(f"{process_prefix}数据集总帧数: {len(data_info)}")
        
        # 确定处理范围
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = len(data_info)
        if max_frames is not None:
            end_frame = min(start_frame + max_frames, end_frame)
        
        frames_to_process = data_info[start_frame:end_frame]
        print(f"{process_prefix}将处理帧 {start_frame} 到 {end_frame-1}，共 {len(frames_to_process)} 帧")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 计算GT速度
        gt_speeds = calculate_gt_speeds(data_root, data_info)
        
        # 初始化检测器和标定
        first_frame = frames_to_process[0]
        camera_intrinsic_path = os.path.join(data_root, first_frame['calib_camera_intrinsic_path'])
        lidar_to_camera_path = os.path.join(data_root, first_frame['calib_virtuallidar_to_camera_path'])
        
        calibration = V2XSpeedCalibration(camera_intrinsic_path, lidar_to_camera_path)
        
        model_path = "yolov8m-seg.pt"
        # 使用支持设备参数的检测器
        detector = SpeedTrackingDetector(model_path, tracking=True, device=device)
        
        print(f"{process_prefix}初始化完成，开始处理...")
        
        # 存储处理后的帧
        gt_frames = []
        pred_frames = []
        bev_frames = []
        speed_stats = []
        
        # 处理每一帧
        for i, frame_info in enumerate(tqdm(frames_to_process, desc=f"{process_prefix}处理帧")):
            try:
                frame_id = frame_info['frame_id']
                
                # 构建文件路径
                image_path = os.path.join(data_root, frame_info['image_path'])
                pointcloud_path = os.path.join(data_root, frame_info['pointcloud_path'])
                
                # 检查文件存在性
                if not os.path.exists(image_path) or not os.path.exists(pointcloud_path):
                    print(f"{process_prefix}跳过帧 {i}: 文件不存在")
                    continue
                
                # 加载图像
                image = cv2.imread(image_path)
                if image is None:
                    print(f"{process_prefix}跳过帧 {i}: 无法读取图像")
                    continue
                
                # 加载点云
                pcd = o3d.io.read_point_cloud(pointcloud_path)
                points = np.asarray(pcd.points, dtype=np.float64)
                
                if len(points) == 0:
                    print(f"{process_prefix}跳过帧 {i}: 点云为空")
                    continue
                
                # 过滤无效点
                if np.any(np.isnan(points)) or np.any(np.isinf(points)):
                    valid_mask = np.isfinite(points).all(axis=1)
                    points = points[valid_mask]
                
                # 加载GT标签并添加速度信息
                gt_objects = load_gt_labels_with_speed(data_root, frame_info)
                
                # 为GT对象添加计算出的速度
                for gt_obj in gt_objects:
                    track_id = gt_obj['track_id']
                    if track_id in gt_speeds and frame_id in gt_speeds[track_id]:
                        gt_obj['speed'] = gt_speeds[track_id][frame_id]
                
                # 运行预测检测
                timestamp = i * 0.1  # 假设10fps
                pred_objects, all_corners_3D, pts_3D, pts_2D, all_filtered_points_of_object, all_object_IDs = detector.process_frame(
                    image, points, calibration, timestamp, erosion_factor=25, depth_factor=20
                )
                
                # 绘制GT速度可视化
                gt_frame = draw_speed_visualization(
                    image, gt_objects, calibration, 
                    f"Ground Truth Speed - Frame {start_frame + i}", is_gt=True
                )
                
                # 绘制预测速度可视化
                pred_frame = draw_speed_visualization(
                    image, pred_objects, calibration, 
                    f"Predicted Speed - Frame {start_frame + i}", is_gt=False
                )
                
                # 创建鸟瞰图
                bev_frame = create_bev_with_enhanced_3d_boxes(points, gt_objects, pred_objects)
                
                # 添加BEV标题
                cv2.putText(bev_frame, f"Bird's Eye View - Frame {start_frame + i}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                gt_frames.append(gt_frame)
                pred_frames.append(pred_frame)
                bev_frames.append(bev_frame)
                
                # 记录速度统计
                gt_speeds_frame = [obj['speed'] for obj in gt_objects if obj['speed'] > 0]
                pred_speeds_frame = [obj['speed'] for obj in pred_objects if obj['speed'] > 0]
                
                speed_stats.append({
                    'frame_id': frame_id,
                    'gt_objects': len(gt_objects),
                    'pred_objects': len(pred_objects),
                    'gt_avg_speed': np.mean(gt_speeds_frame) if gt_speeds_frame else 0,
                    'pred_avg_speed': np.mean(pred_speeds_frame) if pred_speeds_frame else 0,
                    'gt_max_speed': np.max(gt_speeds_frame) if gt_speeds_frame else 0,
                    'pred_max_speed': np.max(pred_speeds_frame) if pred_speeds_frame else 0
                })
                
                # 定期清理内存
                if i % 10 == 0:
                    gc.collect()
                
            except Exception as e:
                print(f"{process_prefix}处理帧 {i} 时出错: {e}")
                continue
        
        print(f"{process_prefix}成功处理 {len(gt_frames)} 帧")
        
        if len(gt_frames) > 0 and len(pred_frames) > 0:
            # 创建单独的视频
            print(f"{process_prefix}创建GT速度视频...")
            gt_video_path = os.path.join(output_dir, "gt_speed_visualization.mp4")
            create_video(gt_frames, gt_video_path)
            
            print(f"{process_prefix}创建预测速度视频...")
            pred_video_path = os.path.join(output_dir, "pred_speed_visualization.mp4")
            create_video(pred_frames, pred_video_path)
            
            # 创建鸟瞰图视频
            if len(bev_frames) > 0:
                print(f"{process_prefix}创建鸟瞰图视频...")
                bev_video_path = os.path.join(output_dir, "bev_3d_boxes_visualization.mp4")
                create_video(bev_frames, bev_video_path)
            
            # 创建对比视频
            print(f"{process_prefix}创建速度对比视频...")
            combined_video_path = os.path.join(output_dir, "speed_comparison.mp4")
            create_combined_video(gt_frames, pred_frames, combined_video_path)
            
            # 创建三合一视频（GT + 预测 + BEV）
            if len(bev_frames) > 0:
                print(f"{process_prefix}创建三合一对比视频...")
                triple_video_path = os.path.join(output_dir, "triple_view_comparison.mp4")
                create_triple_combined_video(gt_frames, pred_frames, bev_frames, triple_video_path)
            
            # 保存速度统计
            stats_path = os.path.join(output_dir, "speed_statistics.json")
            with open(stats_path, 'w') as f:
                json.dump(speed_stats, f, indent=2, default=str)
            
            # 打印统计信息
            print(f"{process_prefix}=== 速度统计分析 ===")
            all_gt_speeds = [stat['gt_avg_speed'] for stat in speed_stats if stat['gt_avg_speed'] > 0]
            all_pred_speeds = [stat['pred_avg_speed'] for stat in speed_stats if stat['pred_avg_speed'] > 0]
            
            if all_gt_speeds:
                print(f"{process_prefix}GT平均速度: {np.mean(all_gt_speeds):.2f} m/s")
                print(f"{process_prefix}GT最大速度: {np.max(all_gt_speeds):.2f} m/s")
            
            if all_pred_speeds:
                print(f"{process_prefix}预测平均速度: {np.mean(all_pred_speeds):.2f} m/s")
                print(f"{process_prefix}预测最大速度: {np.max(all_pred_speeds):.2f} m/s")
            
            print(f"{process_prefix}结果保存在: {output_dir}")
            return True
        else:
            print(f"{process_prefix}没有成功处理任何帧")
            return False
    
    except Exception as e:
        print(f"{process_prefix}处理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_triple_combined_video(gt_frames, pred_frames, bev_frames, output_path, fps=10):
    """创建三合一对比视频（GT + 预测 + BEV）"""
    if len(gt_frames) == 0 or len(pred_frames) == 0 or len(bev_frames) == 0:
        print("没有足够的帧来创建三合一视频")
        return
    
    # 确保三个视频帧数相同
    min_frames = min(len(gt_frames), len(pred_frames), len(bev_frames))
    gt_frames = gt_frames[:min_frames]
    pred_frames = pred_frames[:min_frames]
    bev_frames = bev_frames[:min_frames]
    
    # 获取图像尺寸
    h, w = gt_frames[0].shape[:2]
    bev_h, bev_w = bev_frames[0].shape[:2]
    
    # 调整BEV图像尺寸以匹配相机图像高度
    bev_resized_w = int(bev_w * h / bev_h)
    
    # 创建组合视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    combined_writer = cv2.VideoWriter(output_path, fourcc, fps, (w * 2 + bev_resized_w, h))
    
    print(f"创建三合一视频: {output_path}")
    for i in tqdm(range(min_frames), desc="合并三合一视频帧"):
        # 调整BEV图像尺寸
        bev_resized = cv2.resize(bev_frames[i], (bev_resized_w, h))
        
        # 水平拼接三个帧
        combined_frame = np.hstack([gt_frames[i], pred_frames[i], bev_resized])
        combined_writer.write(combined_frame)
    
    combined_writer.release()
    print(f"三合一视频已保存: {output_path}")


def create_video(frames, output_path, fps=10):
    """创建视频"""
    if len(frames) == 0:
        print("没有帧来创建视频")
        return
    
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame in frames:
        video_writer.write(frame)
    
    video_writer.release()
    print(f"视频已保存: {output_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="V2X速度可视化")
    parser.add_argument('--data-root', type=str, 
                       default="/mnt/disk_2/yuji/voyagepro/yolo-laidar/data/V2X-Seq-SPD-Example/infrastructure-side",
                       help="数据集根目录")
    parser.add_argument('--data-info', type=str,
                       default="/mnt/disk_2/yuji/voyagepro/yolo-laidar/data/V2X-Seq-SPD-Example/infrastructure-side/data_info.json",
                       help="数据信息JSON文件")
    parser.add_argument('--output-dir', type=str, default="./speed_visualization_output_v2x",
                       help="输出目录")
    parser.add_argument('--start-frame', type=int, default=None,
                       help="开始帧索引")
    parser.add_argument('--end-frame', type=int, default=None,
                       help="结束帧索引")
    parser.add_argument('--max-frames', type=int, default=500,
                       help="最大处理帧数")
    
    args = parser.parse_args()
    
    print("V2X速度可视化开始...")
    print(f"数据根目录: {args.data_root}")
    print(f"输出目录: {args.output_dir}")
    print(f"最大帧数: {args.max_frames}")
    
    process_v2x_speed_visualization_with_device(
        data_root=args.data_root,
        data_info_path=args.data_info,
        output_dir=args.output_dir,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        max_frames=args.max_frames
    )


if __name__ == "__main__":
    main() 