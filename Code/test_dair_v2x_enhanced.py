#!/usr/bin/env python3
"""
DAIR-V2X数据集增强版多视角RGB+LiDAR融合检测测试脚本
支持基础设施侧和车辆侧两个视角的3D检测结果返回、人车距离计算
"""

import os
import cv2
import numpy as np
import json
import open3d as o3d
import time
from collections import Counter, deque
import pandas as pd
import argparse
from multiprocessing import Pool, set_start_method, Manager
from tqdm import tqdm
import torch
from PIL import Image
import gc
import traceback
import sys
from typing import Dict, List, Tuple, Optional, Any

# 导入必要的模块
from ultralytics import YOLO

# 设置多进程启动方式为 'spawn'
try:
    set_start_method('spawn')
except RuntimeError:
    pass

# 简化版融合函数（如果原模块不可用）
def lidar_camera_fusion(pts_3D, pts_2D, frame, mask, cls, calibration, erosion_factor=25, depth_factor=20, PCA=False):
    """简化的融合函数"""
    if len(pts_3D) == 0:
        return None
    
    if len(mask) < 3:
        return None
    
    mask_points = mask.astype(np.int32)
    x_min, y_min = np.min(mask_points, axis=0)
    x_max, y_max = np.max(mask_points, axis=0)
    
    # 过滤在掩码区域内的点
    mask_inside = []
    for i, pt_2d in enumerate(pts_2D):
        if x_min <= pt_2d[0] <= x_max and y_min <= pt_2d[1] <= y_max:
            mask_inside.append(i)
    
    if len(mask_inside) == 0:
        return None
        
    filtered_points = pts_3D[mask_inside]
    
    # 生成3D边界框
    min_coords = np.min(filtered_points, axis=0)
    max_coords = np.max(filtered_points, axis=0)
    
    # 8个角点
    corners_3D = np.array([
        [min_coords[0], min_coords[1], min_coords[2]],
        [max_coords[0], min_coords[1], min_coords[2]],
        [max_coords[0], max_coords[1], min_coords[2]],
        [min_coords[0], max_coords[1], min_coords[2]],
        [min_coords[0], min_coords[1], max_coords[2]],
        [max_coords[0], min_coords[1], max_coords[2]],
        [max_coords[0], max_coords[1], max_coords[2]],
        [min_coords[0], max_coords[1], max_coords[2]]
    ])
    
    return filtered_points, corners_3D, 0.0

def get_pred_bbox_edges(corners_2D):
    """生成边界框边缘"""
    if len(corners_2D) < 8:
        return []
    edges = [
        [corners_2D[0], corners_2D[1]], [corners_2D[1], corners_2D[2]],
        [corners_2D[2], corners_2D[3]], [corners_2D[3], corners_2D[0]],
        [corners_2D[4], corners_2D[5]], [corners_2D[5], corners_2D[6]],
        [corners_2D[6], corners_2D[7]], [corners_2D[7], corners_2D[4]],
        [corners_2D[0], corners_2D[4]], [corners_2D[1], corners_2D[5]],
        [corners_2D[2], corners_2D[6]], [corners_2D[3], corners_2D[7]]
    ]
    return edges

def assign_colors_by_depth(points):
    """根据深度分配颜色"""
    if len(points) == 0:
        return []
    depths = points[:, 2]
    normalized_depths = (depths - np.min(depths)) / (np.max(depths) - np.min(depths) + 1e-6)
    colors = []
    for depth in normalized_depths:
        color = [int(255 * (1 - depth)), int(255 * depth), 128]
        colors.append(color)
    return colors


class DAIRCalibration:
    """DAIR-V2X数据集标定类"""
    
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
            
            # 畸变参数（如果有的话）
            self.D = np.array(cam_data.get('cam_D', [0, 0, 0, 0, 0]), dtype=np.float64)
            
            # LiDAR到相机的旋转矩阵 (3x3)
            self.R = np.array(lidar_cam_data['rotation'], dtype=np.float64)
            
            # LiDAR到相机的平移向量 (3x1)
            self.t = np.array(lidar_cam_data['translation'], dtype=np.float64).reshape(3, 1)
            
            # 构建变换矩阵 (4x4)
            self.T = np.eye(4, dtype=np.float64)
            self.T[:3, :3] = self.R
            self.T[:3, 3:4] = self.t
            
            print(f"标定初始化成功")
            print(f"相机内参 K:\n{self.K}")
            print(f"旋转矩阵 R:\n{self.R}")
            print(f"平移向量 t:\n{self.t.flatten()}")
            
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
            
            valid_mask = np.isfinite(points_3D).all(axis=1)
            if not np.any(valid_mask):
                return np.array([]), np.array([])
            
            points_3D_valid = points_3D[valid_mask]
            points_3D_homo = np.hstack([points_3D_valid, np.ones((points_3D_valid.shape[0], 1), dtype=np.float64)])
            points_cam = (self.T @ points_3D_homo.T).T[:, :3]
            
            front_mask = points_cam[:, 2] > 0.1
            points_cam_valid = points_cam[front_mask]
            
            if len(points_cam_valid) == 0:
                return np.array([]), valid_mask
            
            points_2D_homo = (self.K @ points_cam_valid.T).T
            z_coords = points_2D_homo[:, 2:3]
            z_coords = np.where(np.abs(z_coords) < 1e-6, 1e-6, z_coords)
            points_2D = points_2D_homo[:, :2] / z_coords
            
            final_valid_mask = np.zeros(len(points_3D), dtype=bool)
            valid_indices = np.where(valid_mask)[0]
            front_indices = valid_indices[front_mask]
            final_valid_mask[front_indices] = True
            
            return points_2D, final_valid_mask
            
        except Exception as e:
            print(f"3D到2D投影失败: {e}")
            return np.array([]), np.array([])


class DAIREnhanced3DDetector:
    """DAIR-V2X增强版3D检测器"""
    
    def __init__(self, model_path="yolov8m-seg.pt", tracking=True, view_prefix=""):
        print(f"初始化YOLO模型: {model_path}")
        self.model = YOLO(model_path)
        self.model.overrides['conf'] = 0.1  # 降低置信度阈值
        self.model.overrides['iou'] = 0.5
        self.model.overrides['agnostic_nms'] = False
        self.model.overrides['max_det'] = 1000
        # 不限制类别，检测所有目标
        # self.model.overrides['classes'] = [0, 1, 2, 3, 5, 6, 7]  # person, bicycle, car, motorcycle, bus, truck
        
        self.tracking = tracking
        self.view_prefix = view_prefix  # 视角前缀，避免多线程ID冲突
        self.names = self.model.names
        self.colors = np.random.randint(0, 255, size=(80, 3), dtype='uint8')
        
        # 跟踪相关变量
        self.tracking_trajectories = {}
        self.object_metrics = {}
        
        # 类别定义
        self.person_classes = [0]  # person
        self.vehicle_classes = [1, 2, 3, 5, 6, 7]  # bicycle, car, motorcycle, bus, truck
    
    def calculate_3d_distance(self, corners_3D_1, corners_3D_2):
        """计算两个3D边界框之间的距离"""
        if corners_3D_1 is None or corners_3D_2 is None:
            return None
        
        center_1 = np.mean(corners_3D_1, axis=0)
        center_2 = np.mean(corners_3D_2, axis=0)
        distance = np.linalg.norm(center_1 - center_2)
        return distance
    
    def calculate_xy_distance(self, corners_3D_1, corners_3D_2):
        """计算两个3D边界框之间的X-Y平面距离（不包含Z坐标）"""
        if corners_3D_1 is None or corners_3D_2 is None:
            return None
        
        center_1 = np.mean(corners_3D_1, axis=0)
        center_2 = np.mean(corners_3D_2, axis=0)
        # 只使用X-Y坐标计算距离
        distance_xy = np.linalg.norm(center_1[:2] - center_2[:2])
        return distance_xy
    
    def calculate_person_vehicle_distances(self, detection_results):
        """计算人和车辆之间的距离（只使用X-Y坐标）"""
        persons = []
        vehicles = []
        
        # 分类检测结果
        for result in detection_results:
            if result['class'] in self.person_classes:
                persons.append(result)
            elif result['class'] in self.vehicle_classes:
                vehicles.append(result)
        
        # 计算人车距离（只使用X-Y坐标）
        person_vehicle_distances = []
        for person in persons:
            for vehicle in vehicles:
                distance_xy = self.calculate_xy_distance(
                    person['corners_3D'], 
                    vehicle['corners_3D']
                )
                if distance_xy is not None:
                    person_center = np.mean(person['corners_3D'], axis=0)
                    vehicle_center = np.mean(vehicle['corners_3D'], axis=0)
                    
                    person_vehicle_distances.append({
                        'person_id': person['id'],
                        'person_class': self.model.names[person['class']],
                        'vehicle_id': vehicle['id'],
                        'vehicle_class': self.model.names[vehicle['class']],
                        'distance_xy': distance_xy,  # 只使用X-Y距离
                        'person_center_xy': person_center[:2].tolist(),  # 只保存X-Y坐标
                        'vehicle_center_xy': vehicle_center[:2].tolist(),  # 只保存X-Y坐标
                        'person_center_3d': person_center.tolist(),  # 保留完整3D坐标用于其他用途
                        'vehicle_center_3d': vehicle_center.tolist()
                    })
        
        return person_vehicle_distances
    
    def process_frame_enhanced(self, frame, points, calibration, frame_id=0, fps=30) -> Dict[str, Any]:
        """增强版帧处理，返回完整的检测结果字典"""
        try:
            # YOLO推理
            if self.tracking:
                results = self.model.track(
                    source=frame,
                    verbose=False,
                    show=False,
                    persist=True,
                    tracker='bytetrack.yaml'
                )
            else:
                results = self.model.predict(
                    source=frame,
                    verbose=False,
                    show=False,
                )
            
            # 初始化返回结果
            frame_result = {
                'frame_id': frame_id,
                'timestamp': time.time(),
                'detections': [],
                'person_vehicle_distances': [],
                'statistics': {
                    'total_objects': 0,
                    'person_count': 0,
                    'vehicle_count': 0,
                    'min_person_vehicle_distance': None,
                    'avg_person_vehicle_distance': None
                },
                'visualization_data': {
                    'processed_frame': frame.copy(),
                    'bev_frame': None
                }
            }
            
            if not results or results[0] is None:
                print(f"帧 {frame_id}: YOLO没有返回结果")
                return frame_result
            
            r = results[0]
            boxes = r.boxes
            masks = r.masks
            
            # 只在有检测结果时输出信息
            if boxes is not None and len(boxes) > 0:
                print(f"帧 {frame_id}: 检测到 {len(boxes)} 个目标")
                if masks is not None:
                    print(f"帧 {frame_id}: 有 {len(masks)} 个分割掩码")
            
            if boxes is None or len(boxes) == 0:
                return frame_result
            
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
            detection_results = []
            
            for j, cls in enumerate(boxes.cls.tolist()):
                try:
                    conf = boxes.conf.tolist()[j] if boxes.conf is not None else 0.5
                    box_id = int(boxes.id.tolist()[j]) if boxes.id is not None and self.tracking else j
                    bbox_coords = boxes.xyxy[j].tolist()
                    
                    # 添加视角前缀避免多线程ID冲突
                    unique_id = f"{self.view_prefix}_{box_id}" if self.view_prefix else box_id
                    
                    # print(f"处理目标 {j}: 类别={int(cls)} ({self.model.names[int(cls)]}), 置信度={conf:.3f}, ID={unique_id}")
                    
                    # 融合处理
                    fusion_result = None
                    if masks is not None and j < len(masks.xy) and masks.xy[j].size > 0:
                        fusion_result = lidar_camera_fusion(
                            pts_3D, pts_2D, frame, masks.xy[j], int(cls),
                            calibration, erosion_factor=25, depth_factor=20, PCA=False
                        )
                    
                    # 如果没有掩码或融合失败，使用边界框进行3D估算
                    if fusion_result is None:
                        # 使用边界框内的点云进行3D估算
                        x1, y1, x2, y2 = bbox_coords
                        
                        # 找到边界框内的点云
                        if len(pts_2D) > 0:
                            bbox_mask = (
                                (pts_2D[:, 0] >= x1) & (pts_2D[:, 0] <= x2) &
                                (pts_2D[:, 1] >= y1) & (pts_2D[:, 1] <= y2)
                            )
                            bbox_points = pts_3D[bbox_mask]
                        else:
                            bbox_points = np.array([])
                        
                        # 如果边界框内有足够的点云，使用它们
                        if len(bbox_points) >= 5:
                            min_coords = np.min(bbox_points, axis=0)
                            max_coords = np.max(bbox_points, axis=0)
                            
                            corners_3D = np.array([
                                [min_coords[0], min_coords[1], min_coords[2]],
                                [max_coords[0], min_coords[1], min_coords[2]],
                                [max_coords[0], max_coords[1], min_coords[2]],
                                [min_coords[0], max_coords[1], min_coords[2]],
                                [min_coords[0], min_coords[1], max_coords[2]],
                                [max_coords[0], min_coords[1], max_coords[2]],
                                [max_coords[0], max_coords[1], max_coords[2]],
                                [min_coords[0], max_coords[1], max_coords[2]]
                            ])
                            
                            fusion_result = (bbox_points, corners_3D, 0.0)
                        else:
                            # 如果没有足够的点云，创建基于图像的简单3D估算
                            # 假设目标在地面上，使用典型的尺寸
                            bbox_center_x = (x1 + x2) / 2
                            bbox_center_y = (y1 + y2) / 2
                            bbox_width = x2 - x1
                            bbox_height = y2 - y1
                            
                            # 简单的深度估算（基于边界框大小）
                            estimated_depth = max(5.0, 1000.0 / max(bbox_width, bbox_height))
                            
                            # 使用相机参数进行反投影估算
                            if hasattr(calibration, 'K'):
                                # 简单的反投影
                                fx, fy = calibration.K[0,0], calibration.K[1,1]
                                cx, cy = calibration.K[0,2], calibration.K[1,2]
                                
                                # 估算3D位置
                                x_3d = (bbox_center_x - cx) * estimated_depth / fx
                                y_3d = (bbox_center_y - cy) * estimated_depth / fy
                                z_3d = estimated_depth
                                
                                # 估算3D尺寸
                                width_3d = bbox_width * estimated_depth / fx
                                height_3d = bbox_height * estimated_depth / fy
                                depth_3d = max(width_3d, height_3d) * 0.5  # 估算深度
                                
                                # 创建3D边界框
                                corners_3D = np.array([
                                    [x_3d - width_3d/2, y_3d - height_3d/2, z_3d - depth_3d/2],
                                    [x_3d + width_3d/2, y_3d - height_3d/2, z_3d - depth_3d/2],
                                    [x_3d + width_3d/2, y_3d + height_3d/2, z_3d - depth_3d/2],
                                    [x_3d - width_3d/2, y_3d + height_3d/2, z_3d - depth_3d/2],
                                    [x_3d - width_3d/2, y_3d - height_3d/2, z_3d + depth_3d/2],
                                    [x_3d + width_3d/2, y_3d - height_3d/2, z_3d + depth_3d/2],
                                    [x_3d + width_3d/2, y_3d + height_3d/2, z_3d + depth_3d/2],
                                    [x_3d - width_3d/2, y_3d + height_3d/2, z_3d + depth_3d/2]
                                ])
                                
                                fusion_result = (np.array([]), corners_3D, 0.0)
                    
                    if fusion_result is not None:
                        filtered_points, corners_3D, yaw = fusion_result
                        all_corners_3D.append(corners_3D)
                        
                        # 计算3D信息
                        center_3d = np.mean(corners_3D, axis=0)
                        dimensions_3d = np.ptp(corners_3D, axis=0)  # width, height, depth
                        distance_from_origin = np.linalg.norm(center_3d)
                        
                        # 计算速度
                        speed_3d = None
                        if self.tracking and unique_id is not None:
                            if unique_id not in self.tracking_trajectories:
                                self.tracking_trajectories[unique_id] = deque(maxlen=10)
                                self.object_metrics[unique_id] = {
                                    'distances': [], 
                                    'speeds': [], 
                                    'centers_3d': deque(maxlen=5)
                                }
                            
                            self.tracking_trajectories[unique_id].append(
                                ((bbox_coords[0] + bbox_coords[2])/2, (bbox_coords[1] + bbox_coords[3])/2)
                            )
                            self.object_metrics[unique_id]['centers_3d'].append(center_3d)
                            
                            # 计算3D速度
                            if len(self.object_metrics[unique_id]['centers_3d']) >= 2:
                                last_pos = self.object_metrics[unique_id]['centers_3d'][-1]
                                prev_pos = self.object_metrics[unique_id]['centers_3d'][-2]
                                distance_3d = np.linalg.norm(last_pos - prev_pos)
                                speed_ms = distance_3d / (1.0/fps)
                                speed_3d = speed_ms * 3.6  # 转换为km/h
                                
                                self.object_metrics[unique_id]['speeds'].append(speed_3d)
                                if len(self.object_metrics[unique_id]['speeds']) > 3:
                                    self.object_metrics[unique_id]['speeds'].pop(0)
                            
                            self.object_metrics[unique_id]['distances'].append(distance_from_origin)
                            if len(self.object_metrics[unique_id]['distances']) > 5:
                                self.object_metrics[unique_id]['distances'].pop(0)
                        
                        # 构建检测结果
                        detection_result = {
                            'id': unique_id,
                            'class': int(cls),
                            'class_name': self.model.names[int(cls)],
                            'confidence': conf,
                            'bbox_2d': bbox_coords,
                            'corners_3D': corners_3D.tolist(),
                            'center_3d': center_3d.tolist(),
                            'dimensions_3d': dimensions_3d.tolist(),
                            'distance_from_origin': distance_from_origin,
                            'speed_3d_kmh': speed_3d,
                            'filtered_points': filtered_points.tolist() if len(filtered_points) > 0 else [],
                            'yaw': yaw
                        }
                        
                        detection_results.append(detection_result)
                        
                except Exception as e:
                    continue
            
            # 计算人车距离
            person_vehicle_distances = self.calculate_person_vehicle_distances(detection_results)
            
            # 统计信息
            person_count = sum(1 for d in detection_results if d['class'] in self.person_classes)
            vehicle_count = sum(1 for d in detection_results if d['class'] in self.vehicle_classes)
            
            min_distance = None
            avg_distance = None
            if person_vehicle_distances:
                distances = [d['distance_xy'] for d in person_vehicle_distances]
                min_distance = min(distances)
                avg_distance = sum(distances) / len(distances)
            
            # 更新结果
            frame_result.update({
                'detections': detection_results,
                'person_vehicle_distances': person_vehicle_distances,
                'statistics': {
                    'total_objects': len(detection_results),
                    'person_count': person_count,
                    'vehicle_count': vehicle_count,
                    'min_person_vehicle_distance': min_distance,
                    'avg_person_vehicle_distance': avg_distance
                }
            })
            
            # 生成可视化
            result_frame = self.draw_enhanced_results(frame, detection_results, person_vehicle_distances, calibration)
            bev_frame = self.create_bev_view(points, all_corners_3D)
            
            frame_result['visualization_data'] = {
                'processed_frame': result_frame,
                'bev_frame': bev_frame
            }
            
            return frame_result
            
        except Exception as e:
            print(f"处理帧时出错: {e}")
            frame_result['error'] = str(e)
            return frame_result
    
    def draw_enhanced_results(self, image, detection_results, person_vehicle_distances, calibration):
        """绘制增强的检测结果"""
        result_image = image.copy()
        
        # 绘制检测结果
        for result in detection_results:
            try:
                # 绘制3D边界框
                corners_3D = np.array(result['corners_3D'])
                pred_corners_2D, _ = calibration.convert_3D_to_2D(corners_3D)
                
                # 根据类别选择颜色
                if result['class'] in self.person_classes:
                    color = (0, 255, 0)  # 绿色表示人
                else:
                    color = (0, 0, 255)  # 红色表示车辆
                
                if len(pred_corners_2D) >= 8:
                    pred_edges_2D = get_pred_bbox_edges(pred_corners_2D)
                    
                    for pred_edge in pred_edges_2D:
                        pt1 = tuple(np.int32(pred_edge[0]))
                        pt2 = tuple(np.int32(pred_edge[1]))
                        cv2.line(result_image, pt1, pt2, color, 2)
                else:
                    # 如果3D投影失败，绘制2D边界框
                    bbox = result['bbox_2d']
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
                
                # 绘制标签
                bbox = result['bbox_2d']
                label_parts = [
                    f"ID:{result['id']}", 
                    result['class_name'], 
                    f"{result['confidence']*100:.1f}%",
                    f"Dist:{result['distance_from_origin']:.1f}m"
                ]
                
                if result['speed_3d_kmh'] is not None:
                    label_parts.append(f"Speed:{result['speed_3d_kmh']:.1f}km/h")
                
                label = ' '.join(label_parts)
                
                # 绘制标签背景和文字
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                dim, baseline = text_size[0], text_size[1]
                
                cv2.rectangle(result_image, (int(bbox[0]), int(bbox[1]) - dim[1] - 10),
                            (int(bbox[0]) + dim[0] + 5, int(bbox[1]) - 5), (30, 30, 30), cv2.FILLED)
                
                cv2.putText(result_image, label, (int(bbox[0]) + 3, int(bbox[1]) - 8),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
            except Exception as e:
                continue
        
        # 绘制人车距离信息
        if person_vehicle_distances:
            y_offset = 30
            for i, dist_info in enumerate(person_vehicle_distances[:5]):  # 只显示前5个
                text = f"Person{dist_info['person_id']}-Vehicle{dist_info['vehicle_id']}: {dist_info['distance_xy']:.1f}m"
                cv2.putText(result_image, text, (10, y_offset + i*25), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # 绘制轨迹
        if self.tracking:
            for id_, trajectory in self.tracking_trajectories.items():
                if len(trajectory) > 1:
                    for i in range(1, len(trajectory)):
                        pt1 = (int(trajectory[i-1][0]), int(trajectory[i-1][1]))
                        pt2 = (int(trajectory[i][0]), int(trajectory[i][1]))
                        cv2.line(result_image, pt1, pt2, (255, 255, 255), 2)
        
        return result_image
    
    def create_bev_view(self, points, all_corners_3D, max_points=5000):
        """创建鸟瞰图视图"""
        try:
            if len(points) > max_points:
                indices = np.random.choice(len(points), max_points, replace=False)
                sample_points = points[indices]
            else:
                sample_points = points
            
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
            
            # 绘制3D边界框
            for corners_3D in all_corners_3D:
                bottom_indices = np.argsort(corners_3D[:, 2])[:4]
                bottom_corners = corners_3D[bottom_indices]
                
                bev_corners = []
                for corner in bottom_corners:
                    if x_range[0] <= corner[0] <= x_range[1] and y_range[0] <= corner[1] <= y_range[1]:
                        x_bev = int((corner[0] - x_range[0]) / resolution)
                        y_bev = int((corner[1] - y_range[0]) / resolution)
                        bev_corners.append([x_bev, height - 1 - y_bev])
                
                if len(bev_corners) >= 3:
                    bev_corners = np.array(bev_corners, dtype=np.int32)
                    cv2.polylines(bev_image, [bev_corners], True, (0, 255, 0), 2)
            
            # 添加网格和中心点
            grid_spacing = int(10 / resolution)
            for i in range(0, width, grid_spacing):
                cv2.line(bev_image, (i, 0), (i, height-1), (50, 50, 50), 1)
            for i in range(0, height, grid_spacing):
                cv2.line(bev_image, (0, i), (width-1, i), (50, 50, 50), 1)
            
            center_x, center_y = width // 2, height // 2
            cv2.circle(bev_image, (center_x, center_y), 5, (255, 255, 255), -1)
            
            return bev_image
            
        except Exception as e:
            return np.zeros((500, 500, 3), dtype=np.uint8)


def process_dair_view(view_path, view_name, max_frames=100, tracking=True, show=False):
    """处理单个DAIR-V2X视角"""
    print(f"\n=== 开始处理 {view_name} ===")
    
    try:
        # 检查数据结构
        image_dir = os.path.join(view_path, 'image')
        velodyne_dir = os.path.join(view_path, 'velodyne')
        calib_dir = os.path.join(view_path, 'calib')
        
        # 检查标定文件夹结构
        cam_intrinsic_dir = os.path.join(calib_dir, 'camera_intrinsic')
        
        # 判断是基础设施侧还是车辆侧
        if os.path.exists(os.path.join(calib_dir, 'virtuallidar_to_camera')):
            lidar_to_cam_dir = os.path.join(calib_dir, 'virtuallidar_to_camera')
            print(f"{view_name} 使用 virtuallidar_to_camera 标定")
        elif os.path.exists(os.path.join(calib_dir, 'lidar_to_camera')):
            lidar_to_cam_dir = os.path.join(calib_dir, 'lidar_to_camera')
            print(f"{view_name} 使用 lidar_to_camera 标定")
        else:
            raise ValueError(f"找不到LiDAR到相机的标定文件夹")
        
        # 获取文件列表
        image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
        velodyne_files = sorted([f for f in os.listdir(velodyne_dir) if f.endswith('.pcd')])
        
        print(f"找到 {len(image_files)} 个图像文件，{len(velodyne_files)} 个点云文件")
        
        # 限制处理帧数
        if max_frames:
            image_files = image_files[:max_frames]
            velodyne_files = velodyne_files[:max_frames]
        
        # 初始化检测器（使用第一帧的标定信息）
        if len(image_files) == 0:
            print(f"错误：{view_name} 中没有找到图像文件")
            return []
        
        # 获取第一帧的标定文件
        first_frame_id = os.path.splitext(image_files[0])[0]
        cam_intrinsic_path = os.path.join(cam_intrinsic_dir, f"{first_frame_id}.json")
        lidar_to_cam_path = os.path.join(lidar_to_cam_dir, f"{first_frame_id}.json")
        
        if not os.path.exists(cam_intrinsic_path) or not os.path.exists(lidar_to_cam_path):
            print(f"错误：找不到帧 {first_frame_id} 的标定文件")
            return []
        
        # 初始化标定和检测器
        calibration = DAIRCalibration(cam_intrinsic_path, lidar_to_cam_path)
        detector = DAIREnhanced3DDetector(tracking=tracking, view_prefix=view_name)
        
        # 创建输出目录
        output_dir = f'./output/{view_name}'
        os.makedirs(output_dir, exist_ok=True)
        
        # 存储每帧结果
        frame_results = []
        processed_frames = []
        bev_frames = []
        
        print(f"开始处理 {len(image_files)} 帧...")
        
        # 处理每一帧
        for i, (img_file, pcd_file) in enumerate(tqdm(zip(image_files, velodyne_files), 
                                                    desc=f"处理{view_name}", 
                                                    total=len(image_files))):
            try:
                # 加载图像
                img_path = os.path.join(image_dir, img_file)
                image = cv2.imread(img_path)
                if image is None:
                    print(f"跳过帧 {i}: 无法读取图像 {img_path}")
                    continue
                
                # 加载点云
                pcd_path = os.path.join(velodyne_dir, pcd_file)
                pcd = o3d.io.read_point_cloud(pcd_path)
                points = np.asarray(pcd.points, dtype=np.float64)
                
                if len(points) == 0:
                    print(f"跳过帧 {i}: 点云为空")
                    continue
                
                # 过滤无效点
                if np.any(np.isnan(points)) or np.any(np.isinf(points)):
                    valid_mask = np.isfinite(points).all(axis=1)
                    points = points[valid_mask]
                
                # 处理帧
                frame_result = detector.process_frame_enhanced(
                    image, points, calibration, i, fps=10
                )
                
                # 添加视角信息
                frame_result['view_name'] = view_name
                frame_result['image_file'] = img_file
                frame_result['pcd_file'] = pcd_file
                frame_results.append(frame_result)
                
                # 获取可视化结果
                result_frame = frame_result['visualization_data']['processed_frame']
                bev_frame = frame_result['visualization_data']['bev_frame']
                
                # 添加帧信息
                info_text = f'{view_name} Frame: {i} | Objects: {frame_result["statistics"]["total_objects"]} | P-V: {len(frame_result["person_vehicle_distances"])}'
                cv2.putText(result_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(bev_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                processed_frames.append(result_frame)
                bev_frames.append(bev_frame)
                
                # 显示视频
                if show:
                    cv2.imshow(f"{view_name}_Detection", result_frame)
                    cv2.imshow(f"{view_name}_BEV", bev_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # 定期清理内存
                if i % 10 == 0:
                    gc.collect()
                
            except Exception as e:
                print(f"处理帧 {i} 时出错: {e}")
                continue
        
        if show:
            cv2.destroyAllWindows()
        
        # 保存结果
        if frame_results:
            # 保存JSON结果
            results_file = os.path.join(output_dir, f'{view_name}_results.json')
            with open(results_file, 'w') as f:
                json.dump(frame_results, f, indent=2, default=str)
            
            # 创建视频
            if processed_frames:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                
                # 检测视频
                det_video_path = os.path.join(output_dir, f'{view_name}_detection.mp4')
                height, width = processed_frames[0].shape[:2]
                out_det = cv2.VideoWriter(det_video_path, fourcc, 10, (width, height))
                for frame in processed_frames:
                    out_det.write(frame)
                out_det.release()
                
                # BEV视频
                bev_video_path = os.path.join(output_dir, f'{view_name}_bev.mp4')
                height, width = bev_frames[0].shape[:2]
                out_bev = cv2.VideoWriter(bev_video_path, fourcc, 10, (width, height))
                for frame in bev_frames:
                    out_bev.write(frame)
                out_bev.release()
                
                print(f"\n{view_name} 处理完成:")
                print(f"  - 结果文件: {results_file}")
                print(f"  - 检测视频: {det_video_path}")
                print(f"  - BEV视频: {bev_video_path}")
            
            # 统计信息
            total_detections = sum(len(fr['detections']) for fr in frame_results)
            total_person_vehicle_pairs = sum(len(fr['person_vehicle_distances']) for fr in frame_results)
            
            print(f"  - 总帧数: {len(frame_results)}")
            print(f"  - 总检测数: {total_detections}")
            print(f"  - 人车距离对: {total_person_vehicle_pairs}")
            
            if total_person_vehicle_pairs > 0:
                all_distances = []
                for fr in frame_results:
                    all_distances.extend([d['distance_xy'] for d in fr['person_vehicle_distances']])
                
                if all_distances:
                    print(f"  - 人车距离统计:")
                    print(f"    最小: {min(all_distances):.2f}m")
                    print(f"    最大: {max(all_distances):.2f}m")
                    print(f"    平均: {sum(all_distances)/len(all_distances):.2f}m")
        
        # 清理
        del detector
        gc.collect()
        
        return frame_results
        
    except Exception as e:
        print(f"处理 {view_name} 时出错: {e}")
        traceback.print_exc()
        return []


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="DAIR-V2X数据集增强版多视角3D检测测试")
    parser.add_argument('--infrastructure-path', type=str, 
                       default="/mnt/disk_2/yuji/voyagepro/yolo-laidar/data/V2X-Seq-SPD-Example/infrastructure-side/",
                       help="基础设施侧数据路径")
    parser.add_argument('--vehicle-path', type=str,
                       default="/mnt/disk_2/yuji/voyagepro/yolo-laidar/data/V2X-Seq-SPD-Example/infrastructure-side/",
                       help="车辆侧数据路径")
    parser.add_argument('--max-frames', type=int, default=50,
                       help="每个视角最大处理帧数")
    parser.add_argument('--no-tracking', action='store_true',
                       help="禁用目标跟踪")
    parser.add_argument('--show', action='store_true',
                       help="显示处理过程")
    
    args = parser.parse_args()
    
    print("=== DAIR-V2X数据集增强版多视角3D检测测试 ===")
    print(f"基础设施侧路径: {args.infrastructure_path}")
    print(f"车辆侧路径: {args.vehicle_path}")
    print(f"每视角最大帧数: {args.max_frames}")
    print(f"跟踪模式: {'关闭' if args.no_tracking else '开启'}")
    
    # 检查路径存在性
    if not os.path.exists(args.infrastructure_path):
        print(f"错误：基础设施侧路径不存在: {args.infrastructure_path}")
        return
    
    if not os.path.exists(args.vehicle_path):
        print(f"错误：车辆侧路径不存在: {args.vehicle_path}")
        return
    
    # 创建输出目录
    os.makedirs('./output', exist_ok=True)
    
    # 处理两个视角
    all_results = {}
    
    # 使用多线程并行处理两个视角
    print("\n=== 开始多线程并行处理两个视角 ===")
    
    # 准备处理参数
    process_args = [
        (args.infrastructure_path, 'infrastructure-side', args.max_frames, not args.no_tracking, False),
        (args.vehicle_path, 'vehicle-side', args.max_frames, not args.no_tracking, False)
    ]
    
    # 如果用户要求显示，则顺序处理
    if args.show:
        print("检测到--show参数，将顺序处理以支持显示功能")
        # 处理基础设施侧
        infra_results = process_dair_view(
            args.infrastructure_path, 
            'infrastructure-side', 
            max_frames=args.max_frames,
            tracking=not args.no_tracking,
            show=True
        )
        all_results['infrastructure-side'] = infra_results
        
        # 处理车辆侧
        vehicle_results = process_dair_view(
            args.vehicle_path, 
            'vehicle-side', 
            max_frames=args.max_frames,
            tracking=not args.no_tracking,
            show=True
        )
        all_results['vehicle-side'] = vehicle_results
    else:
        # 多线程并行处理
        print("使用多线程并行处理两个视角...")
        with Pool(processes=2) as pool:
            results = pool.starmap(process_dair_view, process_args)
        
        # 整理结果
        all_results['infrastructure-side'] = results[0]
        all_results['vehicle-side'] = results[1]
    
    # 保存汇总结果
    summary_file = './output/dair_v2x_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n=== 所有视角处理完成 ===")
    print(f"汇总结果保存到: {summary_file}")
    
    # 打印总体统计
    for view_name, results in all_results.items():
        if results:
            total_frames = len(results)
            total_detections = sum(len(fr['detections']) for fr in results)
            total_pv_distances = sum(len(fr['person_vehicle_distances']) for fr in results)
            
            print(f"\n{view_name} 总结:")
            print(f"  总帧数: {total_frames}")
            print(f"  总检测数: {total_detections}")
            print(f"  人车距离计算次数: {total_pv_distances}")


if __name__ == "__main__":
    main() 