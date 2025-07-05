#!/usr/bin/env python3
"""
V2X数据集评估脚本
比较预测结果和真实标签，分析融合失败原因
增强版：专注于3D检测框精度评估
"""

import json
import cv2
import numpy as np
import open3d as o3d
import os
import sys
import gc
import traceback
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.spatial.distance import cdist

# 导入必要的模块
from ultralytics import YOLO
from improved_fusion import improved_lidar_camera_fusion
from utils import *
from visualization import *


class V2XEvaluationCalibration:
    """V2X评估标定类"""
    
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
            self.D = np.array(cam_data['cam_D'], dtype=np.float64)
            
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
    
    def convert_3D_to_camera_coords(self, pts_3d_lidar):
        """将3D LiDAR点转换为相机坐标系"""
        if len(pts_3d_lidar) == 0:
            return np.array([])
        
        pts_3d_lidar = np.asarray(pts_3d_lidar)
        
        if pts_3d_lidar.ndim != 2 or pts_3d_lidar.shape[1] != 3:
            return np.array([])
        
        if np.any(np.isnan(pts_3d_lidar)) or np.any(np.isinf(pts_3d_lidar)):
            valid_mask = np.isfinite(pts_3d_lidar).all(axis=1)
            pts_3d_lidar = pts_3d_lidar[valid_mask]
            if len(pts_3d_lidar) == 0:
                return np.array([])
        
        try:
            pts_3d_homo = np.hstack((pts_3d_lidar, np.ones((pts_3d_lidar.shape[0], 1))))
            pts_cam_homo = (self.T @ pts_3d_homo.T).T
            return pts_cam_homo[:, :3]
        except Exception as e:
            print(f"坐标转换时出错: {e}")
            return np.array([])
    
    def project_3D_to_2D(self, pts_3d_camera):
        """将相机坐标系下的3D点投影到2D图像"""
        if len(pts_3d_camera) == 0:
            return np.array([])
        
        pts_2d_homo = (self.K @ pts_3d_camera.T).T
        pts_2d = pts_2d_homo[:, :2] / pts_2d_homo[:, 2:3]
        return pts_2d


class V2XEvaluationDetector:
    """V2X评估检测器"""
    
    def __init__(self, model_path, tracking=False):
        self.model = YOLO(model_path)
        self.tracking = tracking
        self.fusion_stats = {
            'total_detections': 0,
            'successful_fusions': 0,
            'failure_reasons': defaultdict(int)
        }
    
    def process_frame(self, frame, points, calibration, erosion_factor=25, depth_factor=20):
        """处理单帧数据，进行3D检测"""
        try:
            # YOLO推理
            if self.tracking:
                results = self.model.track(
                    source=frame,
                    classes=[0, 1, 2, 3, 5, 6, 7],
                    verbose=False,
                    show=False,
                    persist=True,
                    tracker='bytetrack.yaml'
                )
            else:
                results = self.model.predict(
                    source=frame,
                    classes=[0, 1, 2, 3, 5, 6, 7],
                    verbose=False,
                    show=False,
                )
            
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
            objects3d_data = []
            detection_results = []
            
            if boxes is None or len(boxes) == 0:
                return detection_results, objects3d_data, all_corners_3D, pts_3D, pts_2D, all_filtered_points_of_object, all_object_IDs
            
            for j, cls in enumerate(boxes.cls.tolist()):
                try:
                    conf = boxes.conf.tolist()[j] if boxes.conf is not None else None
                    box_id = int(boxes.id.tolist()[j]) if boxes.id is not None else j
                    
                    # 获取2D边界框
                    xyxy = boxes.xyxy[j].cpu().numpy()
                    
                    # 记录检测结果
                    detection_results.append({
                        'class': int(cls),
                        'confidence': conf,
                        'bbox_2d': xyxy,
                        'id': box_id
                    })
                    
                    self.fusion_stats['total_detections'] += 1
                    all_object_IDs.append(box_id)
                    
                    # 检查掩码
                    if masks is None or j >= len(masks.xy) or masks.xy[j].size == 0:
                        self.fusion_stats['failure_reasons']['no_mask'] += 1
                        continue
                    
                    # 融合处理
                    fusion_result = improved_lidar_camera_fusion(
                        pts_3D, pts_2D, frame, masks.xy[j], int(cls), 
                        calibration, erosion_factor=erosion_factor, 
                        depth_factor=depth_factor, PCA=False
                    )
                    
                    if fusion_result is not None:
                        filtered_points_of_object, corners_3D, yaw = fusion_result
                        all_corners_3D.append(corners_3D)
                        all_filtered_points_of_object.append(filtered_points_of_object)
                        self.fusion_stats['successful_fusions'] += 1
                        
                        # 计算目标信息
                        ROS_type = int(np.int32(cls))
                        bottom_indices = np.argsort(corners_3D[:, 2])[:4]
                        ROS_ground_center = np.mean(corners_3D[bottom_indices], axis=0)
                        ROS_dimensions = np.ptp(corners_3D, axis=0)
                        ROS_points = corners_3D
                        
                        objects3d_data.append([ROS_type, ROS_ground_center, ROS_dimensions, ROS_points])
                        
                        # 更新检测结果
                        detection_results[j]['fusion_success'] = True
                        detection_results[j]['3d_center'] = ROS_ground_center
                        detection_results[j]['3d_dimensions'] = ROS_dimensions
                        detection_results[j]['3d_corners'] = corners_3D
                    else:
                        self.fusion_stats['failure_reasons']['fusion_failed'] += 1
                        detection_results[j]['fusion_success'] = False
                        
                except Exception as e:
                    self.fusion_stats['failure_reasons']['processing_error'] += 1
                    continue
            
            return detection_results, objects3d_data, all_corners_3D, pts_3D, pts_2D, all_filtered_points_of_object, all_object_IDs
            
        except Exception as e:
            print(f"处理帧时出错: {e}")
            traceback.print_exc()
            return [], [], [], np.array([]), np.array([]), [], []


def load_v2x_frame_data(data_root, data_info_path, frame_id):
    """加载指定帧的V2X数据"""
    try:
        # 加载数据信息
        with open(data_info_path, 'r') as f:
            data_info = json.load(f)
        
        # 查找指定帧
        frame_info = None
        for info in data_info:
            if info['frame_id'] == frame_id:
                frame_info = info
                break
        
        if frame_info is None:
            print(f"找不到帧ID: {frame_id}")
            return None
        
        # 构建文件路径
        image_path = os.path.join(data_root, frame_info['image_path'])
        pointcloud_path = os.path.join(data_root, frame_info['pointcloud_path'])
        camera_intrinsic_path = os.path.join(data_root, frame_info['calib_camera_intrinsic_path'])
        lidar_to_camera_path = os.path.join(data_root, frame_info['calib_virtuallidar_to_camera_path'])
        
        # 标签路径
        camera_label_path = os.path.join(data_root, f"label/camera/{frame_id}.json")
        lidar_label_path = os.path.join(data_root, f"label/virtuallidar/{frame_id}.json")
        
        # 检查文件存在性
        for path in [image_path, pointcloud_path, camera_intrinsic_path, lidar_to_camera_path, camera_label_path, lidar_label_path]:
            if not os.path.exists(path):
                print(f"文件不存在: {path}")
                return None
        
        # 加载图像
        image = cv2.imread(image_path)
        if image is None:
            print("无法读取图像")
            return None
        
        # 加载点云
        pcd = o3d.io.read_point_cloud(pointcloud_path)
        points = np.asarray(pcd.points, dtype=np.float64)
        
        if len(points) == 0:
            print("点云为空")
            return None
        
        # 过滤无效点
        if np.any(np.isnan(points)) or np.any(np.isinf(points)):
            valid_mask = np.isfinite(points).all(axis=1)
            points = points[valid_mask]
        
        # 创建标定对象
        calibration = V2XEvaluationCalibration(camera_intrinsic_path, lidar_to_camera_path)
        
        # 加载标签
        with open(camera_label_path, 'r') as f:
            camera_labels = json.load(f)
        
        with open(lidar_label_path, 'r') as f:
            lidar_labels = json.load(f)
        
        return {
            'image': image,
            'points': points,
            'calibration': calibration,
            'frame_info': frame_info,
            'camera_labels': camera_labels,
            'lidar_labels': lidar_labels
        }
        
    except Exception as e:
        print(f"数据加载失败: {e}")
        traceback.print_exc()
        return None


def convert_v2x_type_to_yolo_class(v2x_type):
    """将V2X类型转换为YOLO类别"""
    type_mapping = {
        'Pedestrian': 0,
        'Cyclist': 1,
        'Car': 2,
        'Motorcyclist': 3,
        'Van': 5,  # 将Van映射为公交车类别
        'Truck': 6,
        'Train': 7
    }
    return type_mapping.get(v2x_type, -1)


def calculate_iou_2d(box1, box2):
    """计算2D边界框的IoU"""
    # box格式: [xmin, ymin, xmax, ymax]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def calculate_distance_3d(center1, center2):
    """计算3D中心点距离"""
    return np.linalg.norm(np.array(center1) - np.array(center2))


def calculate_3d_iou(corners1, corners2):
    """计算3D边界框的IoU"""
    try:
        # 将角点转换为numpy数组
        corners1 = np.array(corners1)
        corners2 = np.array(corners2)
        
        if corners1.shape != (8, 3) or corners2.shape != (8, 3):
            return 0.0
        
        # 计算每个边界框的最小和最大坐标
        min1, max1 = np.min(corners1, axis=0), np.max(corners1, axis=0)
        min2, max2 = np.min(corners2, axis=0), np.max(corners2, axis=0)
        
        # 计算交集的最小和最大坐标
        inter_min = np.maximum(min1, min2)
        inter_max = np.minimum(max1, max2)
        
        # 检查是否有交集
        if np.any(inter_min >= inter_max):
            return 0.0
        
        # 计算交集体积
        inter_volume = np.prod(inter_max - inter_min)
        
        # 计算各自体积
        volume1 = np.prod(max1 - min1)
        volume2 = np.prod(max2 - min2)
        
        # 计算并集体积
        union_volume = volume1 + volume2 - inter_volume
        
        # 计算IoU
        iou = inter_volume / union_volume if union_volume > 0 else 0.0
        return iou
        
    except Exception as e:
        print(f"计算3D IoU时出错: {e}")
        return 0.0


def calculate_orientation_error(corners1, corners2):
    """计算3D边界框的朝向误差"""
    try:
        corners1 = np.array(corners1)
        corners2 = np.array(corners2)
        
        if corners1.shape != (8, 3) or corners2.shape != (8, 3):
            return float('inf')
        
        # 计算主方向向量（通过PCA或简单的边向量）
        # 这里使用简化方法：计算最长边的方向
        def get_main_direction(corners):
            # 计算所有边的长度和方向
            edges = []
            for i in range(4):  # 底面的4条边
                edge = corners[(i+1)%4] - corners[i]
                edges.append(edge)
            
            # 找到最长的边
            edge_lengths = [np.linalg.norm(edge) for edge in edges]
            max_idx = np.argmax(edge_lengths)
            main_edge = edges[max_idx]
            
            # 归一化
            return main_edge / np.linalg.norm(main_edge)
        
        dir1 = get_main_direction(corners1)
        dir2 = get_main_direction(corners2)
        
        # 计算角度差（考虑方向可能相反）
        dot_product = np.abs(np.dot(dir1, dir2))
        dot_product = np.clip(dot_product, -1.0, 1.0)
        angle_error = np.arccos(dot_product) * 180 / np.pi
        
        return min(angle_error, 180 - angle_error)
        
    except Exception as e:
        print(f"计算朝向误差时出错: {e}")
        return float('inf')


def calculate_dimension_error(dims1, dims2):
    """计算3D尺寸误差"""
    try:
        dims1 = np.array(dims1)
        dims2 = np.array(dims2)
        
        # 计算相对误差
        relative_errors = np.abs(dims1 - dims2) / (dims2 + 1e-6)
        
        return {
            'length_error': relative_errors[0],
            'width_error': relative_errors[1], 
            'height_error': relative_errors[2],
            'mean_error': np.mean(relative_errors),
            'max_error': np.max(relative_errors)
        }
        
    except Exception as e:
        print(f"计算尺寸误差时出错: {e}")
        return {
            'length_error': float('inf'),
            'width_error': float('inf'),
            'height_error': float('inf'),
            'mean_error': float('inf'),
            'max_error': float('inf')
        }


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


def evaluate_3d_detection_accuracy(detection_results, camera_labels, lidar_labels, 
                                 iou_2d_threshold=0.5, iou_3d_threshold=0.25, 
                                 distance_threshold=5.0):
    """评估3D检测精度"""
    
    evaluation_results = {
        'matches_3d': [],
        'false_positives_3d': [],
        'false_negatives_3d': [],
        'statistics_3d': {
            'total_gt': 0,
            'total_pred_with_3d': 0,
            'true_positives_3d': 0,
            'false_positives_3d': 0,
            'false_negatives_3d': 0
        },
        'accuracy_metrics': {
            'position_errors': [],
            'dimension_errors': [],
            'orientation_errors': [],
            'iou_3d_scores': []
        }
    }
    
    # 转换GT标签格式，包含3D角点
    gt_objects_3d = []
    for i, (cam_label, lidar_label) in enumerate(zip(camera_labels, lidar_labels)):
        yolo_class = convert_v2x_type_to_yolo_class(cam_label['type'])
        if yolo_class >= 0:
            center_3d = [
                lidar_label['3d_location']['x'],
                lidar_label['3d_location']['y'],
                lidar_label['3d_location']['z']
            ]
            dimensions_3d = [
                lidar_label['3d_dimensions']['l'],
                lidar_label['3d_dimensions']['w'],
                lidar_label['3d_dimensions']['h']
            ]
            
            # 获取旋转角度（如果有）
            rotation_y = lidar_label.get('3d_rotation', {}).get('y', 0)
            
            # 生成3D角点
            corners_3d = convert_gt_to_3d_corners(center_3d, dimensions_3d, rotation_y)
            
            gt_objects_3d.append({
                'index': i,
                'class': yolo_class,
                'type': cam_label['type'],
                'bbox_2d': [
                    cam_label['2d_box']['xmin'],
                    cam_label['2d_box']['ymin'],
                    cam_label['2d_box']['xmax'],
                    cam_label['2d_box']['ymax']
                ],
                'center_3d': center_3d,
                'dimensions_3d': dimensions_3d,
                'corners_3d': corners_3d,
                'rotation_y': rotation_y
            })
    
    evaluation_results['statistics_3d']['total_gt'] = len(gt_objects_3d)
    
    # 筛选出有3D信息的预测结果
    predictions_with_3d = [pred for pred in detection_results if pred.get('fusion_success', False)]
    evaluation_results['statistics_3d']['total_pred_with_3d'] = len(predictions_with_3d)
    
    # 匹配预测和GT（基于3D信息）
    used_gt_3d = set()
    
    for pred in predictions_with_3d:
        if not pred.get('fusion_success', False) or '3d_corners' not in pred:
            continue
            
        best_match = None
        best_score = 0
        best_metrics = {}
        
        for gt in gt_objects_3d:
            if gt['index'] in used_gt_3d:
                continue
            
            # 类别必须匹配
            if pred['class'] != gt['class']:
                continue
            
            # 计算2D IoU（作为初步筛选）
            iou_2d = calculate_iou_2d(pred['bbox_2d'], gt['bbox_2d'])
            if iou_2d < iou_2d_threshold:
                continue
            
            # 计算3D指标
            distance_3d = calculate_distance_3d(pred['3d_center'], gt['center_3d'])
            iou_3d = calculate_3d_iou(pred['3d_corners'], gt['corners_3d'])
            orientation_error = calculate_orientation_error(pred['3d_corners'], gt['corners_3d'])
            dimension_error = calculate_dimension_error(pred['3d_dimensions'], gt['dimensions_3d'])
            
            # 综合评分（主要基于3D IoU）
            score = iou_3d
            if distance_3d < distance_threshold:
                score += 0.2
            if orientation_error < 30:  # 朝向误差小于30度
                score += 0.1
            
            if score > best_score and iou_3d >= iou_3d_threshold:
                best_score = score
                best_match = gt
                best_metrics = {
                    'iou_2d': iou_2d,
                    'iou_3d': iou_3d,
                    'distance_3d': distance_3d,
                    'orientation_error': orientation_error,
                    'dimension_error': dimension_error
                }
        
        if best_match:
            used_gt_3d.add(best_match['index'])
            evaluation_results['matches_3d'].append({
                'prediction': pred,
                'ground_truth': best_match,
                'metrics': best_metrics
            })
            evaluation_results['statistics_3d']['true_positives_3d'] += 1
            
            # 收集精度指标
            evaluation_results['accuracy_metrics']['position_errors'].append(best_metrics['distance_3d'])
            evaluation_results['accuracy_metrics']['orientation_errors'].append(best_metrics['orientation_error'])
            evaluation_results['accuracy_metrics']['dimension_errors'].append(best_metrics['dimension_error'])
            evaluation_results['accuracy_metrics']['iou_3d_scores'].append(best_metrics['iou_3d'])
        else:
            evaluation_results['false_positives_3d'].append(pred)
            evaluation_results['statistics_3d']['false_positives_3d'] += 1
    
    # 未匹配的GT为假阴性
    for gt in gt_objects_3d:
        if gt['index'] not in used_gt_3d:
            evaluation_results['false_negatives_3d'].append(gt)
            evaluation_results['statistics_3d']['false_negatives_3d'] += 1
    
    return evaluation_results


def analyze_3d_detection_accuracy(evaluation_results_3d):
    """分析3D检测精度"""
    print("\n" + "="*50)
    print("3D检测精度详细分析")
    print("="*50)
    
    stats = evaluation_results_3d['statistics_3d']
    metrics = evaluation_results_3d['accuracy_metrics']
    
    # 基本统计
    print(f"\n=== 3D检测基本统计 ===")
    print(f"GT总数: {stats['total_gt']}")
    print(f"有3D信息的预测数: {stats['total_pred_with_3d']}")
    print(f"3D真阳性: {stats['true_positives_3d']}")
    print(f"3D假阳性: {stats['false_positives_3d']}")
    print(f"3D假阴性: {stats['false_negatives_3d']}")
    
    # 计算3D检测指标
    if stats['true_positives_3d'] + stats['false_positives_3d'] > 0:
        precision_3d = stats['true_positives_3d'] / (stats['true_positives_3d'] + stats['false_positives_3d'])
    else:
        precision_3d = 0.0
        
    if stats['true_positives_3d'] + stats['false_negatives_3d'] > 0:
        recall_3d = stats['true_positives_3d'] / (stats['true_positives_3d'] + stats['false_negatives_3d'])
    else:
        recall_3d = 0.0
        
    if precision_3d + recall_3d > 0:
        f1_3d = 2 * precision_3d * recall_3d / (precision_3d + recall_3d)
    else:
        f1_3d = 0.0
    
    print(f"\n=== 3D检测性能指标 ===")
    print(f"3D精确率: {precision_3d:.3f}")
    print(f"3D召回率: {recall_3d:.3f}")
    print(f"3D F1分数: {f1_3d:.3f}")
    
    # 精度分析
    if len(metrics['position_errors']) > 0:
        pos_errors = np.array(metrics['position_errors'])
        ori_errors = np.array(metrics['orientation_errors'])
        iou_3d_scores = np.array(metrics['iou_3d_scores'])
        
        print(f"\n=== 3D位置精度分析 ===")
        print(f"平均位置误差: {np.mean(pos_errors):.3f}m")
        print(f"位置误差中位数: {np.median(pos_errors):.3f}m")
        print(f"位置误差标准差: {np.std(pos_errors):.3f}m")
        print(f"最大位置误差: {np.max(pos_errors):.3f}m")
        print(f"位置误差 < 1m 的比例: {np.sum(pos_errors < 1.0) / len(pos_errors) * 100:.1f}%")
        print(f"位置误差 < 2m 的比例: {np.sum(pos_errors < 2.0) / len(pos_errors) * 100:.1f}%")
        
        print(f"\n=== 3D朝向精度分析 ===")
        print(f"平均朝向误差: {np.mean(ori_errors):.1f}°")
        print(f"朝向误差中位数: {np.median(ori_errors):.1f}°")
        print(f"朝向误差标准差: {np.std(ori_errors):.1f}°")
        print(f"朝向误差 < 10° 的比例: {np.sum(ori_errors < 10.0) / len(ori_errors) * 100:.1f}%")
        print(f"朝向误差 < 30° 的比例: {np.sum(ori_errors < 30.0) / len(ori_errors) * 100:.1f}%")
        
        print(f"\n=== 3D IoU分析 ===")
        print(f"平均3D IoU: {np.mean(iou_3d_scores):.3f}")
        print(f"3D IoU中位数: {np.median(iou_3d_scores):.3f}")
        print(f"3D IoU标准差: {np.std(iou_3d_scores):.3f}")
        print(f"3D IoU > 0.5 的比例: {np.sum(iou_3d_scores > 0.5) / len(iou_3d_scores) * 100:.1f}%")
        print(f"3D IoU > 0.7 的比例: {np.sum(iou_3d_scores > 0.7) / len(iou_3d_scores) * 100:.1f}%")
        
        # 尺寸精度分析
        print(f"\n=== 3D尺寸精度分析 ===")
        dim_errors = metrics['dimension_errors']
        if len(dim_errors) > 0:
            length_errors = [d['length_error'] for d in dim_errors if d['length_error'] != float('inf')]
            width_errors = [d['width_error'] for d in dim_errors if d['width_error'] != float('inf')]
            height_errors = [d['height_error'] for d in dim_errors if d['height_error'] != float('inf')]
            
            if length_errors:
                print(f"长度相对误差: {np.mean(length_errors)*100:.1f}% ± {np.std(length_errors)*100:.1f}%")
            if width_errors:
                print(f"宽度相对误差: {np.mean(width_errors)*100:.1f}% ± {np.std(width_errors)*100:.1f}%")
            if height_errors:
                print(f"高度相对误差: {np.mean(height_errors)*100:.1f}% ± {np.std(height_errors)*100:.1f}%")
    
    # 按类别分析
    print(f"\n=== 按类别3D检测精度 ===")
    class_stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0, 'pos_errors': [], 'iou_3d': []})
    class_names = {0: "行人", 1: "骑行者", 2: "汽车", 3: "摩托车", 5: "公交车", 6: "卡车", 7: "火车"}
    
    # 统计匹配结果
    for match in evaluation_results_3d['matches_3d']:
        cls = match['prediction']['class']
        class_stats[cls]['tp'] += 1
        class_stats[cls]['pos_errors'].append(match['metrics']['distance_3d'])
        class_stats[cls]['iou_3d'].append(match['metrics']['iou_3d'])
    
    # 统计假阳性
    for fp in evaluation_results_3d['false_positives_3d']:
        cls = fp['class']
        class_stats[cls]['fp'] += 1
    
    # 统计假阴性
    for fn in evaluation_results_3d['false_negatives_3d']:
        cls = fn['class']
        class_stats[cls]['fn'] += 1
    
    for cls, stats in class_stats.items():
        if stats['tp'] + stats['fp'] + stats['fn'] > 0:
            class_name = class_names.get(cls, f"类别{cls}")
            precision = stats['tp'] / (stats['tp'] + stats['fp']) if (stats['tp'] + stats['fp']) > 0 else 0
            recall = stats['tp'] / (stats['tp'] + stats['fn']) if (stats['tp'] + stats['fn']) > 0 else 0
            
            print(f"\n{class_name}:")
            print(f"  3D精确率: {precision:.3f}")
            print(f"  3D召回率: {recall:.3f}")
            if stats['pos_errors']:
                print(f"  平均位置误差: {np.mean(stats['pos_errors']):.3f}m")
                print(f"  平均3D IoU: {np.mean(stats['iou_3d']):.3f}")


def visualize_3d_detection_results(image, evaluation_results_3d, calibration, output_dir):
    """可视化3D检测结果"""
    
    # 1. 绘制3D检测匹配结果
    result_image = image.copy()
    
    # 绘制成功匹配的3D检测（绿色）
    for match in evaluation_results_3d['matches_3d']:
        pred = match['prediction']
        gt = match['ground_truth']
        metrics = match['metrics']
        
        # 绘制预测的3D边界框（绿色）
        if '3d_corners' in pred:
            corners_2d, _ = calibration.convert_3D_to_2D(pred['3d_corners'])
            if len(corners_2d) >= 8:
                edges = get_pred_bbox_edges(corners_2d)
                for edge in edges:
                    pt1 = tuple(np.int32(edge[0]))
                    pt2 = tuple(np.int32(edge[1]))
                    cv2.line(result_image, pt1, pt2, (0, 255, 0), 2)
        
        # 绘制GT的3D边界框（青色）
        gt_corners_2d, _ = calibration.convert_3D_to_2D(gt['corners_3d'])
        if len(gt_corners_2d) >= 8:
            edges = get_pred_bbox_edges(gt_corners_2d)
            for edge in edges:
                pt1 = tuple(np.int32(edge[0]))
                pt2 = tuple(np.int32(edge[1]))
                cv2.line(result_image, pt1, pt2, (255, 255, 0), 1)
        
        # 添加精度信息
        bbox = pred['bbox_2d'].astype(int)
        label = f"3D IoU:{metrics['iou_3d']:.2f} Dist:{metrics['distance_3d']:.1f}m"
        cv2.putText(result_image, label, (bbox[0], bbox[1]-25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    # 绘制3D假阳性（红色）
    for fp in evaluation_results_3d['false_positives_3d']:
        if '3d_corners' in fp:
            corners_2d, _ = calibration.convert_3D_to_2D(fp['3d_corners'])
            if len(corners_2d) >= 8:
                edges = get_pred_bbox_edges(corners_2d)
                for edge in edges:
                    pt1 = tuple(np.int32(edge[0]))
                    pt2 = tuple(np.int32(edge[1]))
                    cv2.line(result_image, pt1, pt2, (0, 0, 255), 2)
        
        bbox = fp['bbox_2d'].astype(int)
        cv2.putText(result_image, "3D FP", (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    
    # 绘制3D假阴性（蓝色）
    for fn in evaluation_results_3d['false_negatives_3d']:
        gt_corners_2d, _ = calibration.convert_3D_to_2D(fn['corners_3d'])
        if len(gt_corners_2d) >= 8:
            edges = get_pred_bbox_edges(gt_corners_2d)
            for edge in edges:
                pt1 = tuple(np.int32(edge[0]))
                pt2 = tuple(np.int32(edge[1]))
                cv2.line(result_image, pt1, pt2, (255, 0, 0), 2)
        
        # 找到2D边界框位置来放置标签
        bbox = fn['bbox_2d']
        bbox_int = [int(x) for x in bbox]
        cv2.putText(result_image, "3D FN", (bbox_int[0], bbox_int[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    
    cv2.imwrite(os.path.join(output_dir, "04_3d_detection_results.jpg"), result_image)
    
    # 2. 生成精度分析图表
    metrics = evaluation_results_3d['accuracy_metrics']
    if len(metrics['position_errors']) > 0:
        
        # 位置误差分布图
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.hist(metrics['position_errors'], bins=20, alpha=0.7, color='blue')
        plt.xlabel('位置误差 (m)')
        plt.ylabel('频次')
        plt.title('3D位置误差分布')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 2)
        plt.hist(metrics['orientation_errors'], bins=20, alpha=0.7, color='green')
        plt.xlabel('朝向误差 (度)')
        plt.ylabel('频次')
        plt.title('3D朝向误差分布')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 3)
        plt.hist(metrics['iou_3d_scores'], bins=20, alpha=0.7, color='red')
        plt.xlabel('3D IoU')
        plt.ylabel('频次')
        plt.title('3D IoU分布')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 4)
        plt.scatter(metrics['position_errors'], metrics['iou_3d_scores'], alpha=0.6)
        plt.xlabel('位置误差 (m)')
        plt.ylabel('3D IoU')
        plt.title('位置误差 vs 3D IoU')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "05_3d_accuracy_analysis.png"), dpi=300, bbox_inches='tight')
        plt.close()


def evaluate_predictions_vs_gt(detection_results, camera_labels, lidar_labels, iou_threshold=0.5, distance_threshold=5.0):
    """评估预测结果与真实标签（2D评估）"""
    evaluation_results = {
        'matches': [],
        'false_positives': [],
        'false_negatives': [],
        'statistics': {
            'total_gt': len(camera_labels),
            'total_pred': len(detection_results),
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': 0
        }
    }
    
    # 转换GT标签格式
    gt_objects = []
    for i, (cam_label, lidar_label) in enumerate(zip(camera_labels, lidar_labels)):
        yolo_class = convert_v2x_type_to_yolo_class(cam_label['type'])
        if yolo_class >= 0:
            gt_objects.append({
                'index': i,
                'class': yolo_class,
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
                ]
            })
    
    # 匹配预测和GT
    used_gt = set()
    
    for pred in detection_results:
        best_match = None
        best_score = 0
        
        for gt in gt_objects:
            if gt['index'] in used_gt:
                continue
            
            # 类别必须匹配
            if pred['class'] != gt['class']:
                continue
            
            # 计算2D IoU
            iou_2d = calculate_iou_2d(pred['bbox_2d'], gt['bbox_2d'])
            
            # 如果有3D信息，也计算3D距离
            distance_3d = float('inf')
            if pred.get('fusion_success', False) and '3d_center' in pred:
                distance_3d = calculate_distance_3d(pred['3d_center'], gt['center_3d'])
            
            # 综合评分
            score = iou_2d
            if distance_3d < distance_threshold:
                score += 0.5  # 3D距离奖励
            
            if score > best_score and iou_2d >= iou_threshold:
                best_score = score
                best_match = gt
        
        if best_match:
            used_gt.add(best_match['index'])
            evaluation_results['matches'].append({
                'prediction': pred,
                'ground_truth': best_match,
                'iou_2d': calculate_iou_2d(pred['bbox_2d'], best_match['bbox_2d']),
                'distance_3d': calculate_distance_3d(pred['3d_center'], best_match['center_3d']) if pred.get('fusion_success', False) else None
            })
            evaluation_results['statistics']['true_positives'] += 1
        else:
            evaluation_results['false_positives'].append(pred)
            evaluation_results['statistics']['false_positives'] += 1
    
    # 未匹配的GT为假阴性
    for gt in gt_objects:
        if gt['index'] not in used_gt:
            evaluation_results['false_negatives'].append(gt)
            evaluation_results['statistics']['false_negatives'] += 1
    
    return evaluation_results


def visualize_evaluation_results(image, evaluation_results, calibration):
    """可视化2D评估结果"""
    result_image = image.copy()
    
    # 绘制匹配的预测（绿色）
    for match in evaluation_results['matches']:
        pred = match['prediction']
        bbox = pred['bbox_2d'].astype(int)
        cv2.rectangle(result_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        
        # 添加标签
        label = f"TP: {pred['class']} ({pred['confidence']:.2f})"
        cv2.putText(result_image, label, (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # 如果有3D信息，绘制3D边界框
        if pred.get('fusion_success', False) and '3d_corners' in pred:
            corners_2d, _ = calibration.convert_3D_to_2D(pred['3d_corners'])
            if len(corners_2d) >= 8:
                edges = get_pred_bbox_edges(corners_2d)
                for edge in edges:
                    pt1 = tuple(np.int32(edge[0]))
                    pt2 = tuple(np.int32(edge[1]))
                    cv2.line(result_image, pt1, pt2, (0, 255, 0), 1)
    
    # 绘制假阳性（红色）
    for fp in evaluation_results['false_positives']:
        bbox = fp['bbox_2d'].astype(int)
        cv2.rectangle(result_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
        
        label = f"FP: {fp['class']} ({fp['confidence']:.2f})"
        cv2.putText(result_image, label, (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    # 绘制假阴性（蓝色）
    for fn in evaluation_results['false_negatives']:
        bbox = [int(x) for x in fn['bbox_2d']]
        cv2.rectangle(result_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
        
        label = f"FN: {fn['type']}"
        cv2.putText(result_image, label, (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    return result_image


def analyze_fusion_failures(detection_results, fusion_stats):
    """分析融合失败的原因"""
    print("\n=== 融合失败分析 ===")
    
    total_detections = fusion_stats['total_detections']
    successful_fusions = fusion_stats['successful_fusions']
    
    print(f"总检测数: {total_detections}")
    print(f"成功融合数: {successful_fusions}")
    print(f"融合成功率: {successful_fusions/total_detections*100:.1f}%")
    
    print("\n失败原因统计:")
    for reason, count in fusion_stats['failure_reasons'].items():
        percentage = count / total_detections * 100
        print(f"  {reason}: {count} ({percentage:.1f}%)")
    
    # 按类别分析融合成功率
    class_fusion_stats = defaultdict(lambda: {'total': 0, 'success': 0})
    
    for result in detection_results:
        cls = result['class']
        class_fusion_stats[cls]['total'] += 1
        if result.get('fusion_success', False):
            class_fusion_stats[cls]['success'] += 1
    
    print("\n按类别融合成功率:")
    class_names = {0: "行人", 1: "骑行者", 2: "汽车", 3: "摩托车", 5: "公交车", 6: "卡车", 7: "火车"}
    for cls, stats in class_fusion_stats.items():
        class_name = class_names.get(cls, f"类别{cls}")
        success_rate = stats['success'] / stats['total'] * 100 if stats['total'] > 0 else 0
        print(f"  {class_name}: {stats['success']}/{stats['total']} ({success_rate:.1f}%)")


def main():
    """主函数"""
    print("开始V2X数据集3D检测精度评估...")
    
    try:
        # 数据路径
        data_root = "/mnt/disk_2/yuji/voyagepro/yolo-laidar/data/V2X-Seq-SPD-Example/infrastructure-side/"
        data_info_path = "/mnt/disk_2/yuji/voyagepro/yolo-laidar/data/V2X-Seq-SPD-Example/infrastructure-side/data_info.json"
        frame_id = "010699"
        
        # 加载数据
        print(f"加载帧 {frame_id} 的数据...")
        data = load_v2x_frame_data(data_root, data_info_path, frame_id)
        if data is None:
            print("数据加载失败")
            return
        
        image = data['image']
        points = data['points']
        calibration = data['calibration']
        camera_labels = data['camera_labels']
        lidar_labels = data['lidar_labels']
        
        print(f"图像尺寸: {image.shape}")
        print(f"点云点数: {len(points)}")
        print(f"相机标签数: {len(camera_labels)}")
        print(f"LiDAR标签数: {len(lidar_labels)}")
        
        # 初始化检测器
        model_path = "yolov8m-seg.pt"
        detector = V2XEvaluationDetector(model_path, tracking=False)
        
        # 运行检测
        print("\n运行3D检测...")
        detection_results, objects3d_data, all_corners_3D, pts_3D, pts_2D, all_filtered_points_of_object, all_object_IDs = detector.process_frame(
            image, points, calibration, erosion_factor=25, depth_factor=20
        )
        
        print(f"检测到 {len(detection_results)} 个目标")
        print(f"成功融合 {len(objects3d_data)} 个3D目标")
        
        # 评估2D检测结果
        print("\n评估2D检测结果...")
        evaluation_results = evaluate_predictions_vs_gt(detection_results, camera_labels, lidar_labels)
        
        # 评估3D检测精度
        print("\n评估3D检测精度...")
        evaluation_results_3d = evaluate_3d_detection_accuracy(
            detection_results, camera_labels, lidar_labels,
            iou_2d_threshold=0.5, iou_3d_threshold=0.25, distance_threshold=5.0
        )
        
        # 打印2D评估统计
        stats = evaluation_results['statistics']
        precision = stats['true_positives'] / (stats['true_positives'] + stats['false_positives']) if (stats['true_positives'] + stats['false_positives']) > 0 else 0
        recall = stats['true_positives'] / (stats['true_positives'] + stats['false_negatives']) if (stats['true_positives'] + stats['false_negatives']) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\n=== 2D检测评估结果 ===")
        print(f"真阳性 (TP): {stats['true_positives']}")
        print(f"假阳性 (FP): {stats['false_positives']}")
        print(f"假阴性 (FN): {stats['false_negatives']}")
        print(f"精确率: {precision:.3f}")
        print(f"召回率: {recall:.3f}")
        print(f"F1分数: {f1_score:.3f}")
        
        # 分析3D检测精度
        analyze_3d_detection_accuracy(evaluation_results_3d)
        
        # 分析融合失败原因
        analyze_fusion_failures(detection_results, detector.fusion_stats)
        
        # 创建输出目录
        output_dir = "./evaluation_output"
        os.makedirs(output_dir, exist_ok=True)
        
        # 可视化结果
        print("\n生成可视化结果...")
        
        # 1. 原始图像
        cv2.imwrite(os.path.join(output_dir, "01_original_image.jpg"), image)
        
        # 2. 带GT标签的图像
        gt_image = image.copy()
        for cam_label in camera_labels:
            bbox = cam_label['2d_box']
            bbox_int = [int(bbox['xmin']), int(bbox['ymin']), int(bbox['xmax']), int(bbox['ymax'])]
            cv2.rectangle(gt_image, (bbox_int[0], bbox_int[1]), (bbox_int[2], bbox_int[3]), (255, 255, 0), 2)
            cv2.putText(gt_image, cam_label['type'], (bbox_int[0], bbox_int[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.imwrite(os.path.join(output_dir, "02_ground_truth.jpg"), gt_image)
        
        # 3. 2D评估结果可视化
        eval_image = visualize_evaluation_results(image, evaluation_results, calibration)
        cv2.imwrite(os.path.join(output_dir, "03_2d_evaluation_results.jpg"), eval_image)
        
        # 4. 3D检测结果可视化
        visualize_3d_detection_results(image, evaluation_results_3d, calibration, output_dir)
        
        # 5. 详细匹配信息
        print(f"\n=== 3D检测详细匹配信息 ===")
        for i, match in enumerate(evaluation_results_3d['matches_3d']):
            pred = match['prediction']
            gt = match['ground_truth']
            metrics = match['metrics']
            print(f"3D匹配 {i+1}:")
            print(f"  预测: 类别{pred['class']}, 置信度{pred['confidence']:.3f}")
            print(f"  真值: {gt['type']}")
            print(f"  2D IoU: {metrics['iou_2d']:.3f}")
            print(f"  3D IoU: {metrics['iou_3d']:.3f}")
            print(f"  位置误差: {metrics['distance_3d']:.3f}m")
            print(f"  朝向误差: {metrics['orientation_error']:.1f}°")
            if metrics['dimension_error']['mean_error'] != float('inf'):
                print(f"  平均尺寸误差: {metrics['dimension_error']['mean_error']*100:.1f}%")
        
        print(f"\n3D假阳性 ({len(evaluation_results_3d['false_positives_3d'])}):")
        for i, fp in enumerate(evaluation_results_3d['false_positives_3d']):
            print(f"  3D FP {i+1}: 类别{fp['class']}, 置信度{fp['confidence']:.3f}")
        
        print(f"\n3D假阴性 ({len(evaluation_results_3d['false_negatives_3d'])}):")
        for i, fn in enumerate(evaluation_results_3d['false_negatives_3d']):
            print(f"  3D FN {i+1}: {fn['type']}")
        
        print(f"\n所有评估结果已保存到: {output_dir}")
        
    except Exception as e:
        print(f"评估过程失败: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main() 