#!/usr/bin/env python3
"""
DAIR-V2X数据集增强版多视角RGB+LiDAR融合检测测试脚本
使用Code目录中已有的模块和功能
支持基础设施侧和车辆侧两个视角的3D检测结果返回、人车距离计算
"""

import os
import cv2
import numpy as np
import json
import open3d as o3d
import time
from collections import Counter, deque, defaultdict
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

# 导入Code目录中的模块
from ultralytics import YOLO
from detector import YOLOv8Detector
from fusion import lidar_camera_fusion
from improved_fusion import improved_lidar_camera_fusion
from calibration import LiDAR2Camera
from data_processing import *
from utils import *
from visualization import *

# 设置多进程启动方式为 'spawn'
try:
    set_start_method('spawn')
except RuntimeError:
    pass


def simplify_results_for_json(results_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """精简JSON输出，只保留关键车辆信息"""
    simplified_list = []
    vehicle_classes = {1, 2, 3, 5, 6, 7}  # bicycle, car, motorcycle, bus, truck

    for frame_data in results_list:
        simplified_frame = {
            'frame_id': frame_data.get('frame_id'),
            'view_name': frame_data.get('view_name'),
            'vehicles': []
        }
        
        # 创建一个从车辆ID到人车距离信息的映射
        vehicle_distances = defaultdict(list)
        for dist_info in frame_data.get('person_vehicle_distances', []):
            vehicle_id = dist_info['vehicle_id']
            person_id = dist_info['person_id']
            distance = dist_info['distance_xy']
            angle = dist_info.get('angle_degrees', 0.0)
            direction_type = dist_info.get('direction_type', 'UNKNOWN')
            
            vehicle_distances[vehicle_id].append({
                'person_id': person_id,
                'distance_to_person': round(distance, 2),
                'angle_degrees': round(angle, 1),
                'direction_type': direction_type,
                'risk_level': 'high' if angle < 5.0 else 'medium' if angle < 15.0 else 'low'
            })

        # 只处理车辆检测结果
        for det in frame_data.get('detections', []):
            if det['class'] in vehicle_classes:
                vehicle_info = {
                    'id': det['id'],
                    'position_3d': [round(p, 2) for p in det.get('center_3d', [0, 0, 0])],
                    'speed_kmh': round(det['speed_3d_kmh'], 2) if det.get('speed_3d_kmh') is not None else None,
                    'person_relationships': vehicle_distances.get(det['id'], [])
                }
                simplified_frame['vehicles'].append(vehicle_info)
        
        # 只在有车辆时才添加该帧
        if simplified_frame['vehicles']:
            simplified_list.append(simplified_frame)
            
    return simplified_list


def assign_colors_by_depth(points):
    """根据深度分配颜色"""
    if len(points) == 0:
        return []
    depths = points[:, 2]
    if np.max(depths) - np.min(depths) < 1e-6:
        # 如果深度范围很小，使用默认颜色
        return [[128, 128, 255] for _ in range(len(points))]
    
    normalized_depths = (depths - np.min(depths)) / (np.max(depths) - np.min(depths))
    colors = []
    for depth in normalized_depths:
        color = [int(255 * (1 - depth)), int(255 * depth), 128]
        colors.append(color)
    return colors

def safe_to_list(data):
    """安全地将数据转换为list格式"""
    if isinstance(data, list):
        return data
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif hasattr(data, 'tolist'):
        return data.tolist()
    else:
        return data


class V2XEnhancedCalibration:
    """DAIR-V2X数据集增强版标定类"""
    
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
            
            print(f"V2X标定初始化成功")
            print(f"相机内参 K:\n{self.K}")
            print(f"旋转矩阵 R:\n{self.R}")
            print(f"平移向量 t:\n{self.t.flatten()}")
            
        except Exception as e:
            print(f"V2X标定初始化失败: {e}")
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
    
    def convert_3D_to_camera_coords(self, points_3D):
        """将3D点从LiDAR坐标系转换到相机坐标系"""
        try:
            points_3D = np.asarray(points_3D, dtype=np.float64)
            points_3D_homo = np.hstack([points_3D, np.ones((points_3D.shape[0], 1), dtype=np.float64)])
            points_cam = (self.T @ points_3D_homo.T).T[:, :3]
            return points_cam
        except Exception as e:
            print(f"坐标转换失败: {e}")
            return points_3D


class V2XEnhanced3DDetector:
    """DAIR-V2X增强版3D检测器"""
    
    def __init__(self, model_path="yolov8m-seg.pt", tracking=True, view_prefix="", use_improved_fusion=True):
        print(f"初始化YOLO模型: {model_path}")
        self.model = YOLO(model_path)
        self.model.overrides['conf'] = 0.1  # 降低置信度阈值
        self.model.overrides['iou'] = 0.5
        self.model.overrides['agnostic_nms'] = False
        self.model.overrides['max_det'] = 1000
        
        # 创建基于Code模块的检测器
        self.detector = YOLOv8Detector(model_path, tracking=tracking, PCA=False)
        
        self.tracking = tracking
        self.view_prefix = view_prefix  # 视角前缀，避免多线程ID冲突
        self.use_improved_fusion = use_improved_fusion
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
    
    def calculate_person_vehicle_angle(self, person_center_3d, vehicle_center_3d):
        """
        计算人车连线与垂直方向的夹角
        
        Args:
            person_center_3d: 人的3D中心坐标 [x, y, z]
            vehicle_center_3d: 车辆的3D中心坐标 [x, y, z]
            
        Returns:
            angle_degrees: 与垂直方向的夹角（度）
            direction_type: 方向类型描述
        """
        try:
            # 计算人车连线向量（从车辆指向人）
            direction_vector = np.array(person_center_3d) - np.array(vehicle_center_3d)
            
            # 只考虑X-Y平面的方向（忽略Z轴高度差）
            direction_xy = direction_vector[:2]
            
            # 计算向量长度
            vector_length = np.linalg.norm(direction_xy)
            
            if vector_length < 1e-6:  # 如果距离太近，返回0度
                return 0.0, "OVERLAP"
            
            # 垂直方向向量（Y轴正方向）
            vertical_vector = np.array([1, 0])
            
            # 计算夹角（使用点积公式）
            cos_angle = np.dot(direction_xy, vertical_vector) / (vector_length * np.linalg.norm(vertical_vector))
            
            # 限制cos值范围，避免数值误差
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            
            # 计算角度（弧度转度）
            angle_radians = np.arccos(cos_angle)
            angle_degrees = np.degrees(angle_radians)
            
            # 判断方向类型
            if angle_degrees < 5.0:
                direction_type = "DIRECT_APPROACH"
            elif angle_degrees < 15.0:
                direction_type = "BASIC_TOWARD"
            elif angle_degrees < 45.0:
                direction_type = "DIAGONAL_APPROACH"
            elif angle_degrees < 75.0:
                direction_type = "LATERAL_MOTION"
            else:
                direction_type = "BYPASS_MOVEMENT"
            
            return angle_degrees, direction_type
            
        except Exception as e:
            print(f"计算人车夹角时出错: {e}")
            return 0.0, "CALC_ERROR"

    def calculate_person_vehicle_distances(self, detection_results):
        """计算人和车辆之间的距离（只使用X-Y坐标）和相对夹角"""
        persons = []
        vehicles = []
        
        # 分类检测结果
        for result in detection_results:
            if result['class'] in self.person_classes:
                persons.append(result)
            elif result['class'] in self.vehicle_classes:
                vehicles.append(result)
        
        # 计算人车距离和夹角
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
                    
                    # 计算人车相对夹角
                    angle_degrees, direction_type = self.calculate_person_vehicle_angle(
                        person_center, vehicle_center
                    )
                    
                    person_vehicle_distances.append({
                        'person_id': person['id'],
                        'person_class': self.model.names[person['class']],
                        'vehicle_id': vehicle['id'],
                        'vehicle_class': self.model.names[vehicle['class']],
                        'distance_xy': distance_xy,  # 只使用X-Y距离
                        'angle_degrees': angle_degrees,  # 与垂直方向的夹角
                        'direction_type': direction_type,  # 方向类型描述
                        'person_center_xy': safe_to_list(person_center[:2]),  # 只保存X-Y坐标
                        'vehicle_center_xy': safe_to_list(vehicle_center[:2]),  # 只保存X-Y坐标
                        'person_center_3d': safe_to_list(person_center),  # 保留完整3D坐标用于其他用途
                        'vehicle_center_3d': safe_to_list(vehicle_center)
                    })
        
        return person_vehicle_distances
    
    def process_frame_enhanced(self, frame, points, calibration, frame_id=0, fps=30) -> Dict[str, Any]:
        """增强版帧处理，使用Code模块中的检测器和融合方法"""
        try:
            # 使用Code模块中的检测器进行处理
            objects3d_data, all_corners_3D, pts_3D, pts_2D, all_filtered_points_of_object, all_object_IDs = self.detector.process_frame(
                frame, points, calibration, erosion_factor=25, depth_factor=20
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
                    'processed_frame': np.ascontiguousarray(frame.copy(), dtype=np.uint8),
                    'bev_frame': None
                }
            }
            
            # 转换检测结果格式
            detection_results = []
            for i, (corners_3D, filtered_points, object_id) in enumerate(zip(all_corners_3D, all_filtered_points_of_object, all_object_IDs)):
                if i < len(objects3d_data):
                    obj_data = objects3d_data[i]
                    ROS_type, ROS_ground_center, ROS_direction, ROS_dimensions, ROS_velocity, ROS_points = obj_data
                    
                    # 添加视角前缀避免多线程ID冲突
                    unique_id = f"{self.view_prefix}_{object_id}" if self.view_prefix else object_id
                    
                    # 计算3D信息
                    center_3d = ROS_ground_center
                    dimensions_3d = ROS_dimensions
                    distance_from_origin = np.linalg.norm(center_3d)
                    
                    # 计算速度（km/h）
                    speed_3d_kmh = None
                    if ROS_velocity is not None:
                        speed_ms = np.linalg.norm(ROS_velocity)
                        speed_3d_kmh = speed_ms * 3.6
                    
                    # 构建检测结果
                    detection_result = {
                        'id': unique_id,
                        'class': ROS_type,
                        'class_name': self.model.names[ROS_type],
                        'confidence': 0.5,  # 默认置信度
                        'bbox_2d': [0, 0, 0, 0],  # 需要从原始检测结果获取
                        'corners_3D': safe_to_list(corners_3D),
                        'center_3d': safe_to_list(center_3d),
                        'dimensions_3d': safe_to_list(dimensions_3d),
                        'distance_from_origin': distance_from_origin,
                        'speed_3d_kmh': speed_3d_kmh,
                        'filtered_points': safe_to_list(filtered_points) if len(filtered_points) > 0 else []
                    }
                    
                    detection_results.append(detection_result)
            
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
            result_frame = self.draw_enhanced_results(frame, detection_results, person_vehicle_distances, calibration, pts_3D, pts_2D)
            
            # 创建BEV视图，添加错误处理
            try:
                # 合并所有过滤的点云
                all_points = []
                if len(all_filtered_points_of_object) > 0:
                    for points_obj in all_filtered_points_of_object:
                        if len(points_obj) > 0:
                            all_points.append(points_obj)
                
                if len(all_points) > 0:
                    combined_points = np.vstack(all_points)
                else:
                    combined_points = pts_3D if len(pts_3D) > 0 else points[:1000]  # 使用原始点云的子集
                
                bev_frame = self.create_bev_view(combined_points, all_corners_3D, detection_results)
            except Exception as e:
                print(f"创建BEV视图失败: {e}")
                bev_frame = np.zeros((500, 500, 3), dtype=np.uint8)
            
            frame_result['visualization_data'] = {
                'processed_frame': result_frame,
                'bev_frame': bev_frame
            }
            
            return frame_result
            
        except Exception as e:
            print(f"处理帧时出错: {e}")
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
                    'processed_frame': np.ascontiguousarray(frame.copy(), dtype=np.uint8),
                    'bev_frame': np.zeros((500, 500, 3), dtype=np.uint8)
                },
                'error': str(e)
            }
            return frame_result
    
    def draw_enhanced_results(self, image, detection_results, person_vehicle_distances, calibration, pts_3D, pts_2D):
        """绘制增强的检测结果"""
        # 确保图像是正确的 numpy 数组格式
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        
        # 确保图像是连续的内存布局
        result_image = np.ascontiguousarray(image.copy(), dtype=np.uint8)
        
        # 创建一个字典来存储每个对象的角度信息
        object_angle_info = {}
        
        # 预处理人车距离信息，为每个对象收集相关的角度信息
        for dist_info in person_vehicle_distances:
            person_id = dist_info['person_id']
            vehicle_id = dist_info['vehicle_id']
            angle = dist_info.get('angle_degrees', 0)
            direction_type = dist_info.get('direction_type', 'UNKNOWN')
            distance = dist_info['distance_xy']
            
            # 为人员对象添加角度信息
            if person_id not in object_angle_info:
                object_angle_info[person_id] = []
            object_angle_info[person_id].append({
                'target_id': vehicle_id,
                'target_type': 'V',
                'angle': angle,
                'direction': direction_type,
                'distance': distance
            })
            
            # 为车辆对象添加角度信息
            if vehicle_id not in object_angle_info:
                object_angle_info[vehicle_id] = []
            object_angle_info[vehicle_id].append({
                'target_id': person_id,
                'target_type': 'P',
                'angle': angle,
                'direction': direction_type,
                'distance': distance
            })
        
        # 绘制点云
        if len(pts_2D) > 0 and len(pts_3D) > 0:
            colors = assign_colors_by_depth(pts_3D)
            for i, (pt_2d, color) in enumerate(zip(pts_2D.astype(int), colors)):
                if 0 <= pt_2d[0] < result_image.shape[1] and 0 <= pt_2d[1] < result_image.shape[0]:
                    cv2.circle(result_image, tuple(pt_2d), 1, tuple(safe_to_list(color)), -1)
        
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
                    # 使用get_pred_bbox_edges函数绘制3D边界框的所有边
                    pred_edges_2D = get_pred_bbox_edges(pred_corners_2D)
                    
                    # 绘制3D边界框的所有边
                    for pred_edge in pred_edges_2D:
                        pt1 = tuple(np.int32(pred_edge[0]))
                        pt2 = tuple(np.int32(pred_edge[1]))
                        cv2.line(result_image, pt1, pt2, color, 2)
                    
                    # 添加ID和信息标签（参考v2x_speed_visualization.py的标签位置）
                    if len(pred_corners_2D) > 7:
                        top_left_front_corner = pred_corners_2D[7]
                        top_left_front_pt = (int(np.round(top_left_front_corner[0])), 
                                           int(np.round(top_left_front_corner[1])) - 10)
                        
                        # 构建基础标签文本
                        label_parts = [
                            f"ID:{result['id']}", 
                            result['class_name'], 
                            f"Dist:{result['distance_from_origin']:.1f}m"
                        ]
                        
                        if result['speed_3d_kmh'] is not None:
                            label_parts.append(f"Speed:{result['speed_3d_kmh']:.1f}km/h")
                        
                        # 添加角度信息到标签中
                        object_id = result['id']
                        if object_id in object_angle_info:
                            angle_info_list = object_angle_info[object_id]
                            if len(angle_info_list) > 0:
                                # 取最近的一个角度信息
                                closest_angle_info = min(angle_info_list, key=lambda x: x['distance'])
                                angle_text = f"Angle:{closest_angle_info['angle']:.1f}°({closest_angle_info['direction']})"
                                label_parts.append(angle_text)
                        
                        label_text = ' '.join(label_parts)
                        
                        # 绘制带阴影的文本（完全参考v2x_speed_visualization.py）
                        cv2.putText(result_image, label_text, top_left_front_pt, 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 3, cv2.LINE_AA)
                        cv2.putText(result_image, label_text, top_left_front_pt, 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
                        
                        # 在对象上方额外显示角度信息（如果有多个关系）
                        if object_id in object_angle_info and len(object_angle_info[object_id]) > 1:
                            angle_details_y = top_left_front_pt[1] - 25
                            for i, angle_info in enumerate(object_angle_info[object_id][:3]):  # 最多显示3个
                                angle_detail_text = f"→{angle_info['target_type']}{angle_info['target_id']}:{angle_info['angle']:.1f}°"
                                cv2.putText(result_image, angle_detail_text, 
                                          (top_left_front_pt[0], angle_details_y - i*15), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                else:
                    # 如果3D投影失败，绘制2D边界框
                    bbox = result.get('bbox_2d', [0, 0, 100, 100])
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
                    
                    # 绘制简单标签
                    label_parts = [
                        f"ID:{result['id']}", 
                        result['class_name'], 
                        f"Dist:{result['distance_from_origin']:.1f}m"
                    ]
                    
                    if result['speed_3d_kmh'] is not None:
                        label_parts.append(f"Speed:{result['speed_3d_kmh']:.1f}km/h")
                    
                    # 添加角度信息
                    object_id = result['id']
                    if object_id in object_angle_info:
                        angle_info_list = object_angle_info[object_id]
                        if len(angle_info_list) > 0:
                            closest_angle_info = min(angle_info_list, key=lambda x: x['distance'])
                            angle_text = f"Angle:{closest_angle_info['angle']:.1f}°"
                            label_parts.append(angle_text)
                    
                    label = ' '.join(label_parts)
                    
                    # 绘制标签背景和文字
                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    dim, baseline = text_size[0], text_size[1]
                    
                    text_x, text_y = int(bbox[0]), max(int(bbox[1]), 30)
                    cv2.rectangle(result_image, (text_x, text_y - dim[1] - 10),
                                (text_x + dim[0] + 5, text_y - 5), (30, 30, 30), cv2.FILLED)
                    
                    cv2.putText(result_image, label, (text_x + 3, text_y - 8),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
            except Exception as e:
                continue
        
        # 绘制人车之间的连线和距离信息
        if person_vehicle_distances:
            # 为每个人车配对绘制连线
            for i, dist_info in enumerate(person_vehicle_distances):
                try:
                    # 获取人和车辆的3D中心点
                    person_center_3d = np.array(dist_info['person_center_3d'])
                    vehicle_center_3d = np.array(dist_info['vehicle_center_3d'])
                    
                    # 将3D中心点投影到2D图像平面
                    person_2d, person_valid = calibration.convert_3D_to_2D(person_center_3d.reshape(1, -1))
                    vehicle_2d, vehicle_valid = calibration.convert_3D_to_2D(vehicle_center_3d.reshape(1, -1))
                    
                    # 检查投影是否成功
                    if len(person_2d) > 0 and len(vehicle_2d) > 0:
                        person_pt = person_2d[0].astype(int)
                        vehicle_pt = vehicle_2d[0].astype(int)
                        
                        # 检查点是否在图像范围内
                        img_height, img_width = result_image.shape[:2]
                        
                        if (0 <= person_pt[0] < img_width and 0 <= person_pt[1] < img_height and
                            0 <= vehicle_pt[0] < img_width and 0 <= vehicle_pt[1] < img_height):
                            
                            # 根据角度和距离选择连线颜色
                            distance = dist_info['distance_xy']
                            angle = dist_info.get('angle_degrees', 0)
                            
                            # 根据角度选择连线颜色 - 角度越小（越朝向）颜色越红
                            if angle < 5.0:  # 基本垂直，车朝人开
                                line_color = (0, 0, 255)  # 红色 - 危险
                            elif angle < 15.0:  # 基本朝向
                                line_color = (0, 100, 255)  # 橙红色
                            elif angle < 45.0:  # 斜向接近
                                line_color = (0, 255, 255)  # 黄色
                            elif angle < 75.0:  # 侧向运动
                                line_color = (255, 255, 0)  # 青色
                            else:  # 绕开行驶
                                line_color = (255, 0, 0)  # 蓝色 - 安全
                            
                            # 根据距离调整线条粗细
                            if distance < 5.0:
                                line_thickness = 4
                            elif distance < 10.0:
                                line_thickness = 3
                            else:
                                line_thickness = 2
                            
                            # 绘制连线
                            cv2.line(result_image, tuple(person_pt), tuple(vehicle_pt), line_color, line_thickness)
                            
                            # 在连线中点位置绘制距离和角度信息
                            mid_pt = ((person_pt[0] + vehicle_pt[0]) // 2, 
                                     (person_pt[1] + vehicle_pt[1]) // 2)
                            
                            # 构建距离和角度标签 - 增大字体，更醒目
                            distance_label = f"P{dist_info['person_id']}-V{dist_info['vehicle_id']}: {distance:.1f}m"
                            angle_label = f"Angle: {angle:.1f}° ({dist_info.get('direction_type', 'UNKNOWN')})"
                            
                            # 绘制距离标签 - 增大字体
                            text_size = cv2.getTextSize(distance_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                            text_w, text_h = text_size[0][0], text_size[0][1]
                            
                            # 计算标签位置，避免超出图像边界
                            label_x = max(5, min(mid_pt[0] - text_w // 2, img_width - text_w - 5))
                            label_y = max(text_h + 5, min(mid_pt[1] - 15, img_height - 40))
                            
                            # 绘制距离标签背景 - 增大背景
                            cv2.rectangle(result_image, 
                                        (label_x - 3, label_y - text_h - 3),
                                        (label_x + text_w + 3, label_y + 3),
                                        (0, 0, 0), -1)  # 黑色背景
                            
                            # 绘制距离文本 - 增大字体
                            cv2.putText(result_image, distance_label, 
                                      (label_x, label_y), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                            
                            # 绘制角度标签 - 增大字体
                            angle_text_size = cv2.getTextSize(angle_label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                            angle_text_w, angle_text_h = angle_text_size[0][0], angle_text_size[0][1]
                            
                            angle_label_x = max(5, min(mid_pt[0] - angle_text_w // 2, img_width - angle_text_w - 5))
                            angle_label_y = label_y + 20
                            
                            # 绘制角度标签背景 - 增大背景
                            cv2.rectangle(result_image, 
                                        (angle_label_x - 3, angle_label_y - angle_text_h - 3),
                                        (angle_label_x + angle_text_w + 3, angle_label_y + 3),
                                        (0, 0, 0), -1)  # 黑色背景
                            
                            # 根据角度选择文本颜色
                            if angle < 5.0:
                                angle_text_color = (0, 0, 255)  # 红色
                            elif angle < 15.0:
                                angle_text_color = (0, 100, 255)  # 橙红色
                            elif angle < 45.0:
                                angle_text_color = (0, 255, 255)  # 黄色
                            else:
                                angle_text_color = (255, 255, 255)  # 白色
                            
                            # 绘制角度文本 - 增大字体
                            cv2.putText(result_image, angle_label, 
                                      (angle_label_x, angle_label_y), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.55, angle_text_color, 2)
                            
                            # 在人和车辆的中心点绘制更大的圆点标记
                            cv2.circle(result_image, tuple(person_pt), 6, (0, 255, 0), -1)  # 绿色圆点标记人
                            cv2.circle(result_image, tuple(vehicle_pt), 6, (0, 0, 255), -1)  # 红色圆点标记车辆
                            
                            # 绘制更明显的方向箭头（从车辆指向人）
                            arrow_vector = np.array(person_pt) - np.array(vehicle_pt)
                            arrow_length = np.linalg.norm(arrow_vector)
                            
                            if arrow_length > 25:  # 只在距离足够时绘制箭头
                                # 标准化箭头方向
                                arrow_unit = arrow_vector / arrow_length
                                # 箭头起点（距离车辆中心15像素）
                                arrow_start = vehicle_pt + (arrow_unit * 15).astype(int)
                                # 箭头终点（距离人中心15像素）
                                arrow_end = person_pt - (arrow_unit * 15).astype(int)
                                
                                # 绘制更粗的方向箭头
                                cv2.arrowedLine(result_image, tuple(arrow_start), tuple(arrow_end), 
                                              line_color, 3, tipLength=0.25)
                
                except Exception as e:
                    print(f"绘制人车距离连线时出错: {e}")
                    continue
        
        # 在图像左上角绘制距离统计信息 - 增强显示
        if person_vehicle_distances:
            y_offset = 30
            
            # 绘制统计信息标题 - 增大字体
            cv2.putText(result_image, f"Person-Vehicle Relations ({len(person_vehicle_distances)} pairs):", 
                      (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            y_offset += 30
            
            # 显示前几个距离和角度信息 - 增强显示
            for i, dist_info in enumerate(person_vehicle_distances[:5]):  # 显示前5个
                distance = dist_info['distance_xy']
                angle = dist_info.get('angle_degrees', 0)
                direction_type = dist_info.get('direction_type', 'UNKNOWN')
                
                # 构建显示文本 - 更详细的信息
                distance_text = f"P{dist_info['person_id']}-V{dist_info['vehicle_id']}: {distance:.1f}m"
                angle_text = f"   Angle: {angle:.1f}° ({direction_type})"
                
                # 根据角度选择文本颜色
                if angle < 5.0:
                    text_color = (0, 0, 255)  # 红色 - 危险
                elif angle < 15.0:
                    text_color = (0, 100, 255)  # 橙红色
                elif angle < 45.0:
                    text_color = (0, 255, 255)  # 黄色
                elif angle < 75.0:
                    text_color = (255, 255, 0)  # 青色
                else:
                    text_color = (255, 255, 255)  # 白色
                
                # 绘制距离信息 - 增大字体
                cv2.putText(result_image, distance_text, (10, y_offset + i*40), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
                
                # 绘制角度信息 - 增大字体
                cv2.putText(result_image, angle_text, (10, y_offset + i*40 + 18), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 2)
            
            # 如果还有更多距离对，显示省略号
            if len(person_vehicle_distances) > 5:
                cv2.putText(result_image, f"... {len(person_vehicle_distances)-5} more pairs", 
                          (10, y_offset + 5*40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 2)
            
            # 在右上角显示角度分类统计 - 增强显示
            angle_stats = {
                'Direct Approach(<5°)': 0,
                'Basic Toward(5-15°)': 0,
                'Diagonal Approach(15-45°)': 0,
                'Lateral Motion(45-75°)': 0,
                'Bypass Movement(>75°)': 0
            }
            
            for dist_info in person_vehicle_distances:
                angle = dist_info.get('angle_degrees', 0)
                if angle < 5.0:
                    angle_stats['Direct Approach(<5°)'] += 1
                elif angle < 15.0:
                    angle_stats['Basic Toward(5-15°)'] += 1
                elif angle < 45.0:
                    angle_stats['Diagonal Approach(15-45°)'] += 1
                elif angle < 75.0:
                    angle_stats['Lateral Motion(45-75°)'] += 1
                else:
                    angle_stats['Bypass Movement(>75°)'] += 1
            
            # 在右上角显示角度统计 - 增大字体
            stats_x = result_image.shape[1] - 280
            stats_y = 30
            
            cv2.putText(result_image, "Angle Distribution:", 
                      (stats_x, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            stats_y += 25
            colors = [(0, 0, 255), (0, 100, 255), (0, 255, 255), (255, 255, 0), (255, 255, 255)]
            
            for i, (category, count) in enumerate(angle_stats.items()):
                if count > 0:
                    stats_text = f"{category}: {count}"
                    cv2.putText(result_image, stats_text, 
                              (stats_x, stats_y + i*22), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 2)
        
        return result_image

    def create_bev_view(self, points, all_corners_3D, detection_results):
        """创建鸟瞰图视图，参考v2x_speed_visualization.py的实现"""
        try:
            if len(points) > 5000:
                indices = np.random.choice(len(points), 5000, replace=False)
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
            for i, corners_3D in enumerate(all_corners_3D):
                # 获取对应的检测结果（如果存在）
                detection_result = detection_results[i] if i < len(detection_results) else None
                
                # 根据类别选择颜色
                if detection_result and detection_result['class'] in self.person_classes:
                    color = (0, 255, 0)  # 绿色表示人
                else:
                    color = (0, 0, 255)  # 红色表示车辆
                
                # 只使用底面的4个点
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
                    cv2.polylines(bev_image, [bev_corners], True, color, 2)
                    
                    # 添加中心点
                    center = np.mean(bev_corners, axis=0).astype(int)
                    cv2.circle(bev_image, tuple(center), 3, color, -1)
                    
                    # 添加ID标签
                    if detection_result:
                        label_text = f"ID:{detection_result['id']}"
                    else:
                        label_text = f"ID:{i}"
                    
                    cv2.putText(bev_image, label_text, 
                              tuple(center + np.array([5, -5])), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # 绘制人车关系连线（在BEV视图中）
            persons = [det for det in detection_results if det['class'] in self.person_classes]
            vehicles = [det for det in detection_results if det['class'] in self.vehicle_classes]
            
            for person in persons:
                for vehicle in vehicles:
                    # 计算3D中心点
                    person_center = np.mean(person['corners_3D'], axis=0)
                    vehicle_center = np.mean(vehicle['corners_3D'], axis=0)
                    
                    # 检查是否在BEV范围内
                    if (x_range[0] <= person_center[0] <= x_range[1] and y_range[0] <= person_center[1] <= y_range[1] and
                        x_range[0] <= vehicle_center[0] <= x_range[1] and y_range[0] <= vehicle_center[1] <= y_range[1]):
                        
                        # 转换到BEV坐标
                        person_bev = [
                            int((person_center[0] - x_range[0]) / resolution),
                            int(height - 1 - (person_center[1] - y_range[0]) / resolution)
                        ]
                        vehicle_bev = [
                            int((vehicle_center[0] - x_range[0]) / resolution),
                            int(height - 1 - (vehicle_center[1] - y_range[0]) / resolution)
                        ]
                        
                        # 计算角度
                        angle_degrees, direction_type = self.calculate_person_vehicle_angle(
                            person_center, vehicle_center
                        )
                        
                        # 根据角度选择连线颜色
                        if angle_degrees < 5.0:
                            line_color = (0, 0, 255)  # 红色
                        elif angle_degrees < 15.0:
                            line_color = (0, 100, 255)  # 橙红色
                        elif angle_degrees < 45.0:
                            line_color = (0, 255, 255)  # 黄色
                        elif angle_degrees < 75.0:
                            line_color = (255, 255, 0)  # 青色
                        else:
                            line_color = (255, 0, 0)  # 蓝色
                        
                        # 绘制连线
                        cv2.line(bev_image, tuple(person_bev), tuple(vehicle_bev), line_color, 2)
                        
                        # 在连线中点显示角度信息
                        mid_point = [
                            (person_bev[0] + vehicle_bev[0]) // 2,
                            (person_bev[1] + vehicle_bev[1]) // 2
                        ]
                        
                        # 增强角度信息显示
                        angle_text = f"{angle_degrees:.1f}°"
                        direction_text = f"{direction_type}"
                        
                        # 绘制角度信息背景
                        angle_text_size = cv2.getTextSize(angle_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                        angle_text_w, angle_text_h = angle_text_size[0][0], angle_text_size[0][1]
                        
                        # 绘制角度标签背景
                        cv2.rectangle(bev_image, 
                                    (mid_point[0] - angle_text_w//2 - 2, mid_point[1] - angle_text_h - 2),
                                    (mid_point[0] + angle_text_w//2 + 2, mid_point[1] + 2),
                                    (0, 0, 0), -1)  # 黑色背景
                        
                        # 绘制角度文本
                        cv2.putText(bev_image, angle_text, 
                                  (mid_point[0] - angle_text_w//2, mid_point[1]), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                        
                        # 在角度文本下方显示方向类型
                        direction_text_size = cv2.getTextSize(direction_text, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)
                        direction_text_w, direction_text_h = direction_text_size[0][0], direction_text_size[0][1]
                        
                        # 绘制方向标签背景
                        cv2.rectangle(bev_image, 
                                    (mid_point[0] - direction_text_w//2 - 2, mid_point[1] + 5),
                                    (mid_point[0] + direction_text_w//2 + 2, mid_point[1] + 5 + direction_text_h + 2),
                                    (0, 0, 0), -1)  # 黑色背景
                        
                        # 根据角度选择方向文本颜色
                        if angle_degrees < 5.0:
                            direction_color = (0, 0, 255)  # 红色
                        elif angle_degrees < 15.0:
                            direction_color = (0, 100, 255)  # 橙红色
                        elif angle_degrees < 45.0:
                            direction_color = (0, 255, 255)  # 黄色
                        else:
                            direction_color = (255, 255, 255)  # 白色
                        
                        # 绘制方向文本
                        cv2.putText(bev_image, direction_text, 
                                  (mid_point[0] - direction_text_w//2, mid_point[1] + 5 + direction_text_h), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.3, direction_color, 1)
            
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
            
            # 在BEV图像右上角添加角度图例
            legend_x = width - 120
            legend_y = 20
            
            cv2.putText(bev_image, "Angle Legend:", (legend_x, legend_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            legend_items = [
                ("<5°: Direct Approach", (0, 0, 255)),
                ("5-15°: Basic Toward", (0, 100, 255)),
                ("15-45°: Diagonal Approach", (0, 255, 255)),
                ("45-75°: Lateral Motion", (255, 255, 0)),
                (">75°: Bypass Movement", (255, 0, 0))
            ]
            
            for i, (text, color) in enumerate(legend_items):
                y_pos = legend_y + 15 + i * 12
                cv2.putText(bev_image, text, (legend_x, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            
            return bev_image
            
        except Exception as e:
            print(f"BEV可视化失败: {e}")
            return np.zeros((500, 500, 3), dtype=np.uint8)


def process_dair_view(view_path, view_name, max_frames=100, tracking=True, show=False, use_improved_fusion=True):
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
        calibration = V2XEnhancedCalibration(cam_intrinsic_path, lidar_to_cam_path)
        detector = V2XEnhanced3DDetector(
            tracking=tracking, 
            view_prefix=view_name,
            use_improved_fusion=use_improved_fusion
        )
        
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
                
                # 确保result_frame是正确的numpy数组格式
                if not isinstance(result_frame, np.ndarray):
                    result_frame = np.array(result_frame)
                result_frame = np.ascontiguousarray(result_frame, dtype=np.uint8)
                
                # 确保BEV帧是numpy数组
                if isinstance(bev_frame, np.ndarray) and len(bev_frame.shape) == 3:
                    bev_frame = np.ascontiguousarray(bev_frame, dtype=np.uint8)
                else:
                    # 创建默认BEV图像
                    bev_frame = np.zeros((500, 500, 3), dtype=np.uint8)
                
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
            # 精简结果用于JSON输出
            simplified_json_results = simplify_results_for_json(frame_results)
            
            # 保存精简的JSON结果
            results_file = os.path.join(output_dir, f'{view_name}_results.json')
            with open(results_file, 'w') as f:
                json.dump(simplified_json_results, f, indent=2, default=str)
            
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
                       default="/mnt/disk_2/yuji/voyagepro/yolo-laidar/data/V2X-Seq-SPD-Example/infrastructure-side",
                       help="基础设施侧数据路径")
    parser.add_argument('--vehicle-path', type=str,
                       default="/mnt/disk_2/yuji/voyagepro/yolo-laidar/data/V2X-Seq-SPD-Example/infrastructure-side",
                       help="车辆侧数据路径")
    parser.add_argument('--max-frames', type=int, default=1000,
                       help="每个视角最大处理帧数")
    parser.add_argument('--no-tracking', action='store_true',
                       help="禁用目标跟踪")
    parser.add_argument('--show', action='store_true',
                       help="显示处理过程")
    parser.add_argument('--use-improved-fusion', action='store_true',
                       help="使用改进的融合算法")
    
    args = parser.parse_args()
    
    print("=== DAIR-V2X数据集增强版多视角3D检测测试 ===")
    print(f"基础设施侧路径: {args.infrastructure_path}")
    print(f"车辆侧路径: {args.vehicle_path}")
    print(f"每视角最大帧数: {args.max_frames}")
    print(f"跟踪模式: {'关闭' if args.no_tracking else '开启'}")
    print(f"改进融合: {'开启' if args.use_improved_fusion else '关闭'}")
    
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
        (args.infrastructure_path, 'infrastructure-side', args.max_frames, not args.no_tracking, False, args.use_improved_fusion),
        (args.vehicle_path, 'vehicle-side', args.max_frames, not args.no_tracking, False, args.use_improved_fusion)
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
            show=True,
            use_improved_fusion=args.use_improved_fusion
        )
        all_results['infrastructure-side'] = infra_results
        
        # 处理车辆侧
        vehicle_results = process_dair_view(
            args.vehicle_path, 
            'vehicle-side', 
            max_frames=args.max_frames,
            tracking=not args.no_tracking,
            show=True,
            use_improved_fusion=args.use_improved_fusion
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
    
    # 精简汇总结果用于最终JSON输出
    simplified_summary = {
        view_name: simplify_results_for_json(result_list)
        for view_name, result_list in all_results.items()
    }

    # 保存汇总结果
    summary_file = './output/dair_v2x_enhanced_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(simplified_summary, f, indent=2, default=str)
    
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