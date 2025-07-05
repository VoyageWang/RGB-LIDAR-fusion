#!/usr/bin/env python3
"""
DAIR-V2X数据集增强版多视角RGB+LiDAR融合检测测试脚本
使用Code目录中已有的模块和功能
支持基础设施侧和车辆侧两个视角的3D检测结果返回、人车距离计算
支持视频流处理（RGB视频 + LiDAR数据流）
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
import threading
import queue
import glob

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
            vehicle_distances[vehicle_id].append({
                'person_id': person_id,
                'distance_to_person': round(distance, 2)
            })

        # 只处理车辆检测结果
        for det in frame_data.get('detections', []):
            if det['class'] in vehicle_classes:
                vehicle_info = {
                    'id': det['id'],
                    'position_3d': [round(p, 2) for p in det.get('center_3d', [0, 0, 0])],
                    'speed_kmh': round(det['speed_3d_kmh'], 2) if det.get('speed_3d_kmh') is not None else None,
                    'distances_to_persons': vehicle_distances.get(det['id'], [])
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


# def get_pred_bbox_edges(corners_2D):
#     """
#     生成预测的3D边界框边缘.
#     这个实现基于utils.py中的版本，以匹配检测器输出的角点顺序.
#     """
#     if len(corners_2D) < 8:
#         return []
    
#     # 按照 utils.py 中定义的顺序连接角点
#     edges = [
#         # 底面边
#         (corners_2D[0], corners_2D[1]), (corners_2D[1], corners_2D[2]),
#         (corners_2D[2], corners_2D[3]), (corners_2D[3], corners_2D[0]),
#         # 顶面边
#         (corners_2D[4], corners_2D[5]), (corners_2D[5], corners_2D[6]),
#         (corners_2D[6], corners_2D[7]), (corners_2D[7], corners_2D[4]),
#         # 垂直边
#         (corners_2D[0], corners_2D[4]), (corners_2D[1], corners_2D[5]),
#         (corners_2D[2], corners_2D[6]), (corners_2D[3], corners_2D[7])
#     ]
    
#     # 转换为正确的格式
#     return [[np.array(p1), np.array(p2)] for p1, p2 in edges]


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


class CustomCalibration:
    """支持多种标定文件格式的标定类"""
    
    def __init__(self, config_path: str = None, lidar2cam_path: str = None, lidar2world_path: str = None):
        """
        初始化标定参数
        
        Args:
            config_path: 单个配置文件路径（DAIR-V2X格式）
            lidar2cam_path: LiDAR到相机的标定文件路径（RCOOPER格式）
            lidar2world_path: LiDAR到世界坐标系的标定文件路径（RCOOPER格式）
        """
        self.K = None
        self.D = None
        self.R = None
        self.t = None
        self.T = None
        self.lidar2world_T = None
        
        if config_path and os.path.exists(config_path):
            # DAIR-V2X格式的单个配置文件
            self._load_dair_v2x_config(config_path)
        elif lidar2cam_path and os.path.exists(lidar2cam_path):
            # RCOOPER格式的分离标定文件
            self._load_rcooper_config(lidar2cam_path, lidar2world_path)
        else:
            raise ValueError("必须提供有效的标定文件路径")
    
    def _load_dair_v2x_config(self, config_path: str):
        """加载DAIR-V2X格式的标定文件"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)

            # 相机内参矩阵 (3x3)
            self.K = np.array(config['cam_K'], dtype=np.float64).reshape(3, 3)
            
            # 畸变参数
            self.D = np.array(config.get('cam_D', [0, 0, 0, 0, 0]), dtype=np.float64)
            
            # LiDAR到相机的旋转矩阵 (3x3)
            self.R = np.array(config['rotation'], dtype=np.float64)
            
            # LiDAR到相机的平移向量 (3x1)
            self.t = np.array(config['translation'], dtype=np.float64).reshape(3, 1)
            
            # 构建变换矩阵 (4x4)
            self.T = np.eye(4, dtype=np.float64)
            self.T[:3, :3] = self.R
            self.T[:3, 3:4] = self.t
            
            print(f"DAIR-V2X标定加载成功: {config_path}")
            print(f"相机内参 K:\n{self.K}")
            print(f"旋转矩阵 R:\n{self.R}")
            print(f"平移向量 t:\n{self.t.flatten()}")

        except (FileNotFoundError, KeyError) as e:
            print(f"错误: 无法加载或解析DAIR-V2X标定文件 {config_path}")
            print(f"具体错误: {e}")
            raise
    
    def _load_rcooper_config(self, lidar2cam_path: str, lidar2world_path: str = None):
        """加载RCOOPER格式的标定文件"""
        try:
            # 加载LiDAR到相机的标定
            with open(lidar2cam_path, 'r') as f:
                lidar2cam_config = json.load(f)
            
            # 获取cam_0的标定参数
            cam_config = lidar2cam_config['cam_0']
            
            # 相机内参矩阵 (3x3)
            self.K = np.array(cam_config['intrinsic'], dtype=np.float64)
            
            # 默认畸变参数（如果文件中没有提供）
            self.D = np.array([0, 0, 0, 0, 0], dtype=np.float64)
            
            # LiDAR到相机的变换矩阵 (4x4)
            extrinsic_matrix = np.array(cam_config['extrinsic'], dtype=np.float64)
            self.T = extrinsic_matrix
            
            # 提取旋转矩阵和平移向量
            self.R = self.T[:3, :3]
            self.t = self.T[:3, 3:4]
            
            print(f"RCOOPER LiDAR到相机标定加载成功: {lidar2cam_path}")
            print(f"相机内参 K:\n{self.K}")
            print(f"LiDAR到相机变换矩阵 T:\n{self.T}")
            
            # 如果提供了LiDAR到世界坐标系的标定文件，也加载它
            if lidar2world_path and os.path.exists(lidar2world_path):
                with open(lidar2world_path, 'r') as f:
                    lidar2world_config = json.load(f)
                
                # 构建LiDAR到世界坐标系的变换矩阵
                world_R = np.array(lidar2world_config['rotation'], dtype=np.float64)
                world_t = np.array(lidar2world_config['translation'], dtype=np.float64).reshape(3, 1)
                
                self.lidar2world_T = np.eye(4, dtype=np.float64)
                self.lidar2world_T[:3, :3] = world_R
                self.lidar2world_T[:3, 3:4] = world_t
                
                print(f"RCOOPER LiDAR到世界坐标系标定加载成功: {lidar2world_path}")
                print(f"LiDAR到世界坐标系变换矩阵:\n{self.lidar2world_T}")

        except (FileNotFoundError, KeyError) as e:
            print(f"错误: 无法加载或解析RCOOPER标定文件 {lidar2cam_path}")
            print(f"具体错误: {e}")
            raise

    def convert_3D_to_2D(self, points_3D):
        """将3D点云投影到2D图像平面，优化360度LiDAR过滤策略"""
        if points_3D is None or len(points_3D) == 0:
            return np.array([]), np.array([])
        
        points_3D_homo = np.hstack([points_3D, np.ones((points_3D.shape[0], 1))])
        points_cam = (self.T @ points_3D_homo.T).T[:, :3]
        
        # 优化的前方点过滤策略
        # 1. 基本深度过滤：Z > 0.05（降低阈值以保留更多点）
        basic_depth_mask = points_cam[:, 2] > 0.05
        
        # 2. 相机视野角度过滤：保留前方120度范围内的点
        # 使用X/Z比值来判断角度，tan(60°) ≈ 1.732
        angle_mask = np.abs(points_cam[:, 0] / np.maximum(points_cam[:, 2], 1e-6)) < 2.0  # 约120度视野
        
        # 3. 垂直视野过滤：保留合理的垂直角度范围
        vertical_mask = np.abs(points_cam[:, 1] / np.maximum(points_cam[:, 2], 1e-6)) < 1.5  # 约112度垂直视野
        
        # 组合所有过滤条件
        front_mask = basic_depth_mask & angle_mask & vertical_mask
        
        points_cam_valid = points_cam[front_mask]
        
        if len(points_cam_valid) == 0:
            return np.array([]), np.zeros(len(points_3D), dtype=bool)

        # 投影到图像平面
        points_2D_homo = (self.K @ points_cam_valid.T).T
        z_coords = points_2D_homo[:, 2:3]
        points_2D = points_2D_homo[:, :2] / np.where(np.abs(z_coords) < 1e-6, 1e-6, z_coords)
        
        # 图像边界过滤（假设图像尺寸，实际使用时会在detector中进一步过滤）
        img_width, img_height = 1920, 1200  # 默认图像尺寸
        boundary_mask = (
            (points_2D[:, 0] >= -img_width * 0.1) & (points_2D[:, 0] <= img_width * 1.1) &  # 允许稍微超出边界
            (points_2D[:, 1] >= -img_height * 0.1) & (points_2D[:, 1] <= img_height * 1.1)
        )
        
        # 最终有效点
        final_points_2D = points_2D[boundary_mask]
        
        # 构建最终的有效掩码
        final_valid_mask = np.zeros(len(points_3D), dtype=bool)
        front_indices = np.where(front_mask)[0]
        boundary_indices = front_indices[boundary_mask]
        final_valid_mask[boundary_indices] = True
        
        print(f"点云过滤统计: 原始={len(points_3D)}, 深度过滤={np.sum(basic_depth_mask)}, "
              f"角度过滤={np.sum(front_mask)}, 边界过滤={len(final_points_2D)}")
        
        return final_points_2D, final_valid_mask

    def convert_3D_to_camera_coords(self, points_3D):
        """将3D点从LiDAR坐标系转换到相机坐标系"""
        points_3D_homo = np.hstack([points_3D, np.ones((points_3D.shape[0], 1))])
        return (self.T @ points_3D_homo.T).T[:, :3]
    
    def convert_lidar_to_world(self, points_3D):
        """将3D点从LiDAR坐标系转换到世界坐标系（如果有lidar2world标定）"""
        if self.lidar2world_T is None:
            print("警告: 没有LiDAR到世界坐标系的标定信息")
            return points_3D
        
        points_3D_homo = np.hstack([points_3D, np.ones((points_3D.shape[0], 1))])
        return (self.lidar2world_T @ points_3D_homo.T).T[:, :3]
    
    @classmethod
    def create_from_rcooper_id(cls, calib_base_path: str, sensor_id: str):
        """
        根据传感器ID创建RCOOPER标定对象
        
        Args:
            calib_base_path: 标定文件基础路径（如 /path/to/calib）
            sensor_id: 传感器ID（如 "139"）
        """
        lidar2cam_path = os.path.join(calib_base_path, "lidar2cam", f"{sensor_id}.json")
        lidar2world_path = os.path.join(calib_base_path, "lidar2world", f"{sensor_id}.json")
        
        return cls(lidar2cam_path=lidar2cam_path, lidar2world_path=lidar2world_path)


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
    
    def _get_safe_depth_factor(self, depth_factor):
        """
        获取安全的depth_factor值
        原始算法中depth_factor > 1会导致深度范围调整失败
        """
        if depth_factor > 1.0:
            # 将大于1的值映射到0.5-0.9范围
            # depth_factor=20 -> 0.8, depth_factor=10 -> 0.85, depth_factor=5 -> 0.9
            safe_factor = max(0.5, 0.95 - depth_factor / 100.0)
            print(f"调整depth_factor: {depth_factor} -> {safe_factor}")
            return safe_factor
        else:
            return depth_factor
    
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
                frame, points, calibration, erosion_factor=25, depth_factor=self._get_safe_depth_factor(20)
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
                        'filtered_points': safe_to_list(filtered_points) if len(filtered_points) > 0 else [],
                        'yaw': 0.0  # 默认偏航角
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
                
                bev_frame = self.create_bev_view(combined_points, all_corners_3D)
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
                        
                        # 构建标签文本
                        label_parts = [
                            f"ID:{result['id']}", 
                            result['class_name'], 
                            f"Dist:{result['distance_from_origin']:.1f}m"
                        ]
                        
                        if result['speed_3d_kmh'] is not None:
                            label_parts.append(f"Speed:{result['speed_3d_kmh']:.1f}km/h")
                        
                        label_text = ' '.join(label_parts)
                        
                        # 绘制带阴影的文本（完全参考v2x_speed_visualization.py）
                        cv2.putText(result_image, label_text, top_left_front_pt, 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 3, cv2.LINE_AA)
                        cv2.putText(result_image, label_text, top_left_front_pt, 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 1, cv2.LINE_AA)
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
        
        # 绘制人车距离信息
        if person_vehicle_distances:
            y_offset = 30
            for i, dist_info in enumerate(person_vehicle_distances[:5]):  # 只显示前5个
                text = f"Person{dist_info['person_id']}-Vehicle{dist_info['vehicle_id']}: {dist_info['distance_xy']:.1f}m"
                cv2.putText(result_image, text, (10, y_offset + i*25), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        return result_image

    def create_bev_view(self, points, all_corners_3D, max_points=5000):
        """创建鸟瞰图视图，参考v2x_speed_visualization.py的实现"""
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
            for i, corners_3D in enumerate(all_corners_3D):
                # 使用蓝色表示预测的3D边界框
                color = (255, 0, 0)  # 蓝色
                
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
                    cv2.putText(bev_image, str(i), 
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


class VideoStreamProcessor:
    """视频流处理器，支持RGB视频和LiDAR数据流的同步处理"""
    
    def __init__(self, video_path: str, lidar_dir: str, calibration, view_name: str, 
                 output_dir: str = "./output", max_frames: int = None, tracking: bool = True,
                 use_improved_fusion: bool = True, show_realtime: bool = False):
        """
        初始化视频流处理器
        
        Args:
            video_path: RGB视频文件路径
            lidar_dir: LiDAR数据目录路径
            calibration: 标定对象
            view_name: 视角名称
            output_dir: 输出目录
            max_frames: 最大处理帧数
            tracking: 是否启用跟踪
            use_improved_fusion: 是否使用改进的融合算法
            show_realtime: 是否实时显示
        """
        self.video_path = video_path
        self.lidar_dir = lidar_dir
        self.calibration = calibration
        self.view_name = view_name
        self.output_dir = output_dir
        self.max_frames = max_frames
        self.tracking = tracking
        self.use_improved_fusion = use_improved_fusion
        self.show_realtime = show_realtime
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 初始化检测器
        self.detector = V2XEnhanced3DDetector(
            tracking=tracking,
            view_prefix=view_name,
            use_improved_fusion=use_improved_fusion
        )
        
        # 视频和数据队列
        self.frame_queue = queue.Queue(maxsize=30)
        self.lidar_queue = queue.Queue(maxsize=30)
        self.result_queue = queue.Queue()
        
        # 控制标志
        self.stop_flag = threading.Event()
        self.processing_complete = threading.Event()
        
        # 统计信息
        self.frame_count = 0
        self.processed_count = 0
        self.start_time = None
        
        # 结果存储
        self.frame_results = []
        self.processed_frames = []
        self.bev_frames = []
    
    def load_video_frames(self):
        """视频帧加载线程"""
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            print(f"错误: 无法打开视频文件 {self.video_path}")
            self.stop_flag.set()
            return
        
        # 获取视频信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"视频信息: FPS={fps:.1f}, 总帧数={total_frames}")
        
        frame_idx = 0
        while not self.stop_flag.is_set():
            ret, frame = cap.read()
            if not ret:
                print("视频读取完成")
                break
            
            if self.max_frames and frame_idx >= self.max_frames:
                print(f"达到最大帧数限制: {self.max_frames}")
                break
            
            try:
                self.frame_queue.put((frame_idx, frame), timeout=1.0)
                frame_idx += 1
            except queue.Full:
                print("视频帧队列满，跳过当前帧")
                continue
        
        cap.release()
        print(f"视频加载线程结束，共加载 {frame_idx} 帧")
    
    def load_lidar_data(self):
        """LiDAR数据加载线程"""
        # 获取LiDAR文件列表
        lidar_files = sorted(glob.glob(os.path.join(self.lidar_dir, "*.pcd")))
        
        if len(lidar_files) == 0:
            print(f"错误: 在 {self.lidar_dir} 中没有找到PCD文件")
            self.stop_flag.set()
            return
        
        print(f"找到 {len(lidar_files)} 个LiDAR文件")
        
        for idx, pcd_file in enumerate(lidar_files):
            if self.stop_flag.is_set():
                break
            
            if self.max_frames and idx >= self.max_frames:
                break
            
            try:
                # 加载点云
                pcd = o3d.io.read_point_cloud(pcd_file)
                points = np.asarray(pcd.points, dtype=np.float64)
                
                # 过滤无效点
                if len(points) > 0:
                    if np.any(np.isnan(points)) or np.any(np.isinf(points)):
                        valid_mask = np.isfinite(points).all(axis=1)
                        points = points[valid_mask]
                
                self.lidar_queue.put((idx, points), timeout=1.0)
                
            except Exception as e:
                print(f"加载LiDAR文件失败 {pcd_file}: {e}")
                continue
        
        print(f"LiDAR加载线程结束，共加载 {len(lidar_files)} 个文件")
    
    def process_frames(self):
        """帧处理线程"""
        print(f"开始处理 {self.view_name} 视频流...")
        self.start_time = time.time()
        
        while not self.stop_flag.is_set():
            try:
                # 获取视频帧和LiDAR数据
                frame_data = self.frame_queue.get(timeout=2.0)
                lidar_data = self.lidar_queue.get(timeout=2.0)
                
                frame_idx, frame = frame_data
                lidar_idx, points = lidar_data
                
                # 检查帧同步（简单的索引匹配）
                if frame_idx != lidar_idx:
                    print(f"警告: 帧不同步 - 视频帧{frame_idx} vs LiDAR帧{lidar_idx}")
                
                # 处理当前帧
                frame_result = self.detector.process_frame_enhanced(
                    frame, points, self.calibration, frame_idx, fps=30
                )
                
                # 添加视角和文件信息
                frame_result['view_name'] = self.view_name
                frame_result['frame_index'] = frame_idx
                frame_result['processing_time'] = time.time() - self.start_time
                
                # 获取可视化结果
                result_frame = frame_result['visualization_data']['processed_frame']
                bev_frame = frame_result['visualization_data']['bev_frame']
                
                # 确保结果帧格式正确
                if not isinstance(result_frame, np.ndarray):
                    result_frame = np.array(result_frame)
                result_frame = np.ascontiguousarray(result_frame, dtype=np.uint8)
                
                if isinstance(bev_frame, np.ndarray) and len(bev_frame.shape) == 3:
                    bev_frame = np.ascontiguousarray(bev_frame, dtype=np.uint8)
                else:
                    bev_frame = np.zeros((500, 500, 3), dtype=np.uint8)
                
                # 添加帧信息
                info_text = f'{self.view_name} Frame: {frame_idx} | Objects: {frame_result["statistics"]["total_objects"]} | P-V: {len(frame_result["person_vehicle_distances"])}'
                cv2.putText(result_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(bev_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                # 存储结果
                self.frame_results.append(frame_result)
                self.processed_frames.append(result_frame)
                self.bev_frames.append(bev_frame)
                
                self.processed_count += 1
                
                # 实时显示
                if self.show_realtime:
                    cv2.imshow(f"{self.view_name}_Detection", result_frame)
                    cv2.imshow(f"{self.view_name}_BEV", bev_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.stop_flag.set()
                        break
                
                # 进度报告
                if self.processed_count % 10 == 0:
                    elapsed_time = time.time() - self.start_time
                    fps = self.processed_count / elapsed_time
                    print(f"{self.view_name}: 已处理 {self.processed_count} 帧, FPS: {fps:.1f}")
                
                # 内存管理
                if self.processed_count % 50 == 0:
                    gc.collect()
                
            except queue.Empty:
                # 检查是否所有数据都已处理完成
                if self.frame_queue.empty() and self.lidar_queue.empty():
                    print(f"{self.view_name}: 所有帧处理完成")
                    break
                continue
            except Exception as e:
                print(f"处理帧时出错: {e}")
                continue
        
        self.processing_complete.set()
        if self.show_realtime:
            cv2.destroyAllWindows()
        
        print(f"{self.view_name}: 处理线程结束，共处理 {self.processed_count} 帧")
    
    def start_stream_processing(self):
        """启动视频流处理"""
        print(f"启动 {self.view_name} 视频流处理...")
        
        # 启动加载线程
        video_thread = threading.Thread(target=self.load_video_frames, daemon=True)
        lidar_thread = threading.Thread(target=self.load_lidar_data, daemon=True)
        process_thread = threading.Thread(target=self.process_frames, daemon=True)
        
        video_thread.start()
        lidar_thread.start()
        time.sleep(1)  # 让加载线程先启动
        process_thread.start()
        
        # 等待处理完成
        try:
            process_thread.join()
        except KeyboardInterrupt:
            print(f"\n用户中断 {self.view_name} 处理")
            self.stop_flag.set()
        
        # 等待所有线程结束
        video_thread.join(timeout=5)
        lidar_thread.join(timeout=5)
        
        return self.save_results()
    
    def save_results(self):
        """保存处理结果"""
        if not self.frame_results:
            print(f"{self.view_name}: 没有结果需要保存")
            return []
        
        view_output_dir = os.path.join(self.output_dir, self.view_name)
        os.makedirs(view_output_dir, exist_ok=True)
        
        # 精简结果用于JSON输出
        simplified_json_results = simplify_results_for_json(self.frame_results)
        
        # 保存JSON结果
        results_file = os.path.join(view_output_dir, f'{self.view_name}_stream_results.json')
        with open(results_file, 'w') as f:
            json.dump(simplified_json_results, f, indent=2, default=str)
        
        # 创建输出视频
        if self.processed_frames and len(self.processed_frames) > 0:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            
            # 检测视频
            det_video_path = os.path.join(view_output_dir, f'{self.view_name}_stream_detection.mp4')
            height, width = self.processed_frames[0].shape[:2]
            out_det = cv2.VideoWriter(det_video_path, fourcc, 10, (width, height))
            for frame in self.processed_frames:
                out_det.write(frame)
            out_det.release()
            
            # BEV视频
            if self.bev_frames and len(self.bev_frames) > 0:
                bev_video_path = os.path.join(view_output_dir, f'{self.view_name}_stream_bev.mp4')
                height, width = self.bev_frames[0].shape[:2]
                out_bev = cv2.VideoWriter(bev_video_path, fourcc, 10, (width, height))
                for frame in self.bev_frames:
                    out_bev.write(frame)
                out_bev.release()
                
                print(f"\n{self.view_name} 视频流处理完成:")
                print(f"  - 结果文件: {results_file}")
                print(f"  - 检测视频: {det_video_path}")
                print(f"  - BEV视频: {bev_video_path}")
            else:
                print(f"\n{self.view_name} 视频流处理完成:")
                print(f"  - 结果文件: {results_file}")
                print(f"  - 检测视频: {det_video_path}")
        
        # 统计信息
        total_detections = sum(len(fr['detections']) for fr in self.frame_results)
        total_person_vehicle_pairs = sum(len(fr['person_vehicle_distances']) for fr in self.frame_results)
        
        print(f"  - 总帧数: {len(self.frame_results)}")
        print(f"  - 总检测数: {total_detections}")
        print(f"  - 人车距离对: {total_person_vehicle_pairs}")
        
        if total_person_vehicle_pairs > 0:
            all_distances = []
            for fr in self.frame_results:
                all_distances.extend([d['distance_xy'] for d in fr['person_vehicle_distances']])
            
            if all_distances:
                print(f"  - 人车距离统计:")
                print(f"    最小: {min(all_distances):.2f}m")
                print(f"    最大: {max(all_distances):.2f}m")
                print(f"    平均: {sum(all_distances)/len(all_distances):.2f}m")
        
        # 处理性能统计
        if self.start_time:
            total_time = time.time() - self.start_time
            avg_fps = self.processed_count / total_time
            print(f"  - 处理性能: {avg_fps:.1f} FPS, 总耗时: {total_time:.1f}s")
        
        return self.frame_results


def process_dair_view(view_path, view_name, calibration, max_frames=100, tracking=True, show=False, use_improved_fusion=True):
    """处理单个通用视角的数据"""
    print(f"\n=== 开始处理 {view_name} ===")
    
    try:
        # 定义图像和点云目录
        image_dir = os.path.join(view_path, 'cam-0')
        velodyne_dir = os.path.join(view_path, 'lidar')

        if not os.path.isdir(image_dir) or not os.path.isdir(velodyne_dir):
            print(f"错误: 在 {view_path} 中找不到 'image' 或 'velodyne' 文件夹。")
            return []
        
        # 获取文件列表
        image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))])
        velodyne_files = sorted([f for f in os.listdir(velodyne_dir) if f.endswith('.pcd')])
        
        if len(image_files) != len(velodyne_files):
            print("警告: 图像和点云文件数量不匹配。将处理较少数量的帧。")
        
        print(f"找到 {len(image_files)} 个图像文件，{len(velodyne_files)} 个点云文件")
        
        # 限制处理帧数
        if max_frames:
            min_files = min(len(image_files), len(velodyne_files))
            num_frames_to_process = min(max_frames, min_files)
            image_files = image_files[:num_frames_to_process]
            velodyne_files = velodyne_files[:num_frames_to_process]
        
        # 初始化检测器
        if len(image_files) == 0:
            print(f"错误：{view_name} 中没有找到可处理的帧")
            return []

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
    parser = argparse.ArgumentParser(description="可配置的多视角3D检测器，支持一或两个视角，支持DAIR-V2X和RCOOPER数据集，支持视频流处理")
    
    # 处理模式选择
    parser.add_argument('--mode', type=str, choices=['images', 'video'], default='images',
                       help="处理模式: 'images' 处理图像序列, 'video' 处理视频流")
    
    # 视角1参数
    parser.add_argument('--view1-path', type=str, required=True,
                       help="视角1的数据路径，图像模式需包含 'cam-0' 和 'lidar' 子文件夹，视频模式为视频文件路径")
    
    # 视角1 LiDAR路径（视频模式专用）
    parser.add_argument('--view1-lidar-dir', type=str,
                       help="视角1的LiDAR数据目录路径（仅视频模式需要）")
    
    # 视角1标定参数 - 支持两种格式
    calib_group1 = parser.add_mutually_exclusive_group(required=True)
    calib_group1.add_argument('--view1-calib', type=str,
                             help="视角1的相机标定JSON文件路径（DAIR-V2X格式）")
    calib_group1.add_argument('--view1-rcooper-calib', type=str, nargs=2, 
                             metavar=('LIDAR2CAM', 'LIDAR2WORLD'),
                             help="视角1的RCOOPER格式标定文件路径：lidar2cam.json lidar2world.json")
    calib_group1.add_argument('--view1-rcooper-id', type=str, nargs=2,
                             metavar=('CALIB_BASE_PATH', 'SENSOR_ID'),
                             help="视角1的RCOOPER格式标定：标定基础路径 传感器ID（如 /path/to/calib 139）")
    
    parser.add_argument('--view1-name', type=str, default="view1",
                       help="为视角1指定一个名称")

    # 视角2参数 (可选)
    parser.add_argument('--view2-path', type=str,
                       help="视角2的数据路径")
    
    # 视角2 LiDAR路径（视频模式专用）
    parser.add_argument('--view2-lidar-dir', type=str,
                       help="视角2的LiDAR数据目录路径（仅视频模式需要）")
    
    # 视角2标定参数 - 支持两种格式
    calib_group2 = parser.add_mutually_exclusive_group()
    calib_group2.add_argument('--view2-calib', type=str,
                             help="视角2的相机标定JSON文件路径（DAIR-V2X格式）")
    calib_group2.add_argument('--view2-rcooper-calib', type=str, nargs=2,
                             metavar=('LIDAR2CAM', 'LIDAR2WORLD'),
                             help="视角2的RCOOPER格式标定文件路径：lidar2cam.json lidar2world.json")
    calib_group2.add_argument('--view2-rcooper-id', type=str, nargs=2,
                             metavar=('CALIB_BASE_PATH', 'SENSOR_ID'),
                             help="视角2的RCOOPER格式标定：标定基础路径 传感器ID（如 /path/to/calib 139）")
    
    parser.add_argument('--view2-name', type=str, default="view2",
                       help="为视角2指定一个名称")

    # 通用参数
    parser.add_argument('--max-frames', type=int, default=100,
                       help="每个视角最大处理帧数")
    parser.add_argument('--no-tracking', action='store_true',
                       help="禁用目标跟踪")
    parser.add_argument('--show', action='store_true',
                       help="实时显示处理结果")
    parser.add_argument('--use-improved-fusion', action='store_true',
                       help="使用改进的融合算法")
    
    # 视频流专用参数
    parser.add_argument('--realtime-display', action='store_true',
                       help="视频模式下实时显示处理结果")
    parser.add_argument('--parallel-streams', action='store_true',
                       help="并行处理多个视频流（默认为顺序处理）")
    
    args = parser.parse_args()
    
    print(f"=== 可配置的多视角3D检测器启动 ({args.mode}模式) ===")

    # 准备处理任务列表
    tasks = []
    
    # 处理视角1
    if args.view1_path:
        if not os.path.exists(args.view1_path):
            print(f"错误: 视角1路径不存在: {args.view1_path}")
            return
        
        # 视频模式需要额外检查LiDAR目录
        if args.mode == 'video' and not args.view1_lidar_dir:
            print("错误: 视频模式需要提供 --view1-lidar-dir 参数")
            return
        
        if args.mode == 'video' and args.view1_lidar_dir and not os.path.exists(args.view1_lidar_dir):
            print(f"错误: 视角1 LiDAR目录不存在: {args.view1_lidar_dir}")
            return
        
        # 确定标定参数
        calib_config = None
        if args.view1_calib:
            if not os.path.exists(args.view1_calib):
                print(f"错误: 视角1标定文件不存在: {args.view1_calib}")
                return
            calib_config = {'type': 'dair_v2x', 'config_path': args.view1_calib}
        elif args.view1_rcooper_calib:
            lidar2cam_path, lidar2world_path = args.view1_rcooper_calib
            if not os.path.exists(lidar2cam_path):
                print(f"错误: 视角1 lidar2cam文件不存在: {lidar2cam_path}")
                return
            if not os.path.exists(lidar2world_path):
                print(f"错误: 视角1 lidar2world文件不存在: {lidar2world_path}")
                return
            calib_config = {'type': 'rcooper_files', 'lidar2cam_path': lidar2cam_path, 'lidar2world_path': lidar2world_path}
        elif args.view1_rcooper_id:
            calib_base_path, sensor_id = args.view1_rcooper_id
            if not os.path.exists(calib_base_path):
                print(f"错误: 视角1标定基础路径不存在: {calib_base_path}")
                return
            calib_config = {'type': 'rcooper_id', 'calib_base_path': calib_base_path, 'sensor_id': sensor_id}
        
        if calib_config:
            task = {
                "path": args.view1_path,
                "calib_config": calib_config,
                "name": args.view1_name,
                "mode": args.mode
            }
            if args.mode == 'video':
                task["lidar_dir"] = args.view1_lidar_dir
            tasks.append(task)
            print(f"已配置视角1: {args.view1_name} | 路径: {args.view1_path} | 模式: {args.mode} | 标定类型: {calib_config['type']}")

    # 处理视角2
    if args.view2_path:
        if not os.path.exists(args.view2_path):
            print(f"错误: 视角2路径不存在: {args.view2_path}")
            return
        
        # 视频模式需要额外检查LiDAR目录
        if args.mode == 'video' and not args.view2_lidar_dir:
            print("错误: 视频模式需要提供 --view2-lidar-dir 参数")
            return
        
        if args.mode == 'video' and args.view2_lidar_dir and not os.path.exists(args.view2_lidar_dir):
            print(f"错误: 视角2 LiDAR目录不存在: {args.view2_lidar_dir}")
            return
        
        # 确定标定参数
        calib_config = None
        if args.view2_calib:
            if not os.path.exists(args.view2_calib):
                print(f"错误: 视角2标定文件不存在: {args.view2_calib}")
                return
            calib_config = {'type': 'dair_v2x', 'config_path': args.view2_calib}
        elif args.view2_rcooper_calib:
            lidar2cam_path, lidar2world_path = args.view2_rcooper_calib
            if not os.path.exists(lidar2cam_path):
                print(f"错误: 视角2 lidar2cam文件不存在: {lidar2cam_path}")
                return
            if not os.path.exists(lidar2world_path):
                print(f"错误: 视角2 lidar2world文件不存在: {lidar2world_path}")
                return
            calib_config = {'type': 'rcooper_files', 'lidar2cam_path': lidar2cam_path, 'lidar2world_path': lidar2world_path}
        elif args.view2_rcooper_id:
            calib_base_path, sensor_id = args.view2_rcooper_id
            if not os.path.exists(calib_base_path):
                print(f"错误: 视角2标定基础路径不存在: {calib_base_path}")
                return
            calib_config = {'type': 'rcooper_id', 'calib_base_path': calib_base_path, 'sensor_id': sensor_id}
        
        if calib_config:
            task = {
                "path": args.view2_path,
                "calib_config": calib_config,
                "name": args.view2_name,
                "mode": args.mode
            }
            if args.mode == 'video':
                task["lidar_dir"] = args.view2_lidar_dir
            tasks.append(task)
            print(f"已配置视角2: {args.view2_name} | 路径: {args.view2_path} | 模式: {args.mode} | 标定类型: {calib_config['type']}")

    if not tasks:
        print("错误: 没有有效的视角被配置，请至少提供 --view1-path 和相应的标定参数。")
        return

    # 创建输出目录
    os.makedirs('./output', exist_ok=True)
    all_results = {}

    # 根据处理模式选择不同的处理方式
    if args.mode == 'images':
        # 图像序列处理模式（原有逻辑）
        if args.show:
            print("\n检测到 --show 参数，将顺序处理以支持GUI显示...")
            for task in tasks:
                try:
                    calibration = create_calibration_from_config(task["calib_config"])
                    results = process_dair_view(
                        view_path=task["path"],
                        view_name=task["name"],
                        calibration=calibration,
                        max_frames=args.max_frames,
                        tracking=not args.no_tracking,
                        show=True,
                        use_improved_fusion=args.use_improved_fusion
                    )
                    all_results[task["name"]] = results
                except Exception as e:
                    print(f"处理视角 {task['name']} 时发生严重错误: {e}")
                    traceback.print_exc()
        else:
            print(f"\n开始并行处理 {len(tasks)} 个视角...")
            # 准备并行处理的参数
            process_args = []
            for task in tasks:
                process_args.append((
                    task["path"],
                    task["name"],
                    task["calib_config"], # 传递配置字典而非对象
                    args.max_frames,
                    not args.no_tracking,
                    False, # show=False
                    args.use_improved_fusion
                ))
            
            # 使用多进程池
            with Pool(processes=len(tasks)) as pool:
                try:
                    results_list = pool.starmap(processing_wrapper, process_args)
                    for i, task in enumerate(tasks):
                        all_results[task["name"]] = results_list[i]
                except Exception as e:
                    print(f"多进程处理失败: {e}")
    
    elif args.mode == 'video':
        # 视频流处理模式
        print(f"\n开始处理 {len(tasks)} 个视频流...")
        
        if args.parallel_streams and len(tasks) > 1:
            # 并行处理多个视频流
            print("使用并行模式处理多个视频流...")
            
            def process_video_stream_wrapper(task):
                try:
                    calibration = create_calibration_from_config(task["calib_config"])
                    processor = VideoStreamProcessor(
                        video_path=task["path"],
                        lidar_dir=task["lidar_dir"],
                        calibration=calibration,
                        view_name=task["name"],
                        output_dir="./output",
                        max_frames=args.max_frames,
                        tracking=not args.no_tracking,
                        use_improved_fusion=args.use_improved_fusion,
                        show_realtime=False  # 并行模式不支持实时显示
                    )
                    return processor.start_stream_processing()
                except Exception as e:
                    print(f"处理视频流 {task['name']} 时发生错误: {e}")
                    return []
            
            # 使用多进程池处理视频流
            with Pool(processes=len(tasks)) as pool:
                try:
                    results_list = pool.map(process_video_stream_wrapper, tasks)
                    for i, task in enumerate(tasks):
                        all_results[task["name"]] = results_list[i]
                except Exception as e:
                    print(f"并行视频流处理失败: {e}")
        else:
            # 顺序处理视频流
            for task in tasks:
                try:
                    calibration = create_calibration_from_config(task["calib_config"])
                    processor = VideoStreamProcessor(
                        video_path=task["path"],
                        lidar_dir=task["lidar_dir"],
                        calibration=calibration,
                        view_name=task["name"],
                        output_dir="./output",
                        max_frames=args.max_frames,
                        tracking=not args.no_tracking,
                        use_improved_fusion=args.use_improved_fusion,
                        show_realtime=args.realtime_display
                    )
                    results = processor.start_stream_processing()
                    all_results[task["name"]] = results
                except Exception as e:
                    print(f"处理视频流 {task['name']} 时发生严重错误: {e}")
                    traceback.print_exc()

    # 精简并保存最终的汇总结果
    simplified_summary = {
        view_name: simplify_results_for_json(result_list)
        for view_name, result_list in all_results.items() if result_list
    }

    summary_file = f'./output/configurable_v2x_{args.mode}_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(simplified_summary, f, indent=2, default=str)
    
    print(f"\n=== 所有视角处理完成 ===")
    print(f"汇总结果已保存到: {summary_file}")
    
    # 处理性能统计
    total_frames = sum(len(result_list) for result_list in all_results.values())
    total_detections = 0
    total_person_vehicle_pairs = 0
    
    for result_list in all_results.values():
        for frame_result in result_list:
            total_detections += len(frame_result.get('detections', []))
            total_person_vehicle_pairs += len(frame_result.get('person_vehicle_distances', []))
    
    print(f"总体统计:")
    print(f"  - 处理视角数: {len(all_results)}")
    print(f"  - 总处理帧数: {total_frames}")
    print(f"  - 总检测数: {total_detections}")
    print(f"  - 人车距离对: {total_person_vehicle_pairs}")
    
    if args.mode == 'video':
        print(f"  - 处理模式: 视频流")
        print(f"  - 并行处理: {'是' if args.parallel_streams else '否'}")
        print(f"  - 实时显示: {'是' if args.realtime_display else '否'}")
    else:
        print(f"  - 处理模式: 图像序列")
        print(f"  - GUI显示: {'是' if args.show else '否'}")
    

def create_calibration_from_config(calib_config: Dict[str, Any]) -> CustomCalibration:
    """根据配置字典创建标定对象"""
    if calib_config['type'] == 'dair_v2x':
        return CustomCalibration(config_path=calib_config['config_path'])
    elif calib_config['type'] == 'rcooper_files':
        return CustomCalibration(
            lidar2cam_path=calib_config['lidar2cam_path'],
            lidar2world_path=calib_config['lidar2world_path']
        )
    elif calib_config['type'] == 'rcooper_id':
        return CustomCalibration.create_from_rcooper_id(
            calib_config['calib_base_path'],
            calib_config['sensor_id']
        )
    else:
        raise ValueError(f"未知的标定类型: {calib_config['type']}")


def processing_wrapper(view_path, view_name, calib_config, *args):
    """多进程包装函数，在子进程中创建标定对象"""
    try:
        calibration = create_calibration_from_config(calib_config)
        return process_dair_view(view_path, view_name, calibration, *args)
    except Exception as e:
        print(f"子进程 {view_name} 发生错误: {e}")
        return []


if __name__ == "__main__":
    main() 