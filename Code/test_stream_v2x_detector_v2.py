#!/usr/bin/env python3
"""
V2X流式数据检测脚本
结合RGB相机流和LiDAR数据流进行实时3D目标检测
支持双视角固定IP地址配置，只输出JSON检测结果
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import numpy as np
import json
import time
from collections import deque, defaultdict
import argparse
import threading
import queue
import gc
import traceback
import sys
from typing import Dict, List, Tuple, Optional, Any

# 导入流式数据读取模块
from lidar_rgbcamera import RGBCamera, Tanwaylidar

# 导入检测相关模块
from ultralytics import YOLO
from detector import YOLOv8Detector
from calibration import LiDAR2Camera
from data_processing import *
from utils import *
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # 禁用tokenizers并行，避免fork问题
os.environ['OMP_NUM_THREADS'] = '1'  # 限制OpenMP线程数
os.environ['PYTHONHASHSEED'] = '0'  # 设置Python哈希种子为固定值

# 全局变量：维护每个视角最近5帧的检测结果
global_frame_results = {}
global_results_lock = threading.Lock()  # 线程锁保证线程安全

def get_latest_results(view_name: str = None) -> Dict:
    """
    获取最新的检测结果
    
    Args:
        view_name: 视角名称，如果为None则返回所有视角的结果
        
    Returns:
        字典，包含请求的视角结果
    """
    with global_results_lock:
        if view_name:
            return {view_name: list(global_frame_results.get(view_name, deque()))}
        else:
            return {name: list(frames) for name, frames in global_frame_results.items()}

def update_global_results(view_name: str, frame_result: Dict):
    """
    更新全局结果队列
    
    Args:
        view_name: 视角名称
        frame_result: 帧检测结果
    """
    with global_results_lock:
        if view_name not in global_frame_results:
            global_frame_results[view_name] = deque(maxlen=5)  # 最多保存5帧
        
        # 添加新的帧结果
        global_frame_results[view_name].append(frame_result)
        save_global_results_to_json()
def save_global_results_to_json(file_path = "result/global_frame_result.json"):
    """
    将 global_frame_results 保存为 JSON 文件（每次覆盖之前的记录）
    
    Args:
        file_path: 保存文件路径
    """
    # 将 deque 转为 list，否则 json 无法序列化
    serializable_results = {
        view_name: list(frames) for view_name, frames in global_frame_results.items()
    }

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, ensure_ascii=False, indent=4)
# 固定的IP地址配置
VIEW_CONFIGS = {
    'view1': {
        'camera_ip': 'rtsp://admin:Wuhan.123@192.168.20.220',
        'lidar_ip': '192.168.20.221',
        'name': 'infrastructure_view1'
    },
    'view2': {
        'camera_ip': 'rtsp://admin:Wuhan.123@192.168.20.222',
        'lidar_ip': '192.168.20.223',
        'name': 'infrastructure_view2'
    }
}

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


def simplify_results_for_json(results_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """精简JSON输出，只保留关键车辆信息"""
    simplified_list = []
    vehicle_classes = {1, 2, 3, 5, 6, 7}  # bicycle, car, motorcycle, bus, truck

    for frame_data in results_list:
        simplified_frame = {
            'frame_id': frame_data.get('frame_id'),
            'timestamp': frame_data.get('timestamp'),
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
                    'class_name': det['class_name'],
                    'position_3d': [round(p, 2) for p in det.get('center_3d', [0, 0, 0])],
                    'speed_kmh': round(det['speed_3d_kmh'], 2) if det.get('speed_3d_kmh') is not None else None,
                    'distances_to_persons': vehicle_distances.get(det['id'], [])
                }
                simplified_frame['vehicles'].append(vehicle_info)
        
        # 只在有车辆时才添加该帧
        if simplified_frame['vehicles']:
            simplified_list.append(simplified_frame)
            
    return simplified_list


class StreamV2XDetector:
    """流式V2X检测器"""
    
    def __init__(self, model_path="yolov8m-seg.pt", tracking=True, view_prefix=""):
        print(f"初始化YOLO模型: {model_path}")
        self.detector = YOLOv8Detector(model_path, tracking=tracking, PCA=False)
        
        self.tracking = tracking
        self.view_prefix = view_prefix
        self.names = self.detector.model.names
        
        # 类别定义
        self.person_classes = [0]  # person
        self.vehicle_classes = [1, 2, 3, 5, 6, 7]  # bicycle, car, motorcycle, bus, truck
        self.frame_result = {
                'frame_id': None,
                'timestamp': time.time(),
                'detections': [],
                'person_vehicle_distances': [],
                'statistics': {
                    'total_objects': 0,
                    'person_count': 0,
                    'vehicle_count': 0,
                    'min_person_vehicle_distance': None,
                    'avg_person_vehicle_distance': None
                }
            }
    
    def _get_safe_depth_factor(self, depth_factor):
        """获取安全的depth_factor值"""
        if depth_factor > 1.0:
            safe_factor = max(0.5, 0.95 - depth_factor / 100.0)
            return safe_factor
        else:
            return depth_factor
    
    def calculate_xy_distance(self, corners_3D_1, corners_3D_2):
        """计算两个3D边界框之间的X-Y平面距离"""
        if corners_3D_1 is None or corners_3D_2 is None:
            return None
        
        center_1 = np.mean(corners_3D_1, axis=0)
        center_2 = np.mean(corners_3D_2, axis=0)
        distance_xy = np.linalg.norm(center_1[:2] - center_2[:2])
        return distance_xy
    
    def calculate_person_vehicle_angle(self, person_center_3d, vehicle_center_3d):
        """
        计算人车连线与Y轴（前进方向）的夹角
        
        Args:
            person_center_3d: 人的3D中心坐标 [x, y, z]
            vehicle_center_3d: 车辆的3D中心坐标 [x, y, z]
            
        Returns:
            angle_degrees: 与Y轴正方向的夹角（度）
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
            
            # Y轴正方向向量（LiDAR坐标系中的前进方向）
            y_axis_vector = np.array([0, -1])
            
            # 计算夹角（使用点积公式）
            cos_angle = np.dot(direction_xy, y_axis_vector) / (vector_length * np.linalg.norm(y_axis_vector))
            
            # 限制cos值范围，避免数值误差
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            
            # 计算角度（弧度转度）
            angle_radians = np.arccos(cos_angle)
            angle_degrees = np.degrees(angle_radians)
            
            # 判断方向类型
            if angle_degrees < 5.0:
                direction_type = "DIRECT_APPROACH"  # 直接朝向
            elif angle_degrees < 15.0:
                direction_type = "BASIC_TOWARD"     # 基本朝向
            elif angle_degrees < 45.0:
                direction_type = "DIAGONAL_APPROACH"  # 斜向接近
            elif angle_degrees < 75.0:
                direction_type = "LATERAL_MOTION"   # 侧向运动
            else:
                direction_type = "BYPASS_MOVEMENT"  # 绕行运动
            
            return angle_degrees, direction_type
            
        except Exception as e:
            print(f"计算人车夹角时出错: {e}")
            return 0.0, "CALC_ERROR"
    
    def calculate_person_vehicle_distances(self, detection_results):
        """计算人和车辆之间的距离和角度（只使用X-Y坐标）"""
        persons = []
        vehicles = []
        
        # 分类检测结果
        for result in detection_results:
            if result['class'] in self.person_classes:
                persons.append(result)
            elif result['class'] in self.vehicle_classes:
                vehicles.append(result)
        
        # 计算人车距离和角度
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
                    
                    # 计算人车角度
                    angle_degrees, direction_type = self.calculate_person_vehicle_angle(
                        person_center, vehicle_center
                    )
                    
                    person_vehicle_distances.append({
                        'person_id': person['id'],
                        'person_class': self.names[person['class']],
                        'vehicle_id': vehicle['id'],
                        'vehicle_class': self.names[vehicle['class']],
                        'distance_xy': distance_xy,
                        'angle_degrees': round(angle_degrees, 2),
                        'direction_type': direction_type,
                        'person_center_xy': safe_to_list(person_center[:2]),
                        'vehicle_center_xy': safe_to_list(vehicle_center[:2]),
                        'person_center_3d': safe_to_list(person_center),
                        'vehicle_center_3d': safe_to_list(vehicle_center)
                    })
        
        return person_vehicle_distances
    
    def process_frame(self, frame, points, calibration, frame_id=0) -> Dict[str, Any]:
        """处理单帧数据"""
        frame_start_time = time.time()
        
        try:
            # 检测和融合处理
            objects3d_data, all_corners_3D, pts_3D, pts_2D, all_filtered_points_of_object, all_object_IDs = self.detector.process_frame(
                frame, points, calibration, erosion_factor=25, depth_factor=self._get_safe_depth_factor(20)
            )
            
            # 数据结构初始化
            
            
            # 检测结果转换
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
                    # 优化：不保存完整的点云数据，只保存统计信息以减少内存和序列化开销
                    detection_result = {
                        'id': unique_id,
                        'class': ROS_type,
                        'class_name': self.names[ROS_type],
                        'confidence': 0.5,
                        'corners_3D': safe_to_list(corners_3D),
                        'center_3d': safe_to_list(center_3d),
                        'dimensions_3d': safe_to_list(dimensions_3d),
                        'distance_from_origin': distance_from_origin,
                        'speed_3d_kmh': speed_3d_kmh,
                        'filtered_points_count': len(filtered_points) if len(filtered_points) > 0 else 0,  # 只保存数量，不保存完整点云
                        'yaw': 0.0
                    }
                    
                    detection_results.append(detection_result)
            
            # 计算人车距离
            person_vehicle_distances = self.calculate_person_vehicle_distances(detection_results)
            
            # 统计计算
            person_count = sum(1 for d in detection_results if d['class'] in self.person_classes)
            vehicle_count = sum(1 for d in detection_results if d['class'] in self.vehicle_classes)
            
            min_distance = None
            avg_distance = None
            if person_vehicle_distances:
                distances = [d['distance_xy'] for d in person_vehicle_distances]
                min_distance = min(distances)
                avg_distance = sum(distances) / len(distances)
            
            # 更新结果
            self.frame_result.update({
                'frame_id': frame_id,
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
            
            # 计算处理时间
            total_time = time.time() - frame_start_time
            self.frame_result['processing_time'] = total_time
            
            return self.frame_result
            
        except Exception as e:
            total_time = time.time() - frame_start_time
            print(f"处理帧 {frame_id} 时出错 (耗时 {total_time*1000:.1f}ms): {e}")
            
            self.frame_result = {
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
                'processing_time': total_time,
                'error': str(e)
            }
            return self.frame_result


class StreamProcessor:
    """流式数据处理器"""
    
    def __init__(self, view_key: str, calibration, output_dir: str = "./output", 
                 max_frames: int = None, tracking: bool = True):
        """
        初始化流式数据处理器
        
        Args:
            view_key: 视角配置键名 ('view1' 或 'view2')
            calibration: 标定对象
            output_dir: 输出目录
            max_frames: 最大处理帧数
            tracking: 是否启用跟踪
        """
        if view_key not in VIEW_CONFIGS:
            raise ValueError(f"无效的视角配置键: {view_key}，支持的键: {list(VIEW_CONFIGS.keys())}")
        
        self.view_config = VIEW_CONFIGS[view_key]
        self.camera_ip = self.view_config['camera_ip']
        self.lidar_ip = self.view_config['lidar_ip']
        self.view_name = self.view_config['name']
        
        self.calibration = calibration
        self.output_dir = output_dir
        self.max_frames = max_frames
        self.tracking = tracking
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 初始化检测器
        self.detector = StreamV2XDetector(
            tracking=tracking,
            view_prefix=self.view_name
        )
        
        # 初始化RGB相机和LiDAR
        self.rgb_camera = None
        self.lidar = None
        
        # 控制标志
        self.stop_flag = threading.Event()
        self.processing_complete = threading.Event()
        
        # 统计信息
        self.frame_count = 0
        self.processed_count = 0
        self.start_time = None
        
        # 结果存储
        self.frame_results = []
    
    def initialize_sensors(self):
        """初始化传感器"""
        try:
            print(f"{self.view_name}: 初始化RGB相机 - {self.camera_ip}")
            self.rgb_camera = RGBCamera(frame_interval=0.1, camera_rstp=self.camera_ip)
            
            print(f"{self.view_name}: 初始化LiDAR - {self.lidar_ip}")
            self.lidar = Tanwaylidar(lidar_ip=self.lidar_ip)
            
            # # 启动数据采集
            self.rgb_camera.start()
            # self.lidar.start()
            
            print(f"{self.view_name}: 传感器初始化完成")
            return True
            
        except Exception as e:
            print(f"{self.view_name}: 传感器初始化失败 - {e}")
            return False
    
    def cleanup_sensors(self):
        """清理传感器资源"""
        try:
            if self.rgb_camera:
                self.rgb_camera.stop()
            if self.lidar:
                self.lidar.stop()
            print(f"{self.view_name}: 传感器资源清理完成")
        except Exception as e:
            print(f"{self.view_name}: 传感器资源清理失败 - {e}")
    
    def process_stream(self):
        """处理流式数据"""
        print(f"{self.view_name}: 开始处理流式数据...")
        self.start_time = time.time()
        
        try:
            # 初始化传感器
            if not self.initialize_sensors():
                return []
            
            # 等待数据稳定
            time.sleep(2)
            # import pcl
            frame_id = 0
            last_process_time = time.time()
            pcd = o3d.geometry.PointCloud()
            
            while not self.stop_flag.is_set():
                file_path_rgb = f"/home/nebula/RGB-LIDAR-fusion/Data/{frame_id}.jpg"
                file_path_pcd = f"/home/nebula/RGB-LIDAR-fusion/Data/{frame_id}.pcd"
                try:
                    # 检查是否达到最大帧数}
                    if self.max_frames and frame_id >= self.max_frames:
                        print(f"{self.view_name}: 达到最大帧数限制 {self.max_frames}")
                        break
                    
                    # 获取当前帧数据
                    current_time = time.time()
                    
                    # 控制处理频率（每0.1秒处理一次）
                    if current_time - last_process_time < 0.1:
                        time.sleep(0.01)
                        continue
                    
                    # 获取RGB图像
                    frame, frame_timestamp = self.rgb_camera.get_one_frame()
                    if frame is None:
                        print(f"{self.view_name}: 获取RGB帧失败")
                        continue
                    # cv2.imwrite(file_path_rgb,frame)
                    # 获取点云数据
                    points_raw = self.lidar.get_one_pcd()
                    if points_raw is None or len(points_raw) == 0:
                        print(f"{self.view_name}: 获取点云数据失败")
                        continue
                    
                    # 提取XYZ坐标（前3列）
                    points = points_raw[:, :3].astype(np.float64)
                    # pcd.points = o3d.utility.Vector3dVector(points)
                    
                    # 过滤无效点
                    if np.any(np.isnan(points)) or np.any(np.isinf(points)):
                        valid_mask = np.isfinite(points).all(axis=1)
                        points = points[valid_mask]
                    
                    if len(points) == 0:
                        print(f"{self.view_name}: 点云数据为空")
                        continue
                    
                    # 处理当前帧
                    frame_result = self.detector.process_frame(
                        frame, points, self.calibration, frame_id
                    )
                    
                    # 添加视角和时间信息
                    frame_result['view_name'] = self.view_name
                    frame_result['camera_ip'] = self.camera_ip
                    frame_result['lidar_ip'] = self.lidar_ip
                    frame_result['frame_timestamp'] = frame_timestamp
                    
                    # 获取当前帧的精简结果并打印（优化：减少打印频率以降低延迟）
                    # 注意：simplify_results_for_json仍需要计算用于全局结果更新
                    current_frame_simplified = simplify_results_for_json([frame_result])
                    if frame_id % 5 == 0:  # 每5帧打印一次，而不是每帧（减少I/O延迟）
                        print(current_frame_simplified)
                    
                    # 更新全局结果队列
                    if current_frame_simplified:
                        update_global_results(self.view_name, current_frame_simplified[0])
                    
                    # 存储结果
                    self.frame_results.append(frame_result)
                    
                    frame_id += 1
                    self.processed_count += 1
                    last_process_time = current_time
                    
                    # 进度报告
                    if frame_id % 10 == 0:
                        elapsed_time = time.time() - self.start_time
                        fps = self.processed_count / elapsed_time
                        avg_processing_time = frame_result.get('processing_time', 0)
                        
                        print(f"{self.view_name}: 已处理 {self.processed_count} 帧, "
                              f"FPS: {fps:.1f}, 平均处理时间: {avg_processing_time*1000:.1f}ms")
                    
                    # 内存管理
                    if frame_id % 50 == 0:
                        gc.collect()
                
                except Exception as e:
                    print(f"{self.view_name}: 处理帧时出错 - {e}")
                    continue
            
        except KeyboardInterrupt:
            print(f"\n{self.view_name}: 用户中断处理")
        except Exception as e:
            print(f"{self.view_name}: 处理过程中发生错误 - {e}")
            traceback.print_exc()
        finally:
            # 清理传感器资源
            self.cleanup_sensors()
        
        self.processing_complete.set()
        print(f"{self.view_name}: 处理线程结束，共处理 {self.processed_count} 帧")
        
        # 保存结果
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
        
        print(f"\n{self.view_name} 流式数据处理完成:")
        print(f"  - 结果文件: {results_file}")
        
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
            
            all_processing_times = [fr.get('processing_time', 0) for fr in self.frame_results if fr.get('processing_time')]
            if all_processing_times:
                avg_processing_time = sum(all_processing_times) / len(all_processing_times)
                print(f"  - 处理性能: {avg_fps:.1f} FPS, 总耗时: {total_time:.1f}s")
                print(f"  - 平均单帧处理时间: {avg_processing_time*1000:.1f}ms")
        
        return self.frame_results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="V2X流式数据检测器 - 固定IP地址配置")
    
    # 视角1标定参数
    parser.add_argument('--view1-calib', type=str, default = "/home/nebula/RGB-LIDAR-fusion/view1_calib.json",
                       help="视角1的DAIR-V2X标定JSON文件路径")
    
    # 视角2标定参数 (可选)
    parser.add_argument('--view2-calib', type=str, default = "/home/nebula/RGB-LIDAR-fusion/view2_calib.json",
                       help="视角2的DAIR-V2X标定JSON文件路径")

    # 通用参数
    parser.add_argument('--max-frames', type=int, default=100000000,
                       help="每个视角最大处理帧数")
    parser.add_argument('--no-tracking', action='store_true',
                       help="禁用目标跟踪")
    parser.add_argument('--parallel', action='store_true',
                       help="并行处理多个视角（默认为顺序处理）")
    parser.add_argument('--output-dir', type=str, default='./output',
                       help="输出目录")

    args = parser.parse_args()
    
    print("=== V2X流式数据检测器启动 ===")
    print("固定IP配置:")
    for view_key, config in VIEW_CONFIGS.items():
        print(f"  {config['name']}: 相机={config['camera_ip']}, LiDAR={config['lidar_ip']}")
    
    # 准备处理任务
    tasks = []
    
    # 视角1
    if args.view1_calib:
        if not os.path.exists(args.view1_calib):
            print(f"错误: 视角1标定文件不存在: {args.view1_calib}")
            return
        
        tasks.append({
            'view_key': 'view1',
            'calib_path': args.view1_calib,
        })
        print(f"已配置视角1: {VIEW_CONFIGS['view1']['name']} | 标定文件: {args.view1_calib}")

    # 视角2
    if args.view2_calib:
        if not os.path.exists(args.view2_calib):
            print(f"错误: 视角2标定文件不存在: {args.view2_calib}")
            return
        
        tasks.append({
            'view_key': 'view2',
            'calib_path': args.view2_calib,
        })
        print(f"已配置视角2: {VIEW_CONFIGS['view2']['name']} | 标定文件: {args.view2_calib}")

    if not tasks:
        print("错误: 没有有效的视角被配置")
        return

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    all_results = {}

    # 处理流式数据
    if args.parallel and len(tasks) > 1:
        # 并行处理多个视角
        print(f"并行处理 {len(tasks)} 个视角...")
        
        def process_stream_wrapper(task):
            try:
                calibration = CustomCalibration(task['calib_path'])
                processor = StreamProcessor(
                    view_key=task['view_key'],
                    calibration=calibration,
                    output_dir=args.output_dir,
                    max_frames=args.max_frames,
                    tracking=not args.no_tracking
                )
                return processor.process_stream()
            except Exception as e:
                view_name = VIEW_CONFIGS[task['view_key']]['name']
                print(f"处理视角 {view_name} 时发生错误: {e}")
                return []
        
        # 使用多线程处理
        threads = []
        results = {}
        
        def worker(task):
            view_name = VIEW_CONFIGS[task['view_key']]['name']
            results[view_name] = process_stream_wrapper(task)
        
        for task in tasks:
            thread = threading.Thread(target=worker, args=(task,), daemon=True)
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        all_results = results
    else:
        # 顺序处理
        for task in tasks:
            try:
                view_name = VIEW_CONFIGS[task['view_key']]['name']
                calibration = CustomCalibration(task['calib_path'])
                processor = StreamProcessor(
                    view_key=task['view_key'],
                    calibration=calibration,
                    output_dir=args.output_dir,
                    max_frames=args.max_frames,
                    tracking=not args.no_tracking
                )
                results = processor.process_stream()
                all_results[view_name] = results
            except Exception as e:
                view_name = VIEW_CONFIGS[task['view_key']]['name']
                print(f"处理视角 {view_name} 时发生严重错误: {e}")
                traceback.print_exc()

    # 保存汇总结果
    simplified_summary = {
        view_name: simplify_results_for_json(result_list)
        for view_name, result_list in all_results.items() if result_list
    }

    summary_file = os.path.join(args.output_dir, 'stream_v2x_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(simplified_summary, f, indent=2, default=str)
    
    print(f"\n=== 所有视角流式处理完成 ===")
    print(f"汇总结果已保存到: {summary_file}")
    
    # 总体统计
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
    print(f"  - 并行处理: {'是' if args.parallel else '否'}")


if __name__ == "__main__":
    main() 
