#!/usr/bin/env python3
"""
集成的车路协同3D检测系统
结合路端LiDAR感知和车端GPS消息，实现车辆识别和匹配
"""

import os
import cv2
import numpy as np
import json
import open3d as o3d
import time
import threading
from collections import defaultdict, deque
import argparse
from tqdm import tqdm
import gc
import traceback
from typing import Dict, List, Tuple, Optional, Any

# 导入现有模块
from Code.test_dair_v2x_enhanced import V2XEnhanced3DDetector, V2XEnhancedCalibration
from v2x_message_processor import V2XMessageProcessor, MatchedVehicle, VehicleMessage


class IntegratedV2XDetectionSystem:
    """集成的车路协同检测系统"""
    
    def __init__(self, 
                 lidar_gps_lat: float, 
                 lidar_gps_lon: float, 
                 lidar_gps_alt: float = 0.0,
                 v2x_port: int = 8888,
                 model_path: str = "yolov8m-seg.pt"):
        """
        初始化集成系统
        
        Args:
            lidar_gps_lat: LiDAR传感器GPS纬度
            lidar_gps_lon: LiDAR传感器GPS经度
            lidar_gps_alt: LiDAR传感器GPS高度
            v2x_port: V2X消息监听端口
            model_path: YOLO模型路径
        """
        
        # V2X消息处理器
        self.v2x_processor = V2XMessageProcessor(
            lidar_gps_lat, lidar_gps_lon, lidar_gps_alt, v2x_port
        )
        
        # 3D检测器（稍后初始化，需要标定信息）
        self.detector_3d = None
        self.calibration = None
        
        # 匹配结果存储
        self.matched_vehicles = {}
        self.detection_history = deque(maxlen=100)
        self.v2x_message_history = deque(maxlen=100)
        
        # 统计信息
        self.stats = {
            'total_frames': 0,
            'total_detections': 0,
            'total_v2x_messages': 0,
            'successful_matches': 0,
            'start_time': time.time()
        }
        
        print(f"集成V2X检测系统初始化完成")
        print(f"LiDAR GPS位置: ({lidar_gps_lat:.6f}, {lidar_gps_lon:.6f}, {lidar_gps_alt:.2f})")
        print(f"V2X监听端口: {v2x_port}")
    
    def initialize_detector(self, camera_intrinsic_path: str, lidar_to_camera_path: str):
        """初始化3D检测器"""
        try:
            # 初始化标定
            self.calibration = V2XEnhancedCalibration(camera_intrinsic_path, lidar_to_camera_path)
            
            # 初始化3D检测器
            self.detector_3d = V2XEnhanced3DDetector(
                tracking=True, 
                view_prefix="integrated",
                use_improved_fusion=True
            )
            
            print("3D检测器初始化成功")
            
        except Exception as e:
            print(f"初始化3D检测器失败: {e}")
            raise
    
    def start(self):
        """启动集成系统"""
        try:
            # 启动V2X消息处理器
            self.v2x_processor.start()
            print("V2X消息处理器已启动")
            
        except Exception as e:
            print(f"启动集成系统失败: {e}")
            raise
    
    def stop(self):
        """停止集成系统"""
        self.v2x_processor.stop()
        self._print_final_stats()
        print("集成系统已停止")
    
    def process_frame_with_v2x(self, image: np.ndarray, points: np.ndarray, frame_id: int = 0) -> Dict[str, Any]:
        """
        处理帧并结合V2X信息
        
        Args:
            image: 输入图像
            points: 点云数据
            frame_id: 帧ID
            
        Returns:
            包含检测结果和V2X匹配信息的字典
        """
        if self.detector_3d is None or self.calibration is None:
            raise RuntimeError("3D检测器未初始化，请先调用initialize_detector()")
        
        try:
            # 使用3D检测器处理帧
            detection_result = self.detector_3d.process_frame_enhanced(
                image, points, self.calibration, frame_id
            )
            
            # 将检测结果添加到V2X处理器中
            for detection in detection_result['detections']:
                self.v2x_processor.add_detection_result(
                    detection_id=detection['id'],
                    position_3d=np.array(detection['center_3d']),
                    velocity_3d=np.array([0, 0, 0]),  # 如果有速度信息可以添加
                    confidence=detection['confidence'],
                    class_name=detection['class_name']
                )
            
            # 获取V2X匹配结果
            v2x_matches = self.v2x_processor.get_matches()
            unmatched_detections = self.v2x_processor.get_unmatched_detections()
            
            # 增强检测结果
            enhanced_detections = self._enhance_detections_with_v2x(
                detection_result['detections'], v2x_matches
            )
            
            # 构建返回结果
            integrated_result = {
                'frame_id': frame_id,
                'timestamp': time.time(),
                'detections': enhanced_detections,
                'v2x_matches': [self._match_to_dict(match) for match in v2x_matches],
                'unmatched_detections': len(unmatched_detections),
                'person_vehicle_distances': detection_result['person_vehicle_distances'],
                'statistics': detection_result['statistics'],
                'visualization_data': detection_result['visualization_data']
            }
            
            # 增强可视化
            enhanced_frame = self._draw_v2x_info(
                detection_result['visualization_data']['processed_frame'],
                v2x_matches,
                enhanced_detections
            )
            
            integrated_result['visualization_data']['processed_frame'] = enhanced_frame
            
            # 更新统计
            self.stats['total_frames'] += 1
            self.stats['total_detections'] += len(enhanced_detections)
            self.stats['successful_matches'] += len(v2x_matches)
            
            # 保存历史
            self.detection_history.append({
                'frame_id': frame_id,
                'timestamp': time.time(),
                'detections': enhanced_detections,
                'v2x_matches': len(v2x_matches)
            })
            
            return integrated_result
            
        except Exception as e:
            print(f"处理帧时出错: {e}")
            traceback.print_exc()
            return {
                'frame_id': frame_id,
                'timestamp': time.time(),
                'detections': [],
                'v2x_matches': [],
                'unmatched_detections': 0,
                'person_vehicle_distances': [],
                'statistics': {'total_objects': 0, 'person_count': 0, 'vehicle_count': 0},
                'visualization_data': {
                    'processed_frame': np.ascontiguousarray(image.copy(), dtype=np.uint8),
                    'bev_frame': np.zeros((500, 500, 3), dtype=np.uint8)
                },
                'error': str(e)
            }
    
    def _enhance_detections_with_v2x(self, detections: List[Dict], v2x_matches: List[MatchedVehicle]) -> List[Dict]:
        """使用V2X信息增强检测结果"""
        # 创建匹配映射
        match_map = {match.detection_id: match for match in v2x_matches}
        
        enhanced_detections = []
        for detection in detections:
            enhanced_detection = detection.copy()
            
            # 如果有V2X匹配信息
            if detection['id'] in match_map:
                match = match_map[detection['id']]
                enhanced_detection.update({
                    'v2x_matched': True,
                    'vehicle_id': match.vehicle_id,
                    'gps_position': match.gps_position,
                    'v2x_speed_kmh': match.speed_kmh,
                    'v2x_heading': match.heading,
                    'match_confidence': match.match_confidence,
                    'position_source': 'lidar_gps_fused'
                })
            else:
                enhanced_detection.update({
                    'v2x_matched': False,
                    'vehicle_id': None,
                    'gps_position': None,
                    'v2x_speed_kmh': None,
                    'v2x_heading': None,
                    'match_confidence': 0.0,
                    'position_source': 'lidar_only'
                })
            
            enhanced_detections.append(enhanced_detection)
        
        return enhanced_detections
    
    def _match_to_dict(self, match: MatchedVehicle) -> Dict:
        """将匹配结果转换为字典格式"""
        return {
            'vehicle_id': match.vehicle_id,
            'detection_id': match.detection_id,
            'lidar_position': match.lidar_position.tolist(),
            'gps_position': match.gps_position,
            'speed_kmh': match.speed_kmh,
            'heading': match.heading,
            'match_confidence': match.match_confidence,
            'timestamp': match.timestamp
        }
    
    def _draw_v2x_info(self, image: np.ndarray, v2x_matches: List[MatchedVehicle], detections: List[Dict]) -> np.ndarray:
        """在图像上绘制V2X信息"""
        result_image = image.copy()
        
        # 绘制V2X匹配信息
        y_offset = 60
        for i, match in enumerate(v2x_matches):
            # V2X匹配信息
            v2x_text = f"V2X: {match.vehicle_id} -> {match.detection_id} (conf:{match.match_confidence:.2f})"
            cv2.putText(result_image, v2x_text, (10, y_offset + i*20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # GPS位置信息
            gps_text = f"GPS: ({match.gps_position[0]:.6f}, {match.gps_position[1]:.6f})"
            cv2.putText(result_image, gps_text, (10, y_offset + i*20 + 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        # 在检测框上标注V2X信息
        for detection in detections:
            if detection.get('v2x_matched', False):
                # 找到检测框位置（这里需要根据实际的边界框格式调整）
                if 'bbox_2d' in detection:
                    x1, y1, x2, y2 = detection['bbox_2d']
                    # 绘制V2X标记
                    cv2.rectangle(result_image, (int(x1)-2, int(y1)-2), (int(x2)+2, int(y2)+2), (0, 255, 255), 2)
                    
                    # V2X车辆ID标签
                    v2x_label = f"V2X:{detection['vehicle_id']}"
                    cv2.putText(result_image, v2x_label, (int(x1), int(y1)-25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        return result_image
    
    def get_v2x_statistics(self) -> Dict:
        """获取V2X统计信息"""
        runtime = time.time() - self.stats['start_time']
        
        return {
            'runtime_seconds': runtime,
            'total_frames': self.stats['total_frames'],
            'total_detections': self.stats['total_detections'],
            'total_v2x_messages': self.v2x_processor.stats['messages_received'],
            'successful_matches': self.stats['successful_matches'],
            'match_rate': self.stats['successful_matches'] / max(1, self.stats['total_detections']),
            'fps': self.stats['total_frames'] / max(1, runtime),
            'v2x_message_rate': self.v2x_processor.stats['messages_received'] / max(1, runtime)
        }
    
    def _print_final_stats(self):
        """打印最终统计信息"""
        stats = self.get_v2x_statistics()
        
        print("\n=== 集成V2X检测系统统计 ===")
        print(f"运行时间: {stats['runtime_seconds']:.1f}秒")
        print(f"处理帧数: {stats['total_frames']}")
        print(f"总检测数: {stats['total_detections']}")
        print(f"V2X消息数: {stats['total_v2x_messages']}")
        print(f"成功匹配数: {stats['successful_matches']}")
        print(f"匹配成功率: {stats['match_rate']:.2%}")
        print(f"处理帧率: {stats['fps']:.1f} FPS")
        print(f"V2X消息率: {stats['v2x_message_rate']:.1f} msg/s")


def process_dair_with_v2x(view_path: str, view_name: str, 
                         lidar_gps_lat: float, lidar_gps_lon: float, lidar_gps_alt: float = 0.0,
                         v2x_port: int = 8888, max_frames: int = 100, show: bool = False):
    """
    使用V2X增强功能处理DAIR-V2X数据集
    
    Args:
        view_path: 数据路径
        view_name: 视角名称
        lidar_gps_lat: LiDAR GPS纬度
        lidar_gps_lon: LiDAR GPS经度
        lidar_gps_alt: LiDAR GPS高度
        v2x_port: V2X消息端口
        max_frames: 最大处理帧数
        show: 是否显示处理过程
    """
    
    print(f"\n=== 开始V2X增强处理 {view_name} ===")
    
    try:
        # 初始化集成系统
        integrated_system = IntegratedV2XDetectionSystem(
            lidar_gps_lat, lidar_gps_lon, lidar_gps_alt, v2x_port
        )
        
        # 检查数据结构
        image_dir = os.path.join(view_path, 'image')
        velodyne_dir = os.path.join(view_path, 'velodyne')
        calib_dir = os.path.join(view_path, 'calib')
        
        # 获取标定文件
        cam_intrinsic_dir = os.path.join(calib_dir, 'camera_intrinsic')
        
        if os.path.exists(os.path.join(calib_dir, 'virtuallidar_to_camera')):
            lidar_to_cam_dir = os.path.join(calib_dir, 'virtuallidar_to_camera')
        elif os.path.exists(os.path.join(calib_dir, 'lidar_to_camera')):
            lidar_to_cam_dir = os.path.join(calib_dir, 'lidar_to_camera')
        else:
            raise ValueError(f"找不到LiDAR到相机的标定文件夹")
        
        # 获取文件列表
        image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])[:max_frames]
        velodyne_files = sorted([f for f in os.listdir(velodyne_dir) if f.endswith('.pcd')])[:max_frames]
        
        if not image_files:
            print(f"错误：{view_name} 中没有找到图像文件")
            return []
        
        # 初始化检测器
        first_frame_id = os.path.splitext(image_files[0])[0]
        cam_intrinsic_path = os.path.join(cam_intrinsic_dir, f"{first_frame_id}.json")
        lidar_to_cam_path = os.path.join(lidar_to_cam_dir, f"{first_frame_id}.json")
        
        integrated_system.initialize_detector(cam_intrinsic_path, lidar_to_cam_path)
        
        # 启动系统
        integrated_system.start()
        
        # 创建输出目录
        output_dir = f'./output/v2x_{view_name}'
        os.makedirs(output_dir, exist_ok=True)
        
        # 处理结果存储
        frame_results = []
        processed_frames = []
        
        print(f"开始V2X增强处理 {len(image_files)} 帧...")
        print(f"V2X消息监听端口: {v2x_port}")
        print("请确保车辆正在发送V2X消息到该端口")
        
        # 处理每一帧
        for i, (img_file, pcd_file) in enumerate(tqdm(zip(image_files, velodyne_files), 
                                                    desc=f"V2X处理{view_name}", 
                                                    total=len(image_files))):
            try:
                # 加载图像
                img_path = os.path.join(image_dir, img_file)
                image = cv2.imread(img_path)
                if image is None:
                    continue
                
                # 加载点云
                pcd_path = os.path.join(velodyne_dir, pcd_file)
                pcd = o3d.io.read_point_cloud(pcd_path)
                points = np.asarray(pcd.points, dtype=np.float64)
                
                if len(points) == 0:
                    continue
                
                # 使用集成系统处理帧
                frame_result = integrated_system.process_frame_with_v2x(image, points, i)
                
                # 添加文件信息
                frame_result.update({
                    'view_name': view_name,
                    'image_file': img_file,
                    'pcd_file': pcd_file
                })
                
                frame_results.append(frame_result)
                
                # 获取处理后的图像
                result_frame = frame_result['visualization_data']['processed_frame']
                processed_frames.append(result_frame)
                
                # 显示处理结果
                if show:
                    cv2.imshow(f"V2X_{view_name}", result_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # 定期输出统计信息
                if i % 10 == 0 and i > 0:
                    stats = integrated_system.get_v2x_statistics()
                    print(f"  帧 {i}: 匹配率 {stats['match_rate']:.1%}, V2X消息 {stats['total_v2x_messages']}")
                
                # 内存管理
                if i % 20 == 0:
                    gc.collect()
                
            except Exception as e:
                print(f"处理帧 {i} 时出错: {e}")
                continue
        
        if show:
            cv2.destroyAllWindows()
        
        # 停止系统
        integrated_system.stop()
        
        # 保存结果
        if frame_results:
            # 保存完整结果
            results_file = os.path.join(output_dir, f'{view_name}_v2x_results.json')
            with open(results_file, 'w') as f:
                json.dump(frame_results, f, indent=2, default=str)
            
            # 创建视频
            if processed_frames:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_path = os.path.join(output_dir, f'{view_name}_v2x_detection.mp4')
                height, width = processed_frames[0].shape[:2]
                out = cv2.VideoWriter(video_path, fourcc, 10, (width, height))
                for frame in processed_frames:
                    out.write(frame)
                out.release()
                
                print(f"\nV2X增强处理完成:")
                print(f"  - 结果文件: {results_file}")
                print(f"  - 检测视频: {video_path}")
            
            # 最终统计
            final_stats = integrated_system.get_v2x_statistics()
            print(f"  - 总帧数: {final_stats['total_frames']}")
            print(f"  - 总检测数: {final_stats['total_detections']}")
            print(f"  - V2X消息数: {final_stats['total_v2x_messages']}")
            print(f"  - 成功匹配数: {final_stats['successful_matches']}")
            print(f"  - 匹配成功率: {final_stats['match_rate']:.1%}")
        
        return frame_results
        
    except Exception as e:
        print(f"V2X增强处理 {view_name} 时出错: {e}")
        traceback.print_exc()
        return []


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="集成V2X车路协同3D检测系统")
    parser.add_argument('--data-path', type=str, required=True,
                       help="DAIR-V2X数据路径")
    parser.add_argument('--view-name', type=str, default="infrastructure-side",
                       help="视角名称")
    parser.add_argument('--lidar-gps-lat', type=float, required=True,
                       help="LiDAR传感器GPS纬度")
    parser.add_argument('--lidar-gps-lon', type=float, required=True,
                       help="LiDAR传感器GPS经度")
    parser.add_argument('--lidar-gps-alt', type=float, default=0.0,
                       help="LiDAR传感器GPS高度")
    parser.add_argument('--v2x-port', type=int, default=8888,
                       help="V2X消息监听端口")
    parser.add_argument('--max-frames', type=int, default=100,
                       help="最大处理帧数")
    parser.add_argument('--show', action='store_true',
                       help="显示处理过程")
    
    args = parser.parse_args()
    
    print("=== 集成V2X车路协同3D检测系统 ===")
    print(f"数据路径: {args.data_path}")
    print(f"LiDAR GPS位置: ({args.lidar_gps_lat:.6f}, {args.lidar_gps_lon:.6f}, {args.lidar_gps_alt:.2f})")
    print(f"V2X监听端口: {args.v2x_port}")
    
    # 检查路径
    if not os.path.exists(args.data_path):
        print(f"错误：数据路径不存在: {args.data_path}")
        return
    
    # 创建输出目录
    os.makedirs('./output', exist_ok=True)
    
    # 处理数据
    results = process_dair_with_v2x(
        args.data_path, 
        args.view_name,
        args.lidar_gps_lat,
        args.lidar_gps_lon, 
        args.lidar_gps_alt,
        args.v2x_port,
        args.max_frames,
        args.show
    )
    
    print(f"\n处理完成，共处理 {len(results)} 帧")


if __name__ == "__main__":
    main() 