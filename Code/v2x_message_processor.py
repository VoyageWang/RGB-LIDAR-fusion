#!/usr/bin/env python3
"""
车路协同消息处理器
功能：
1. 实时接收车端XML消息（包含GPS位置、速度等信息）
2. GPS坐标转换到LiDAR坐标系
3. 与路端3D检测结果进行位置匹配
4. 输出匹配结果和增强信息
"""

import socket
import threading
import time
import xml.etree.ElementTree as ET
import json
import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict, deque
import logging
from datetime import datetime
import pyproj
import requests
from concurrent.futures import ThreadPoolExecutor
import queue

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class VehicleMessage:
    """车辆消息数据结构"""
    vehicle_id: str
    timestamp: float
    gps_lat: float
    gps_lon: float
    gps_alt: float
    speed_kmh: float
    heading: float  # 航向角（度）
    acceleration: float = 0.0
    raw_xml: str = ""


@dataclass
class LiDARDetection:
    """LiDAR检测结果数据结构"""
    detection_id: str
    position_3d: np.ndarray  # LiDAR坐标系下的位置
    velocity_3d: np.ndarray  # 3D速度向量
    dimensions: np.ndarray   # 尺寸
    confidence: float
    timestamp: float
    class_name: str


@dataclass
class MatchedVehicle:
    """匹配的车辆信息"""
    vehicle_id: str
    detection_id: str
    lidar_position: np.ndarray
    gps_position: Tuple[float, float, float]
    speed_kmh: float
    heading: float
    match_confidence: float
    timestamp: float


class GPSToLiDARConverter:
    """GPS坐标到LiDAR坐标系转换器"""
    
    def __init__(self, lidar_gps_lat: float, lidar_gps_lon: float, lidar_gps_alt: float = 0.0):
        """
        初始化转换器
        
        Args:
            lidar_gps_lat: LiDAR传感器的GPS纬度
            lidar_gps_lon: LiDAR传感器的GPS经度  
            lidar_gps_alt: LiDAR传感器的GPS高度
        """
        self.lidar_gps_lat = lidar_gps_lat
        self.lidar_gps_lon = lidar_gps_lon
        self.lidar_gps_alt = lidar_gps_alt
        
        # 创建投影变换器（使用UTM投影）
        self.wgs84 = pyproj.CRS('EPSG:4326')  # WGS84经纬度
        
        # 根据LiDAR位置确定UTM区域
        utm_zone = self._get_utm_zone(lidar_gps_lon)
        hemisphere = 'north' if lidar_gps_lat >= 0 else 'south'
        utm_epsg = f'EPSG:{32600 + utm_zone if hemisphere == "north" else 32700 + utm_zone}'
        
        self.utm = pyproj.CRS(utm_epsg)
        self.transformer = pyproj.Transformer.from_crs(self.wgs84, self.utm, always_xy=True)
        
        # 计算LiDAR在UTM坐标系下的位置（作为原点）
        self.lidar_utm_x, self.lidar_utm_y = self.transformer.transform(lidar_gps_lon, lidar_gps_lat)
        
        logger.info(f"GPS转换器初始化完成:")
        logger.info(f"  LiDAR GPS位置: ({lidar_gps_lat:.6f}, {lidar_gps_lon:.6f}, {lidar_gps_alt:.2f})")
        logger.info(f"  UTM区域: {utm_epsg}")
        logger.info(f"  LiDAR UTM位置: ({self.lidar_utm_x:.2f}, {self.lidar_utm_y:.2f})")
    
    def _get_utm_zone(self, longitude: float) -> int:
        """根据经度计算UTM区域"""
        return int((longitude + 180) / 6) + 1
    
    def gps_to_lidar(self, gps_lat: float, gps_lon: float, gps_alt: float = 0.0) -> np.ndarray:
        """
        将GPS坐标转换为LiDAR坐标系
        
        Args:
            gps_lat: GPS纬度
            gps_lon: GPS经度
            gps_alt: GPS高度
            
        Returns:
            LiDAR坐标系下的3D位置 [x, y, z]
        """
        try:
            # 转换到UTM坐标系
            utm_x, utm_y = self.transformer.transform(gps_lon, gps_lat)
            
            # 相对于LiDAR位置的偏移
            x_offset = utm_x - self.lidar_utm_x
            y_offset = utm_y - self.lidar_utm_y
            z_offset = gps_alt - self.lidar_gps_alt
            
            # 转换到LiDAR坐标系（通常：X向前，Y向左，Z向上）
            # 这里假设UTM的X轴对应LiDAR的X轴，Y轴对应LiDAR的Y轴
            lidar_x = x_offset
            lidar_y = y_offset
            lidar_z = z_offset
            
            return np.array([lidar_x, lidar_y, lidar_z])
            
        except Exception as e:
            logger.error(f"GPS坐标转换失败: {e}")
            return np.array([0.0, 0.0, 0.0])
    
    def calculate_distance(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        """计算两个3D位置之间的距离"""
        return np.linalg.norm(pos1 - pos2)


class VehicleMessageReceiver:
    """车辆消息接收器"""
    
    def __init__(self, port: int = 8888, buffer_size: int = 4096):
        self.port = port
        self.buffer_size = buffer_size
        self.socket = None
        self.running = False
        self.message_queue = queue.Queue()
        self.receive_thread = None
        
    def start(self):
        """启动消息接收"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.bind(('', self.port))
            self.socket.settimeout(1.0)  # 设置超时，便于优雅退出
            
            self.running = True
            self.receive_thread = threading.Thread(target=self._receive_loop)
            self.receive_thread.start()
            
            logger.info(f"消息接收器启动，监听端口: {self.port}")
            
        except Exception as e:
            logger.error(f"启动消息接收器失败: {e}")
            raise
    
    def stop(self):
        """停止消息接收"""
        self.running = False
        if self.receive_thread:
            self.receive_thread.join()
        if self.socket:
            self.socket.close()
        logger.info("消息接收器已停止")
    
    def _receive_loop(self):
        """消息接收循环"""
        while self.running:
            try:
                data, addr = self.socket.recvfrom(self.buffer_size)
                message_xml = data.decode('utf-8')
                
                # 解析XML消息
                vehicle_msg = self._parse_xml_message(message_xml)
                if vehicle_msg:
                    self.message_queue.put(vehicle_msg)
                    logger.debug(f"收到车辆消息: {vehicle_msg.vehicle_id} from {addr}")
                
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    logger.error(f"接收消息时出错: {e}")
    
    def _parse_xml_message(self, xml_data: str) -> Optional[VehicleMessage]:
        """解析XML消息"""
        try:
            root = ET.fromstring(xml_data)
            
            # 根据实际XML格式调整解析逻辑
            vehicle_id = root.find('vehicle_id').text if root.find('vehicle_id') is not None else 'unknown'
            timestamp = float(root.find('timestamp').text) if root.find('timestamp') is not None else time.time()
            
            # GPS位置信息
            gps_element = root.find('gps') or root.find('position')
            if gps_element is not None:
                gps_lat = float(gps_element.find('latitude').text or gps_element.find('lat').text)
                gps_lon = float(gps_element.find('longitude').text or gps_element.find('lon').text)
                gps_alt = float(gps_element.find('altitude').text or gps_element.find('alt').text or 0.0)
            else:
                # 直接从根节点获取
                gps_lat = float(root.find('latitude').text or root.find('lat').text)
                gps_lon = float(root.find('longitude').text or root.find('lon').text)
                gps_alt = float(root.find('altitude').text or root.find('alt').text or 0.0)
            
            # 速度和航向信息
            speed_kmh = float(root.find('speed').text or 0.0)
            heading = float(root.find('heading').text or root.find('yaw').text or 0.0)
            acceleration = float(root.find('acceleration').text or 0.0)
            
            return VehicleMessage(
                vehicle_id=vehicle_id,
                timestamp=timestamp,
                gps_lat=gps_lat,
                gps_lon=gps_lon,
                gps_alt=gps_alt,
                speed_kmh=speed_kmh,
                heading=heading,
                acceleration=acceleration,
                raw_xml=xml_data
            )
            
        except Exception as e:
            logger.error(f"解析XML消息失败: {e}")
            logger.debug(f"XML内容: {xml_data}")
            return None
    
    def get_message(self, timeout: float = 0.1) -> Optional[VehicleMessage]:
        """获取消息"""
        try:
            return self.message_queue.get(timeout=timeout)
        except queue.Empty:
            return None


class VehicleDetectionMatcher:
    """车辆消息与检测结果匹配器"""
    
    def __init__(self, gps_converter: GPSToLiDARConverter, 
                 max_distance_threshold: float = 10.0,
                 max_time_diff: float = 2.0):
        """
        初始化匹配器
        
        Args:
            gps_converter: GPS坐标转换器
            max_distance_threshold: 最大距离阈值（米）
            max_time_diff: 最大时间差阈值（秒）
        """
        self.gps_converter = gps_converter
        self.max_distance_threshold = max_distance_threshold
        self.max_time_diff = max_time_diff
        
        # 历史数据
        self.vehicle_messages = defaultdict(lambda: deque(maxlen=10))
        self.detection_results = defaultdict(lambda: deque(maxlen=10))
        self.matched_vehicles = {}
        
    def add_vehicle_message(self, msg: VehicleMessage):
        """添加车辆消息"""
        self.vehicle_messages[msg.vehicle_id].append(msg)
        
    def add_detection_result(self, detection: LiDARDetection):
        """添加检测结果"""
        self.detection_results[detection.detection_id].append(detection)
        
    def match_vehicles(self, current_time: float) -> List[MatchedVehicle]:
        """匹配车辆消息和检测结果"""
        matches = []
        
        # 获取最近的车辆消息
        recent_vehicles = {}
        for vehicle_id, messages in self.vehicle_messages.items():
            if messages:
                latest_msg = messages[-1]
                if current_time - latest_msg.timestamp <= self.max_time_diff:
                    recent_vehicles[vehicle_id] = latest_msg
        
        # 获取最近的检测结果
        recent_detections = {}
        for detection_id, detections in self.detection_results.items():
            if detections:
                latest_detection = detections[-1]
                if current_time - latest_detection.timestamp <= self.max_time_diff:
                    recent_detections[detection_id] = latest_detection
        
        # 进行匹配
        used_detections = set()
        
        for vehicle_id, vehicle_msg in recent_vehicles.items():
            # 将GPS坐标转换为LiDAR坐标
            lidar_pos = self.gps_converter.gps_to_lidar(
                vehicle_msg.gps_lat, vehicle_msg.gps_lon, vehicle_msg.gps_alt
            )
            
            best_match = None
            best_distance = float('inf')
            
            # 寻找最佳匹配的检测结果
            for detection_id, detection in recent_detections.items():
                if detection_id in used_detections:
                    continue
                
                distance = self.gps_converter.calculate_distance(lidar_pos, detection.position_3d)
                
                if distance < self.max_distance_threshold and distance < best_distance:
                    best_distance = distance
                    best_match = (detection_id, detection)
            
            # 如果找到匹配
            if best_match:
                detection_id, detection = best_match
                used_detections.add(detection_id)
                
                # 计算匹配置信度
                match_confidence = max(0.0, 1.0 - (best_distance / self.max_distance_threshold))
                
                matched_vehicle = MatchedVehicle(
                    vehicle_id=vehicle_id,
                    detection_id=detection_id,
                    lidar_position=detection.position_3d,
                    gps_position=(vehicle_msg.gps_lat, vehicle_msg.gps_lon, vehicle_msg.gps_alt),
                    speed_kmh=vehicle_msg.speed_kmh,
                    heading=vehicle_msg.heading,
                    match_confidence=match_confidence,
                    timestamp=current_time
                )
                
                matches.append(matched_vehicle)
                self.matched_vehicles[vehicle_id] = matched_vehicle
                
                logger.info(f"匹配成功: 车辆{vehicle_id} <-> 检测{detection_id}, 距离={best_distance:.2f}m, 置信度={match_confidence:.2f}")
        
        return matches
    
    def get_unmatched_detections(self, current_time: float) -> List[LiDARDetection]:
        """获取未匹配的检测结果"""
        unmatched = []
        matched_detection_ids = {mv.detection_id for mv in self.matched_vehicles.values()}
        
        for detection_id, detections in self.detection_results.items():
            if detections and detection_id not in matched_detection_ids:
                latest_detection = detections[-1]
                if current_time - latest_detection.timestamp <= self.max_time_diff:
                    unmatched.append(latest_detection)
        
        return unmatched


class V2XMessageProcessor:
    """车路协同消息处理器主类"""
    
    def __init__(self, lidar_gps_lat: float, lidar_gps_lon: float, lidar_gps_alt: float = 0.0,
                 listen_port: int = 8888):
        """
        初始化处理器
        
        Args:
            lidar_gps_lat: LiDAR传感器GPS纬度
            lidar_gps_lon: LiDAR传感器GPS经度
            lidar_gps_alt: LiDAR传感器GPS高度
            listen_port: 监听端口
        """
        self.gps_converter = GPSToLiDARConverter(lidar_gps_lat, lidar_gps_lon, lidar_gps_alt)
        self.message_receiver = VehicleMessageReceiver(listen_port)
        self.matcher = VehicleDetectionMatcher(self.gps_converter)
        
        self.running = False
        self.process_thread = None
        
        # 统计信息
        self.stats = {
            'messages_received': 0,
            'detections_processed': 0,
            'matches_found': 0,
            'start_time': time.time()
        }
    
    def start(self):
        """启动处理器"""
        try:
            self.message_receiver.start()
            self.running = True
            self.process_thread = threading.Thread(target=self._process_loop)
            self.process_thread.start()
            
            logger.info("V2X消息处理器启动成功")
            
        except Exception as e:
            logger.error(f"启动V2X消息处理器失败: {e}")
            raise
    
    def stop(self):
        """停止处理器"""
        self.running = False
        if self.process_thread:
            self.process_thread.join()
        self.message_receiver.stop()
        
        logger.info("V2X消息处理器已停止")
        self._print_stats()
    
    def _process_loop(self):
        """主处理循环"""
        while self.running:
            try:
                # 接收车辆消息
                vehicle_msg = self.message_receiver.get_message()
                if vehicle_msg:
                    self.matcher.add_vehicle_message(vehicle_msg)
                    self.stats['messages_received'] += 1
                    logger.debug(f"处理车辆消息: {vehicle_msg.vehicle_id}")
                
                # 这里可以添加接收检测结果的逻辑
                # 例如从另一个队列或文件中读取
                
                time.sleep(0.01)  # 避免CPU占用过高
                
            except Exception as e:
                logger.error(f"处理循环中出错: {e}")
    
    def add_detection_result(self, detection_id: str, position_3d: np.ndarray, 
                           velocity_3d: np.ndarray = None, dimensions: np.ndarray = None,
                           confidence: float = 1.0, class_name: str = "vehicle"):
        """添加检测结果"""
        if velocity_3d is None:
            velocity_3d = np.array([0.0, 0.0, 0.0])
        if dimensions is None:
            dimensions = np.array([4.0, 2.0, 1.5])  # 默认车辆尺寸
        
        detection = LiDARDetection(
            detection_id=detection_id,
            position_3d=position_3d,
            velocity_3d=velocity_3d,
            dimensions=dimensions,
            confidence=confidence,
            timestamp=time.time(),
            class_name=class_name
        )
        
        self.matcher.add_detection_result(detection)
        self.stats['detections_processed'] += 1
    
    def get_matches(self) -> List[MatchedVehicle]:
        """获取当前匹配结果"""
        current_time = time.time()
        matches = self.matcher.match_vehicles(current_time)
        self.stats['matches_found'] += len(matches)
        return matches
    
    def get_unmatched_detections(self) -> List[LiDARDetection]:
        """获取未匹配的检测结果"""
        return self.matcher.get_unmatched_detections(time.time())
    
    def _print_stats(self):
        """打印统计信息"""
        runtime = time.time() - self.stats['start_time']
        logger.info("=== V2X消息处理统计 ===")
        logger.info(f"运行时间: {runtime:.1f}秒")
        logger.info(f"接收消息数: {self.stats['messages_received']}")
        logger.info(f"处理检测数: {self.stats['detections_processed']}")
        logger.info(f"成功匹配数: {self.stats['matches_found']}")
        if runtime > 0:
            logger.info(f"消息接收率: {self.stats['messages_received']/runtime:.1f} msg/s")


def create_test_xml_message(vehicle_id: str, lat: float, lon: float, speed: float, heading: float) -> str:
    """创建测试用的XML消息"""
    xml_template = f"""<?xml version="1.0" encoding="UTF-8"?>
<vehicle_message>
    <vehicle_id>{vehicle_id}</vehicle_id>
    <timestamp>{time.time()}</timestamp>
    <gps>
        <latitude>{lat}</latitude>
        <longitude>{lon}</longitude>
        <altitude>0.0</altitude>
    </gps>
    <speed>{speed}</speed>
    <heading>{heading}</heading>
    <acceleration>0.0</acceleration>
</vehicle_message>"""
    return xml_template


def send_test_message(target_ip: str = "127.0.0.1", target_port: int = 8888):
    """发送测试消息"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    # 创建测试消息
    test_msg = create_test_xml_message("TEST_VEHICLE_001", 39.9042, 116.4074, 60.0, 90.0)
    
    try:
        sock.sendto(test_msg.encode('utf-8'), (target_ip, target_port))
        print(f"测试消息已发送到 {target_ip}:{target_port}")
    except Exception as e:
        print(f"发送测试消息失败: {e}")
    finally:
        sock.close()


if __name__ == "__main__":
    # 示例用法
    
    # 路端LiDAR的GPS位置（需要根据实际情况设置）
    LIDAR_GPS_LAT = 39.9042  # 北京天安门纬度
    LIDAR_GPS_LON = 116.4074  # 北京天安门经度
    LIDAR_GPS_ALT = 10.0      # 高度10米
    
    # 创建处理器
    processor = V2XMessageProcessor(
        lidar_gps_lat=LIDAR_GPS_LAT,
        lidar_gps_lon=LIDAR_GPS_LON,
        lidar_gps_alt=LIDAR_GPS_ALT,
        listen_port=8888
    )
    
    try:
        # 启动处理器
        processor.start()
        
        # 模拟添加一些检测结果
        processor.add_detection_result("DET_001", np.array([10.0, 5.0, 0.0]))
        processor.add_detection_result("DET_002", np.array([-5.0, 10.0, 0.0]))
        
        print("处理器已启动，等待车辆消息...")
        print("可以运行 send_test_message() 来发送测试消息")
        
        # 主循环
        while True:
            matches = processor.get_matches()
            if matches:
                print(f"\n找到 {len(matches)} 个匹配:")
                for match in matches:
                    print(f"  车辆 {match.vehicle_id} -> 检测 {match.detection_id}")
                    print(f"    LiDAR位置: {match.lidar_position}")
                    print(f"    GPS位置: {match.gps_position}")
                    print(f"    速度: {match.speed_kmh} km/h")
                    print(f"    匹配置信度: {match.match_confidence:.2f}")
            
            unmatched = processor.get_unmatched_detections()
            if unmatched:
                print(f"\n未匹配的检测: {len(unmatched)} 个")
            
            time.sleep(1.0)
            
    except KeyboardInterrupt:
        print("\n正在停止处理器...")
        processor.stop()
        print("处理器已停止") 