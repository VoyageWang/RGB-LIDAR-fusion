#!/usr/bin/env python3
"""
V2X消息发送器
用于模拟车辆发送XML消息，包含GPS位置、速度等信息
"""

import socket
import time
import threading
import argparse
import json
import math
import random
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class VehicleSimConfig:
    """车辆模拟配置"""
    vehicle_id: str
    start_lat: float
    start_lon: float
    speed_kmh: float
    heading: float  # 度
    send_interval: float = 0.5  # 发送间隔（秒）


class VehicleSimulator:
    """车辆模拟器"""
    
    def __init__(self, config: VehicleSimConfig):
        self.config = config
        self.current_lat = config.start_lat
        self.current_lon = config.start_lon
        self.current_speed = config.speed_kmh
        self.current_heading = config.heading
        self.current_alt = 0.0
        
        # 地球半径（米）
        self.EARTH_RADIUS = 6371000
        
    def update_position(self, time_delta: float):
        """更新车辆位置"""
        # 将速度从km/h转换为m/s
        speed_ms = self.current_speed / 3.6
        
        # 计算移动距离
        distance = speed_ms * time_delta
        
        # 将航向角从度转换为弧度
        heading_rad = math.radians(self.current_heading)
        
        # 计算经纬度变化
        # 纬度变化
        lat_change = distance * math.cos(heading_rad) / self.EARTH_RADIUS
        self.current_lat += math.degrees(lat_change)
        
        # 经度变化
        lon_change = distance * math.sin(heading_rad) / (self.EARTH_RADIUS * math.cos(math.radians(self.current_lat)))
        self.current_lon += math.degrees(lon_change)
        
        # 添加一些随机噪声来模拟真实的GPS误差
        self.current_lat += random.gauss(0, 0.000001)  # 约0.1米的误差
        self.current_lon += random.gauss(0, 0.000001)
        
        # 速度也可以有一些变化
        self.current_speed += random.gauss(0, 1)  # 速度变化
        self.current_speed = max(0, min(120, self.current_speed))  # 限制在0-120km/h
        
        # 航向角也可以有一些变化
        self.current_heading += random.gauss(0, 2)  # 航向角变化
        self.current_heading = self.current_heading % 360
    
    def get_current_state(self) -> dict:
        """获取当前状态"""
        return {
            'vehicle_id': self.config.vehicle_id,
            'latitude': self.current_lat,
            'longitude': self.current_lon,
            'altitude': self.current_alt,
            'speed_kmh': self.current_speed,
            'heading': self.current_heading,
            'timestamp': time.time()
        }


class V2XMessageSender:
    """V2X消息发送器"""
    
    def __init__(self, target_ip: str = "127.0.0.1", target_port: int = 8888):
        self.target_ip = target_ip
        self.target_port = target_port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.running = False
        self.send_threads = []
        self.simulators = []
        
    def add_vehicle(self, config: VehicleSimConfig):
        """添加车辆"""
        simulator = VehicleSimulator(config)
        self.simulators.append(simulator)
        print(f"添加车辆: {config.vehicle_id}")
    
    def create_xml_message(self, vehicle_state: dict) -> str:
        """创建XML消息"""
        xml_template = f"""<?xml version="1.0" encoding="UTF-8"?>
<vehicle_message>
    <vehicle_id>{vehicle_state['vehicle_id']}</vehicle_id>
    <timestamp>{vehicle_state['timestamp']}</timestamp>
    <gps>
        <latitude>{vehicle_state['latitude']:.8f}</latitude>
        <longitude>{vehicle_state['longitude']:.8f}</longitude>
        <altitude>{vehicle_state['altitude']:.2f}</altitude>
    </gps>
    <speed>{vehicle_state['speed_kmh']:.2f}</speed>
    <heading>{vehicle_state['heading']:.2f}</heading>
    <acceleration>0.0</acceleration>
</vehicle_message>"""
        return xml_template
    
    def send_message(self, xml_message: str):
        """发送消息"""
        try:
            self.socket.sendto(xml_message.encode('utf-8'), (self.target_ip, self.target_port))
        except Exception as e:
            print(f"发送消息失败: {e}")
    
    def vehicle_sender_thread(self, simulator: VehicleSimulator):
        """单个车辆的发送线程"""
        last_time = time.time()
        
        while self.running:
            current_time = time.time()
            time_delta = current_time - last_time
            
            # 更新车辆位置
            simulator.update_position(time_delta)
            
            # 获取当前状态
            vehicle_state = simulator.get_current_state()
            
            # 创建并发送XML消息
            xml_message = self.create_xml_message(vehicle_state)
            self.send_message(xml_message)
            
            print(f"发送 {vehicle_state['vehicle_id']}: "
                  f"({vehicle_state['latitude']:.6f}, {vehicle_state['longitude']:.6f}) "
                  f"速度:{vehicle_state['speed_kmh']:.1f}km/h "
                  f"航向:{vehicle_state['heading']:.1f}°")
            
            last_time = current_time
            time.sleep(simulator.config.send_interval)
    
    def start(self):
        """启动发送器"""
        if not self.simulators:
            print("没有车辆需要模拟")
            return
        
        self.running = True
        
        # 为每个车辆创建发送线程
        for simulator in self.simulators:
            thread = threading.Thread(target=self.vehicle_sender_thread, args=(simulator,))
            thread.start()
            self.send_threads.append(thread)
        
        print(f"V2X消息发送器启动，目标: {self.target_ip}:{self.target_port}")
        print(f"模拟车辆数: {len(self.simulators)}")
    
    def stop(self):
        """停止发送器"""
        self.running = False
        
        # 等待所有线程结束
        for thread in self.send_threads:
            thread.join()
        
        self.socket.close()
        print("V2X消息发送器已停止")


def create_preset_vehicles(lidar_gps_lat: float, lidar_gps_lon: float) -> List[VehicleSimConfig]:
    """创建预设的车辆配置"""
    vehicles = []
    
    # 车辆1：从LiDAR位置向北行驶
    vehicles.append(VehicleSimConfig(
        vehicle_id="CAR_001",
        start_lat=lidar_gps_lat - 0.001,  # 南边约100米
        start_lon=lidar_gps_lon,
        speed_kmh=50.0,
        heading=0.0,  # 向北
        send_interval=0.5
    ))
    
    # 车辆2：从LiDAR位置向东行驶
    vehicles.append(VehicleSimConfig(
        vehicle_id="CAR_002", 
        start_lat=lidar_gps_lat,
        start_lon=lidar_gps_lon - 0.001,  # 西边约100米
        speed_kmh=40.0,
        heading=90.0,  # 向东
        send_interval=0.6
    ))
    
    # 车辆3：从LiDAR位置向南行驶
    vehicles.append(VehicleSimConfig(
        vehicle_id="CAR_003",
        start_lat=lidar_gps_lat + 0.0015,  # 北边约150米
        start_lon=lidar_gps_lon,
        speed_kmh=60.0,
        heading=180.0,  # 向南
        send_interval=0.4
    ))
    
    return vehicles


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="V2X消息发送器")
    parser.add_argument('--target-ip', type=str, default="127.0.0.1",
                       help="目标IP地址")
    parser.add_argument('--target-port', type=int, default=8888,
                       help="目标端口")
    parser.add_argument('--lidar-gps-lat', type=float, required=True,
                       help="LiDAR传感器GPS纬度")
    parser.add_argument('--lidar-gps-lon', type=float, required=True,
                       help="LiDAR传感器GPS经度")
    parser.add_argument('--vehicle-config', type=str,
                       help="车辆配置JSON文件路径")
    parser.add_argument('--duration', type=int, default=300,
                       help="运行时长（秒）")
    
    args = parser.parse_args()
    
    print("=== V2X消息发送器 ===")
    print(f"目标地址: {args.target_ip}:{args.target_port}")
    print(f"LiDAR GPS位置: ({args.lidar_gps_lat:.6f}, {args.lidar_gps_lon:.6f})")
    
    # 创建发送器
    sender = V2XMessageSender(args.target_ip, args.target_port)
    
    # 加载车辆配置
    if args.vehicle_config and os.path.exists(args.vehicle_config):
        # 从JSON文件加载配置
        with open(args.vehicle_config, 'r') as f:
            config_data = json.load(f)
        
        for vehicle_data in config_data['vehicles']:
            config = VehicleSimConfig(**vehicle_data)
            sender.add_vehicle(config)
    else:
        # 使用预设车辆配置
        print("使用预设车辆配置")
        preset_vehicles = create_preset_vehicles(args.lidar_gps_lat, args.lidar_gps_lon)
        for config in preset_vehicles:
            sender.add_vehicle(config)
    
    try:
        # 启动发送器
        sender.start()
        
        # 运行指定时长
        print(f"运行 {args.duration} 秒...")
        time.sleep(args.duration)
        
    except KeyboardInterrupt:
        print("\n接收到中断信号，正在停止...")
    
    finally:
        # 停止发送器
        sender.stop()
        print("程序结束")


def create_sample_config_file():
    """创建示例配置文件"""
    sample_config = {
        "vehicles": [
            {
                "vehicle_id": "CAR_001",
                "start_lat": 39.9042,
                "start_lon": 116.4074,
                "speed_kmh": 50.0,
                "heading": 0.0,
                "send_interval": 0.5
            },
            {
                "vehicle_id": "CAR_002",
                "start_lat": 39.9052,
                "start_lon": 116.4084,
                "speed_kmh": 40.0,
                "heading": 90.0,
                "send_interval": 0.6
            }
        ]
    }
    
    with open('sample_vehicle_config.json', 'w') as f:
        json.dump(sample_config, f, indent=2)
    
    print("示例配置文件已创建: sample_vehicle_config.json")


if __name__ == "__main__":
    import os
    
    # 如果没有参数，创建示例配置文件
    if len(os.sys.argv) == 1:
        create_sample_config_file()
        print("\n使用示例:")
        print("python v2x_message_sender.py --lidar-gps-lat 39.9042 --lidar-gps-lon 116.4074")
        print("python v2x_message_sender.py --lidar-gps-lat 39.9042 --lidar-gps-lon 116.4074 --vehicle-config sample_vehicle_config.json")
    else:
        main() 