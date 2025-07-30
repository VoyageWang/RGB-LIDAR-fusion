import rospy
import open3d as o3d
import numpy as np
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2

import os
import socket
import threading
import queue
import time
import struct
import math

Save_fps = 1 #几帧保留一次pcd文件
Save_path = "velodyne"

"""
接口：
get_pcd()
get_points_list()
"""
class lidar_listener_base_ros:
    def __init__(self,topic):
        rospy.init_node('lidar_listener',anonymous = True)#初始化节点
        rospy.Subscriber(topic, PointCloud2, self.callback,queue_size = 1)#订阅话题 数据格式为ros的pcd
        self.frame_counter = 0
        self.points_list = [] 
        self.save_path = Save_path
        self.save_number = 0
        self.accept_flag = 0 #判断新接收到消息与否
        self.save_fps = Save_fps
        self.pcd = o3d.geometry.PointCloud()
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
    def callback(self,data):
        """回调函数"""
        points = []
        # for point in pc2.read_points(data, field_names=("x", "y", "z"), skip_nans=True): #读取Ros点云文件
        #     points.append([point[0], point[1], point[2]])
        points = np.array(list(pc2.read_points(data, field_names=("x", "y", "z"), skip_nans=True)))
        points = points.astype(np.float32)
        print(points)
        if len(points) == 0:#如果点云为空
            rospy.logwarn("Received empty pointcloud.")
            return
        elif self.save_fps != 1:
            self.points_list.extend(points)
            self.frame_counter += 1
        if self.save_fps == 1:#实时输入
            self.points_list = points
            self.pcd.points = o3d.utility.Vector3dVector(points)
        elif self.frame_counter > 0 and self.frame_counter % self.save_fps == 0:
            points_np = (self.points_list)
            self.save_number += 1
            self.save_pcd(points_np)#保存pcd
            self.points_list = []
        self.accept_flag = 1 
    def save_pcd(self,points_np):
        """保存点云"""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_np)
        file_path = os.path.join(self.save_path,f"{self.save_number:06d}.pcd")
        o3d.io.write_point_cloud(file_path, pcd)
        rospy.loginfo("Saved PCD: %s", file_path)
    def spin(self):
        """等待接收信息"""
        rospy.spin()
    def get_pcd(self):
        pcd = self.pcd
        return pcd
    def get_points_list(self):
        return self.points_list
"""
输入目标监听端口后，创建两个类，self.start()开始监听
使用self.pcd可直接调用点云
"""
class lidar_listener_base_port:
    def __init__(self,local_ip = "192.168.20.228",listen_port = [5601,5701]):
        self.vertical_angle = [
            -12.368, -11.986, -11.603, -11.219, -10.834, -10.448, -10.061,
            -9.674,  -9.285,  -8.896,  -8.505,  -8.115,  -7.723,  -7.331,
            -6.938,  -6.545,  -6.151,  -5.756,  -5.361,  -4.966,  -4.570,
            -4.174,  -3.777,  -3.381,  -2.983,  -2.586,  -2.189,  -1.791,
            -1.393,  -0.995,  -0.597,  -0.199,   0.199,   0.597,   0.995,
            1.393,   1.791,   2.189,   2.586,   2.983,   3.381,   3.777,
            4.174,   4.570,   4.966,   5.361,   5.756,   6.151,   6.545,
            6.938,   7.331,   7.723,   8.115,   8.505,   8.896,   9.285,
            9.674,  10.061,  10.448,  10.834,  11.219,  11.603,  11.986,
            12.368]
        self.local_ip = local_ip
        self.listen_port = listen_port
        self._running = False
        self._thread = None
        self.data = None
        self.skewing = [1.0, 1.0, 1.0]
        self.frame_id = 0
        self.points = []
        self.mirror_offset = 0.0
        self.pcd = o3d.geometry.PointCloud()
        self._lock = threading.Lock()
    def start(self):
        """启动监听线程"""
        self._running = True
        self._thread1 = threading.Thread(target=self._listen_thread_pcf, daemon=True)
        self._thread2 = threading.Thread(target=self._listen_thread_dif, daemon=True)
        self._thread1.start()
        self._thread2.start()
        print(f"[Lidar_Listener_base_port] 正在监听 {self.local_ip}:{self.listen_port[0]},{self.local_ip}:{self.listen_port[1]}")
    def stop(self):
        """停止监听"""
        self._running = False
        if self._thread1:
            self._thread1.join()
        if self._thread2:
            self._thread2.join()
        print("[Lidar_Listener_base_port] 已停止")
    def _listen_thread_dif(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((self.local_ip, self.listen_port[1]))
        while self._running:
            try:
                data, _ = sock.recvfrom(65536)
                if data:
                    with self._lock:
                        self.parse_dif(data)
            except socket.timeout:
                continue
    def _listen_thread_pcf(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((self.local_ip, self.listen_port[0]))
        sock.settimeout(1)
        while self._running:
            try:
                data, _ = sock.recvfrom(65536)
                if data:
                    with self._lock:
                        frame_id,total_packet,points = self.parse_udp(data)
                        self.points = points
                        self.data2pcd()
                        if self.frame_id == frame_id:
                            self.points.extend(points)
                        else:
                            self.data2pcd()
                            self.frame_id = frame_id
                            self.points = points
            except socket.timeout:
                self.data2pcd()
                continue
    def parse_dif(self,data):
        pitch_angle_data = struct.unpack_from('<I', data, offset=564)[0]
        skewing = [
            ((pitch_angle_data >> 16) & 0xFF) * 0.04,
            ((pitch_angle_data >> 8) & 0xFF) * 0.04,
            (pitch_angle_data & 0xFF) * 0.04
        ]
        correct_skewing = [
            skewing[0] - 0.3,
            skewing[1] - 0.2,
            skewing[2] - 0.1
        ]
        self.skewing = [
            math.cos(math.radians(correct_skewing[0])),
            math.cos(math.radians(correct_skewing[1])),
            math.sin(math.radians(correct_skewing[2])),
        ]
    def parse_udp(self,data):
        """
        解析雷达 UDP 数据包，返回帧计数、包总数和点云列表。
        每个点包含  horizon_angle, echo1 距离/置信度, echo2 距离/置信度。
        """
        ETH_HEADER_LEN = 42
        HEADER_LEN = 32
        TAIL_LEN = 4
        BLOCK_NUM = 8
        CHANNELS_PER_BLOCK = 16
        BLOCK_HEADER_LEN = 4
        BLOCK_DATA_LEN = 10
        expected_len = 1348 
        if len(data) != expected_len:
            print(f"长度错误：收到 {len(data)} bytes，预期 {expected_len} bytes")
            return None
        # 去掉以太网头
        # data = data[ETH_HEADER_LEN:]
        header = data[:HEADER_LEN]
        payload = data[HEADER_LEN:-TAIL_LEN]

        frame_id = struct.unpack_from('<H', header, 4)[0]         # 第4~5字节
        total_packet = struct.unpack_from('<H', header, 21)[0]    # 第21~22字节
        points = []
        for block_idx in range(BLOCK_NUM):
            block_start = block_idx * (BLOCK_HEADER_LEN + CHANNELS_PER_BLOCK * BLOCK_DATA_LEN)
            block_data = payload[block_start:block_start + BLOCK_HEADER_LEN + CHANNELS_PER_BLOCK * BLOCK_DATA_LEN]
            offset = BLOCK_HEADER_LEN  # 跳过 block header
            for channel in range(CHANNELS_PER_BLOCK):
                vertical_channel = 65 - (16 * (block_idx - 4 if block_idx >= 4 else block_idx) + channel + 1)
                point_offset = offset + channel * BLOCK_DATA_LEN
                point = block_data[point_offset:point_offset + BLOCK_DATA_LEN]
                if len(point) < BLOCK_DATA_LEN:
                    continue  # 数据不完整
                horizon_angle, dist1, _, conf1_flag, dist2, _, conf2_flag = struct.unpack_from('<HHBBHBB', point, 0)
                points.append({
                    "block": block_idx,
                    "channel": channel,
                    "vertical_angle":self.vertical_angle[vertical_channel-1],
                    "horizon_angle":  horizon_angle * 0.01+self.mirror_offset,# 角度          
                    "distance_echo1": dist1 * 0.004796,# 回波一距离        
                    "confidence_echo1": conf1_flag & 0x01,#判断置信度
                    "distance_echo2": dist2 * 0.004796,
                    "confidence_echo2": conf2_flag & 0x01,
                })
        return frame_id,total_packet,points
    def data2pcd(self):
        pointcloud = []
        skewing = self.skewing
        points = self.points
        for point in points:
            print(point)
            L = np.array([
                math.cos(math.radians(point["vertical_angle"])),
                0,
                math.sin(math.radians(point["vertical_angle"]))
            ])
            N = np.array([
                skewing[0]*math.cos(math.radians(point["horizon_angle"])/2),
                skewing[1]*math.sin(math.radians(point["horizon_angle"])/2),
                skewing[2]
            ])
            dot_product = np.dot(L, N)
            if point["confidence_echo1"]:
                L_2 = point["distance_echo1"]
            elif point["confidence_echo2"]:
                L_2 = point["distance_echo2"]
            else:
                continue
            if L_2 <= 0 or not np.isfinite(L_2):
                continue
            [x,y,z] = [
                L_2*(L[0]-2*dot_product*N[0]),
                L_2*(L[1]-2*dot_product*N[1]),
                L_2*(L[2]-2*dot_product*N[2])
            ]
            print([x,y,z])
            pointcloud.append([x,y,z])
        pc_array = np.array(pointcloud, dtype=np.float32)
        self.pcd.points = o3d.utility.Vector3dVector(pc_array)
        # print(f"[Frame {self.frame_id}] 点云生成完成，共 {len(pointcloud)} 点")
    def get_pcd(self):
        return self.pcd
if __name__ == '__main__':
    # listener_ros = lidar_listener_base_ros("/tanwaylidar_pointcloud")
    # listener_ros.spin()
    listener_port = lidar_listener_base_port()
    listener_port.start()
    while(True):
        pass
