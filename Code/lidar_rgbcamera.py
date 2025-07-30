import time
from tanway import Lidar
import open3d as o3d
import numpy as np
import threading
import cv2
import datetime
import time
from collections import deque

class RGBCamera:
    def __init__(self,frame_interval = 0.07,camera_rstp = "rtsp://admin:Wuhan.123@192.168.20.220"):
        self.camera_rstp = camera_rstp
        self.status = False
        self.frame = None
        self.cap = cv2.VideoCapture(self.camera_rstp)
        self._running = False
        self._lock = threading.Lock()
        if not self.cap.isOpened():
            raise Exception(f"无法连接到RTSP流: {self.camera_rstp}")
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 10.0  # 如果无法获取FPS，默认使用10
        print(f"视频尺寸: {self.width}x{self.height}")
        print(f"FPS: {self.fps}")
    def get_one_frame(self):
        with self._lock:
            if self.status:
                current_time = time.time()
                return self.frame,current_time
            return None,None
    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self.update, daemon=True)
        self._thread.start()
    def update(self):
        while self._running:
            ret, frame = self.cap.read()
            if not ret:
                print("错误: 无法读取帧")
                continue
            with self._lock:
                self.status = ret
                self.frame = frame
    def stop(self):
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join()
        if self.cap:
            self.cap.release()

class Tanwaylidar:
    def __init__(self,lidar_ip = "192.168.20.221"):
        self.pcd = deque(maxlen=20)
        self._running = False
        self._lock = threading.Lock()
        self.lidar = Lidar()
        if lidar_ip == "192.168.20.221":
            self.lidar.create_online("192.168.20.221", "192.168.20.228",5601,
                                    5701, "FocusB2")
        else:
            self.lidar.create_online("192.168.20.223", "192.168.20.228",5602,
                                    5702, "FocusB2")
        #解析算法的参数表，目前还不完善，只有scope256的阳光噪点算法                                
        self.lidar.parse_algo_config("algorithms.json")
        #启动lidar
        print ('Lidar start return ' +  str(self.lidar.start()))
    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
    def stop(self):
        self._running = False
        if self._thread.is_alive():
            self._thread.join()
    def _run(self):
        while self._running:
            time.sleep(0.1) 
            point_cloud = self.lidar.capture()
            if( point_cloud.is_empty):
                print("empty")
                continue
            pointsxyzid = point_cloud.copy_data("xyzall")
            pointsxyzid_data = pointsxyzid.reshape(-1, 14)
            with self._lock:
                self.pcd.append(pointsxyzid_data)
    def get_one_pcd(self):
        while True:
            point_cloud = self.lidar.capture()
            if(point_cloud.is_empty):
                time.sleep(0.05)
                continue
            else:
                break
        pointsxyzid = point_cloud.copy_data("xyzall")
        pointsxyzid_data = pointsxyzid.reshape(-1, 14)
        return pointsxyzid_data
    def return_pcd(self):
        with self._lock:
            return list(self.pcd)
'''
        xyz: 
            ndarray(Height,Width,3) of float
        xyzi:   x,y,z,intensity
            ndarray(Height,Width,4) of float    
        xyzid:  x,y,z,intensity,distance
            ndarray(Height,Width,5) of float    
        xyzall: x,y,z,intensity,distance,channel,angle,pulse,echo,mirror,left_right,block,t_sec,t_usec
            ndarray(Height,Width,14) of float 
        - `t_sec`：时间戳的秒部分。
        - `t_usec`：时间戳的微秒部分。
'''


if __name__ == "__main__":
    b = RGBCamera()
    for i in range(5):
        frame,_ = b.get_one_frame()
