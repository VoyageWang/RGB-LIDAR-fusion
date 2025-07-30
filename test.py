# from ultralytics import YOLO
# import time

# # Load a YOLO11n PyTorch model
# model = YOLO("/home/nebula/RGB-LIDAR-fusion/Code/yolov8m-seg.pt")

# # Export the model to TensorRT
# model.export(format="engine")  # creates 'yolo11n.engine'

# # Load the exported TensorRT model
# trt_model = YOLO("yolo11n.engine")

# # Run inference
# print(time.time())
# results = trt_model("/home/nebula/RGB-LIDAR-fusion/V2X-Seq-SPD-Example/infrastructure-side/image/010699.jpg")
# results = trt_model("/home/nebula/RGB-LIDAR-fusion/V2X-Seq-SPD-Example/infrastructure-side/image/010700.jpg")
# results = trt_model("/home/nebula/RGB-LIDAR-fusion/V2X-Seq-SPD-Example/infrastructure-side/image/010701.jpg")
# results = trt_model("/home/nebula/RGB-LIDAR-fusion/V2X-Seq-SPD-Example/infrastructure-side/image/010702.jpg")
# results = trt_model("/home/nebula/RGB-LIDAR-fusion/V2X-Seq-SPD-Example/infrastructure-side/image/010703.jpg")
# orint(time.time())


# from ultralytics import YOLO

# # Load a model
# model = YOLO('/home/nebula/RGB-LIDAR-fusion/Code/yolov8m-seg.pt')  # load a custom trained
# # Export the model
# model.export(format='engine',half=True,simplify=True)

#出现requirements报错，直接ctrl+c跳过
# import cv2
# from ultralytics import YOLO
# from cv2 import getTickCount, getTickFrequency
# # 加载 YOLOv8 模型
# model = YOLO("/home/nebula/RGB-LIDAR-fusion/Code/yolov8m-seg.engine")
# imgpath="/home/nebula/RGB-LIDAR-fusion/V2X-Seq-SPD-Example/infrastructure-side/image/010703.jpg"
# results = model.predict(source=imgpath,save=True) # 对当前帧进行目标检测并显示结果  

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
        self.frames = deque(maxlen=20)
        self.status = False
        self.timestamps = deque(maxlen=20)
        self.camera_rstp = camera_rstp
        self.interval = frame_interval
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
        if self.status:
            return self.frame
    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self.update, daemon=True)
        self._thread.start()
    def update(self):
        while self._running:
            if self.cap.isOpened():
                (self.status, self.frame) = self.cap.read()
    def stop(self):
        self._running = False
        if self._thread.is_alive():
            self._thread.join()
        if self.cap:
            self.cap.release()
    def _run(self):
        last_save_time = 0
        while self._running:
            ret, frame = self.cap.read()
            if not ret:
                print("错误: 无法读取帧")
                continue
            current_time = time.time()
            if current_time - last_save_time >= self.interval:
                with self._lock:
                    self.frames.append(frame)
                    self.timestamps.append(current_time)
                last_save_time = current_time
            time.sleep(0.03)
    def return_frame(self):
        with self._lock:
            return list(self.frames),list(self.timestamps)

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
            pointsxyzid = point_cloud.copy_data("xyz")
            pointsxyzid_data = pointsxyzid.reshape(-1, 3)
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
        pointsxyzid = point_cloud.copy_data("xyz")
        pointsxyzid_data = pointsxyzid.reshape(-1, 3)
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
    d = RGBCamera(camera_rstp = "rtsp://admin:Wuhan.123@192.168.20.222")
    a = Tanwaylidar()
    c = Tanwaylidar(lidar_ip = "192.168.20.223")
    i = 0
    pcd = o3d.geometry.PointCloud()
    pcd2 = o3d.geometry.PointCloud()
    b.start()
    d.start()
    time.sleep(2)
    while True:
        output_path1 = f"Data/frame/view1/{i}.jpg"
        output_path2 = f"Data/pointcloud/view1/{i}.pcd"
        output_path3 = f"Data/frame/view2/{i}.jpg"
        output_path4 = f"Data/pointcloud/view2/{i}.pcd"
        frame1= b.get_one_frame()
        point1 = a.get_one_pcd()
        point2 = c.get_one_pcd()
        frame2 = d.get_one_frame()
        pcd.points = o3d.utility.Vector3dVector(point1)
        cv2.imwrite(output_path1,frame1)
        pcd2.points = o3d.utility.Vector3dVector(point2)
        cv2.imwrite(output_path3,frame2)
        o3d.io.write_point_cloud(output_path2,pcd)
        o3d.io.write_point_cloud(output_path4,pcd2)
        print(i)
        i = i+1
        time.sleep(0.1)


    

    










# import torch
# import torchvision
# #查看版本

# print(torch.__version__)
# print(torchvision.__version__)
# #查看gpu是否可用

# print(torch.cuda.is_available())

# #返回设备gpu个数

# print(torch.cuda.device_count())

# # 查看对应CUDA的版本号

# print(torch.backends.cudnn.version())

# print(torch.version.cuda)

# #退出python

# quit()

