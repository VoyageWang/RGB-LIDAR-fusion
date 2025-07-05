#!/usr/bin/env python3
"""
改进的V2X可视化脚本
修复投影问题并添加完整的3D检测可视化
"""

import json
import cv2
import numpy as np
import open3d as o3d
import os
import sys
import gc
import traceback

# 导入必要的模块
from ultralytics import YOLO
from fusion import *
from utils import *
from visualization import *


class ImprovedV2XCalibration:
    """改进的V2X标定类"""
    
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
            
            print("标定参数加载成功")
            
        except Exception as e:
            print(f"标定初始化失败: {e}")
            raise
    
    def convert_3D_to_2D(self, points_3D):
        """将3D点云投影到2D图像平面"""
        try:
            if points_3D is None or len(points_3D) == 0:
                return np.array([]), np.array([])
            
            points_3D = np.asarray(points_3D, dtype=np.float64)
            
            # 检查输入有效性
            if points_3D.ndim != 2 or points_3D.shape[1] != 3:
                print(f"点云形状错误: {points_3D.shape}")
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
            front_mask = points_cam[:, 2] > 0.1  # 至少0.1米远
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


class SafeYOLODetector:
    """安全的YOLO检测器，支持3D融合"""
    
    def __init__(self, model_path, tracking=False):
        self.model = YOLO(model_path)
        self.tracking = tracking
        self.last_ground_center_of_id = {}
    
    def process_frame(self, frame, points, calibration, erosion_factor=25, depth_factor=20):
        """处理单帧数据，进行3D检测"""
        try:
            print("开始YOLO推理...")
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
            
            print(f"YOLO检测到 {len(boxes) if boxes else 0} 个目标")
            
            # 点云投影
            print("进行点云投影...")
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
            
            print(f"有效投影点数: {len(pts_2D)}")
            
            # 处理检测结果
            all_corners_3D = []
            all_filtered_points_of_object = []
            all_object_IDs = []
            objects3d_data = []
            
            if boxes is None or len(boxes) == 0:
                return objects3d_data, all_corners_3D, pts_3D, pts_2D, all_filtered_points_of_object, all_object_IDs
            
            print("开始3D融合处理...")
            for j, cls in enumerate(boxes.cls.tolist()):
                try:
                    conf = boxes.conf.tolist()[j] if boxes.conf is not None else None
                    box_id = int(boxes.id.tolist()[j]) if boxes.id is not None else j
                    
                    all_object_IDs.append(box_id)
                    
                    print(f"处理目标 {j+1}/{len(boxes)}: 类别{int(cls)}, 置信度{conf:.2f}")
                    
                    # 检查掩码
                    if masks is None or j >= len(masks.xy) or masks.xy[j].size == 0:
                        print(f"  目标 {j} 没有有效掩码，跳过")
                        continue
                    
                    # 融合处理
                    print(f"  进行LiDAR-相机融合...")
                    fusion_result = lidar_camera_fusion(
                        pts_3D, pts_2D, frame, masks.xy[j], int(cls), 
                        calibration, erosion_factor=erosion_factor, 
                        depth_factor=depth_factor, PCA=False
                    )
                    
                    if fusion_result is not None:
                        filtered_points_of_object, corners_3D, yaw = fusion_result
                        all_corners_3D.append(corners_3D)
                        all_filtered_points_of_object.append(filtered_points_of_object)
                        
                        print(f"  成功生成3D边界框，点数: {len(filtered_points_of_object)}")
                        
                        # 计算目标信息
                        ROS_type = int(np.int32(cls))
                        bottom_indices = np.argsort(corners_3D[:, 2])[:4]
                        ROS_ground_center = np.mean(corners_3D[bottom_indices], axis=0)
                        ROS_dimensions = np.ptp(corners_3D, axis=0)
                        ROS_points = corners_3D
                        
                        objects3d_data.append([ROS_type, ROS_ground_center, ROS_dimensions, ROS_points])
                    else:
                        print(f"  目标 {j} 融合失败")
                        
                except Exception as e:
                    print(f"处理目标 {j} 时出错: {e}")
                    continue
            
            print(f"3D检测完成，共生成 {len(all_corners_3D)} 个3D边界框")
            return objects3d_data, all_corners_3D, pts_3D, pts_2D, all_filtered_points_of_object, all_object_IDs
            
        except Exception as e:
            print(f"处理帧时出错: {e}")
            traceback.print_exc()
            return [], [], np.array([]), np.array([]), [], []


def load_v2x_data():
    """加载V2X数据"""
    print("=== 加载V2X数据 ===")
    
    try:
        data_root = "/mnt/disk_3/yuji/voyage_Pro/YOLO-LiDAR-Fusion/data/V2X-Seq-SPD-Example/infrastructure-side/"
        data_info_path = "/mnt/disk_3/yuji/voyage_Pro/YOLO-LiDAR-Fusion/data/V2X-Seq-SPD-Example/infrastructure-side/data_info.json"
        frame_id = "010699"
        
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
        
        print(f"图像路径: {image_path}")
        print(f"点云路径: {pointcloud_path}")
        
        # 检查文件存在性
        for path in [image_path, pointcloud_path, camera_intrinsic_path, lidar_to_camera_path]:
            if not os.path.exists(path):
                print(f"文件不存在: {path}")
                return None
        
        # 加载图像
        print("加载图像...")
        image = cv2.imread(image_path)
        if image is None:
            print("无法读取图像")
            return None
        print(f"图像尺寸: {image.shape}")
        
        # 加载点云
        print("加载点云...")
        pcd = o3d.io.read_point_cloud(pointcloud_path)
        points = np.asarray(pcd.points, dtype=np.float64)
        print(f"原始点云点数: {len(points)}")
        
        if len(points) == 0:
            print("点云为空")
            return None
        
        # 过滤无效点
        if np.any(np.isnan(points)) or np.any(np.isinf(points)):
            print("发现无效点，正在过滤...")
            valid_mask = np.isfinite(points).all(axis=1)
            points = points[valid_mask]
            print(f"过滤后点数: {len(points)}")
        
        # 创建标定对象
        print("创建标定对象...")
        calibration = ImprovedV2XCalibration(camera_intrinsic_path, lidar_to_camera_path)
        
        print("数据加载成功!")
        return image, points, calibration, frame_info
        
    except Exception as e:
        print(f"数据加载失败: {e}")
        traceback.print_exc()
        return None


def visualize_point_cloud_projection(image, points, calibration, max_points=5000):
    """可视化点云投影"""
    print(f"\n=== 可视化点云投影 ===")
    
    try:
        # 限制点数以提高性能
        if len(points) > max_points:
            indices = np.random.choice(len(points), max_points, replace=False)
            sample_points = points[indices]
        else:
            sample_points = points
        
        print(f"使用 {len(sample_points)} 个点进行投影")
        
        # 投影点云
        pts_2D, valid_mask = calibration.convert_3D_to_2D(sample_points)
        
        print(f"投影后2D点数: {len(pts_2D)}")
        print(f"有效点数: {np.sum(valid_mask)}")
        
        if len(pts_2D) == 0:
            print("没有有效的投影点")
            return image.copy()
        
        # 过滤图像边界内的点
        img_height, img_width = image.shape[:2]
        valid_2d_mask = (
            (pts_2D[:, 0] >= 0) & (pts_2D[:, 0] < img_width) &
            (pts_2D[:, 1] >= 0) & (pts_2D[:, 1] < img_height)
        )
        
        valid_2d_points = pts_2D[valid_2d_mask]
        print(f"图像内的点数: {len(valid_2d_points)}")
        
        # 创建结果图像
        result_image = image.copy()
        
        if len(valid_2d_points) > 0:
            # 获取对应的3D点用于颜色编码
            valid_indices = np.where(valid_mask)[0][valid_2d_mask]
            valid_3d_points = sample_points[valid_indices]
            
            # 根据距离进行颜色编码
            distances = np.linalg.norm(valid_3d_points, axis=1)
            max_dist = np.percentile(distances, 95)  # 使用95%分位数避免异常值
            min_dist = np.percentile(distances, 5)
            
            for i, (x, y) in enumerate(valid_2d_points):
                # 颜色编码：近处为红色，远处为蓝色
                dist_norm = (distances[i] - min_dist) / (max_dist - min_dist)
                dist_norm = np.clip(dist_norm, 0, 1)
                
                color = (
                    int(255 * (1 - dist_norm)),  # B
                    int(255 * (1 - abs(dist_norm - 0.5) * 2)),  # G
                    int(255 * dist_norm)  # R
                )
                
                cv2.circle(result_image, (int(x), int(y)), 1, color, -1)
        
        return result_image
        
    except Exception as e:
        print(f"点云投影可视化失败: {e}")
        traceback.print_exc()
        return image.copy()


def run_3d_detection_and_visualization(image, points, calibration):
    """运行3D检测并可视化"""
    print(f"\n=== 3D检测和可视化 ===")
    
    try:
        # 初始化检测器
        model_path = "yolov8m-seg.pt"
        print(f"初始化3D检测器: {model_path}")
        detector = SafeYOLODetector(model_path, tracking=False)
        
        # 运行3D检测
        print("运行3D检测...")
        objects3d_data, all_corners_3D, pts_3D, pts_2D, all_filtered_points_of_object, all_object_IDs = detector.process_frame(
            image, points, calibration, erosion_factor=25, depth_factor=20
        )
        
        print(f"检测到 {len(objects3d_data)} 个3D目标")
        
        # 创建结果图像
        result_image = image.copy()
        
        # 绘制3D边界框
        print("绘制3D边界框...")
        for j, pred_corner_3D in enumerate(all_corners_3D):
            object_id = all_object_IDs[j] if j < len(all_object_IDs) else None
            # 修复：正确处理convert_3D_to_2D的返回值
            try:
                pred_corners_2D, _ = calibration.convert_3D_to_2D(pred_corner_3D)
                if len(pred_corners_2D) >= 8:  # 确保有足够的点
                    pred_edges_2D = get_pred_bbox_edges(pred_corners_2D)
                    
                    for pred_edge in pred_edges_2D:
                        pt1 = tuple(np.int32(pred_edge[0]))
                        pt2 = tuple(np.int32(pred_edge[1]))
                        cv2.line(result_image, pt1, pt2, (0, 0, 255), 2)
                    
                    if object_id is not None and len(pred_corners_2D) > 7:
                        # top corners: 1, 4, 6, 7 
                        top_left_front_corner = pred_corners_2D[7]
                        top_left_front_pt = (int(np.round(top_left_front_corner[0])), int(np.round(top_left_front_corner[1])) - 10)
                        
                        # Write the ID at the top-left corner
                        cv2.putText(result_image, f'ID: {object_id}', top_left_front_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 3, cv2.LINE_AA)
                        cv2.putText(result_image, f'ID: {object_id}', top_left_front_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 1, cv2.LINE_AA)
                else:
                    print(f"  目标 {j} 投影点数不足: {len(pred_corners_2D)}")
            except Exception as e:
                print(f"  绘制目标 {j} 的3D边界框时出错: {e}")
        
        # 绘制点云投影
        print("绘制目标点云...")
        if len(all_filtered_points_of_object) > 0:
            all_filtered_points_combined = np.vstack(all_filtered_points_of_object)
            # 修复：正确处理点云绘制
            try:
                if len(all_filtered_points_combined) > 0:
                    pts_to_draw_2D, _ = calibration.convert_3D_to_2D(all_filtered_points_combined)
                    
                    # 获取颜色
                    colors = assign_colors_by_depth(pts_3D)
                    
                    # 绘制点云
                    for i, pt_2d in enumerate(pts_to_draw_2D):
                        if i < len(colors):
                            color = colors[i % len(colors)]
                            pt = (int(np.round(pt_2d[0])), int(np.round(pt_2d[1])))
                            # 检查点是否在图像范围内
                            if 0 <= pt[0] < result_image.shape[1] and 0 <= pt[1] < result_image.shape[0]:
                                cv2.circle(result_image, pt, 2, color=(int(color[0]), int(color[1]), int(color[2])), thickness=-1)
            except Exception as e:
                print(f"  绘制点云时出错: {e}")
        
        # 清理内存
        del detector
        gc.collect()
        
        return result_image, all_corners_3D, all_filtered_points_of_object, objects3d_data
        
    except Exception as e:
        print(f"3D检测失败: {e}")
        traceback.print_exc()
        return image.copy(), [], [], []


def create_bev_visualization_with_3d_boxes(points, all_corners_3D, max_points=10000):
    """创建包含3D边界框的鸟瞰图可视化"""
    print(f"\n=== 创建3D鸟瞰图 ===")
    
    try:
        # 限制点数
        if len(points) > max_points:
            indices = np.random.choice(len(points), max_points, replace=False)
            sample_points = points[indices]
        else:
            sample_points = points
        
        # 设置BEV参数
        x_range = (-50, 50)  # 米
        y_range = (-50, 50)  # 米
        resolution = 0.2  # 米/像素
        
        # 计算图像尺寸
        width = int((x_range[1] - x_range[0]) / resolution)
        height = int((y_range[1] - y_range[0]) / resolution)
        
        # 创建BEV图像
        bev_image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # 绘制点云
        valid_mask = (
            (sample_points[:, 0] >= x_range[0]) & (sample_points[:, 0] <= x_range[1]) &
            (sample_points[:, 1] >= y_range[0]) & (sample_points[:, 1] <= y_range[1])
        )
        
        valid_points = sample_points[valid_mask]
        print(f"BEV范围内的点数: {len(valid_points)}")
        
        if len(valid_points) > 0:
            # 转换到图像坐标
            x_img = ((valid_points[:, 0] - x_range[0]) / resolution).astype(int)
            y_img = ((valid_points[:, 1] - y_range[0]) / resolution).astype(int)
            
            # 根据高度进行颜色编码
            heights = valid_points[:, 2]
            height_norm = (heights - np.min(heights)) / (np.max(heights) - np.min(heights) + 1e-6)
            
            for i in range(len(valid_points)):
                if 0 <= x_img[i] < width and 0 <= y_img[i] < height:
                    # 颜色编码：低处为蓝色，高处为红色
                    color = (
                        int(255 * (1 - height_norm[i])),  # B
                        int(255 * (1 - abs(height_norm[i] - 0.5) * 2)),  # G
                        int(255 * height_norm[i])  # R
                    )
                    cv2.circle(bev_image, (x_img[i], height - 1 - y_img[i]), 1, color, -1)
        
        # 绘制3D边界框
        print(f"绘制 {len(all_corners_3D)} 个3D边界框...")
        for corners_3D in all_corners_3D:
            # 只使用底面的4个点
            bottom_indices = np.argsort(corners_3D[:, 2])[:4]
            bottom_corners = corners_3D[bottom_indices]
            
            # 转换到BEV坐标
            bev_corners = []
            for corner in bottom_corners:
                if x_range[0] <= corner[0] <= x_range[1] and y_range[0] <= corner[1] <= y_range[1]:
                    x_bev = int((corner[0] - x_range[0]) / resolution)
                    y_bev = int((corner[1] - y_range[0]) / resolution)
                    bev_corners.append([x_bev, height - 1 - y_bev])
            
            if len(bev_corners) >= 3:  # 至少需要3个点来绘制多边形
                bev_corners = np.array(bev_corners, dtype=np.int32)
                cv2.polylines(bev_image, [bev_corners], True, (0, 255, 0), 2)
        
        # 添加网格线
        grid_spacing = int(10 / resolution)  # 每10米一条线
        for i in range(0, width, grid_spacing):
            cv2.line(bev_image, (i, 0), (i, height-1), (50, 50, 50), 1)
        for i in range(0, height, grid_spacing):
            cv2.line(bev_image, (0, i), (width-1, i), (50, 50, 50), 1)
        
        # 添加中心点
        center_x, center_y = width // 2, height // 2
        cv2.circle(bev_image, (center_x, center_y), 5, (255, 255, 255), -1)
        
        return bev_image
        
    except Exception as e:
        print(f"BEV可视化失败: {e}")
        traceback.print_exc()
        return np.zeros((500, 500, 3), dtype=np.uint8)


def main():
    """主函数"""
    print("开始改进的V2X 3D检测可视化...")
    
    try:
        # 加载数据
        data_result = load_v2x_data()
        if data_result is None:
            print("数据加载失败")
            return
        
        image, points, calibration, frame_info = data_result
        
        # 创建输出目录
        output_dir = "./visualization_output"
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 原始图像
        cv2.imwrite(os.path.join(output_dir, "01_original_image.jpg"), image)
        print("保存原始图像")
        
        # 2. 点云投影可视化
        projection_image = visualize_point_cloud_projection(image, points, calibration)
        cv2.imwrite(os.path.join(output_dir, "02_point_cloud_projection.jpg"), projection_image)
        print("保存点云投影图像")
        
        # 3. 3D检测结果
        detection_image, all_corners_3D, all_filtered_points_of_object, objects3d_data = run_3d_detection_and_visualization(image, points, calibration)
        cv2.imwrite(os.path.join(output_dir, "03_3d_detection.jpg"), detection_image)
        print("保存3D检测图像")
        
        # 4. 包含3D边界框的鸟瞰图
        bev_image = create_bev_visualization_with_3d_boxes(points, all_corners_3D)
        cv2.imwrite(os.path.join(output_dir, "04_bev_with_3d_boxes.jpg"), bev_image)
        print("保存3D鸟瞰图")
        
        # 5. 组合图像
        # 调整图像尺寸以便组合
        h, w = image.shape[:2]
        target_h, target_w = 540, 960  # 缩放到一半
        
        img1 = cv2.resize(image, (target_w, target_h))
        img2 = cv2.resize(projection_image, (target_w, target_h))
        img3 = cv2.resize(detection_image, (target_w, target_h))
        img4 = cv2.resize(bev_image, (target_w, target_h))
        
        # 创建2x2组合图像
        top_row = np.hstack([img1, img2])
        bottom_row = np.hstack([img3, img4])
        combined_image = np.vstack([top_row, bottom_row])
        
        # 添加标题
        cv2.putText(combined_image, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(combined_image, "Point Cloud Projection", (target_w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(combined_image, "3D Detection", (10, target_h + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(combined_image, "3D Bird's Eye View", (target_w + 10, target_h + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.imwrite(os.path.join(output_dir, "05_combined_3d_visualization.jpg"), combined_image)
        print("保存组合3D可视化图像")
        
        # 打印检测统计
        print(f"\n=== 3D检测统计 ===")
        print(f"检测到 {len(objects3d_data)} 个3D目标")
        for i, obj_data in enumerate(objects3d_data):
            obj_type, ground_center, dimensions, points_3d = obj_data
            class_names = {0: "行人", 1: "骑行者", 2: "汽车", 3: "摩托车", 5: "公交车", 6: "卡车", 7: "火车"}
            class_name = class_names.get(obj_type, f"类别{obj_type}")
            print(f"目标 {i+1}: {class_name}")
            print(f"  中心位置: ({ground_center[0]:.2f}, {ground_center[1]:.2f}, {ground_center[2]:.2f})")
            print(f"  尺寸: {dimensions[0]:.2f} x {dimensions[1]:.2f} x {dimensions[2]:.2f}")
        
        print(f"\n所有3D可视化结果已保存到: {output_dir}")
        print("文件列表:")
        print("- 01_original_image.jpg: 原始图像")
        print("- 02_point_cloud_projection.jpg: 点云投影")
        print("- 03_3d_detection.jpg: 3D检测结果")
        print("- 04_bev_with_3d_boxes.jpg: 3D鸟瞰图")
        print("- 05_combined_3d_visualization.jpg: 组合3D可视化")
        
    except Exception as e:
        print(f"可视化过程失败: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main() 