#!/usr/bin/env python3
"""
使用改进融合算法的V2X视频序列3D检测脚本
解决融合失败和近距离物体3D框过大的问题
"""

import json
import cv2
import numpy as np
import open3d as o3d
import os
import sys
import gc
import traceback
import argparse
from tqdm import tqdm

# 导入必要的模块
from ultralytics import YOLO
from improved_fusion import improved_lidar_camera_fusion
from utils import *
from visualization import *


class V2XImprovedCalibration:
    """V2X改进标定类"""
    
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


class ImprovedVideoYOLODetector:
    """改进的视频YOLO检测器，使用改进的融合算法"""
    
    def __init__(self, model_path, tracking=True):
        self.model = YOLO(model_path)
        self.tracking = tracking
        self.last_ground_center_of_id = {}
        
        # 统计信息
        self.fusion_stats = {
            'total_detections': 0,
            'successful_fusions': 0,
            'failed_fusions': 0,
            'failure_reasons': {}
        }
    
    def process_frame(self, frame, points, calibration, erosion_factor=25, depth_factor=20):
        """处理单帧数据，进行改进的3D检测"""
        try:
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
            
            # 点云投影
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
            
            # 处理检测结果
            all_corners_3D = []
            all_filtered_points_of_object = []
            all_object_IDs = []
            objects3d_data = []
            
            if boxes is None or len(boxes) == 0:
                return objects3d_data, all_corners_3D, pts_3D, pts_2D, all_filtered_points_of_object, all_object_IDs
            
            print(f"检测到 {len(boxes)} 个目标，开始融合处理...")
            
            for j, cls in enumerate(boxes.cls.tolist()):
                try:
                    self.fusion_stats['total_detections'] += 1
                    
                    conf = boxes.conf.tolist()[j] if boxes.conf is not None else None
                    box_id = int(boxes.id.tolist()[j]) if boxes.id is not None else j
                    
                    all_object_IDs.append(box_id)
                    
                    # 检查掩码
                    if masks is None or j >= len(masks.xy) or masks.xy[j].size == 0:
                        print(f"  目标 {j} (类别: {int(cls)}): 无有效掩码")
                        self.fusion_stats['failed_fusions'] += 1
                        self._record_failure_reason("无有效掩码")
                        continue
                    
                    print(f"  处理目标 {j} (类别: {int(cls)}, ID: {box_id})...")
                    
                    # 使用改进的融合处理
                    fusion_result = improved_lidar_camera_fusion(
                        pts_3D, pts_2D, frame, masks.xy[j], int(cls), 
                        calibration, erosion_factor=erosion_factor, 
                        depth_factor=depth_factor, PCA=False
                    )
                    
                    if fusion_result is not None:
                        filtered_points_of_object, corners_3D, yaw = fusion_result
                        all_corners_3D.append(corners_3D)
                        all_filtered_points_of_object.append(filtered_points_of_object)
                        
                        # 计算目标信息
                        ROS_type = int(np.int32(cls))
                        bottom_indices = np.argsort(corners_3D[:, 2])[:4]
                        ROS_ground_center = np.mean(corners_3D[bottom_indices], axis=0)
                        ROS_dimensions = np.ptp(corners_3D, axis=0)
                        ROS_points = corners_3D
                        
                        objects3d_data.append([ROS_type, ROS_ground_center, ROS_dimensions, ROS_points])
                        
                        self.fusion_stats['successful_fusions'] += 1
                        print(f"    融合成功！点数: {len(filtered_points_of_object)}, 尺寸: {ROS_dimensions}")
                        
                    else:
                        self.fusion_stats['failed_fusions'] += 1
                        print(f"    融合失败")
                        
                except Exception as e:
                    self.fusion_stats['failed_fusions'] += 1
                    self._record_failure_reason(f"异常: {str(e)}")
                    print(f"  目标 {j} 处理异常: {e}")
                    continue
            
            # 打印融合统计
            success_rate = (self.fusion_stats['successful_fusions'] / 
                          max(1, self.fusion_stats['total_detections'])) * 100
            print(f"融合成功率: {success_rate:.1f}% ({self.fusion_stats['successful_fusions']}/{self.fusion_stats['total_detections']})")
            
            return objects3d_data, all_corners_3D, pts_3D, pts_2D, all_filtered_points_of_object, all_object_IDs
            
        except Exception as e:
            print(f"处理帧时出错: {e}")
            return [], [], np.array([]), np.array([]), [], []
    
    def _record_failure_reason(self, reason):
        """记录融合失败原因"""
        if reason not in self.fusion_stats['failure_reasons']:
            self.fusion_stats['failure_reasons'][reason] = 0
        self.fusion_stats['failure_reasons'][reason] += 1
    
    def print_fusion_statistics(self):
        """打印融合统计信息"""
        print("\n=== 融合统计信息 ===")
        print(f"总检测数: {self.fusion_stats['total_detections']}")
        print(f"成功融合: {self.fusion_stats['successful_fusions']}")
        print(f"失败融合: {self.fusion_stats['failed_fusions']}")
        
        if self.fusion_stats['total_detections'] > 0:
            success_rate = (self.fusion_stats['successful_fusions'] / 
                          self.fusion_stats['total_detections']) * 100
            print(f"成功率: {success_rate:.2f}%")
        
        if self.fusion_stats['failure_reasons']:
            print("\n失败原因统计:")
            for reason, count in self.fusion_stats['failure_reasons'].items():
                print(f"  {reason}: {count}")


def draw_improved_3d_detection_results(image, all_corners_3D, all_filtered_points_of_object, all_object_IDs, calibration, pts_3D):
    """绘制改进的3D检测结果"""
    result_image = image.copy()
    
    # 绘制3D边界框
    for j, pred_corner_3D in enumerate(all_corners_3D):
        object_id = all_object_IDs[j] if j < len(all_object_IDs) else None
        try:
            pred_corners_2D, _ = calibration.convert_3D_to_2D(pred_corner_3D)
            if len(pred_corners_2D) >= 8:
                pred_edges_2D = get_pred_bbox_edges(pred_corners_2D)
                
                # 使用不同颜色表示不同目标
                colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
                color = colors[j % len(colors)]
                
                for pred_edge in pred_edges_2D:
                    pt1 = tuple(np.int32(pred_edge[0]))
                    pt2 = tuple(np.int32(pred_edge[1]))
                    cv2.line(result_image, pt1, pt2, color, 2)
                
                if object_id is not None and len(pred_corners_2D) > 7:
                    top_left_front_corner = pred_corners_2D[7]
                    top_left_front_pt = (int(np.round(top_left_front_corner[0])), int(np.round(top_left_front_corner[1])) - 10)
                    
                    cv2.putText(result_image, f'ID: {object_id}', top_left_front_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 3, cv2.LINE_AA)
                    cv2.putText(result_image, f'ID: {object_id}', top_left_front_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 1, cv2.LINE_AA)
        except Exception as e:
            continue
    
    # 绘制目标点云
    if len(all_filtered_points_of_object) > 0:
        try:
            all_filtered_points_combined = np.vstack(all_filtered_points_of_object)
            if len(all_filtered_points_combined) > 0:
                pts_to_draw_2D, _ = calibration.convert_3D_to_2D(all_filtered_points_combined)
                
                # 获取颜色
                if len(pts_3D) > 0:
                    colors = assign_colors_by_depth(pts_3D)
                    
                    # 绘制点云
                    for i, pt_2d in enumerate(pts_to_draw_2D):
                        if i < len(colors):
                            color = colors[i % len(colors)]
                            pt = (int(np.round(pt_2d[0])), int(np.round(pt_2d[1])))
                            if 0 <= pt[0] < result_image.shape[1] and 0 <= pt[1] < result_image.shape[0]:
                                cv2.circle(result_image, pt, 2, color=(int(color[0]), int(color[1]), int(color[2])), thickness=-1)
        except Exception as e:
            pass
    
    return result_image


def process_improved_v2x_video_sequence(data_root, data_info_path, output_dir, start_frame=None, end_frame=None, max_frames=None, tracking=True):
    """处理V2X视频序列（使用改进的融合算法）"""
    print("=== 开始处理V2X视频序列（改进版本）===")
    
    try:
        # 加载数据信息
        with open(data_info_path, 'r') as f:
            data_info = json.load(f)
        
        print(f"数据集总帧数: {len(data_info)}")
        
        # 确定处理范围
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = len(data_info)
        if max_frames is not None:
            end_frame = min(start_frame + max_frames, end_frame)
        
        frames_to_process = data_info[start_frame:end_frame]
        print(f"将处理帧 {start_frame} 到 {end_frame-1}，共 {len(frames_to_process)} 帧")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 初始化检测器（使用第一帧的标定信息）
        first_frame = frames_to_process[0]
        camera_intrinsic_path = os.path.join(data_root, first_frame['calib_camera_intrinsic_path'])
        lidar_to_camera_path = os.path.join(data_root, first_frame['calib_virtuallidar_to_camera_path'])
        
        calibration = V2XImprovedCalibration(camera_intrinsic_path, lidar_to_camera_path)
        
        model_path = "yolov8m-seg.pt"
        detector = ImprovedVideoYOLODetector(model_path, tracking=tracking)
        
        print(f"初始化完成，开始处理...")
        
        # 存储处理后的帧
        processed_frames = []
        bev_frames = []
        detection_stats = []
        
        # 处理每一帧
        for i, frame_info in enumerate(tqdm(frames_to_process, desc="处理帧")):
            try:
                # 构建文件路径
                image_path = os.path.join(data_root, frame_info['image_path'])
                pointcloud_path = os.path.join(data_root, frame_info['pointcloud_path'])
                
                # 检查文件存在性
                if not os.path.exists(image_path) or not os.path.exists(pointcloud_path):
                    print(f"跳过帧 {i}: 文件不存在")
                    continue
                
                # 加载图像
                image = cv2.imread(image_path)
                if image is None:
                    print(f"跳过帧 {i}: 无法读取图像")
                    continue
                
                # 加载点云
                pcd = o3d.io.read_point_cloud(pointcloud_path)
                points = np.asarray(pcd.points, dtype=np.float64)
                
                if len(points) == 0:
                    print(f"跳过帧 {i}: 点云为空")
                    continue
                
                # 过滤无效点
                if np.any(np.isnan(points)) or np.any(np.isinf(points)):
                    valid_mask = np.isfinite(points).all(axis=1)
                    points = points[valid_mask]
                
                print(f"\n处理帧 {i} (ID: {frame_info['frame_id']})...")
                
                # 运行改进的3D检测
                objects3d_data, all_corners_3D, pts_3D, pts_2D, all_filtered_points_of_object, all_object_IDs = detector.process_frame(
                    image, points, calibration, erosion_factor=25, depth_factor=20
                )
                
                # 绘制检测结果
                result_image = draw_improved_3d_detection_results(
                    image, all_corners_3D, all_filtered_points_of_object, all_object_IDs, calibration, pts_3D
                )
                
                # 创建BEV视图
                bev_image = create_bev_with_3d_boxes(points, all_corners_3D)
                
                # 添加帧信息
                frame_text = f"Frame: {start_frame + i} | Objects: {len(objects3d_data)} | Success Rate: {detector.fusion_stats['successful_fusions']}/{detector.fusion_stats['total_detections']}"
                cv2.putText(result_image, frame_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(bev_image, frame_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                processed_frames.append(result_image)
                bev_frames.append(bev_image)
                
                # 记录检测统计
                detection_stats.append({
                    'frame_id': frame_info['frame_id'],
                    'objects_count': len(objects3d_data),
                    'objects_data': objects3d_data,
                    'fusion_success_rate': detector.fusion_stats['successful_fusions'] / max(1, detector.fusion_stats['total_detections'])
                })
                
                # 定期清理内存
                if i % 10 == 0:
                    gc.collect()
                
            except Exception as e:
                print(f"处理帧 {i} 时出错: {e}")
                continue
        
        print(f"成功处理 {len(processed_frames)} 帧")
        
        # 打印最终统计
        detector.print_fusion_statistics()
        
        if len(processed_frames) > 0:
            # 创建视频
            print("创建检测结果视频...")
            detection_video_path = os.path.join(output_dir, "improved_v2x_3d_detection.mp4")
            create_video(processed_frames, detection_video_path)
            
            print("创建BEV视频...")
            bev_video_path = os.path.join(output_dir, "improved_v2x_bev_view.mp4")
            create_video(bev_frames, bev_video_path)
            
            print("创建组合视频...")
            combined_video_path = os.path.join(output_dir, "improved_v2x_combined_3d_detection.mp4")
            create_combined_video(processed_frames, bev_frames, combined_video_path)
            
            # 保存检测统计
            stats_path = os.path.join(output_dir, "improved_detection_statistics.json")
            with open(stats_path, 'w') as f:
                json.dump({
                    'detection_stats': detection_stats,
                    'fusion_stats': detector.fusion_stats
                }, f, indent=2, default=str)
            
            print(f"\n=== 处理完成 ===")
            print(f"输出目录: {output_dir}")
            print(f"- 改进3D检测视频: improved_v2x_3d_detection.mp4")
            print(f"- 改进BEV视频: improved_v2x_bev_view.mp4")
            print(f"- 改进组合视频: improved_v2x_combined_3d_detection.mp4")
            print(f"- 改进检测统计: improved_detection_statistics.json")
            
            # 打印总体统计
            total_objects = sum(stat['objects_count'] for stat in detection_stats)
            avg_objects = total_objects / len(detection_stats) if detection_stats else 0
            avg_success_rate = sum(stat['fusion_success_rate'] for stat in detection_stats) / len(detection_stats) if detection_stats else 0
            
            print(f"\n总检测目标数: {total_objects}")
            print(f"平均每帧目标数: {avg_objects:.2f}")
            print(f"平均融合成功率: {avg_success_rate:.2%}")
        
        # 清理
        del detector
        gc.collect()
        
    except Exception as e:
        print(f"处理视频序列失败: {e}")
        traceback.print_exc()


def create_bev_with_3d_boxes(points, all_corners_3D, max_points=5000):
    """创建包含3D边界框的鸟瞰图"""
    try:
        # 限制点数
        if len(points) > max_points:
            indices = np.random.choice(len(points), max_points, replace=False)
            sample_points = points[indices]
        else:
            sample_points = points
        
        # 设置BEV参数
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
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        for idx, corners_3D in enumerate(all_corners_3D):
            color = colors[idx % len(colors)]
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
        
        # 添加网格线
        grid_spacing = int(10 / resolution)
        for i in range(0, width, grid_spacing):
            cv2.line(bev_image, (i, 0), (i, height-1), (50, 50, 50), 1)
        for i in range(0, height, grid_spacing):
            cv2.line(bev_image, (0, i), (width-1, i), (50, 50, 50), 1)
        
        # 添加中心点
        center_x, center_y = width // 2, height // 2
        cv2.circle(bev_image, (center_x, center_y), 5, (255, 255, 255), -1)
        
        return bev_image
        
    except Exception as e:
        return np.zeros((500, 500, 3), dtype=np.uint8)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="改进的V2X视频序列3D检测")
    parser.add_argument('--data-root', type=str, 
                       default="/mnt/disk_3/yuji/voyage_Pro/YOLO-LiDAR-Fusion/data/V2X-Seq-SPD-Example/infrastructure-side/",
                       help="数据集根目录")
    parser.add_argument('--data-info', type=str,
                       default="/mnt/disk_3/yuji/voyage_Pro/YOLO-LiDAR-Fusion/data/V2X-Seq-SPD-Example/infrastructure-side/data_info.json",
                       help="数据信息JSON文件")
    parser.add_argument('--output-dir', type=str, default="./improved_video_output",
                       help="输出目录")
    parser.add_argument('--start-frame', type=int, default=None,
                       help="开始帧索引")
    parser.add_argument('--end-frame', type=int, default=None,
                       help="结束帧索引")
    parser.add_argument('--max-frames', type=int, default=3000,
                       help="最大处理帧数")
    parser.add_argument('--no-tracking', action='store_true',
                       help="禁用目标跟踪")
    
    args = parser.parse_args()
    
    print("改进的V2X视频序列3D检测开始...")
    print(f"数据根目录: {args.data_root}")
    print(f"输出目录: {args.output_dir}")
    print(f"最大帧数: {args.max_frames}")
    print(f"跟踪模式: {'关闭' if args.no_tracking else '开启'}")
    
    process_improved_v2x_video_sequence(
        data_root=args.data_root,
        data_info_path=args.data_info,
        output_dir=args.output_dir,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        max_frames=args.max_frames,
        tracking=not args.no_tracking
    )


if __name__ == "__main__":
    main() 