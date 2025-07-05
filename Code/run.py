import argparse
import os
import json
import cv2
import numpy as np
import open3d as o3d
from random import randint
from detector import *
from utils import *
from evaluation import *
from visualization import *


class V2XCalibration:
    """适配V2X数据集的标定类"""
    
    def __init__(self, camera_intrinsic_path, lidar_to_camera_path):
        # 读取相机内参
        with open(camera_intrinsic_path, 'r') as f:
            cam_data = json.load(f)
        
        # 读取LiDAR到相机的外参
        with open(lidar_to_camera_path, 'r') as f:
            lidar_cam_data = json.load(f)
        
        # 相机内参矩阵 (3x3)
        self.K = np.array(cam_data['cam_K']).reshape(3, 3)
        
        # 畸变参数
        self.D = np.array(cam_data['cam_D'])
        
        # LiDAR到相机的旋转矩阵 (3x3)
        self.R = np.array(lidar_cam_data['rotation'])
        
        # LiDAR到相机的平移向量 (3x1)
        self.t = np.array(lidar_cam_data['translation']).reshape(3, 1)
        
        # 构建变换矩阵 (4x4)
        self.T = np.eye(4)
        self.T[:3, :3] = self.R
        self.T[:3, 3:4] = self.t
        
        print(f"相机内参矩阵 K:\n{self.K}")
        print(f"LiDAR到相机变换矩阵 T:\n{self.T}")
    
    def convert_3D_to_2D(self, points_3D):
        """将3D点云投影到2D图像平面"""
        if len(points_3D) == 0:
            return np.array([]), np.array([])
        
        # 转换为齐次坐标
        points_3D_homo = np.hstack([points_3D, np.ones((points_3D.shape[0], 1))])
        
        # 应用LiDAR到相机的变换
        points_cam = (self.T @ points_3D_homo.T).T[:, :3]
        
        # 过滤掉相机后方的点
        valid_mask = points_cam[:, 2] > 0
        points_cam_valid = points_cam[valid_mask]
        
        if len(points_cam_valid) == 0:
            return np.array([]), valid_mask
        
        # 投影到图像平面
        points_2D_homo = (self.K @ points_cam_valid.T).T
        points_2D = points_2D_homo[:, :2] / points_2D_homo[:, 2:3]
        
        return points_2D, valid_mask
    
    def convert_3D_to_camera_coords(self, pts_3d_lidar):
        """
        将3D LiDAR点转换为相机坐标系
        Input: 3D Points in LiDAR Coordinates (N x 3)
        Output: 3D Points in Camera Coordinates (N x 3)
        """
        if len(pts_3d_lidar) == 0:
            return np.array([])
        
        # 确保输入是numpy数组
        pts_3d_lidar = np.asarray(pts_3d_lidar)
        
        # 检查输入形状
        if pts_3d_lidar.ndim != 2 or pts_3d_lidar.shape[1] != 3:
            print(f"警告: 输入点云形状不正确: {pts_3d_lidar.shape}")
            return np.array([])
        
        # 检查是否有无效值
        if np.any(np.isnan(pts_3d_lidar)) or np.any(np.isinf(pts_3d_lidar)):
            print("警告: 点云数据包含NaN或无穷大值，正在过滤...")
            valid_mask = np.isfinite(pts_3d_lidar).all(axis=1)
            pts_3d_lidar = pts_3d_lidar[valid_mask]
            if len(pts_3d_lidar) == 0:
                return np.array([])
        
        try:
            # 转换为齐次坐标
            pts_3d_homo = np.hstack((pts_3d_lidar, np.ones((pts_3d_lidar.shape[0], 1))))
            
            # 应用变换矩阵
            pts_cam_homo = (self.T @ pts_3d_homo.T).T
            
            # 返回3D相机坐标
            return pts_cam_homo[:, :3]
        except Exception as e:
            print(f"坐标转换时出错: {e}")
            return np.array([])
    
    def project_3D_to_2D(self, pts_3d_camera):
        """
        将相机坐标系下的3D点投影到2D图像
        Input: 3D Points in Camera Coordinates (N x 3)
        Output: 2D Pixels in Image Coordinates (N x 2)
        """
        if len(pts_3d_camera) == 0:
            return np.array([])
        
        # 投影到图像平面
        pts_2d_homo = (self.K @ pts_3d_camera.T).T
        
        # 归一化得到像素坐标
        pts_2d = pts_2d_homo[:, :2] / pts_2d_homo[:, 2:3]
        
        return pts_2d


def load_v2x_frame_data(data_root, frame_info):
    """加载V2X数据集的单帧数据"""
    
    # 构建文件路径
    image_path = os.path.join(data_root, frame_info['image_path'])
    pointcloud_path = os.path.join(data_root, frame_info['pointcloud_path'])
    
    # 标定文件路径 - 适配实际的字段名
    if 'calib_camera_intrinsic_path' in frame_info:
        camera_intrinsic_path = os.path.join(data_root, frame_info['calib_camera_intrinsic_path'])
    else:
        # 备用字段名
        camera_intrinsic_path = os.path.join(data_root, frame_info.get('camera_intrinsic_path', ''))
    
    if 'calib_virtuallidar_to_camera_path' in frame_info:
        lidar_to_camera_path = os.path.join(data_root, frame_info['calib_virtuallidar_to_camera_path'])
    else:
        # 备用字段名
        lidar_to_camera_path = os.path.join(data_root, frame_info.get('virtuallidar_to_camera_path', ''))
    
    # 检查文件是否存在
    for path, name in [(image_path, '图像'), (pointcloud_path, '点云'), 
                       (camera_intrinsic_path, '相机内参'), (lidar_to_camera_path, 'LiDAR外参')]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name}文件不存在: {path}")
    
    # 加载图像
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像文件: {image_path}")
    
    # 加载点云
    try:
        pcd = o3d.io.read_point_cloud(pointcloud_path)
        points = np.asarray(pcd.points)
        if len(points) == 0:
            raise ValueError("点云文件为空")
    except Exception as e:
        raise ValueError(f"无法读取点云文件 {pointcloud_path}: {e}")
    
    # 创建标定对象
    calibration = V2XCalibration(camera_intrinsic_path, lidar_to_camera_path)
    
    return image, points, calibration


def process_v2x_frame(frame_id, data_root, data_info_path, detector, erosion_factor, depth_factor, output_path=None):
    """处理V2X数据集的单个帧"""
    
    # 加载数据信息
    with open(data_info_path, 'r') as f:
        data_info = json.load(f)
    
    # 查找指定帧ID的信息
    frame_info = None
    for info in data_info:
        if info['frame_id'] == frame_id:
            frame_info = info
            break
    
    if frame_info is None:
        raise ValueError(f"找不到帧ID: {frame_id}")
    
    print(f"处理帧 {frame_id}...")
    
    # 加载帧数据
    image, points, calibration = load_v2x_frame_data(data_root, frame_info)
    
    print(f"图像尺寸: {image.shape}")
    print(f"点云点数: {len(points)}")
    
    # 运行检测
    objects3d_data, all_corners_3D, pts_3D, pts_2D, all_filtered_points_of_object, all_object_IDs = detector.process_frame(
        image, points, calibration, erosion_factor, depth_factor
    )
    
    print(f"检测到 {len(objects3d_data)} 个3D目标")
    
    # 绘制预测边界框
    for j, pred_corner_3D in enumerate(all_corners_3D):
        object_id = all_object_IDs[j] if j < len(all_object_IDs) else None
        plot_projected_pred_bounding_boxes(calibration, image, pred_corner_3D, (0, 0, 255), object_id)
    
    # 绘制投影的3D点
    if len(all_filtered_points_of_object) > 0:
        all_filtered_points_combined = np.vstack(all_filtered_points_of_object)
        draw_projected_3D_points(calibration, image, pts_3D, pts_2D, all_filtered_points_combined)
    
    # 创建BEV表示
    bev = create_BEV(all_filtered_points_of_object, all_corners_3D)
    
    # 保存或显示结果
    if output_path:
        os.makedirs(output_path, exist_ok=True)
        output_file = os.path.join(output_path, f"result_{frame_id}.png")
        create_combined_image(image, bev, output_path=output_file)
        print(f"结果已保存到: {output_file}")
    else:
        # 显示结果
        combined_img = create_combined_image(image, bev, output_path=None)
        cv2.imshow(f"V2X检测结果 - 帧 {frame_id}", combined_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return objects3d_data


def process_random_frames(data_root, data_info_path, detector, erosion_factor, depth_factor, frame_amount=5, output_path=None):
    """随机处理多个帧"""
    
    # 加载数据信息
    with open(data_info_path, 'r') as f:
        data_info = json.load(f)
    
    # 随机选择帧
    total_frames = len(data_info)
    selected_indices = [randint(0, total_frames-1) for _ in range(min(frame_amount, total_frames))]
    
    print(f"从 {total_frames} 个帧中随机选择 {len(selected_indices)} 个进行处理...")
    
    all_results = []
    for i, idx in enumerate(selected_indices):
        frame_info = data_info[idx]
        frame_id = frame_info['frame_id']
        
        try:
            print(f"\n处理第 {i+1}/{len(selected_indices)} 个帧...")
            result = process_v2x_frame(frame_id, data_root, data_info_path, detector, erosion_factor, depth_factor, output_path)
            all_results.append((frame_id, result))
        except Exception as e:
            print(f"处理帧 {frame_id} 时出错: {e}")
            continue
    
    print(f"\n成功处理了 {len(all_results)} 个帧")
    return all_results


def main():
    parser = argparse.ArgumentParser(description='V2X数据集YOLO-LiDAR融合3D目标检测')
    parser.add_argument('frame_id', help='要处理的帧ID，或使用 "random" 处理随机帧')
    parser.add_argument('--data-root', default='data/V2X-Seq-SPD-Example/infrastructure-side/', 
                       help='V2X数据集根目录路径')
    parser.add_argument('--data-info', default='data/V2X-Seq-SPD-Example/infrastructure-side/data_info.json',
                       help='数据信息JSON文件路径')
    parser.add_argument('--mode', choices=['detect', 'track'], default='detect',
                       help='检测模式: detect(检测) 或 track(跟踪)')
    parser.add_argument('--model-size', choices=['n', 's', 'm', 'l', 'x'], default='m',
                       help='YOLOv8模型大小')
    parser.add_argument('--erosion', type=int, default=25, help='腐蚀因子')
    parser.add_argument('--depth', type=int, default=20, help='深度过滤因子')
    parser.add_argument('--pca', type=bool, default=False, help='是否使用PCA创建3D边界框')
    parser.add_argument('--output-path', help='输出路径，为空则显示结果')
    parser.add_argument('--frame-amount', type=int, default=5, help='随机处理的帧数量')
    
    args = parser.parse_args()
    
    # 检查数据路径
    if not os.path.exists(args.data_root):
        print(f"错误: 数据根目录不存在: {args.data_root}")
        return
    
    if not os.path.exists(args.data_info):
        print(f"错误: 数据信息文件不存在: {args.data_info}")
        return
    
    # 初始化检测器
    model_path = f"yolov8{args.model_size}-seg.pt"
    tracking = (args.mode == 'track')
    
    print(f"初始化YOLOv8检测器 (模型: {model_path}, 跟踪: {tracking})...")
    detector = YOLOv8Detector(model_path, tracking=tracking, PCA=args.pca)
    
    try:
        if args.frame_id.lower() == 'random':
            # 处理随机帧
            process_random_frames(
                args.data_root, args.data_info, detector, 
                args.erosion, args.depth, args.frame_amount, args.output_path
            )
        else:
            # 处理指定帧
            process_v2x_frame(
                args.frame_id, args.data_root, args.data_info, detector,
                args.erosion, args.depth, args.output_path
            )
    except Exception as e:
        print(f"处理过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 