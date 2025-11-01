import numpy as np
from ultralytics import YOLO
from fusion import *
from improved_fusion import improved_lidar_camera_fusion
from  ..Code.utils import *
import open3d as o3d
import time
import os
folder_name = "yolo"
os.makedirs(folder_name,exist_ok = True)


class Voxelization:
    """
    点云Voxel化处理类
    使用Open3D进行体素降采样，减少点云数量以提高处理效率
    """
    
    def __init__(self, voxel_size=(0.5, 0.5, 0.5), enable=True):
        """
        初始化Voxelization类
        
        Args:
            voxel_size: 体素大小，格式为(x, y, z)或单个数值，单位：米
            enable: 是否启用voxel化，默认True
        """
        # 保存原始voxel_size（用于set_voxel_size等方法）
        self.voxel_size = voxel_size
        
        # 在初始化时直接转换为单个数值（Open3D需要单个值）
        if isinstance(voxel_size, (tuple, list)):
            # 如果是元组或列表，计算平均值
            self.avg_voxel_size = sum(voxel_size) / len(voxel_size)
        else:
            # 如果已经是单个数值，直接使用
            self.avg_voxel_size = float(voxel_size)
        
        self.enable = enable
        self.pcd = o3d.geometry.PointCloud()
        print(f"Voxelization初始化: voxel_size={voxel_size}, avg_voxel_size={self.avg_voxel_size:.3f}, enable={enable}")
    
    def downsample(self, point_cloud):
        """
        对点云进行voxel降采样
        
        Args:
            point_cloud: numpy数组，形状为(N, 3)的3D点云
            
        Returns:
            downsampled_points: numpy数组，降采样后的点云，形状为(M, 3)，M <= N
        """
        if not self.enable:
            return point_cloud
        
        if point_cloud is None or len(point_cloud) == 0:
            return point_cloud
        
        try:
            # 检查点云维度
            if point_cloud.ndim != 2 or point_cloud.shape[1] != 3:
                print(f"警告: 点云维度不正确，shape={point_cloud.shape}，跳过voxel化")
                return point_cloud
            
            # 检查点云数量，如果点太少则不需要降采样
            if len(point_cloud) < 1000:
                return point_cloud
            
            # 创建Open3D点云对象
            self.pcd.points = o3d.utility.Vector3dVector(point_cloud)
            
            # 直接使用初始化时计算好的avg_voxel_size
            # 执行voxel降采样
            downsampled_pcd = self.pcd.voxel_down_sample(voxel_size=self.avg_voxel_size)
            
            # 转换回numpy数组
            downsampled_points = np.asarray(downsampled_pcd.points)
            
            # 打印统计信息
            original_count = len(point_cloud)
            downsampled_count = len(downsampled_points)
            reduction_ratio = (1 - downsampled_count / original_count) * 100
            print(f"Voxel降采样: {original_count} -> {downsampled_count} 点 "
                  f"(减少 {reduction_ratio:.1f}%)")
            
            return downsampled_points
            
        except Exception as e:
            print(f"Voxel降采样失败: {e}，返回原始点云")
            return point_cloud
    
    def set_enable(self, enable):
        """
        设置是否启用voxel化
        
        Args:
            enable: bool，True启用，False禁用
        """
        self.enable = enable
        print(f"Voxelization enable设置为: {enable}")
    
    def set_voxel_size(self, voxel_size):
        """
        设置体素大小
        
        Args:
            voxel_size: 体素大小，格式为(x, y, z)或单个数值，单位：米
        """
        self.voxel_size = voxel_size
        
        # 同时更新avg_voxel_size
        if isinstance(voxel_size, (tuple, list)):
            self.avg_voxel_size = sum(voxel_size) / len(voxel_size)
        else:
            self.avg_voxel_size = float(voxel_size)
        
        print(f"Voxelization voxel_size设置为: {voxel_size}, avg_voxel_size={self.avg_voxel_size:.3f}")


class YOLOv8Detector:
    def __init__(self, model_path, tracking=False, PCA=False, use_improved_fusion=True):
        self.model = YOLO("/home/nebula/RGB-LIDAR-fusion/Code/yolov8m-seg.engine")
        self.model.overrides['conf'] = 0.5
        self.model.overrides['iou'] = 0.5
        self.model.overrides['agnostic_nms'] = False
        self.model.overrides['max_det'] = 1000
        self.tracking = tracking
        self.pca = PCA
        self.use_improved_fusion = use_improved_fusion
        self.last_ground_center_of_id = {}
        self.use_boxes = True
        
        # 初始化Voxelization类，voxel大小为(0.5, 0.5, 0.5)
        self.voxelizer = Voxelization(voxel_size=(0.2, 0.2, 0.2), enable=True)    

    def process_frame(self, frame, pts, lidar2camera, erosion_factor, depth_factor):
        #frame size (1920, 1080)->(960, 540)
        start_time = time.time()
        if self.tracking: #TODO 重新训一个更小size的yolo模型
            results = self.model.track(
                source=frame,
                classes=[0, 1, 2, 3, 5, 6, 7],
                verbose=False,
                show=False,
                persist=True,
                tracker='bytetrack.yaml',
                conf = 0.5
            )
        else:
            results = self.model.predict(
                source=frame,
                classes=[0, 1, 2, 3, 5, 6, 7],
                verbose=False,
                show=False,
                save = True,
                conf = 0.5
            )
        end_time = time.time()
        # 打印这个for用时ms级
        print(f"处理帧用时 yolov8: {(end_time - start_time)*1000:.2f}ms")
        # Get the results from the YOLOv8-seg model
        r = results[0]
        boxes = r.boxes  # Boxes object for bbox outputs
        if len(boxes) != 0:
            print(boxes.id)
            print(boxes.cls)
        masks = r.masks  # Masks object for segment masks outputs

        # Preprocess LiDAR point cloud - 支持不同格式
        try:
            if isinstance(pts, str):
                if pts.endswith('.pcd'):
                    # 加载PCD文件 (V2X数据集)
                    pcd = o3d.io.read_point_cloud(pts)
                    points = np.asarray(pcd.points)
                    print(f"从PCD文件加载了 {len(points)} 个点")
                else:
                    # 加载二进制文件 (KITTI数据集)
                    points = np.fromfile(pts, dtype=np.float32).reshape((-1, 4))[:, 0:3]
                    print(f"从二进制文件加载了 {len(points)} 个点")
            else:
                # 直接传入点云数组
                points = pts
                print(f"直接使用点云数组，包含 {len(points)} 个点")
            
            # 先转换为numpy数组
            point_cloud = np.asarray(points)
            
            # 检查点云数据的有效性
            if len(point_cloud) == 0:
                print("警告: 点云为空")
                return [], [], np.array([]), np.array([]), [], []
            
            # 检查点云数据是否包含无效值（必须在voxel化之前过滤）
            if np.any(np.isnan(point_cloud)) or np.any(np.isinf(point_cloud)):
                print("警告: 点云包含无效值，正在过滤...")
                valid_mask = np.isfinite(point_cloud).all(axis=1)
                point_cloud = point_cloud[valid_mask]
                print(f"过滤后剩余 {len(point_cloud)} 个有效点")
            
            # 在过滤无效值之后进行voxel降采样（提高效率且避免Open3D处理无效值）
            point_cloud = self.voxelizer.downsample(point_cloud)
            
        except Exception as e:
            print(f"加载点云时出错: {e}")
            return [], [], np.array([]), np.array([]), [], []
        
        # 使用适当的过滤函数
        try:
            if hasattr(lidar2camera, 'convert_3D_to_2D'):
                # V2X标定类
                pts_2D, valid_mask = lidar2camera.convert_3D_to_2D(point_cloud)
                if len(pts_2D) > 0:
                    # 过滤图像边界内的点
                    img_width, img_height = frame.shape[1], frame.shape[0]
                    valid_2d_mask = (
                        (pts_2D[:, 0] >= 0) & (pts_2D[:, 0] < img_width) &
                        (pts_2D[:, 1] >= 0) & (pts_2D[:, 1] < img_height)
                    )
                    valid_indices = np.where(valid_mask)[0][valid_2d_mask]
                    pts_3D = point_cloud[valid_indices]
                    pts_2D = pts_2D[valid_2d_mask]
                    # 移除调试输出：投影统计
                    # print(f"投影后得到 {len(pts_3D)} 个有效的3D点")
                else:
                    pts_3D, pts_2D = np.array([]), np.array([])
                    print("投影后没有有效点")
            else:
                # KITTI标定类
                pts_3D, pts_2D = filter_lidar_points(lidar2camera, point_cloud, (frame.shape[1], frame.shape[0]))
                print(f"KITTI过滤后得到 {len(pts_3D)} 个有效的3D点")
        except Exception as e:
            print(f"点云投影时出错: {e}")
            return [], [], np.array([]), np.array([]), [], []

        # For each object detected by the YOLOv8 model, fuse and process it
        all_corners_3D = []
        all_filtered_points_of_object = []
        all_object_IDs = []
        objects3d_data = []
        
        if boxes is None or len(boxes) == 0:
            print("没有检测到任何目标")
            return objects3d_data, all_corners_3D, pts_3D, pts_2D, all_filtered_points_of_object, all_object_IDs
        
        # 打印这个for用时
        
        start_time = time.time()

        for j, cls in enumerate(boxes.cls.tolist()):
            try:
                conf = boxes.conf.tolist()[j] if boxes.conf is not None else None
                box_id = int(boxes.id.tolist()[j]) if boxes.id is not None else None

                all_object_IDs.append(box_id)

                # 根据use_boxes参数选择输入类型
                if self.use_improved_fusion:
                    if self.use_boxes:
                        # 使用边界框模式（快速模式）
                        # 获取边界框坐标 [x1, y1, x2, y2]
                        box_coords = boxes.xyxy[j]
                        if hasattr(box_coords, 'cpu'):
                            box_coords = box_coords.cpu().numpy()
                        elif hasattr(box_coords, 'numpy'):
                            box_coords = box_coords.numpy()
                        else:
                            box_coords = np.array(box_coords)
                        
                        box = [float(box_coords[0]), float(box_coords[1]), 
                               float(box_coords[2]), float(box_coords[3])]
                        
                        fusion_result = improved_lidar_camera_fusion(
                            pts_3D, pts_2D, frame, box, int(cls), lidar2camera,
                            erosion_factor=erosion_factor, depth_factor=depth_factor,
                            PCA=self.pca, use_boxes=True
                        )
                    else:
                        # 使用多边形掩码模式（精确模式）
                        # Check if the mask is empty before processing
                        if masks is None or j >= len(masks.xy) or masks.xy[j].size == 0:
                            print(f"目标 {j} 的掩码为空，跳过")
                            continue
                        
                        fusion_result = improved_lidar_camera_fusion(
                            pts_3D, pts_2D, frame, masks.xy[j], int(cls), lidar2camera,
                            erosion_factor=erosion_factor, depth_factor=depth_factor,
                            PCA=self.pca, use_boxes=False
                        )
                else:
                    # 使用原始fusion函数
                    # Check if the mask is empty before processing
                    if masks is None or j >= len(masks.xy) or masks.xy[j].size == 0:
                        print(f"目标 {j} 的掩码为空，跳过")
                        continue
                    
                    fusion_result = lidar_camera_fusion(
                        pts_3D, pts_2D, frame, masks.xy[j], int(cls), lidar2camera,
                        erosion_factor=erosion_factor, depth_factor=depth_factor, PCA=self.pca
                    )

                # If the fusion is successfull, retrieve relevant bbox data (e.g. for RoboCar)
                if fusion_result is not None:
                    filtered_points_of_object, corners_3D, yaw = fusion_result

                    all_corners_3D.append(corners_3D)
                    all_filtered_points_of_object.append(filtered_points_of_object)

                    # Retrieve the ROS data (e.g. relevant for RoboCar)
                    ROS_type = int(np.int32(cls))
                    bottom_indices = np.argsort(corners_3D[:, 2])[:4]
                    ROS_ground_center = np.mean(corners_3D[bottom_indices], axis=0)
                    ROS_dimensions = np.ptp(corners_3D, axis=0)                
                    ROS_points = corners_3D
                    time_between_frames = 0.1

                    # Compute the velocity and direction (only available with tracking)
                    if box_id in self.last_ground_center_of_id and not np.array_equal(self.last_ground_center_of_id[box_id], ROS_ground_center):
                        ROS_direction, ROS_velocity = compute_relative_object_velocity(self.last_ground_center_of_id[box_id], ROS_ground_center, time_between_frames)
                    else:
                        # 为第一帧或静止目标提供默认值
                        ROS_direction = np.array([0.0, 0.0, 0.0])  # 默认朝向
                        ROS_velocity = np.array([0.0, 0.0, 0.0])   # 静止状态

                    self.last_ground_center_of_id[box_id] = ROS_ground_center

                    # Save the ROS information of the current object and append it to an array that contains all information of all objects in the frame
                    if ROS_type is not None and ROS_ground_center is not None and ROS_direction is not None and ROS_dimensions is not None and ROS_velocity is not None and ROS_points is not None:
                        objects3d_data.append([ROS_type, ROS_ground_center, ROS_direction, ROS_dimensions, ROS_velocity, ROS_points])
                else:
                    print(f"目标 {j} 的融合失败")
            except Exception as e:
                print(f"处理目标 {j} 时出错: {e}")
                continue
        
        print(f"成功处理了 {len(all_corners_3D)} 个目标")
        end_time = time.time()
        # 打印这个for用时ms级
        print(f"处理帧用时 for循环融合: {(end_time - start_time)*1000:.2f}ms")
        return objects3d_data, all_corners_3D, pts_3D, pts_2D, all_filtered_points_of_object, all_object_IDs
    
    def get_IoU_results(self, frame, pts, lidar2camera, erosion_factor, depth_factor):
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

        # Get the results from the YOLOv8-seg model
        r = results[0]
        boxes = r.boxes  # Boxes object for bbox outputs
        masks = r.masks  # Masks object for segment masks outputs

        # Preprocess LiDAR point cloud - 支持不同格式
        if isinstance(pts, str):
            if pts.endswith('.pcd'):
                # 加载PCD文件 (V2X数据集)
                pcd = o3d.io.read_point_cloud(pts)
                points = np.asarray(pcd.points)
            else:
                # 加载二进制文件 (KITTI数据集)
                points = np.fromfile(pts, dtype=np.float32).reshape((-1, 4))[:, 0:3]
        else:
            # 直接传入点云数组
            points = pts
        
        point_cloud = np.asarray(points)
        
        # 使用适当的过滤函数
        if hasattr(lidar2camera, 'convert_3D_to_2D'):
            # V2X标定类
            pts_2D, valid_mask = lidar2camera.convert_3D_to_2D(point_cloud)
            if len(pts_2D) > 0:
                # 过滤图像边界内的点
                img_width, img_height = frame.shape[1], frame.shape[0]
                valid_2d_mask = (
                    (pts_2D[:, 0] >= 0) & (pts_2D[:, 0] < img_width) &
                    (pts_2D[:, 1] >= 0) & (pts_2D[:, 1] < img_height)
                )
                valid_indices = np.where(valid_mask)[0][valid_2d_mask]
                pts_3D = point_cloud[valid_indices]
                pts_2D = pts_2D[valid_2d_mask]
            else:
                pts_3D, pts_2D = np.array([]), np.array([])
        else:
            # KITTI标定类
            pts_3D, pts_2D = filter_lidar_points(lidar2camera, point_cloud, (frame.shape[1], frame.shape[0]))

        # For each object detected by the YOLOv8 model, fuse and process it
        all_corners_3D = []
        all_filtered_points_of_object = []
        objects3d_data = []
        for j, cls in enumerate(boxes.cls.tolist()):
            conf = boxes.conf.tolist()[j] if boxes.conf is not None else None
            box_id = int(boxes.id.tolist()[j]) if boxes.id is not None else None

            # Check if the mask is empty before processing
            if masks.xy[j].size == 0:
                continue

            # Pass the segmentation mask to the fusion function
            fusion_result = lidar_camera_fusion(pts_3D, pts_2D, frame, masks.xy[j], int(cls), lidar2camera, erosion_factor=erosion_factor, depth_factor=depth_factor, PCA=self.pca)

            # If the fusion is successfull, retrieve the relevant data for the IoU computation with KITTI GT boxes
            if fusion_result is not None:
                filtered_points_of_object, corners_3D, yaw = fusion_result

                all_corners_3D.append(corners_3D)
                all_filtered_points_of_object.append(filtered_points_of_object)

                if cls == 0:
                    type = "Pedestrian"
                elif cls == 1:
                    type = "Cyclist"
                elif cls == 2:
                    type = "Car"
                else:
                    type = "DontCare"

                # Ground Center is the center of the bottom bbox side, thus of the 4 corners with the lowest z value (in LiDAR coordinates) 
                bottom_indices = np.argsort(corners_3D[:, 2])[:4]
                ground_center = np.mean(corners_3D[bottom_indices], axis=0)

                # Get the bbox dimensions in l, w, h format
                dimensions = np.ptp(corners_3D, axis=0)

                # Append relevant information to array that is later returned
                objects3d_data.append([type, ground_center, dimensions, yaw])

        return objects3d_data, all_corners_3D, pts_3D, pts_2D, all_filtered_points_of_object
def compute_relative_object_velocity(ground_center_frame1, ground_center_frame2, time_between_frames):
    # Compute displacement vector between ground centers
    displacement = ground_center_frame2 - ground_center_frame1

    # Compute relative velocity components
    relative_velocity_x = displacement[0] / time_between_frames
    relative_velocity_y = displacement[1] / time_between_frames
    relative_velocity_z = displacement[2] / time_between_frames

    # Return velocity vector
    return displacement, np.array([relative_velocity_x, relative_velocity_y, relative_velocity_z])