import numpy as np
from ultralytics import YOLO
from fusion import *
from utils import *
import open3d as o3d


class YOLOv8Detector:
    def __init__(self, model_path, tracking=False, PCA=False):
        self.model = YOLO(model_path)
        self.tracking = tracking
        self.pca = PCA
        self.last_ground_center_of_id = {}    

    def process_frame(self, frame, pts, lidar2camera, erosion_factor, depth_factor):
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
            
            point_cloud = np.asarray(points)
            
            # 检查点云数据的有效性
            if len(point_cloud) == 0:
                print("警告: 点云为空")
                return [], [], np.array([]), np.array([]), [], []
            
            # 检查点云数据是否包含无效值
            if np.any(np.isnan(point_cloud)) or np.any(np.isinf(point_cloud)):
                print("警告: 点云包含无效值，正在过滤...")
                valid_mask = np.isfinite(point_cloud).all(axis=1)
                point_cloud = point_cloud[valid_mask]
                print(f"过滤后剩余 {len(point_cloud)} 个有效点")
            
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
                    print(f"投影后得到 {len(pts_3D)} 个有效的3D点")
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
        
        for j, cls in enumerate(boxes.cls.tolist()):
            try:
                conf = boxes.conf.tolist()[j] if boxes.conf is not None else None
                box_id = int(boxes.id.tolist()[j]) if boxes.id is not None else None

                all_object_IDs.append(box_id)

                # Check if the mask is empty before processing
                if masks is None or j >= len(masks.xy) or masks.xy[j].size == 0:
                    print(f"目标 {j} 的掩码为空，跳过")
                    continue

                # Pass the segmentation mask to the fusion function
                fusion_result = lidar_camera_fusion(pts_3D, pts_2D, frame, masks.xy[j], int(cls), lidar2camera, erosion_factor=erosion_factor, depth_factor=depth_factor, PCA=self.pca)

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
                        ROS_direction = None
                        ROS_velocity = None

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