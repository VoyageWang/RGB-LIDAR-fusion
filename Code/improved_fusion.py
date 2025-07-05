#!/usr/bin/env python3
"""
改进的LiDAR-相机融合模块
解决融合失败和近距离物体3D框过大的问题
"""

import numpy as np
import cv2
from sklearn.cluster import DBSCAN
import open3d as o3d
from calibration import *
from data_processing import *


def improved_lidar_camera_fusion(pts_3D, pts_2D, frame, seg_mask, obj_class, lidar2cam, erosion_factor=25, depth_factor=20, PCA=False):
    """
    改进的LiDAR-相机融合函数
    解决融合失败和3D框过大的问题
    """
    
    # 1. 改进的多边形腐蚀
    eroded_seg = improved_erode_polygon(seg_mask, frame, erosion_factor, obj_class)
    
    if eroded_seg is None or len(eroded_seg) <= 2:
        print(f"  融合失败: 腐蚀后多边形太小 (类别: {obj_class})")
        return None
    
    # 2. 改进的点云筛选
    all_points_of_object = extract_points_in_polygon(pts_3D, pts_2D, eroded_seg, frame)
    
    if len(all_points_of_object) == 0:
        print(f"  融合失败: 多边形内无点云 (类别: {obj_class})")
        return None
    
    # 3. 改进的深度聚类过滤
    filtered_points_of_object = improved_depth_clustering(all_points_of_object, depth_factor, obj_class)
    
    if len(filtered_points_of_object) < 4:
        print(f"  融合失败: 过滤后点数不足 ({len(filtered_points_of_object)} < 4, 类别: {obj_class})")
        return None
    
    if is_coplanar(filtered_points_of_object):
        print(f"  融合失败: 点云共面 (类别: {obj_class})")
        return None
    
    # 4. 改进的3D边界框创建
    bbox_corners_3D, yaw = create_improved_bbox_3D(filtered_points_of_object, lidar2cam, obj_class, PCA)
    
    if bbox_corners_3D is None:
        print(f"  融合失败: 无法创建3D边界框 (类别: {obj_class})")
        return None
    
    # 5. 边界框合理性检查
    if not is_bbox_reasonable(bbox_corners_3D, obj_class):
        print(f"  融合失败: 3D边界框不合理 (类别: {obj_class})")
        return None
    
    return filtered_points_of_object, bbox_corners_3D, yaw


def improved_erode_polygon(polygon, img, erosion_factor=25, obj_class=0):
    """
    改进的多边形腐蚀，根据目标类别调整腐蚀参数
    """
    if len(polygon) == 0:
        return None
    
    # 根据目标类别调整腐蚀因子
    class_erosion_factors = {
        0: erosion_factor * 0.8,  # 行人：较小腐蚀
        1: erosion_factor * 0.9,  # 骑行者：较小腐蚀
        2: erosion_factor * 1.0,  # 汽车：标准腐蚀
        3: erosion_factor * 0.9,  # 摩托车：较小腐蚀
        5: erosion_factor * 1.2,  # 公交车：较大腐蚀
        6: erosion_factor * 1.2,  # 卡车：较大腐蚀
        7: erosion_factor * 1.5   # 火车：最大腐蚀
    }
    
    adjusted_erosion_factor = class_erosion_factors.get(obj_class, erosion_factor)
    
    # 计算多边形面积
    polygon_area = cv2.contourArea(polygon.astype(np.int32))
    
    # 如果多边形太小，减少腐蚀
    if polygon_area < 1000:  # 小目标
        adjusted_erosion_factor *= 0.5
    elif polygon_area < 5000:  # 中等目标
        adjusted_erosion_factor *= 0.7
    
    # 计算缩放腐蚀因子
    scaled_erosion_factor = np.sqrt(polygon_area) / adjusted_erosion_factor
    
    # 确保最小腐蚀核大小
    kernel_size = max(3, int(2 * scaled_erosion_factor + 1))
    
    # 创建多边形掩码
    mask = np.zeros_like(img[:, :, 0], dtype=np.uint8)
    cv2.fillPoly(mask, [polygon.astype(np.int32)], color=255)
    
    # 应用腐蚀
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    eroded_mask = cv2.erode(mask, kernel, iterations=1)
    
    # 查找轮廓
    contours, _ = cv2.findContours(eroded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return None
    
    # 获取最大轮廓
    largest_contour = max(contours, key=cv2.contourArea)
    
    # 检查腐蚀后面积是否合理
    eroded_area = cv2.contourArea(largest_contour)
    if eroded_area < polygon_area * 0.1:  # 腐蚀后面积太小
        return None
    
    eroded_polygon = np.squeeze(largest_contour).astype(np.float32)
    
    return eroded_polygon


def extract_points_in_polygon(pts_3D, pts_2D, polygon, frame):
    """
    提取多边形内的点云，添加边界检查
    """
    # 创建多边形掩码
    mask = np.zeros_like(frame[:, :, 0], dtype=np.uint8)
    cv2.fillPoly(mask, [polygon.astype(np.int32)], color=1)
    
    # 转换2D点为整数坐标
    pts_2D_int = pts_2D.astype(np.int32)
    
    # 边界检查
    h, w = frame.shape[:2]
    valid_indices = (
        (pts_2D_int[:, 0] >= 0) & (pts_2D_int[:, 0] < w) &
        (pts_2D_int[:, 1] >= 0) & (pts_2D_int[:, 1] < h)
    )
    
    if not np.any(valid_indices):
        return np.array([])
    
    # 筛选有效点
    valid_pts_2D = pts_2D_int[valid_indices]
    valid_pts_3D = pts_3D[valid_indices]
    
    # 使用掩码筛选多边形内的点
    inside_mask_indices = mask[valid_pts_2D[:, 1], valid_pts_2D[:, 0]] == 1
    
    return valid_pts_3D[inside_mask_indices]


def improved_depth_clustering(points_of_object, depth_factor=20, obj_class=0):
    """
    改进的深度聚类，根据目标类别调整参数
    """
    if len(points_of_object) < 4:
        return points_of_object
    
    # 根据目标类别调整聚类参数
    class_params = {
        0: {'eps': 0.3, 'min_samples_ratio': 0.05},  # 行人：更紧密聚类
        1: {'eps': 0.4, 'min_samples_ratio': 0.04},  # 骑行者
        2: {'eps': 0.5, 'min_samples_ratio': 0.03},  # 汽车：标准参数
        3: {'eps': 0.4, 'min_samples_ratio': 0.04},  # 摩托车
        5: {'eps': 0.6, 'min_samples_ratio': 0.02},  # 公交车：更宽松聚类
        6: {'eps': 0.6, 'min_samples_ratio': 0.02},  # 卡车
        7: {'eps': 0.8, 'min_samples_ratio': 0.01}   # 火车：最宽松聚类
    }
    
    params = class_params.get(obj_class, {'eps': 0.5, 'min_samples_ratio': 0.03})
    
    # 计算最小样本数
    min_samples = max(3, int(len(points_of_object) * params['min_samples_ratio']))
    
    # 计算深度值
    depths = np.sqrt(points_of_object[:, 0]**2 + points_of_object[:, 1]**2)
    
    # DBSCAN聚类
    dbscan = DBSCAN(eps=params['eps'], min_samples=min_samples)
    clusters = dbscan.fit_predict(depths.reshape(-1, 1))
    
    # 获取最大聚类
    unique_clusters, cluster_counts = np.unique(clusters[clusters != -1], return_counts=True)
    
    if len(unique_clusters) == 0:
        return points_of_object  # 如果没有有效聚类，返回原始点
    
    # 选择点数最多的聚类
    largest_cluster = unique_clusters[np.argmax(cluster_counts)]
    cluster_mask = clusters == largest_cluster
    
    # 获取聚类内的点
    cluster_points = points_of_object[cluster_mask]
    cluster_depths = depths[cluster_mask]
    
    # 改进的深度过滤
    depth_median = np.median(cluster_depths)
    depth_std = np.std(cluster_depths)
    
    # 根据目标类别调整深度范围
    depth_tolerance = {
        0: 1.5,  # 行人：较小深度范围
        1: 2.0,  # 骑行者
        2: 2.5,  # 汽车：标准深度范围
        3: 2.0,  # 摩托车
        5: 3.0,  # 公交车：较大深度范围
        6: 3.0,  # 卡车
        7: 4.0   # 火车：最大深度范围
    }.get(obj_class, 2.5)
    
    # 深度过滤
    depth_range = depth_tolerance * depth_std
    min_depth = depth_median - depth_range
    max_depth = depth_median + depth_range
    
    depth_filter_mask = (cluster_depths >= min_depth) & (cluster_depths <= max_depth)
    
    return cluster_points[depth_filter_mask]


def create_improved_bbox_3D(pts_3D, lidar2cam, obj_class=0, PCA=False):
    """
    创建改进的3D边界框，解决近距离物体框过大的问题
    """
    if len(pts_3D) < 4:
        return None, 0
    
    try:
        # 根据目标类别选择边界框创建方法
        if obj_class == 0:  # 行人总是使用PCA
            return create_constrained_bbox_3D_PCA(pts_3D, lidar2cam, obj_class)
        elif PCA:
            return create_constrained_bbox_3D_PCA_no_z_rotation(pts_3D, lidar2cam, obj_class)
        else:
            return create_constrained_bbox_3D(pts_3D, lidar2cam, obj_class)
    except Exception as e:
        print(f"  创建3D边界框失败: {e}")
        return None, 0


def create_constrained_bbox_3D(pts_3D, lidar2cam, obj_class=0):
    """
    创建约束的简单3D边界框
    """
    # 移除异常值
    pts_3D_filtered = remove_outliers(pts_3D, obj_class)
    
    if len(pts_3D_filtered) < 4:
        pts_3D_filtered = pts_3D
    
    # 计算边界框
    min_point = np.min(pts_3D_filtered, axis=0)
    max_point = np.max(pts_3D_filtered, axis=0)
    
    # 根据目标类别约束边界框尺寸
    dimensions = max_point - min_point
    constrained_dimensions = constrain_bbox_dimensions(dimensions, obj_class)
    
    # 重新计算边界框中心和尺寸
    center = (min_point + max_point) / 2
    half_dims = constrained_dimensions / 2
    
    new_min = center - half_dims
    new_max = center + half_dims
    
    # 定义边界框角点
    corners_3D = np.array([
        [new_min[0], new_min[1], new_min[2]],  # Corner 0
        [new_min[0], new_min[1], new_max[2]],  # Corner 1
        [new_min[0], new_max[1], new_min[2]],  # Corner 2
        [new_max[0], new_min[1], new_min[2]],  # Corner 3
        [new_max[0], new_max[1], new_max[2]],  # Corner 4
        [new_max[0], new_max[1], new_min[2]],  # Corner 5
        [new_max[0], new_min[1], new_max[2]],  # Corner 6
        [new_min[0], new_max[1], new_max[2]]   # Corner 7
    ])
    
    return corners_3D, 0


def create_constrained_bbox_3D_PCA(pts_3D, lidar2cam, obj_class=0):
    """
    创建约束的PCA 3D边界框
    """
    # 移除异常值
    pts_3D_filtered = remove_outliers(pts_3D, obj_class)
    
    if len(pts_3D_filtered) < 4:
        pts_3D_filtered = pts_3D
    
    # 转换到相机坐标系计算yaw
    pts_3D_camera_coord = lidar2cam.convert_3D_to_camera_coords(pts_3D_filtered)
    
    if len(pts_3D_camera_coord) == 0:
        return create_constrained_bbox_3D(pts_3D, lidar2cam, obj_class)
    
    # 创建点云并计算OBB
    point_cloud_cam = o3d.geometry.PointCloud()
    point_cloud_cam.points = o3d.utility.Vector3dVector(pts_3D_camera_coord)
    
    obb_cam = point_cloud_cam.get_oriented_bounding_box()
    R_camera_coord = np.array(obb_cam.R)
    yaw = np.arctan2(R_camera_coord[0, 0], R_camera_coord[2, 0])
    
    # 在LiDAR坐标系中创建OBB
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pts_3D_filtered)
    
    obb = point_cloud.get_oriented_bounding_box()
    
    # 约束边界框尺寸
    extent = np.array(obb.extent)
    constrained_extent = constrain_bbox_dimensions(extent, obj_class)
    
    # 创建新的OBB
    obb_constrained = o3d.geometry.OrientedBoundingBox(obb.center, obb.R, constrained_extent)
    corners_3D = np.asarray(obb_constrained.get_box_points())
    
    return corners_3D, yaw


def create_constrained_bbox_3D_PCA_no_z_rotation(pts_3D, lidar2cam, obj_class=0):
    """
    创建约束的PCA 3D边界框（无Z轴旋转）
    """
    # 移除异常值
    pts_3D_filtered = remove_outliers(pts_3D, obj_class)
    
    if len(pts_3D_filtered) < 4:
        pts_3D_filtered = pts_3D
    
    # 转换到相机坐标系计算yaw
    pts_3D_camera_coord = lidar2cam.convert_3D_to_camera_coords(pts_3D_filtered)
    
    if len(pts_3D_camera_coord) == 0:
        return create_constrained_bbox_3D(pts_3D, lidar2cam, obj_class)
    
    point_cloud_cam = o3d.geometry.PointCloud()
    point_cloud_cam.points = o3d.utility.Vector3dVector(pts_3D_camera_coord)
    
    obb_cam = point_cloud_cam.get_oriented_bounding_box()
    R_camera_coord = np.array(obb_cam.R)
    yaw = np.arctan2(R_camera_coord[0, 0], R_camera_coord[2, 0])
    
    # 在LiDAR坐标系中创建OBB
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pts_3D_filtered)
    
    obb = point_cloud.get_oriented_bounding_box()
    
    # 修改旋转矩阵以移除Z轴旋转
    R = np.array(obb.R)
    R[:, 0] = [R[0, 0], R[1, 0], 0]
    R[:, 1] = [R[0, 1], R[1, 1], 0]
    R[:, 2] = [0, 0, 1]
    
    # 约束边界框尺寸
    extent = np.array(obb.extent)
    constrained_extent = constrain_bbox_dimensions(extent, obj_class)
    
    # 创建新的OBB
    obb_constrained = o3d.geometry.OrientedBoundingBox(obb.center, R, constrained_extent)
    corners_3D = np.asarray(obb_constrained.get_box_points())
    
    return corners_3D, yaw


def remove_outliers(pts_3D, obj_class=0):
    """
    移除异常值点云
    """
    if len(pts_3D) < 10:
        return pts_3D
    
    # 计算距离
    distances = np.sqrt(np.sum(pts_3D**2, axis=1))
    
    # 使用IQR方法移除异常值
    Q1 = np.percentile(distances, 25)
    Q3 = np.percentile(distances, 75)
    IQR = Q3 - Q1
    
    # 根据目标类别调整异常值阈值
    outlier_factors = {
        0: 1.5,  # 行人：较严格
        1: 1.8,  # 骑行者
        2: 2.0,  # 汽车：标准
        3: 1.8,  # 摩托车
        5: 2.5,  # 公交车：较宽松
        6: 2.5,  # 卡车
        7: 3.0   # 火车：最宽松
    }.get(obj_class, 2.0)
    
    lower_bound = Q1 - outlier_factors * IQR
    upper_bound = Q3 + outlier_factors * IQR
    
    # 过滤异常值
    valid_mask = (distances >= lower_bound) & (distances <= upper_bound)
    
    return pts_3D[valid_mask]


def constrain_bbox_dimensions(dimensions, obj_class=0):
    """
    根据目标类别约束边界框尺寸
    """
    # 定义各类别的合理尺寸范围 (长, 宽, 高)
    size_constraints = {
        0: {'min': [0.3, 0.3, 1.0], 'max': [1.0, 1.0, 2.2]},  # 行人
        1: {'min': [0.5, 0.5, 1.0], 'max': [2.5, 1.5, 2.2]},  # 骑行者
        2: {'min': [2.0, 1.5, 1.0], 'max': [6.0, 2.5, 2.5]},  # 汽车
        3: {'min': [1.0, 0.8, 1.0], 'max': [3.0, 1.5, 2.0]},  # 摩托车
        5: {'min': [6.0, 2.0, 2.5], 'max': [15.0, 3.5, 4.0]}, # 公交车
        6: {'min': [4.0, 2.0, 2.0], 'max': [12.0, 3.0, 4.5]}, # 卡车
        7: {'min': [10.0, 2.5, 3.0], 'max': [50.0, 4.0, 5.0]} # 火车
    }
    
    constraints = size_constraints.get(obj_class, {'min': [0.5, 0.5, 0.5], 'max': [10.0, 5.0, 3.0]})
    
    min_dims = np.array(constraints['min'])
    max_dims = np.array(constraints['max'])
    
    # 约束尺寸
    constrained_dims = np.clip(dimensions, min_dims, max_dims)
    
    return constrained_dims


def is_bbox_reasonable(bbox_corners_3D, obj_class=0):
    """
    检查3D边界框是否合理
    """
    if bbox_corners_3D is None or len(bbox_corners_3D) != 8:
        return False
    
    # 计算边界框尺寸
    min_point = np.min(bbox_corners_3D, axis=0)
    max_point = np.max(bbox_corners_3D, axis=0)
    dimensions = max_point - min_point
    
    # 检查尺寸是否合理
    if np.any(dimensions <= 0) or np.any(dimensions > 100):  # 尺寸过大或过小
        return False
    
    # 检查体积是否合理
    volume = np.prod(dimensions)
    if volume < 0.1 or volume > 1000:  # 体积过大或过小
        return False
    
    # 根据类别检查特定约束
    class_volume_limits = {
        0: (0.1, 5.0),    # 行人
        1: (0.5, 10.0),   # 骑行者
        2: (5.0, 50.0),   # 汽车
        3: (1.0, 15.0),   # 摩托车
        5: (20.0, 200.0), # 公交车
        6: (15.0, 150.0), # 卡车
        7: (50.0, 500.0)  # 火车
    }
    
    if obj_class in class_volume_limits:
        min_vol, max_vol = class_volume_limits[obj_class]
        if volume < min_vol or volume > max_vol:
            return False
    
    return True 