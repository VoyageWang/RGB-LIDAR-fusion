#!/usr/bin/env python3
"""
调试融合失败的脚本
分析RCOOPER数据集中融合失败的具体原因
"""

import os
import cv2
import numpy as np
import open3d as o3d
import json
from configurable_v2x_detector import CustomCalibration, V2XEnhanced3DDetector
from ultralytics import YOLO

def analyze_depth_clustering_issue(all_points_of_object, depth_factor=20):
    """分析深度聚类的问题"""
    print(f"    深度聚类分析 (depth_factor={depth_factor}):")
    
    from sklearn.cluster import DBSCAN
    
    # 计算深度值
    depths = np.sqrt(all_points_of_object[:, 0]**2 + all_points_of_object[:, 1]**2)
    print(f"      深度范围: {np.min(depths):.2f} - {np.max(depths):.2f}m")
    print(f"      深度标准差: {np.std(depths):.2f}m")
    
    # DBSCAN聚类
    min_samples = max(5, int(len(all_points_of_object) * 0.01))
    dbscan = DBSCAN(eps=0.5, min_samples=min_samples)
    clusters = dbscan.fit_predict(depths.reshape(-1, 1))
    
    unique_clusters, cluster_counts = np.unique(clusters, return_counts=True)
    print(f"      聚类结果: {len(unique_clusters)} 个聚类")
    for cluster_id, count in zip(unique_clusters, cluster_counts):
        if cluster_id == -1:
            print(f"        噪声点: {count} 个")
        else:
            cluster_depths = depths[clusters == cluster_id]
            print(f"        聚类 {cluster_id}: {count} 个点, 深度范围 {np.min(cluster_depths):.2f}-{np.max(cluster_depths):.2f}m")
    
    # 分析深度范围调整
    if len(unique_clusters) > 1 and unique_clusters[0] != -1:  # 有有效聚类
        # 选择最大的聚类
        valid_clusters = unique_clusters[unique_clusters != -1]
        if len(valid_clusters) > 0:
            cluster_counts_dict = dict(zip(unique_clusters, cluster_counts))
            largest_cluster = max(valid_clusters, key=lambda x: cluster_counts_dict[x])
            
            cluster_depth_values = depths[clusters == largest_cluster]
            min_depth = np.min(cluster_depth_values)
            max_depth = np.max(cluster_depth_values)
            range_length = max_depth - min_depth
            
            print(f"      最大聚类深度范围: {min_depth:.2f} - {max_depth:.2f}m (范围长度: {range_length:.2f}m)")
            
            # 原始深度范围调整公式的问题
            min_depth_adjusted = min_depth + (1 - depth_factor) * range_length / 2
            max_depth_adjusted = max_depth - (1 - depth_factor) * range_length / 2
            
            print(f"      调整后深度范围: {min_depth_adjusted:.2f} - {max_depth_adjusted:.2f}m")
            
            if min_depth_adjusted >= max_depth_adjusted:
                print(f"      ⚠️ 问题发现: 调整后范围无效! (min >= max)")
                print(f"      原因: depth_factor={depth_factor} 太大，建议使用 depth_factor <= 1")
                
                # 建议合适的参数
                suggested_factor = 0.8  # 保留80%的深度范围
                suggested_min = min_depth + (1 - suggested_factor) * range_length / 2
                suggested_max = max_depth - (1 - suggested_factor) * range_length / 2
                print(f"      建议 depth_factor=0.8: 范围 {suggested_min:.2f} - {suggested_max:.2f}m")
    
    return len(unique_clusters) > 1

def analyze_fusion_failure():
    """分析融合失败的原因"""
    
    # 测试数据路径
    test_data_path = "/mnt/disk_2/yuji/RCOOPER/rcooper"
    calib_base_path = "/mnt/disk_2/yuji/RCOOPER/rcooper/calib"
    sensor_id = "139"
    
    # 数据路径
    image_dir = os.path.join(test_data_path, f"data/136-137-138-139/139/seq-9//cam-0")
    lidar_dir = os.path.join(test_data_path, f"data/136-137-138-139/139/seq-9//lidar")
    
    if not os.path.exists(image_dir) or not os.path.exists(lidar_dir):
        print(f"数据路径不存在: {image_dir} 或 {lidar_dir}")
        return
    
    # 创建标定对象
    try:
        calibration = CustomCalibration.create_from_rcooper_id(calib_base_path, sensor_id)
        print("✓ 标定对象创建成功")
    except Exception as e:
        print(f"✗ 标定对象创建失败: {e}")
        return
    
    # 获取测试文件
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))])
    lidar_files = sorted([f for f in os.listdir(lidar_dir) if f.endswith('.pcd')])
    
    if len(image_files) == 0 or len(lidar_files) == 0:
        print("没有找到测试文件")
        return
    
    print(f"找到 {len(image_files)} 个图像文件，{len(lidar_files)} 个点云文件")
    
    # 加载YOLO模型
    model = YOLO("yolov8m-seg.pt")
    
    # 只测试第一帧，详细分析
    print(f"\n=== 详细分析第一帧 ===")
    
    # 加载图像
    img_path = os.path.join(image_dir, image_files[0])
    image = cv2.imread(img_path)
    if image is None:
        print(f"无法加载图像: {img_path}")
        return
        
    print(f"图像尺寸: {image.shape}")
    
    # 加载点云
    pcd_path = os.path.join(lidar_dir, lidar_files[0])
    pcd = o3d.io.read_point_cloud(pcd_path)
    points = np.asarray(pcd.points, dtype=np.float64)
    
    if len(points) == 0:
        print("点云为空")
        return
        
    print(f"原始点云数量: {len(points)}")
    
    # 分析点云分布
    distances = np.sqrt(points[:, 0]**2 + points[:, 1]**2 + points[:, 2]**2)
    print(f"点云距离分布:")
    print(f"  距离范围: {np.min(distances):.2f} - {np.max(distances):.2f}m")
    print(f"  平均距离: {np.mean(distances):.2f}m")
    print(f"  距离标准差: {np.std(distances):.2f}m")
    
    # 过滤无效点
    if np.any(np.isnan(points)) or np.any(np.isinf(points)):
        valid_mask = np.isfinite(points).all(axis=1)
        points = points[valid_mask]
        print(f"过滤后点云数量: {len(points)}")
    
    # YOLO检测
    results = model.predict(
        source=image,
        classes=[0, 1, 2, 3, 5, 6, 7],
        verbose=False,
        show=False,
    )
    
    r = results[0]
    boxes = r.boxes
    masks = r.masks
    
    if boxes is None or len(boxes) == 0:
        print("没有检测到目标")
        return
        
    print(f"检测到 {len(boxes)} 个目标")
    
    # 点云投影
    pts_2D, valid_mask = calibration.convert_3D_to_2D(points)
    if len(pts_2D) > 0:
        img_width, img_height = image.shape[1], image.shape[0]
        valid_2d_mask = (
            (pts_2D[:, 0] >= 0) & (pts_2D[:, 0] < img_width) &
            (pts_2D[:, 1] >= 0) & (pts_2D[:, 1] < img_height)
        )
        valid_indices = np.where(valid_mask)[0][valid_2d_mask]
        pts_3D = points[valid_indices]
        pts_2D = pts_2D[valid_2d_mask]
    else:
        pts_3D, pts_2D = np.array([]), np.array([])
    
    print(f"投影后有效点数: {len(pts_3D)}")
    
    if len(pts_3D) == 0:
        print("⚠️ 关键问题：投影后没有有效点！")
        return
    
    # 分析每个检测目标（只分析前3个）
    for j in range(min(3, len(boxes.cls.tolist()))):
        cls = boxes.cls.tolist()[j]
        print(f"\n=== 详细分析目标 {j} (类别: {int(cls)}) ===")
        
        # 检查掩码
        if masks is None or j >= len(masks.xy) or masks.xy[j].size == 0:
            print("    ✗ 无有效掩码")
            continue
        
        seg_mask = masks.xy[j]
        print(f"    掩码点数: {len(seg_mask)}")
        
        # 分析掩码区域
        mask_area = cv2.contourArea(seg_mask.astype(np.int32))
        print(f"    掩码面积: {mask_area:.0f} 像素")
        
        # 测试腐蚀
        from data_processing import erode_polygon
        eroded_seg = erode_polygon(seg_mask, image, erosion_factor=25)
        
        if eroded_seg is None or len(eroded_seg) <= 2:
            print("    ✗ 腐蚀后多边形太小")
            continue
        
        eroded_area = cv2.contourArea(eroded_seg.astype(np.int32))
        print(f"    腐蚀后面积: {eroded_area:.0f} 像素 (保留率: {eroded_area/mask_area*100:.1f}%)")
        
        # 测试点云提取
        mask = np.zeros_like(image[:, :, 0], dtype=np.uint8)
        cv2.fillPoly(mask, [eroded_seg.astype(np.int32)], color=1)
        
        pts_2D_int = pts_2D.astype(np.int32)
        inside_mask_indices = mask[pts_2D_int[:, 1], pts_2D_int[:, 0]] == 1
        all_points_of_object = pts_3D[inside_mask_indices]
        
        print(f"    多边形内点数: {len(all_points_of_object)}")
        
        if len(all_points_of_object) == 0:
            print("    ✗ 多边形内无点云")
            continue
        
        # 详细分析深度聚类问题
        has_valid_clusters = analyze_depth_clustering_issue(all_points_of_object, depth_factor=20)
        
        if has_valid_clusters:
            print("    建议解决方案:")
            print("      1. 减小 depth_factor 到 0.5-1.0 之间")
            print("      2. 或者修改深度聚类算法的范围调整公式")
            print("      3. 增加腐蚀因子到 15-20")
        
        # 测试更合适的参数
        print(f"\n    测试建议参数 (depth_factor=0.8):")
        from data_processing import filter_points_with_depth_clustering
        
        # 修改depth_factor测试
        test_filtered_points = test_improved_depth_clustering(all_points_of_object, depth_factor=0.8)
        print(f"      改进后点数: {len(test_filtered_points)}")
        
        if len(test_filtered_points) >= 4:
            from data_processing import is_coplanar
            if not is_coplanar(test_filtered_points):
                print("    ✓ 使用改进参数后融合条件满足！")
            else:
                print("    ⚠️ 点云仍然共面")
        else:
            print("    ⚠️ 改进后点数仍然不足")

def test_improved_depth_clustering(points_of_object, eps=0.5, depth_factor=0.8):
    """测试改进的深度聚类"""
    from sklearn.cluster import DBSCAN
    
    # Compute min_samples as a percentage of number of points
    min_samples = max(5, int(len(points_of_object) * 0.01))
    
    # Compute the depth values using the Euclidean Distance
    depths = np.sqrt(points_of_object[:, 0]**2 + points_of_object[:, 1]**2)

    # Density-Based Clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(depths.reshape(-1, 1))

    # Calculate the number of points in each cluster
    unique_clusters, cluster_counts = np.unique(clusters, return_counts=True)
    cluster_counts_dict = dict(zip(unique_clusters, cluster_counts))

    # Sort clusters based on the number of points they contain
    sorted_clusters = sorted(cluster_counts_dict, key=cluster_counts_dict.get, reverse=True)

    # Keep only the clusters with the most points
    top_cluster = sorted_clusters[:1]

    # 改进的深度范围调整
    filtered_points_of_object = []
    for cluster_label in top_cluster:
        if cluster_label == -1:
            continue  # Skip noise points
        cluster_depth_values = depths[clusters == cluster_label]
        min_depth = np.min(cluster_depth_values)
        max_depth = np.max(cluster_depth_values)
        
        # 改进的范围调整公式
        range_length = max_depth - min_depth
        shrink_factor = (1 - depth_factor) / 2  # 从两端各收缩
        min_depth_adjusted = min_depth + shrink_factor * range_length
        max_depth_adjusted = max_depth - shrink_factor * range_length
        
        # 确保范围有效
        if min_depth_adjusted < max_depth_adjusted:
            valid_cluster_mask = clusters == cluster_label
            valid_cluster_indices = np.where(valid_cluster_mask)[0]
            cluster_points = points_of_object[valid_cluster_indices]
            cluster_depths = depths[valid_cluster_indices]
            depth_mask = (cluster_depths >= min_depth_adjusted) & (cluster_depths <= max_depth_adjusted)
            filtered_points_of_object.extend(cluster_points[depth_mask])

    return np.array(filtered_points_of_object)

def main():
    print("=== RCOOPER融合失败详细分析 ===")
    analyze_fusion_failure()

if __name__ == "__main__":
    main() 