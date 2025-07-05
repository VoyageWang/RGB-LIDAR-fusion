#!/usr/bin/env python3
"""
批量处理V2X视频数据集的速度可视化
为每个序列生成速度可视化结果
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
from collections import defaultdict
import glob
import shutil

# 导入必要的模块
from ultralytics import YOLO
from fusion import *
from utils import *
from visualization import *


class V2XBatchSpeedProcessor:
    """V2X批量速度处理器"""
    
    def __init__(self, data_root, output_root, model_path="yolov8m-seg.pt"):
        self.data_root = data_root
        self.output_root = output_root
        self.model_path = model_path
        
        # 创建输出根目录
        os.makedirs(output_root, exist_ok=True)
        
        # 初始化模型
        self.model = YOLO(model_path)
        
    def find_sequences(self):
        """查找所有序列目录"""
        sequence_dirs = []
        for item in os.listdir(self.data_root):
            if item.startswith('sequence_') and os.path.isdir(os.path.join(self.data_root, item)):
                sequence_dirs.append(item)
        
        sequence_dirs.sort()
        return sequence_dirs
    
    def create_data_info_for_sequence(self, sequence_dir):
        """为序列创建data_info.json文件"""
        sequence_path = os.path.join(self.data_root, sequence_dir)
        
        # 检查必要的目录
        image_dir = os.path.join(sequence_path, "image")
        velodyne_dir = os.path.join(sequence_path, "velodyne")
        calib_dir = os.path.join(sequence_path, "calib")
        
        if not all(os.path.exists(d) for d in [image_dir, velodyne_dir, calib_dir]):
            print(f"跳过序列 {sequence_dir}: 缺少必要目录")
            return None
        
        # 获取图像文件列表
        image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
        
        # 生成data_info
        data_info = []
        for img_file in image_files:
            frame_id = os.path.splitext(img_file)[0]
            pcd_file = f"{frame_id}.pcd"
            
            # 检查对应的点云文件是否存在
            pcd_path = os.path.join(velodyne_dir, pcd_file)
            if not os.path.exists(pcd_path):
                continue
            
            frame_info = {
                "frame_id": frame_id,
                "image_path": f"image/{img_file}",
                "pointcloud_path": f"velodyne/{pcd_file}",
                "calib_camera_intrinsic_path": f"calib/camera_intrinsic/{frame_id}.json",
                "calib_virtuallidar_to_camera_path": f"calib/virtuallidar_to_camera/{frame_id}.json"
            }
            data_info.append(frame_info)
        
        # 保存data_info.json
        data_info_path = os.path.join(sequence_path, "data_info.json")
        with open(data_info_path, 'w') as f:
            json.dump(data_info, f, indent=2)
        
        print(f"为序列 {sequence_dir} 创建了 data_info.json，包含 {len(data_info)} 帧")
        return data_info_path
    
    def process_single_sequence(self, sequence_dir, max_frames=None):
        """处理单个序列"""
        print(f"\n=== 开始处理序列: {sequence_dir} ===")
        
        sequence_path = os.path.join(self.data_root, sequence_dir)
        sequence_output_dir = os.path.join(self.output_root, sequence_dir)
        
        # 创建序列输出目录
        os.makedirs(sequence_output_dir, exist_ok=True)
        
        try:
            # 检查已有的data_info.json是否存在且路径正确
            data_info_path = os.path.join(sequence_path, "data_info.json")
            need_recreate = False
            
            if os.path.exists(data_info_path):
                # 检查已有data_info.json的路径格式是否正确
                with open(data_info_path, 'r') as f:
                    data_info = json.load(f)
                
                if len(data_info) > 0:
                    first_frame = data_info[0]
                    # 检查标定文件路径格式
                    cam_intrinsic_path = first_frame['calib_camera_intrinsic_path']
                    if cam_intrinsic_path.endswith('camera_intrinsic.json'):
                        # 路径格式不正确，需要重新创建
                        need_recreate = True
                        print(f"检测到序列 {sequence_dir} 的data_info.json路径格式不正确，将重新创建")
            else:
                need_recreate = True
            
            if need_recreate:
                data_info_path = self.create_data_info_for_sequence(sequence_dir)
                if data_info_path is None:
                    return False
            
            # 重新加载数据信息（如果重新创建了）
            with open(data_info_path, 'r') as f:
                data_info = json.load(f)
            
            print(f"序列总帧数: {len(data_info)}")
            
            # 限制处理帧数
            if max_frames is not None:
                data_info = data_info[:max_frames]
                print(f"限制处理帧数为: {len(data_info)}")
            
            # 检查第一帧的标定文件
            first_frame = data_info[0]
            camera_intrinsic_path = os.path.join(sequence_path, first_frame['calib_camera_intrinsic_path'])
            lidar_to_camera_path = os.path.join(sequence_path, first_frame['calib_virtuallidar_to_camera_path'])
            
            if not os.path.exists(camera_intrinsic_path) or not os.path.exists(lidar_to_camera_path):
                print(f"跳过序列 {sequence_dir}: 缺少标定文件")
                print(f"  缺少: {camera_intrinsic_path if not os.path.exists(camera_intrinsic_path) else lidar_to_camera_path}")
                return False
            
            # 调用速度可视化处理
            from v2x_speed_visualization import process_v2x_speed_visualization
            
            success = process_v2x_speed_visualization(
                data_root=sequence_path,
                data_info_path=data_info_path,
                output_dir=sequence_output_dir,
                start_frame=None,
                end_frame=None,
                max_frames=max_frames
            )
            
            print(f"序列 {sequence_dir} 处理完成")
            return True
            
        except Exception as e:
            print(f"处理序列 {sequence_dir} 时出错: {e}")
            traceback.print_exc()
            return False
    
    def process_all_sequences(self, max_frames_per_sequence=None, start_from=None, end_at=None):
        """处理所有序列"""
        print("=== 开始批量处理V2X视频数据集 ===")
        
        # 查找所有序列
        sequences = self.find_sequences()
        print(f"找到 {len(sequences)} 个序列")
        
        # 过滤序列范围
        if start_from is not None:
            sequences = [s for s in sequences if s >= start_from]
        if end_at is not None:
            sequences = [s for s in sequences if s <= end_at]
        
        print(f"将处理 {len(sequences)} 个序列")
        
        # 记录处理结果
        results = {
            'total_sequences': len(sequences),
            'processed': [],
            'failed': [],
            'skipped': []
        }
        
        # 处理每个序列
        for i, sequence_dir in enumerate(sequences):
            print(f"\n[{i+1}/{len(sequences)}] 处理序列: {sequence_dir}")
            
            try:
                # 检查输出目录是否已存在且有内容
                sequence_output_dir = os.path.join(self.output_root, sequence_dir)
                if os.path.exists(sequence_output_dir) and len(os.listdir(sequence_output_dir)) > 0:
                    print(f"序列 {sequence_dir} 已存在输出，跳过")
                    results['skipped'].append(sequence_dir)
                    continue
                
                success = self.process_single_sequence(sequence_dir, max_frames_per_sequence)
                
                if success:
                    results['processed'].append(sequence_dir)
                    print(f"✓ 序列 {sequence_dir} 处理成功")
                else:
                    results['failed'].append(sequence_dir)
                    print(f"✗ 序列 {sequence_dir} 处理失败")
                
                # 定期清理内存
                if i % 3 == 0:
                    gc.collect()
                    
            except Exception as e:
                print(f"处理序列 {sequence_dir} 时发生异常: {e}")
                results['failed'].append(sequence_dir)
                continue
        
        # 保存处理结果统计
        results_path = os.path.join(self.output_root, "batch_processing_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # 打印总结
        print(f"\n=== 批量处理完成 ===")
        print(f"总序列数: {results['total_sequences']}")
        print(f"成功处理: {len(results['processed'])}")
        print(f"处理失败: {len(results['failed'])}")
        print(f"跳过序列: {len(results['skipped'])}")
        
        if results['failed']:
            print(f"失败的序列: {results['failed']}")
        
        print(f"结果保存在: {self.output_root}")
        print(f"处理统计保存在: {results_path}")
        
        return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="批量处理V2X视频数据集的速度可视化")
    parser.add_argument('--data-root', type=str, 
                       default="/mnt/disk_2/yuji/voyagepro/yolo-laidar/data/split-infrastructure-side-test",
                       help="数据集根目录")
    parser.add_argument('--output-root', type=str, 
                       default="./batch_speed_visualization_output123",
                       help="输出根目录")
    parser.add_argument('--model-path', type=str, default="yolov8m-seg.pt",
                       help="YOLO模型路径")
    parser.add_argument('--max-frames', type=int, default=500,
                       help="每个序列最大处理帧数")
    parser.add_argument('--start-from', type=str, default=None,
                       help="从指定序列开始处理")
    parser.add_argument('--end-at', type=str, default=None,
                       help="处理到指定序列结束")
    parser.add_argument('--single-sequence', type=str, default=None,
                       help="只处理指定的单个序列")
    
    args = parser.parse_args()
    
    print("V2X视频数据集批量速度可视化开始...")
    print(f"数据根目录: {args.data_root}")
    print(f"输出根目录: {args.output_root}")
    print(f"每序列最大帧数: {args.max_frames}")
    
    # 检查数据根目录
    if not os.path.exists(args.data_root):
        print(f"错误: 数据根目录不存在: {args.data_root}")
        return
    
    # 创建处理器
    processor = V2XBatchSpeedProcessor(
        data_root=args.data_root,
        output_root=args.output_root,
        model_path=args.model_path
    )
    
    # 处理序列
    if args.single_sequence:
        # 处理单个序列
        print(f"处理单个序列: {args.single_sequence}")
        success = processor.process_single_sequence(args.single_sequence, args.max_frames)
        if success:
            print("单个序列处理成功")
        else:
            print("单个序列处理失败")
    else:
        # 批量处理所有序列
        results = processor.process_all_sequences(
            max_frames_per_sequence=args.max_frames,
            start_from=args.start_from,
            end_at=args.end_at
        )


if __name__ == "__main__":
    main() 