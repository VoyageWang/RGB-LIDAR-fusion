# V2X车路协同3D检测系统

## 项目概述

这是一个基于DAIR-V2X数据集的多视角RGB+LiDAR融合检测系统，支持基础设施侧和车辆侧两个视角的3D目标检测、人车距离计算和车路协同功能。

## 核心文件

### 主要检测器
- `Code/configurable_v2x_detector.py` - 可配置的多视角3D检测器（主要版本）
- `Code/configurable_v2x_detector copy.py` - 检测器的备份版本
- `Code/test_dair_v2x_enhanced_o.py` - 增强版DAIR-V2X检测测试脚本

### 核心依赖模块
- `Code/detector.py` - YOLO检测器封装
- `Code/fusion.py` - LiDAR-相机融合算法
- `Code/improved_fusion.py` - 改进的融合算法
- `Code/calibration.py` - 相机标定处理
- `Code/utils.py` - 工具函数
- `Code/visualization.py` - 可视化功能
- `Code/data_processing.py` - 数据处理模块

### V2X车路协同模块
- `Code/v2x_message_processor.py` - V2X消息处理器
- `Code/v2x_message_sender.py` - V2X消息发送器
- `Code/integrated_v2x_detection.py` - 集成V2X检测系统

## 主要功能

1. **多视角3D检测**: 支持基础设施侧和车辆侧数据处理
2. **RGB-LiDAR融合**: 结合相机图像和LiDAR点云进行3D目标检测
3. **人车距离计算**: 计算人员和车辆之间的安全距离
4. **目标跟踪**: 多帧目标跟踪和速度估计
5. **车路协同**: GPS坐标转换和车端消息匹配
6. **可视化输出**: 生成检测视频和BEV鸟瞰图

## 使用方法

### 基本用法
```bash
# 处理单个视角
python Code/configurable_v2x_detector.py \
    --view1-path /path/to/data \
    --view1-calib /path/to/calib.json \
    --view1-name infrastructure-side \
    --max-frames 100

# 处理双视角
python Code/configurable_v2x_detector.py \
    --view1-path /path/to/infra/data \
    --view1-calib /path/to/infra/calib.json \
    --view1-name infrastructure-side \
    --view2-path /path/to/vehicle/data \
    --view2-calib /path/to/vehicle/calib.json \
    --view2-name vehicle-side \
    --max-frames 100
```

### V2X车路协同模式
```bash
python Code/integrated_v2x_detection.py \
    --data-path /path/to/data \
    --view-name infrastructure-side \
    --lidar-gps-lat 39.9042 \
    --lidar-gps-lon 116.4074 \
    --lidar-gps-alt 10.0 \
    --v2x-port 8888 \
    --max-frames 100
```

## 依赖要求

### Python包
```bash
pip install ultralytics opencv-python numpy open3d tqdm torch pillow pyproj
```

### 数据格式
- 图像: `.jpg`, `.png`
- 点云: `.pcd`
- 标定: `.json`

## 输出结果

- **检测视频**: `output/{view_name}_detection.mp4`
- **BEV视频**: `output/{view_name}_bev.mp4`
- **结果数据**: `output/{view_name}_results.json`
- **汇总报告**: `output/configurable_v2x_summary.json`

## 算法特点

1. **多格式标定支持**: 支持DAIR-V2X和RCOOPER数据格式
2. **并行处理**: 支持多视角并行处理
3. **实时显示**: 支持实时可视化显示
4. **内存优化**: 包含内存管理和垃圾回收
5. **错误处理**: 完善的异常处理机制

## 注意事项

1. 需要下载YOLO模型文件 (`yolov8m-seg.pt`)
2. 确保数据路径结构正确 (`image/`, `velodyne/`, `calib/`)
3. 标定文件格式需要与数据集匹配
4. 大文件和输出目录已在`.gitignore`中排除

## 许可证

请遵循相关开源许可证要求。
