import cv2
import os
import datetime
import time
import argparse

class VideoRecorder:
    def __init__(self, rtsp_url, save_mode='video', duration=60, interval=5, max_frames=100):
        """
        初始化视频录制器
        
        Args:
            rtsp_url: RTSP流地址
            save_mode: 保存模式 ('video', 'frames', 'both')
            duration: 录制时长（秒），仅对video模式有效
            interval: 帧保存间隔（秒），仅对frames模式有效
            max_frames: 最大帧数，仅对frames模式有效
        """
        self.rtsp_url = rtsp_url
        self.save_mode = save_mode
        self.duration = duration
        self.interval = interval
        self.max_frames = max_frames
        
        # 创建VideoCapture对象
        self.cap = cv2.VideoCapture(rtsp_url)
        
        if not self.cap.isOpened():
            raise Exception(f"无法连接到RTSP流: {rtsp_url}")
        
        # 获取视频属性
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 10.0  # 如果无法获取FPS，默认使用10
        
        print(f"视频尺寸: {self.width}x{self.height}")
        print(f"FPS: {self.fps}")
        
        self.output_video = None
        self.frame_count = 0
        self.last_frame_time = 0

    def start_video_recording(self):
        """开始录制视频"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f'video_{timestamp}.mp4'
        
        # 使用H264编码器，兼容性更好
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.output_video = cv2.VideoWriter(output_filename, fourcc, self.fps, (self.width, self.height))
        
        if not self.output_video.isOpened():
            print("警告: 无法初始化VideoWriter，尝试其他编码器...")
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            output_filename = f'video_{timestamp}.avi'
            self.output_video = cv2.VideoWriter(output_filename, fourcc, self.fps, (self.width, self.height))
        
        print(f"开始录制视频: {output_filename}")
        return output_filename

    def save_frame(self, frame):
        """保存单帧图片"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # 包含毫秒
        output_filename = f'frame_{timestamp}.jpg'
        cv2.imwrite(output_filename, frame)
        print(f"保存帧: {output_filename}")
        return output_filename

    def run_video_mode(self):
        """视频录制模式"""
        output_filename = self.start_video_recording()
        start_time = time.time()
        frame_count = 0
        
        print(f"开始录制视频，时长: {self.duration}秒")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("错误: 无法读取帧")
                break
            
            # 写入视频帧
            if self.output_video is not None:
                self.output_video.write(frame)
                frame_count += 1
            
            # 检查是否达到录制时长
            if time.time() - start_time >= self.duration:
                break
            
            # 显示进度（每秒一次）
            if frame_count % int(self.fps) == 0:
                elapsed = time.time() - start_time
                remaining = self.duration - elapsed
                print(f"录制进度: {elapsed:.1f}s / {self.duration}s (剩余: {remaining:.1f}s)")
        
        if self.output_video is not None:
            self.output_video.release()
            self.output_video = None
        
        print(f"视频录制完成: {output_filename}, 总帧数: {frame_count}")

    def run_frames_mode(self):
        """帧保存模式"""
        print(f"开始保存帧，间隔: {self.interval}秒，最大帧数: {self.max_frames}")
        
        saved_frames = 0
        last_save_time = 0
        
        while saved_frames < self.max_frames:
            ret, frame = self.cap.read()
            if not ret:
                print("错误: 无法读取帧")
                break
            
            current_time = time.time()
            
            # 检查是否到了保存帧的时间
            if current_time - last_save_time >= self.interval:
                self.save_frame(frame)
                saved_frames += 1
                last_save_time = current_time
                print(f"已保存帧数: {saved_frames}/{self.max_frames}")
            
            # 短暂休眠避免CPU占用过高
            time.sleep(0.03)  # 约30fps的检查频率
        
        print(f"帧保存完成，总共保存了 {saved_frames} 帧")

    def run_both_mode(self):
        """同时保存视频和帧"""
        output_filename = self.start_video_recording()
        start_time = time.time()
        frame_count = 0
        saved_frames = 0
        last_frame_save_time = 0
        
        print(f"开始同时录制视频和保存帧")
        print(f"视频时长: {self.duration}秒, 帧间隔: {self.interval}秒")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("错误: 无法读取帧")
                break
            
            current_time = time.time()
            
            # 写入视频帧
            if self.output_video is not None:
                self.output_video.write(frame)
                frame_count += 1
            
            # 保存独立帧
            if current_time - last_frame_save_time >= self.interval:
                self.save_frame(frame)
                saved_frames += 1
                last_frame_save_time = current_time
            
            # 检查是否达到录制时长
            if time.time() - start_time >= self.duration:
                break
            
            # 显示进度
            if frame_count % int(self.fps) == 0:
                elapsed = time.time() - start_time
                remaining = self.duration - elapsed
                print(f"进度: {elapsed:.1f}s/{self.duration}s, 视频帧: {frame_count}, 保存帧: {saved_frames}")
        
        if self.output_video is not None:
            self.output_video.release()
            self.output_video = None
        
        print(f"录制完成 - 视频: {output_filename}, 视频帧数: {frame_count}, 保存帧数: {saved_frames}")

    def run(self):
        """根据模式运行相应的功能"""
        try:
            print(f"连接到RTSP流: {self.rtsp_url}")
            print(f"运行模式: {self.save_mode}")
            
            if self.save_mode == 'video':
                self.run_video_mode()
            elif self.save_mode == 'frames':
                self.run_frames_mode()
            elif self.save_mode == 'both':
                self.run_both_mode()
            else:
                print(f"不支持的模式: {self.save_mode}")
                
        except KeyboardInterrupt:
            print("\n用户中断录制")
        except Exception as e:
            print(f"发生错误: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        """清理资源"""
        if self.cap is not None:
            self.cap.release()
        if self.output_video is not None:
            self.output_video.release()
        print("资源清理完成")

def main():
    parser = argparse.ArgumentParser(description='无界面RTSP视频录制工具')
    parser.add_argument('--rtsp', default="rtsp://admin:Wuhan.123@192.168.20.220", 
                        help='RTSP流地址')
    parser.add_argument('--mode', choices=['video', 'frames', 'both'], default='video',
                        help='保存模式: video(录制视频), frames(保存帧), both(两者都保存)')
    parser.add_argument('--duration', type=int, default=60,
                        help='录制视频时长(秒), 默认60秒')
    parser.add_argument('--interval', type=float, default=5.0,
                        help='保存帧的时间间隔(秒), 默认5秒')
    parser.add_argument('--max-frames', type=int, default=100,
                        help='最大保存帧数, 默认100帧')
    
    args = parser.parse_args()
    
    print("=== RTSP视频录制工具 ===")
    print(f"RTSP地址: {args.rtsp}")
    print(f"保存模式: {args.mode}")
    
    if args.mode in ['video', 'both']:
        print(f"视频录制时长: {args.duration}秒")
    
    if args.mode in ['frames', 'both']:
        print(f"帧保存间隔: {args.interval}秒")
        print(f"最大帧数: {args.max_frames}")
    
    print("按 Ctrl+C 可以随时停止录制")
    print("=" * 30)
    
    # 创建录制器并开始录制
    recorder = VideoRecorder(
        rtsp_url=args.rtsp,
        save_mode=args.mode,
        duration=args.duration,
        interval=args.interval,
        max_frames=args.max_frames
    )
    
    recorder.run()

if __name__ == "__main__":
    main() 