import time
import socket
import struct
from nebulalink import perceptron3_0_5_pb2
from nebulalink.perceptron3_0_5_pb2 import PerceptronSet

FIXED_HEADER = bytes([0xDA, 0xDB, 0xDC, 0xDD])            
FRAME_TYPE = 0x01        
PERCEPTION_TYPE = 0x07

def wrap_message_with_header(event) -> bytes:
    payload = event.SerializeToString()
    length = len(payload)

    # struct.pack 只打包后面 4 字节：FrameType(1), PerceptionType(1), Length(2)
    tail_header = struct.pack('>BBH', FRAME_TYPE, PERCEPTION_TYPE, length)

    return FIXED_HEADER + tail_header + payload

def build_perceptron_set(event,test_count) -> PerceptronSet:
    ps = PerceptronSet()
    ps.devide_id = b"F2421023"
    ps.devide_is_true = True
    ps.time_stamp = int(time.time() * 1000)
    ps.number_frame = test_count  # 可根据实际递增或系统计数
    # ps.perception_gps 可填充设备位置信息

    ps.event_list.append(event)  # 添加你生成的 Eventlist
    return ps

def build_rsi_event(event_status,test_count, event_type=0, event_desc=""):
    Eventlist = perceptron3_0_5_pb2.Eventlist
    # Eventlist = perceptron_pb2.Eventlist

    event = Eventlist()
    event.event_id = event_type
    event.event_status = event_status
    event.event_type = event_type
    event.rte_source = 5
    event.event_radius = 200.0
    event.event_desc = event_desc.encode("ascii")

    # GPS示例
    event.event_gps.object_latitude = 39.9042
    event.event_gps.object_longitude = 116.4074
    event.event_gps.object_elevation = 45.0

    print(f"Sent: event_type={event.event_type}, status={event.event_status}")
    ps = build_perceptron_set(event,test_count)

    return ps

def get_message(test_count,event_status=0, event_type=0, event_desc=""):
    event = build_rsi_event(event_status, test_count,event_type, event_desc)
    return wrap_message_with_header(event)

