import time
import json
import socket
import struct
# from google.protobuf.message import Message 
from nebulalink import perceptron3_0_5_pb2
from nebulalink.perceptron3_0_5_pb2 import PerceptronSet
# from nebulalink import perceptron_pb2  # 假设包含 PerceptionFrame

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

RSU_ADDR = ("192.168.20.224", 10086)  # 可替换为你的RSU地址
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

def send_to_rsu_func(event):
    msg = wrap_message_with_header(event)
    sock.sendto(msg, RSU_ADDR)
    # print(msg)
    # print(f"Sent: event_type={event.event_list.event_type}, status={event.event_list.event_status}")

def test_decision(test_count):
    if test_count < 100:
        # 模拟车辆未出现
        event_status = 0
        event_type = 0
        desc=""
        return build_rsi_event(event_status,test_count, event_type, desc)
    elif 100 <= test_count < 200:
        # 模拟车辆出现
        event_status = 3
        event_type = 913
        desc = ""
        return build_rsi_event(event_status,test_count, event_type, desc)
    elif 200 <= test_count < 300:
        # 模拟车辆超速靠近行人
        event_status = 3
        event_type = 912
        desc = "20"
        return build_rsi_event(event_status, test_count,event_type, desc)
    elif 300 <= test_count < 400:
        event_status = 3
        event_type = 911
        desc = ""
        return build_rsi_event(event_status, test_count,event_type, desc)
    else:
        event_status = 2
        event_type = 0
        desc = ""
        return build_rsi_event(event_status, test_count,event_type, desc)
    

def run_decision_loop():
    test_count = 301
    while True:
        event = test_decision(test_count)
        if event:
            send_to_rsu_func(event)
        time.sleep(0.1)  # 10Hz
        # test_count += 1

if __name__ == "__main__":
    run_decision_loop()
