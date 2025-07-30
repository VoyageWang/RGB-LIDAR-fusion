import time
import socket
import struct
from nebulalink import perceptron3_0_5_pb2
from decision.decision_making import DecisionEngine
from decision.msg_warp import get_message
from decision.get_bsm import get_bsm_valid_info
from utils.utils import get_json_info

RSU_ADDR = ("192.168.20.224", 10086)  # 可替换为你的RSU地址
UDP_IP = "0.0.0.0"
UDP_PORT = 4000
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
vehicle_id = "DFRD0001"
json_file = "result/global_frame_result.json"

def send_to_rsu_func(msg):
    sock.sendto(msg, RSU_ADDR)

def run_decision_loop():
    decision_engine = DecisionEngine()
    test_count = 0
    while True:
        start_time = time.time()

        view_data = get_json_info(json_file)
        # if view_data is None:
        #     print("无法读取 view_data.json 文件，跳过当前循环。")
        #     time.sleep(0.1)
        #     continue
        bsm_data = get_bsm_valid_info(sock, vehicle_id, timeout=0.1)

        event_status, event_type, event_desc = decision_engine.run_decision(view_data, bsm_data)
        print(f"Decision Result: event_status={event_status}, event_type={event_type}, event_desc={event_desc}")
        msg = get_message(test_count, event_status, event_type, event_desc)
        if msg:
            send_to_rsu_func(msg)

        elapsed = time.time() - start_time
        sleep_time = max(0, 0.1 - elapsed)
        time.sleep(sleep_time)
        test_count += 1

if __name__ == "__main__":
    run_decision_loop()
