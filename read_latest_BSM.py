# UDP接受解码BSM消息相关的模块
import socket
from nebulalink.nebulalink.perceptron3_0_5_pb2 import PerceptronSet
# import binascii
# from j2735decoder.src.CAVmessages import J2735_decode

# 初始化 UDP socket
UDP_IP = "0.0.0.0"
local_ip = "192.168.20.228"
UDP_PORT = 40118
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
sock.setblocking(False)  # 非阻塞模式 :contentReference[oaicite:4]{index=4}
latest_bsm = None  # 缓存最新一条 raw BSM 数据
OUT_PUT_FILE = "test_rsu_bsm.txt"


def get_latest_bsm():
    """
    尝试从 UDP 缓冲区读取所有待处理的 BSM 报文，只保留最新一条。
    不阻塞调用者，若无新报文则返回之前缓存。
    """
    global latest_bsm
    while True:
        try:
            data, addr = sock.recvfrom(4096)
            latest_bsm = data
            print(f"收到来自 {addr} 的 BSM 报文")
        except BlockingIOError:
            break
    return latest_bsm

def decode_bsm(raw_bytes):
    """
    将 ASN.1 UPER 编码的 raw_bytes 解码成 JSON 结构。
    使用 usdot‑fhwa‑stol/j2735decoder 的 J2735_decode。
    """
    message = PerceptronSet()
    message.ParseFromString(raw_bytes)
    return message

# 示例：主逻辑按需调用
def process_cycle():
    raw = get_latest_bsm()
    if raw is None:
        print("无新 BSM")
        return
    try:
        j = decode_bsm(raw)
        # core = j['BSM']['bsmFrame']
        print(j)
        print("==============================================")
        # with open(OUT_PUT_FILE, "a", encoding="utf-8") as f:
        #     f.write(f"{raw}\n")
        # print(raw)
        # print("==============================================")
    except Exception as e:
        print("解码或字段访问失败：", e)



if __name__ == "__main__":
    import time
    # 假设以 5–6 Hz 频率调用 process_cycle
    for _ in range(60):
        process_cycle()
        time.sleep(0.18)  # 约 5.5 Hz
    
    