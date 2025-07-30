import re
import socket
from nebulalink.perceptron3_0_5_pb2 import PerceptronSet

block_pattern = re.compile(r'v2x_obus\s*\{(.*?)\n\}', re.DOTALL)

def get_latest_bsm(sock: socket.socket, timeout=0.1):
    """
    忽略之前缓存的所有报文，等待并返回下一条新报文。
    如果 timeout 内没有收到新报文，返回 None。
    """
    # 1. 非阻塞模式清空缓冲
    sock.setblocking(False)
    while True:
        try:
            sock.recvfrom(4096)
        except BlockingIOError:
            break

    # 2. 切换为阻塞+timeout，等待下一条
    sock.settimeout(timeout)
    try:
        data, addr = sock.recvfrom(4096)
        return data
    except socket.timeout:
        return None
    finally:
        sock.settimeout(None)

def decode_bsm(raw_bytes):
    if raw_bytes == None:
        return None
    message = PerceptronSet()  
    message.ParseFromString(raw_bytes)
    return message

def parse_block_to_dict(block_text: str) -> dict:
    """
    将单个 v2x_obus 块文本解析为 dict。
    支持一级嵌套：obu_point、obu_size、p_accel_4way。
    """
    result = {}
    lines = block_text.splitlines()
    stack = [result]
    for line in lines:
        line = line.strip()
        if not line or line.startswith('//'):
            continue
        if line.endswith('{'):
            key = line[:-1].strip()
            new = {}
            stack[-1][key] = new
            stack.append(new)
        elif line == '}':
            stack.pop()
        else:
            m = re.match(r'(\w+):\s*(.+)', line)
            if m:
                k, v = m.group(1), m.group(2)
                v = v.strip().strip('"')
                # 尝试转换数字
                try:
                    if '.' in v:
                        val = float(v)
                    else:
                        val = int(v)
                except:
                    val = v
                stack[-1][k] = val
    return result

def filter_bsm_data(raw_text, target_deviceid):
    """
    从 raw_text 提取匹配 target_deviceid 的那一块，并返回其 dict。
    若无匹配返回 None。
    """
    try:
        if raw_text is None:
            return None
        if not isinstance(raw_text, str):
            raw_text = str(raw_text)
        # 匹配目标设备的块
        for m in block_pattern.finditer(raw_text):
            body = m.group(1)
            d = parse_block_to_dict(body)
            dev = d.get('obu_deviceid')
            if dev == target_deviceid:
                return d
        return None
    except Exception as e:
        print(f"BSM数据解析过滤失败: {e}")
        return None

def get_bsm_valid_info(sock: socket.socket, vehicle_id: str, timeout=0.1):
    """
    从 BSM 数据中提取有效信息。
    返回一个包含速度、加速度等信息的字典。
    """
    bsm_raw = get_latest_bsm(sock, timeout)
    if bsm_raw is None:
        return None
    full_bsm_data = decode_bsm(bsm_raw)
    if full_bsm_data is None:
        return None
    bsm_data = filter_bsm_data(str(full_bsm_data), vehicle_id)
    if bsm_data is None:
        return None

    # print(bsm_data)

    speed_ms = bsm_data['obu_point'].get("p_speed",0)

    # 提取速度和加速度等信息
    speed_ms = int(speed_ms)  # 转换为 m/s
    # bsm_data['obu_point']["p_speed"]
    speed_bsm_kmh = speed_ms * 3.6

    return {"speed_bsm_kmh": speed_bsm_kmh,
            "vehicle_id": vehicle_id}

