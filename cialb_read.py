import yaml
import numpy

file_path = "/mnt/disk_2/yuji/RCOOPER/new_data/data/intersection/val/117-118-120-119_seq-002/0/00000.yaml"

try:
    with open(file_path, 'r') as stream:
        data_loaded = yaml.unsafe_load(stream)
        print("文件内容：")
        print(data_loaded)
except FileNotFoundError:
    print(f"错误：找不到文件 '{file_path}'")
except yaml.YAMLError as exc:
    print(exc)
