import json

def get_json_info(file_path):
    """
    读取 JSON 文件并返回其内容。
    
    :param file_path: JSON 文件的路径
    :return: JSON 文件内容的字典
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"文件 {file_path} 未找到。")
        return None
    except json.JSONDecodeError:
        print(f"文件 {file_path} 不是有效的 JSON 格式。")
        return None
