from src.CurveSlice import Slice
import matplotlib.pyplot as plt
import hashlib
import os


def get_ture_chp(data):
    last = None
    change_points = [0]
    idx = 0
    for now in data:
        if last is not None and last != now:
            change_points.append(idx)
        idx += 1
        last = now
    change_points.append(len(data))
    return change_points


def dict_hash(obj):
    sorted_items = sorted(obj.items())
    hash_obj = hashlib.sha256()
    for item in sorted_items:
        hash_obj.update(str(item).encode())
    return hash_obj.hexdigest()


def get_hash_code(json_file, config):
    mark = json_file.copy()
    if mark.get('config') is not None:
        mark.pop('config')
    mark['dt'] = config['dt']
    mark['total_time'] = config['total_time']
    return dict_hash(mark)


def check_data_update(hash_code, data_path):
    hash_path = os.path.join(data_path, 'info.sha256')
    ori_hash_code = ''
    if os.path.exists(hash_path):
        with open(hash_path, 'r') as f:
            ori_hash_code = f.read()
            f.close()
    return ori_hash_code != hash_code


def save_hash_code(hash_code, data_path):
    hash_path = os.path.join(data_path, 'info.sha256')
    with open(hash_path, 'w') as f:
        f.write(hash_code)
        f.close()
