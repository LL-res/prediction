import csv
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess


def read_from_csv(container_id):
    """
    Parameters:
      container_id - 容器id号，格式为“c_%d”

    Returns:
        cpu使用率序列，内存使用率序列

    """
    cpu_usage = []
    memory_usage = []
    cached = False
    cache_path = '/home/liao/Desktop/P/prediction/ali_data_prepare/cache.csv'
    #先读缓存
    with open(cache_path, 'r') as cache:
        reader = csv.reader(cache)
        for line in reader:
            if line[0] != container_id:
                continue
            cached = True
            cpu_usage = line[1:]
            memory_usage = next(reader)[1:]
            cpu_usage = [int(x) for x in cpu_usage]
            memory_usage = [int(x) for x in memory_usage]
            break
        if cached:
            return cpu_usage, memory_usage
    #缓存未击中则读取源数据
    prefix = '/home/liao/Desktop/P/data/'
    file_name = 'container_usage.csv'
    with open(prefix + file_name, 'r') as file:
        reader = csv.reader(file)
        recorded = False
        for line in reader:
            if line[0] != container_id:
                if recorded:
                    break
                else:
                    continue
            recorded = True
            cpu_val = int(line[3])
            cpu_usage.append(cpu_val)
            mem_val = int(line[4])
            memory_usage.append(mem_val)
    cpu_to_cache = [container_id]
    mem_to_cache = [container_id]
    with open(cache_path, 'a', newline='') as file:
        csv_writer = csv.writer(file)
        data_to_write = [cpu_to_cache+cpu_usage,mem_to_cache+memory_usage]
        csv_writer.writerows(data_to_write)
    return cpu_usage, memory_usage


def smooth(data_list,frac,draw_pic = True):
    """
    Parameters:
      data_list - 使用率数据list
      frac - 越接近0,越与原序列特征类似

    Returns:
        平滑后的序列

    """
    x_values = np.arange(len(data_list))
    # 使用 lowess 进行平滑,越接近0,越与原序列特征类似
    smoothed_result = lowess(data_list, x_values, frac=frac)
    smoothed_values = smoothed_result[:, 1]
    if draw_pic:
        plt.figure(figsize=(40, 10))
        plt.plot(data_list, label='Original Data')
        # plt.scatter(x_values, y_values, label='Original Data', color='blue', alpha=0.5)
        plt.plot(x_values, smoothed_values, label='Loess Smoothed', color='red', linewidth=2)
        plt.legend()
        plt.title('Lowess Smoothing Example')
        plt.show()
    return smoothed_values

def compress(data_list,new_size):
    """
    Parameters:
      data_list - 待处理数据序列
      new_size - 压缩到的size

    Returns:
        压缩后的数据序列

    """
    segment_length = len(data_list) // new_size
    # 平均采样
    averaged_list = [np.mean(data_list[i * segment_length: (i + 1) * segment_length]) for i in range(new_size)]
    return averaged_list

def extend(data_list,new_size):
    """
    Parameters:
      data_list - 待处理数据序列
      new_size - 加长后的size

    Returns:
        加长后的数据序列

    """
    result = []
    for i in range(new_size):
        result.append(data_list[i%len(data_list)])
    return result

def get_data(container_id,compress_to_size,extend_to_size,data_type='cpu',smooth_frac=0.05,draw_pic=True):
    """
    Parameters:
      container_id - 容器id号，格式为“c_%d”
      compress_to_size - 保留特征并压缩到的长度
      extend_to_size - 把序列加长后的长度
      data_type - 可选：‘cpu’，‘mem’
      smooth_frac - 平滑系数
    Returns:
        可以用来预测或训练的数据
    Doc:
        把一个延展很长的序列先保留特征压缩，在使用这个压缩的序列拼接出一个长序列

    """
    cpu_usage,mem_usage = read_from_csv(container_id)
    list_to_process = []
    if data_type == 'cpu':
        list_to_process = cpu_usage
    elif data_type == 'mem':
        list_to_process = mem_usage
    else:
        return list_to_process
    list_to_process = smooth(list_to_process,frac=smooth_frac,draw_pic=draw_pic)
    list_to_process = compress(list_to_process,compress_to_size)
    list_to_process = extend(list_to_process,extend_to_size)

    if draw_pic:
        plt.plot(list_to_process)
        plt.show()

    return list_to_process



