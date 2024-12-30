from pyexpat import features

from src.ChangePoints import getFeature
import numpy as np

def clustering_learning(trace, x, ud, num_var, num_ud):
    seg_index, seg_index_var = compute_segments(trace, num_var, num_ud)
    seg_index_var = list(seg_index_var.values())

    # 计算 similarity matrix
    combined_metric = compute_similarity_matrix(x, seg_index_var[0:num_var], num_var)

    # label = []
    # for i in range(len(trace)):
    #     for j in trace[i]['chpoints']:
    #         if j != 1001:
    #             label.append(trace[i]['mode'][j - 1])

    # label_me = np.zeros((len(label), len(label)))
    # for i in range(0):
    #     for j in range(len(label)):
    #         if label[i] == label[j]:
    #             label_me[i, j] = 1
    #         else:
    #             label_me[i, j] = 0


    if num_ud:
        combined_metric_ud = compute_similarity_matrix(ud, seg_index_var[num_var:num_var + num_ud], num_ud)
        combined_metric.append(combined_metric_ud)


    # 创建每个输出变量的局部聚类
    cluster_segs, trace = compute_clusters_local(trace, combined_metric, seg_index_var, num_var, num_ud)
    # fg = check_vectors_equal(label, cluster_segs[0])
    # 将局部聚类合并为全局聚类
    cluster_global, trace = compute_clusters_global(x, ud, trace, cluster_segs, seg_index, seg_index_var, num_var, num_ud)

    # 将全局结果保存到 trace 数据结构中
    labels_num = np.unique(cluster_global[:, 0])

    # 更新 trace 中的标签
    for i in range(len(trace)):
        chpoints = trace[i]['chpoints']
        len_segs = len(chpoints) - 1
        trace[i]['labels_num'] = labels_num
        trace[i]['labels_trace'] = cluster_global[:len_segs, 0]
        cluster_global = cluster_global[len_segs:, :]

    return trace


def check_vectors_equal(A, B):
    if len(A) != len(B):
        raise ValueError("The vectors must have the same length.")

    # 使用逐个元素比较
    return np.all(A == B)


def compute_clusters_global(xs, ud, trace, cluster_segs, seg_index, seg_index_var, num_var, num_ud):
    indices = np.ones((num_var + num_ud, 1), dtype=int)  # 每个局部变量的当前索引
    next_id = 1  # 下一个全局聚类ID
    cluster_global = np.zeros((len(seg_index), 1), dtype=int)  # 全局聚类结果
    M = {}

    # 遍历所有的全局切片
    for i in range(len(seg_index)):
        key = []  # 当前全局切片的局部聚类ID的组合
        for k in range(num_var + num_ud):
            # 生成当前局部聚类ID的key
            cluster_curr = cluster_segs[k][indices[k][0] - 1]  # 获取当前局部聚类ID
            key.append(str(cluster_curr))
            seg_curr = seg_index_var[k]

            # 如果下一个全局切片需要考虑下一个局部切片，则更新局部索引
            if seg_index[i][1] == seg_curr[indices[k][0] - 1][1]:
                indices[k][0] += 1

        # 连接所有局部聚类ID作为key
        key = '-'.join(key)

        # 如果字典中没有该key，则为其分配一个新的全局聚类ID
        if key not in M:
            M[key] = next_id
            next_id += 1

        # 保存全局聚类ID
        cluster_global[i] = M[key]

    # 更新 trace 中的聚类标签
    begin = 0

    for t in trace:
        for k in range(num_var + num_ud):
            chpoints_var = np.array(t['chpoints_per_var'][k])  # 当前变量的切片信息
            len_segs = len(chpoints_var) - 1  # 计算切片数量
            t['labels_trace_per_var'][k] = cluster_global[begin:len_segs+begin]  # 更新聚类标签
            begin = len_segs+begin

    return cluster_global, trace


def compute_clusters_local(trace, combined_metric, seg_index_var, num_var, num_ud, similarity_threshold=0.2):
    cluster_segs = [None] * (num_var + num_ud)  # 用于保存每个变量的聚类结果

    for t in trace:
        t['labels_trace_per_var'] = [None] * (num_var + num_ud)

    for k in range(num_var + num_ud):
        seg_index_curr = np.array(seg_index_var[k])  # 获取当前变量的切片索引
        num_segments = len(seg_index_curr)

        clusters = []  # 存储所有的聚类
        already_clustered = np.zeros(num_segments)  # 存储每个切片的聚类标签，0表示未聚类

        # 对每对切片进行比较
        for i in range(num_segments):
            # 跳过已经聚类的切片
            if already_clustered[i] > 0:
                continue

            # 找到当前切片与其他切片相似度大于阈值的切片
            cluster_id = len(clusters) + 1  # 新的聚类ID
            clusters.append([i])  # 新聚类中先加入当前切片

            for j in range(i + 1, num_segments):
                # 如果 i 和 j 的相似度大于阈值，认为它们是同一类
                if combined_metric[k][i, j] < similarity_threshold:
                    clusters[-1].append(j)  # 将 j 加入当前聚类
                    already_clustered[j] = cluster_id  # 标记 j 已经被聚类

            # 标记 i 已经聚类
            already_clustered[i] = cluster_id

        # 保存聚类结果
        cluster_segs[k] = already_clustered

        # 更新 trace 数据结构中的聚类信息
        len_begin = 0
        for t in trace:
            chpoints_var = np.array(t['chpoints_per_var'][k])  # 当前变量的切片信息
            len_segs = len(chpoints_var) - 1  # 计算切片数量
            t['labels_trace_per_var'][k] = already_clustered[len_begin:len_begin + len_segs]
            len_begin += len_segs

    return cluster_segs, trace


def compute_similarity_matrix(data, seg_index_var, num):

    similarity_matrix = []

    for i in range(num):
        seg_index_temp = seg_index_var[i]
        data_now = data[i]
        feature_list = []

        for j in range(seg_index_temp.shape[0]):
            begin = seg_index_temp[j][0] - 1
            end = seg_index_temp[j][1] - 1
            feature = getFeature(data_now[begin:end], 3, True)
            feature_list.append(feature)
        similarity_matrix_temp = compute_feature_distance(feature_list)
        similarity_matrix.append(similarity_matrix_temp)

    return similarity_matrix



def compute_feature_distance(feature_list):

    # all_elements = [item for sublist in feature_list for item in sublist]

    # max_value = max(all_elements)
    # min_value = min(all_elements)

    transposed_feature = np.array(feature_list)
    # min_values = np.min(transposed_feature, axis=0)
    # max_values = np.max(transposed_feature, axis=0)
    # normalized_feature = (feature_list - min_values ) / (max_values - min_values + 1)
    for i in range(transposed_feature.shape[1]):
        current_row = transposed_feature[:, i]
        min_val = np.min(current_row)
        max_val = np.max(current_row)
        if max_val - min_val < 1e-6:
            current_row = current_row / current_row
        else:
            current_row = (current_row - min_val) / (max_val - min_val)

        # 将归一化后的当前行存储回结果矩阵
        transposed_feature[:, i] = current_row


    distance_matrix = np.zeros((len(feature_list), len(feature_list)))

    for i in range(len(transposed_feature)):
        for j in range(len(transposed_feature)):
            # f_i = min_max_normalize(feature_list[i], max_value, min_value)
            # f_j = min_max_normalize(feature_list[j], max_value, min_value)
            f_i = transposed_feature[i]
            f_j = transposed_feature[j]

            distance_matrix[i, j] =  np.sum(np.abs(f_i - f_j))  # + 0.7 * (1 - np.dot(f_i, f_j) / (np.linalg.norm(f_i) * np.linalg.norm(f_j)))

            #np.dot(feature_list[i], feature_list[j]) / (np.linalg.norm(feature_list[i]) * np.linalg.norm(feature_list[j])))
    return distance_matrix


def compute_segments(trace, num_var, num_ud):

    # Initialize variables
    seg_index = np.array([[0, 0]])  # Global segments
    seg_index_var = {i: [[0, 0]] for i in range(num_var + num_ud)}  # Local segments per variable

    for i in range(len(trace)):
        # Global segments
        chpoints = trace[i]['chpoints']
        chsegments = np.column_stack((chpoints[:-1], chpoints[1:]))
        chsegments[1:, 0] += 1  # Increase start point by 1 to avoid overlapping segments

        seg_index = np.vstack((seg_index, seg_index[-1, 1] + chsegments))

        # Local segments (per output variable)
        chp_var = trace[i]['chpoints_per_var']
        for j in range(num_var + num_ud):
            chp_curr = np.array(chp_var[j])
            chsegments_var = np.column_stack((chp_curr[:-1], chp_curr[1:]))
            chsegments_var[1:, 0] += 1  # Increase start by 1 as done above
            # Apply offsets for consistency with x (and ud)
            seg_index_var_temp = np.array(seg_index_var[j])
            seg_index_var[j] = np.vstack((seg_index_var_temp, seg_index_var_temp[-1, 1] + chsegments_var))

    # Delete dummy line (used to offset first append as 0)
    seg_index = seg_index[1:]
    for j in range(num_var + num_ud):
        seg_index_var_temp = np.array(seg_index_var[j])
        seg_index_var[j] = seg_index_var_temp[1:]  # Remove the first line

    return seg_index, seg_index_var

