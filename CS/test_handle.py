import numpy as np
import random
import pandas as pd
from sklearn.preprocessing import StandardScaler


# 标准化汉明距离函数，时间复杂度为O（k），k为字符串长度
def normalized_hamming_distance(s1, s2):
    return np.sum(np.array(list(s1)) != np.array(list(s2))) / max(len(s1), len(s2))

def find_basis_strings(sis_data, threshold, i):
    temp_basis_strings = []  # 存储基字符串的列表
    chosen_indices = []  # 存储这些基字符串在原始数据 sis_data 中的索引

    # 创建 i_sis_data，只包含在第 (i-1) 列为 0 的行
    i_sis_data = []
    for index, row in enumerate(sis_data):
        if row[i - 1] == '0':  # 检查第 (i-1) 列的值是否为 0
            # 将第 i 列的值去掉并将行转换为字符串
            modified_row = ''.join(row[:i - 1] + row[i:])  # 拼接去掉第 i 列的字符串
            i_sis_data.append((modified_row, index))  # 将修改后的行及其原始索引一起添加

    if not i_sis_data:  # 如果没有满足条件的行，则返回空列表
        print("No valid rows found in i_sis_data.")
        return temp_basis_strings, chosen_indices

    # 随机选择一个字符串的下标
    first_index = random.randint(0, len(i_sis_data) - 1)
    first_basis_string, original_index = i_sis_data[first_index]
    temp_basis_strings.append(first_basis_string)  # 将第一个基字符串添加到列表
    chosen_indices.append(original_index)  # 存储原始索引

    # 遍历 i_sis_data 来选择基字符串
    for modified_row, original_index in i_sis_data:
        if modified_row not in temp_basis_strings:  # 确保当前字符串未被选中
            # 仅在 basis_strings 不为空时才进行比较
            if all(normalized_hamming_distance(modified_row, temp_basis_strings[k]) > threshold for k in
                   range(len(temp_basis_strings))):
                temp_basis_strings.append(modified_row)  # 添加基字符串
                chosen_indices.append(original_index)  # 存储原始索引

    # 根据 chosen_indices 获取原始字符串
    basis_strings = [sis_data[index] for index in chosen_indices]

    # 打印输出 basis_strings 和 chosen_indices
    #print("Basis Strings:", basis_strings)  # 打印原始字符串
    #print("Chosen Indices:", chosen_indices)
    print("选择的基字符串个数为：",len(basis_strings))


    return basis_strings , chosen_indices  # 返回原始字符串和其下标


def compute_Y(basis, sis_data, chosen_indices, threshold_Y, node_i):
    """
    计算每个基字符串的匹配字符串及其平均值，用于生成 Y 矩阵。

    参数:
    - basis: 基字符串列表（来自原始 sis_data）
    - sis_data: 原始时间序列数据（列表形式）
    - chosen_indices: 基字符串对应的下标
    - threshold_Y: 汉明距离的阈值
    - node_i: 要去掉的列索引（0-based）

    返回:
    - Y_values_advance: 生成的 Y 矩阵的值
    - matching_indices: 每个基字符串对应的匹配字符串的下标
    """
    Y_values_advance = []  # 用于存储 Y 矩阵的值
    matching_indices = {}  # 用于存储每个基字符串匹配字符串的下标

    # 创建新的 sis_data_new，去掉第 node_i 列的值
    sis_data_new = []
    for row in sis_data:
        modified_row = ''.join(row[:node_i] + row[node_i + 1:])  # 去掉第 node_i 列的值
        sis_data_new.append(modified_row)

    # 遍历每个基字符串
    for basis_string in basis:
        matching_indices[basis_string] = []  # 初始化匹配字符串下标列表

        # 去掉 basis_string 的第 node_i 个字符
        modified_basis_string = basis_string[:node_i] + basis_string[node_i + 1:]

        for index, s in enumerate(sis_data_new):  # 遍历 sis_data_new 列表中的每个字符串
            s = str(s)  # 确保将元素转换为字符串

            # 跳过基字符串本身
            if index in chosen_indices:
                continue

            # 比较长度，确保基字符串与去掉列后的字符串长度一致
            if len(modified_basis_string) == len(s):  # 两者长度应相同
                if normalized_hamming_distance(modified_basis_string, s) < threshold_Y:
                    matching_indices[basis_string].append(index)  # 存储原始下标

    # 计算 Y 矩阵的值
    for basis_string, indices in matching_indices.items():
        if indices:  # 确保有匹配字符串
            # 获取基字符串的对应值（对应的时刻 ta+1 的状态）
            chosen_index = chosen_indices[basis.index(basis_string)]  # 获取基字符串的下标
            Si_ta_plus_1 = int(sis_data[chosen_index][node_i])  # Si(ta+1) 的值

            # 对于匹配字符串 Si(tv+1)，累加其值
            matching_values = [int(sis_data[idx][node_i]) for idx in indices]
            l = len(matching_values)  # 匹配字符串的个数

            if l > 0:  # 防止除以零
                Y_value = (Si_ta_plus_1 + sum(matching_values)) / (1 + l)
            else:
                Y_value = Si_ta_plus_1  # 若没有匹配字符串，则直接取 Si(ta+1) 的值

            Y_values_advance.append(Y_value)  # 将计算的值添加到列表中

            print(f"基字符串: {basis_string}, 匹配字符串索引: {indices}")

    return Y_values_advance, matching_indices




# `time_series_data` 是时间序列数据，对时间序列数据进行预处理，分为5等分的矩阵，每个子矩阵用于后续的计算
scaler = StandardScaler()  #这一行创建了一个 StandardScaler 对象。StandardScaler 是来自 sklearn.preprocessing 模块的一个类，用于对数据进行标准化处理，使其均值为 0，标准差为 1。
normalized_data = pd.read_excel('../SIS_Dynamic/Teststate_nodes.xlsx')

# 使用 numpy 的 array_split 函数将其分成 5 份
split_data = np.array_split(normalized_data, 5)  # split_data 是一个列表，包含 5 个 (10000, 50) 的子矩阵

# 测试split_data中的数据
#for i, sub_matrix in enumerate(split_data):
    #print(f"Matrix {i + 1} shape: {sub_matrix.shape}")
    #print(sub_matrix)  # 输出子矩阵内容

first_sub_matrix = split_data[0]  # 获取第一个子矩阵
sis_data = first_sub_matrix.astype(str).agg(lambda x: ''.join(x), axis=1).tolist()  # 时间复杂度为O（p*k），p为行数；sis_data为有关列表形式
#print("sis_data大小：", len(sis_data))
#print(sis_data)

