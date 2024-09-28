import numpy as np
import random
import pandas as pd
from sklearn.preprocessing import StandardScaler


# 标准化汉明距离函数，时间复杂度为O（k），k为字符串长度
def normalized_hamming_distance(s1, s2):
    return np.sum(np.array(list(s1)) != np.array(list(s2))) / max(len(s1), len(s2))


# 查找一组基字符串函数，时间复杂度为O（n*n*k）
def find_basis_strings(sis_data, threshold):
    basis_strings = []
    first_index = random.randint(0, len(sis_data) - 1)
    basis_strings.append(str(sis_data[first_index]))  # 确保是字符串

    for s in sis_data:
        s = str(s)  # 确保每个元素都是字符串
        if s not in basis_strings:
            if all(normalized_hamming_distance(s, b) < threshold for b in basis_strings):
                basis_strings.append(s)

    return basis_strings


# `time_series_data` 是时间序列数据，对时间序列数据进行预处理，分为5等分的矩阵，每个子矩阵用于后续的计算
scaler = StandardScaler()  #这一行创建了一个 StandardScaler 对象。StandardScaler 是来自 sklearn.preprocessing 模块的一个类，用于对数据进行标准化处理，使其均值为 0，标准差为 1。
normalized_data = pd.read_excel('../SIS_Dynamic/state_nodes.xlsx')

# 使用 numpy 的 array_split 函数将其分成 5 份
split_data = np.array_split(normalized_data, 5)

# split_data 是一个列表，包含 5 个 (10000, 50) 的子矩阵
#for i, sub_matrix in enumerate(split_data):
    #print(f"Matrix {i + 1} shape: {sub_matrix.shape}")
    #print(sub_matrix)  # 输出子矩阵内容

first_sub_matrix = split_data[0]  # 获取第一个子矩阵
sis_data = first_sub_matrix.astype(str).agg(lambda x: ''.join(x), axis=1).tolist()  #时间复杂度为O（p*k），p为行数
#print(sis_data)


# 设置threshold_X阈值,使用find_basis_strings生成基字符串
threshold_X = 0.13
basis = find_basis_strings(sis_data, threshold_X)

basis_count=len(basis)
print("选择的基字符串个数:", basis_count)


# 已有一组基字符串后，设置threshold_Y阈值，对于其中每一个基字符串，使用如下代码选择其对应的相关字符串；基字符串 ：'basis'  时间序列数据 ：'sis_data'
threshold_Y = 0.1  # 设置阈值

# 创建一个包含不在 basis 中的字符串的新列表
sis_data_new = [s for s in sis_data if s not in basis]

# 创建一个字典来存储每个基字符串对应的满足条件的字符串
matching_strings = {}

# 对于每一个基字符串，遍历时间序列数据来寻找对应的相关字符串
# 遍历每个基字符串
# 初始化 Si_ta 变量，用于存储结果
Si_ta = {}

# 遍历每个基字符串
for basis_string in basis:
    matching_strings[basis_string] = []  # 初始化匹配字符串列表

    for s in sis_data_new:
        if normalized_hamming_distance(basis_string, s) < threshold_Y:
            matching_strings[basis_string].append(s)

# 查看生成了哪些基字符串，及其匹配的对应字符串个数
for basis_string, matches in matching_strings.items():
    match_count = len(matches)
    print(f"基字符串 '{basis_string}' 匹配的字符串个数: {match_count}")


    # 计算平均值Si(ta+1)，保持时间序列的形式
    if matching_strings[basis_string]:
        basis_values = np.array([float(x) for x in basis_string])  # 基字符串转换为数值数组
        match_values = np.array([float(x) for match in matching_strings[basis_string] for x in match])

        average_value = (basis_values + match_values.reshape(-1, len(basis_values))).mean(axis=0)

        # 将平均值转换为字符串形式
        Si_ta[basis_string] = [str(value) for value in average_value]

# 输出结果
for basis_string, avg in Si_ta.items():
    print(f"基字符串 '{basis_string}' 的平均值: {avg}")
