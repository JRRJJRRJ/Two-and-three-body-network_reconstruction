import numpy as np
import test_handle  # 导入主函数模块
import pandas as pd
from sklearn.metrics import f1_score
import os
from sklearn.linear_model import Ridge

# 调用主函数以获取参数值
sis_data = test_handle.sis_data

# 使用岭回归计算 A 矩阵
def compute_A_ridge(XJ, Y, lambda_i):
    ln_factor = np.log(1 - lambda_i)
    ridge_reg = Ridge(alpha=1.0)  # alpha 为正则化参数，可以进行调整
    A_i = ridge_reg.fit(XJ, Y).coef_ / ln_factor
    return A_i

# 生成矩阵X，命名为XJ
def compute_XJ(sis_data, basis, chosen_indices, chosen_indicesJ, matching_indices, node_i):
    n = len(sis_data[0])  # 取第一个字符串的长度作为 n
    m = len(basis)  # 基字符串的个数
    XJ = np.zeros((m, n - 1))  # 初始化 XJ 矩阵

    for basis_idx, basis_string in enumerate(basis):
        # 提取 ta 和 ta+1 的值
        ta_plus_1 = chosen_indices[basis_idx]
        ta = chosen_indicesJ[basis_idx]

        for j in range(n):  # j 的范围是 0 到 N-1
            if j == node_i:
                continue  # 跳过当前重构的节点 i

            # 输出 j 的值用于调试
            #print(f"当前基字符串索引: {basis_idx}, j 的值: {j}","节点i的值:",{node_i})

            # 计算 Sjv1
            try:
                Sjv1 = int(sis_data[ta][j])
            except IndexError as e:
                print(f"数据提取错误: ta = {ta}, j = {j}，错误信息: {e}")
                continue

            # 累加 Sjv2
            Sjv2 = 0
            for idx in matching_indices[basis_string]:
                Sjv2 += int(sis_data[idx][j])

            # 计算 Sjv0
            Sjv0 = (Sjv1 + Sjv2) / (len(matching_indices[basis_string]) + 1)

            # 列索引计算
            col_index = j if j < node_i else j - 1
            XJ[basis_idx, col_index] = Sjv0  # 将结果存储到 XJ 的相应位置
            #print(XJ)

            #print("XJ矩阵生成完毕")

    return XJ


# 用i=1来测试compute_XJ的功能
#XJ=compute_XJ(sis_data, basis, chosen_indices, chosen_indicesJ, matching_indices, 1)
#print(type(XJ))


connection_net = []  # 用于存储整个网络的邻接矩阵

num_nodes = len(sis_data[0])  # 节点的数量
print("num_node:",num_nodes)
# 假设已经定义了 basis, sis_data, chosen_indices, threshold_Y
# 你需要确保这些变  量在调用前已经被赋值

# 如果 sis_data 中可能包含非字符串数据，确保将其转换为字符串
sis_data = [str(s) for s in sis_data]

# 设置threshold_X阈值,使用find_basis_strings生成基字符串，同时得到生成的基字符串在时间序列数据即sis_data中的下标
threshold_X = 0.15

# 已有一组基字符串后，设置threshold_Y阈值，对于其中每一个基字符串，使用如下代码选择其对应的相关字符串；基字符串 ：'basis'  时间序列数据 ：'sis_data'
threshold_Y = 0.23


for node_i in range(num_nodes):  # 遍历每个节点

    # 调用find_basis_strings寻找节点i的基字符串及其对应的下标
    basis_strings, chosen_indices=test_handle.find_basis_strings(sis_data, threshold_X, node_i)

    # 测试内容
    #basis_count = len(basis_string)
    #print("选择的基字符串个数:", basis_count)
    #print("选择的基字符串下标:", chosen_indices)  # 输出选择的基字符串下标

    # 基字符串下标对应i节点在sis_data中的时刻ta+1，使用其去计算j节点在sis_data中对应的时刻ta，用chosen_indicesJ存储
    chosen_indicesJ = [max(0, i - 1) for i in chosen_indices]

    # 调用 compute_Y 函数，计算 Y_values_advance
    Y_values_advance, matching_indices = test_handle.compute_Y(basis_strings, sis_data, chosen_indices, threshold_Y, node_i)

    # 将 Y_values_advance 转换为 numpy 数组
    Y_values_advance = np.array(Y_values_advance)

    # 计算 Y 矩阵（m*1 列向量）
    Y = np.log(1 - Y_values_advance)  # 注意确保 Y_values_advance 中的值都在 (0, 1) 范围内

    # 计算当前节点的 XJ 矩阵
    XJ = compute_XJ(sis_data, basis_strings, chosen_indices, chosen_indicesJ, matching_indices, node_i)


    # 求解 A 矩阵
    lambda_i = 0.6  # 假设为 0.6
    A_i = compute_A_ridge(XJ, Y, lambda_i)

    # 动态阈值设置
    threshold = np.mean(np.abs(A_i)) * 0.9  # 或者其他动态阈值设定

    # 初始化 connection_Aij 列表，包含 num_nodes 个节点的连接情况
    connection_Aij = []
    

    for j in range(num_nodes):  # 遍历所有节点
        if j == node_i:
            connection_Aij.append(0)  # i 和 i 的自连接默认为 1
        elif j < len(A_i) and abs(A_i[j]) > threshold:  # 使用动态阈值
            connection_Aij.append(1)  # 若 A_i,j 的值接近阈值，表示有连接
        else:
            connection_Aij.append(0)  # 否则没有连接

    # 确保每个 connection_Aij 的长度始终是 num_nodes
    connection_Aij = connection_Aij[:num_nodes] + [0] * (num_nodes - len(connection_Aij))

    # 将 connection_Aij 添加到全局邻接矩阵中
    connection_net.append(connection_Aij)







# 确保邻接矩阵结构一致
connection_net = np.array(connection_net)

# 确保 connection_net 的结构一致
try:
    connection_net = np.array(connection_net)
    print("邻接矩阵构建成功")
except ValueError as e:
    print(f"转换为 numpy 数组时出错: {e}")
    for row in connection_net:
        print(f"行长度: {len(row)}")  # 输出每行的长度以检查不一致

# 输出邻接矩阵
print("矩阵大小：",connection_net.shape)

# 将生成的邻接矩阵保存为 .xlsx 文件
output_folder = 'Generate_file'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

output_file_path = os.path.join(output_folder, 'connection_net.xlsx')

# 将 numpy 数组转换为 pandas DataFrame 并保存为 Excel 文件
df = pd.DataFrame(connection_net)
df.to_excel(output_file_path, index=False, header=False)

print(f"邻接矩阵已保存到 {output_file_path}")

actual_matrix = pd.read_excel('../SIS_Dynamic/TestBASC-w.xlsx',header=None).values

# 假设生成的邻接矩阵为 connection_net
# 比较两个矩阵，计算F1分数

# 展开矩阵为一维数组
actual_flat = actual_matrix.flatten()
predicted_flat = connection_net.flatten()

# 计算F1分数
f1 = f1_score(actual_flat, predicted_flat)

print("生成矩阵格式：",type(connection_net))
print("原始矩阵格式：",type(actual_matrix))
print(f"F1 Score: {f1}")

# 计算误差
numerator = np.sum([np.linalg.norm(connection_net[i] - actual_matrix[i]) ** 2 for i in range(len(connection_net))])
denominator = np.sum([np.linalg.norm(actual_matrix[i]) ** 2 for i in range(len(actual_matrix))])

# 计算误差公式 E^2
error = numerator / denominator
print("误差 E^2:", error)
'''
print("XJ 矩阵:")
print(XJ)
print("Y 向量:")
print(Y)
'''