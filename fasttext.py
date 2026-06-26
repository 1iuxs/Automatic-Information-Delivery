
#
# import fasttext
# train_data_path = 'train_fast.txt'
# test_data_path = 'test_fast.txt'
#
# # 开启模型训练
# model = fasttext.train_supervised(input=train_data_path, wordNgrams=2)
# #supervised 监督
# #unsupervised 无监督
# # 开启模型测试
# result = model.test(test_data_path)
# print(result)
#
#
#

import random

# ===================== 配置参数 =====================
# 原始数据文件路径
raw_file_path = "train_fast_2.0.txt"
# 输出文件名称
train_out = "train.txt"
val_out = "val.txt"
test_out = "test.txt"
# 划分比例
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15
# 设置随机种子保证每次划分结果一致
random.seed(42)


# ====================================================

def split_dataset():
    # 1. 读取所有数据行
    all_lines = []
    with open(raw_file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:  # 跳过空行
                all_lines.append(line)

    # 2. 随机打乱数据
    random.shuffle(all_lines)
    total_num = len(all_lines)
    print(f"总数据量：{total_num} 条")

    # 3. 计算分割下标
    train_end = int(total_num * train_ratio)
    val_end = train_end + int(total_num * val_ratio)

    # 切分数据集
    train_data = all_lines[:train_end]
    val_data = all_lines[train_end:val_end]
    test_data = all_lines[val_end:]

    print(f"训练集：{len(train_data)} 条")
    print(f"验证集：{len(val_data)} 条")
    print(f"测试集：{len(test_data)} 条")

    # 4. 写入文件
    def write_file(save_path, data_list):
        with open(save_path, "w", encoding="utf-8") as fw:
            fw.write("\n".join(data_list))

    write_file(train_out, train_data)
    write_file(val_out, val_data)
    write_file(test_out, test_data)
    print("数据集划分完成！已生成 train.txt / val.txt / test.txt")


if __name__ == "__main__":
    split_dataset()








