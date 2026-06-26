

import fasttext
train_data_path = 'train_fast.txt'
test_data_path = 'test_fast.txt'

# 开启模型训练
model = fasttext.train_supervised(input=train_data_path, wordNgrams=2)
#supervised 监督
#unsupervised 无监督
# 开启模型测试
result = model.test(test_data_path)
print(result)









