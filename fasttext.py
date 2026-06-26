

import fasttext
import time

train_data_path = 'train.txt'
test_data_path = 'test.txt'
dev_data_path = 'dev.txt'
# 开启模型训练
model = fasttext.train_supervised(
    input=train_data_path,
    autotuneValidationFile=dev_data_path,
    autotuneDuration=300,
    wordNgrams=2,
    verbose=3)
#supervised 监督
#unsupervised 无监督
# 开启模型测试
result = model.test(test_data_path)
print(result)


#模型保存
time1 = int(time.time())
model_save_path = 'model_fast.bin_2.0'
model.save_model(model_save_path)

