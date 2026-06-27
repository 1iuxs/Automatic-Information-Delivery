import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from sklearn import metrics
import time
from utils import build_time
from tqdm import tqdm
from torch.optim import AdamW


#编写训练函数

def loss_fn(outputs, labels):
    return nn.CrossEntropyLoss()(outputs, labels)

def train(config,model,train_iter,dev_iter):
    start_time = time.time()
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01
        },
        {
            "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0
        }
    ]
    optimizer = AdamW(optimizer_grouped_parameters,lr=config.learning_rate)
    total_batch = 0
    dev_best_loss = float('inf')
    last_improve = 0
    flag = False
    model.train()
    for epoch in range(config.num_epochs):
        total_batch =0
        print("Epoch [{}/{}]".format(epoch + 1, config.num_epochs))
        for i,(trains,labels) in enumerate(tqdm(train_iter)):
            outputs = model(trains)
            model.zero_grad()#梯度清零
            loss = loss_fn(outputs,labels)
            loss.backward()
            optimizer.step()
            if total_batch % 200 == 0 and total_batch != 0:
                # 每多少轮输出在训练集和验证集上的效果
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)
                    improve = "*"
                    last_improve = total_batch
                else:
                    improve = ""
                time_dif = build_time.get_time_dif(start_time)
                msg = "Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},"
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                # 评估完成后将模型置于训练模式，更新参数
                model.train()

            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                print("No more improvements, stop training.")
                flag = True
                break
        if flag:
            break

#编写测试函数
def test(config,model,test_iter):
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
    msg = "Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}"
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = build_time.get_time_dif(start_time)
    print("Time usage:", time_dif)





#编写验证函数
def evaluate(config, model, data_iter, test=False):
    # 采用量化模型进行推理时需要关闭
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    #模型只进行前向传播，不进行反向传播
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)

















