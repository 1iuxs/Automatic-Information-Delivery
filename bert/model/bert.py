# coding: UTF-8
import torch
import os
from transformers import BertModel, BertTokenizer, BertConfig
import torch.nn as nn



class Config(object):
    def __init__(self, dataset):
        self.model_name = "bert"
        self.data_path = "data/"
        self.train_path = self.data_path + "train.txt"  # 训练集
        self.dev_path = self.data_path + "dev.txt"      # 验证集
        self.test_path = self.data_path + "test.txt"    # 测试集
        self.class_list = [
            x.strip() for x in open(self.data_path + "class.txt").readlines()
        ]  # 类别名单
        self.save_path = '/'
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        self.save_path += "/" + self.model_name + ".pt"  # 模型训练结果
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.require_improvement = 1000  # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)+1  # 类别数
        self.num_epochs = 3  # epoch数
        self.batch_size = 128  # mini-batch大小
        self.pad_size = 32  # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-5  # 学习率
        self.bert_path = r"./bert-base-chinese"
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.bert_config = BertConfig.from_pretrained(self.bert_path+ '/config.json' )
        self.hidden_size = 768




class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path, config=config.bert_config)

        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        # 输入的句子
        context = x[0]
        # 对padding部分进行mask，和句子一个size，padding部分用0表示，比如[1, 1, 1, 1, 0,
        mask = x[2]
        #_位置代表bert 最后一层隐藏层的输出张量，shape = [128,512,768]
        #pooled位置代表bert最后一层cls位置的输出张量,[128,768]
        bert_out = self.bert(context, attention_mask=mask)
        pooled = bert_out.pooler_output
        out = self.fc(pooled)

        return out
