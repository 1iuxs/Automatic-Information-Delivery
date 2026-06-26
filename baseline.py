# 导入文本向量化工具TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
# 导入数据集划分工具：训练集/测试集分割
from sklearn.model_selection import train_test_split
# 导入随机森林分类器（本次baseline基线模型）
from sklearn.ensemble import RandomForestClassifier
# 导入pandas读取csv数据集
import pandas as pd
# icecream 打印美化工具，方便调试输出
from icecream import ic
import jieba
# 分类评价指标：准确率、召回率、精确率、F1分数
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

# 数据集文件路径：分词预处理完成的csv
TRAIN_CORPUS = 'train_new.csv'
# 停用词文件路径
STOP_WORDS = 'stopwords.txt'
# 存储分词文本的列名（预处理后words列，空格分隔分词）
WORDS_COLUMN = 'words'

# 读取预处理后的全量文本数据集
content = pd.read_csv(TRAIN_CORPUS)
# 取出分词文本列，作为模型输入语料
corpus = content[WORDS_COLUMN].values

# 超参数定义
stop_words_size = 766         # 限定使用前766个停用词
WORDS_LONG_TAIL_BEGIN = 10000   # 特征词上限基数
WORDS_SIZE = WORDS_LONG_TAIL_BEGIN - stop_words_size  # TF-IDF最大保留特征词数量

# 读取停用词文件，按空格分割，只取前749个停用词
# stop_words = open(STOP_WORDS,encoding="utf-8").read().split()[:stop_words_size]

with open(STOP_WORDS, encoding="utf-8") as f:
    raw_stop = f.read()
stop_tokens = jieba.cut(raw_stop)
stop_words = list(set([w.strip() for w in stop_tokens if w.strip()]))[:stop_words_size]

# 初始化TF-IDF向量化器
# max_features：只保留词频最高的WORDS_SIZE个特征词
# stop_words：传入停用词列表，分词时过滤无意义虚词
tfidf = TfidfVectorizer(max_features=WORDS_SIZE, stop_words=stop_words)
# 对全部语料拟合TF-IDF词典，并将文本转为数值向量 训练词表
text_vectors = tfidf.fit_transform(corpus)


# 1. 取出标签列
targets = content["label"].values
# 2. 划分训练集、测试集，8:2拆分
x_train, x_test, y_train, y_test = train_test_split(text_vectors, targets, test_size=0.2, random_state=42)

# 3. 初始化随机森林分类器
model = RandomForestClassifier(n_jobs=-1)
# 4. 模型训练
print("开始训练！！！！！！！")
model.fit(x_train, y_train)


print("开始预算！！！！！！")
acc = accuracy_score(model.predict(x_test), y_test)
ic(f"准确率 accuracy: {acc:.4f}")


#准确率有 80%

