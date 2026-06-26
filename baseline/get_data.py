
import pandas as pd
from collections import Counter
import jieba


#处理原始数据
label_map = {"100": 1,"101": 2, "102": 3, "103": 4, "104": 5,  "106": 6, "107":7, "108": 8, "109": 9, "110": 10,
             "112": 11, "113": 12, "114": 13,"115": 14,"116": 15,}
IN_PATH = "toutiao_data.txt"
OUT_PATH = "train.txt"
with open(IN_PATH, "r", encoding="utf8", buffering=1024*1024) as f_in, \
     open(OUT_PATH, "w", encoding="utf8", buffering=1024*1024) as f_out:
    buf = []
    for line in f_in:
        sp = line.split("_!_")
        if len(sp) < 4:
            continue
        c, t = sp[1], sp[3].strip()
        # 核心修复：清除标题内所有制表符，避免分割异常
        t = t.replace("\t", " ").replace("\n", "").replace("\r", "")
        if c in label_map and len(t) > 2:
            buf.append(f"{t}\t{label_map[c]}\n")
        if len(buf) >= 5000:
            f_out.writelines(buf)
            buf.clear()
    if buf:
        f_out.writelines(buf)
print("处理完毕，输出文件：", OUT_PATH)


#随机森林数据
content = pd.read_csv('../toutiao_data/train.txt', sep='\t')
count = Counter(content.label.values)
def cut_sentence(s):
    return list(jieba.cut(s))
# 1. 生成分词列表存入words列
content['words'] = content['sentence'].apply(cut_sentence)
print(content.head(10))
# 2. 将分词列表拼接为空格分隔的字符串
content['words'] = content['sentence'].apply(lambda s: ' '.join(cut_sentence(s)))
print(content.head(10))
# 3. 分词后截断，只保留前30个词
content['words'] = content['words'].apply(lambda s: ' '.join(s.split())[:30])
#保存预处理完成的数据
content.to_csv('train_new.csv')






#构造fasttext训练数据
id_to_label = {}
idx = 1
with open('../toutiao_data/class.txt', 'r', encoding='utf-8') as f1:
    for line in f1.readlines():
        line = line.strip('\n').strip()
        id_to_label[idx] = line
        idx += 1

print(id_to_label)

count = 0
train_data = []
with open('../toutiao_data/train.txt', 'r', encoding='utf-8') as f2:
    for line in f2.readlines():

        line = line.strip('\n').strip()
        sentence, label = line.split('\t')

        # 1: 首先处理标签部分
        label_id = int(label)
        label_name = id_to_label[label_id]
        new_label = '__label__' + label_name

        # 2: 然后处理文本部分，为了便于后续增加n-gram特性，可以按字划分，也可以按词划分
        #字向量
        #sent_char = ' '.join(list(sentence))
        #词向量
        sent_char = ' '.join(jieba.cut(sentence))

        # 3: 将文本和标签组合成fasttext规定的格式
        new_sentence = new_label + ' ' + sent_char
        train_data.append(new_sentence)

        count += 1
        if count % 10000 == 0:
            print('count=', count)

with open('train_fast.txt', 'w', encoding='utf-8') as f3:
    for data in train_data:
        f3.write(data + '\n')
print('FastText训练数据预处理完毕！')




















