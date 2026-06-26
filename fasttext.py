

import time
import jieba
import fasttext

from flask import Flask
from flask import request


app = Flask(__name__)
# jieba.load_userdict("stopwords.txt")

stop_words = set()
# 替换成你的stopwords.txt相对路径
with open('stopwords.txt', 'r', encoding='utf-8') as f:
    for line in f:
        word = line.strip()
        if word:  # 跳过空行
            stop_words.add(word)

model_save_path = "model_fast.bin_2.0"
model = fasttext.load_model(model_save_path)

print("模型加载完成")



@app.route('/v1/main_server', methods=['POST'])
def main_server():
    uid = request.form['uid']
    text = request.form['text']
    input_text = ' '.join(jieba.cut(text))
    res = model.predict(input_text)
    predict_name = res[0][0]
    return predict_name     


if __name__ == '__main__':
    app.run(host="127.0.0.1", port=5000)






