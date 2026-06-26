import requests
import time


url = "http://127.0.0.1:5000/v1/main_server"
data = {"uid":"AI-Cloud-001","text":"齐佳佳喜欢运动"}
start_time = time.time()
res = requests.post(url, data=data)

cost_time = time.time() - start_time

print(res.text)

print(cost_time*1000)
