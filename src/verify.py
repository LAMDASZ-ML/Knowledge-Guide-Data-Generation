import json
import numpy as np
import argparse
import sys
import json, os
import tqdm
from openai import OpenAI
import multiprocessing
import time

parser = argparse.ArgumentParser()
parser.add_argument('--cores',   default=16, type=int)
parser.add_argument('--nums',    default=1000000, type=int)
parser.add_argument('--suffix',  default=None, type=str)

args = parser.parse_args()

CORRECT = """给你一个 JSON 格式的法律领域的问题及其答案。其中，instruction 字段指导如何回答问题，question 字段中包含问题，answer 字段中包含答案，reference 字段中包含法律法条的内容，reasoning 包含推理过程。

{JSON}

请你判断数据中的推理过程与答案是否正确，请以 JSON 格式返回你的判断结果。JSON格式数据中包含一个 verify 字段，取值为正确或错误，也包含一个 message 字段，表示你判断的理由。
"""

def generate(prompt):
    client = OpenAI(api_key="Your Token", base_url="https://api.deepseek.com")
    messages=[
        {"role": "system", "content": "你是一个法律领域专家。"},
        {"role": "user", "content": prompt},
    ]
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        stream=False
    )
    return response.choices[0].message.content

def toprob(item):
    ans = {}
    ans["instruction"] = item["instruction"]
    ans["reference"] = item["reference"]
    ans["question"] = item["question"]
    ans["reasoning"] = item["reasoning"]
    ans["answer"] = item["answer"]
    return ans

def tojson(reply):
    reply = reply.replace("'", '"')
    if reply.startswith("```json"): reply = reply[7: -3]
    if reply.startswith("```"): reply = reply[3: -3]
    print(reply)
    return json.loads(reply.strip())

def solve(data):
    try:
        prob = toprob(data)
        ver = CORRECT.format(JSON=data)
        res = tojson(generate(ver))
        data["verify"] = res["verify"]
        data["message"] = res["message"]
        return data
    except Exception as err:
        print(err)
        return None

ori_path = f"./generation/POL{'-' + args.suffix if args.suffix is not None else ''}.json"
tar_path = f"./generation/VER{'-' + args.suffix if args.suffix is not None else ''}.json"
with open(ori_path, "r") as fr:
    data = json.load(fr)
n = len(data)
print("#Data = ",len(data))

inputs = []
for i in range(min(args.nums, n)):
    inputs.append(data[i])

start_time = time.time()
with multiprocessing.Pool(processes=args.cores) as pool:
    res = pool.map(solve, inputs)

results = []
oks = 0
cor = 0
for item in res:
    if item is None: continue
    results.append(item)
    oks += 1
    if item["verify"] == "正确": cor += 1
print(f"Time: {time.time() - start_time:.2f}s", f"OK: {oks}/{len(inputs)}", f"Veify: {cor}/{oks}")

with open(tar_path, "w") as fw:
    json.dump(results, fw, ensure_ascii=False, indent=4)