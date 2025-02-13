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
parser.add_argument('--nums',    default=1, type=int)
parser.add_argument('--suffix',  default=None, type=str)

args = parser.parse_args()

def get_xingshi():
    n = 186197 # Set the number of xingshi
    while True:
        try:
            index = np.random.randint(0, n - 1)
            with open(f"./reference/xingshi/{index}.json", "r") as fr:
                xingshi = json.load(fr)
            ans  = "".join(xingshi["本院查明"]).strip()
            ans += "\n" + "".join(xingshi["本院认为"]).strip()
            ans += "\n" + "".join(xingshi["裁判结果"]).strip()
            return index, ans
        except: pass

def get_minshi():
    n = 152452
    while True:
        try:
            index = np.random.randint(0, n - 1)
            with open(f"./reference/minshi/{index}.json", "r") as fr:
                minshi = json.load(fr)
            ans  = "".join(minshi["本院查明"]).strip()
            ans += "\n" + "".join(minshi["本院认为"]).strip()
            ans += "\n" + "".join(minshi["裁判结果"]).strip()
            return index, ans
        except: pass

CONTROL = """给你一个 JSON 格式的法律领域的问题及其答案。其中，instruction 字段指导如何回答问题，question 字段中包含问题，answer 字段中包含答案。

{JSON}

现在请你根据法律文书数据生成类似的问题，请问你需要什么类型的文书数据。可以选择的类型有：刑事法律文书、民事法律文书。请你选择一项并以 JSON 格式在 type 字段中返回。
"""

GENERATE = """给你一个 JSON 格式的法律领域的问题及其答案。其中，instruction 字段指导如何回答问题，question 字段中包含问题，answer 字段中包含答案。

{JSON}

请你以如下法律文书的内容为原型，按照相同的 JSON 格式和问题形式，在 instruction 不变的情况下，编造一个新问题与对应的答案。
请增加一个 reasoning 字段，此字段是一个字符串，表示得出答案的推理过程。
请增加一个 reference 字段，此字段是一个字典，Key 为推理过程中涉及的法律法条，Value 表示法律法条的具体内容。
请适当改写法律文书的内容，不要包含与答案无关的内容，不要直接复述法律文书的内容。
请修改问题与答案中的姓名、企业名称、地点等涉及隐私的内容。
answer 字段的内容应该完全按照 instruction 中的对答案的格式要求给出。

{DOCS}
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

def tojson(reply):
    if reply.startswith("```json"): reply = reply[7: -3]
    if reply.startswith("```"): reply = reply[3: -3]
    return json.loads(reply.strip())

def solve(seed):
    try:
        ctrl = CONTROL.format(JSON=seed)
        typ = tojson(generate(ctrl))["type"]
        doc = None
        idx = None
        if typ == "刑事法律文书": idx, doc = get_xingshi()
        if typ == "民事法律文书": idx, doc = get_minshi()
        if doc is None: return None
        
        gene = GENERATE.format(JSON=seed, DOCS=doc)
        res  = tojson(generate(gene))
        res["type"] = typ
        res["refi"] = idx
        return res
    except: return None

with open(f"./seed.json", "r") as fr:
    problems = json.load(fr)
problems = [problems[i] for i in range(len(problems)) if i % 1 == 0]
n = len(problems)
print("#Seed Questions = ",len(problems))

inputs = []
for i in range(args.nums):
    inputs.append(problems[i % n])

start_time = time.time()
with multiprocessing.Pool(processes=args.cores) as pool:
    res = pool.map(solve, inputs)

results = []
path = f"./generation/GEN{'-' + args.suffix if args.suffix is not None else ''}.json"
try:
    with open(path, "r") as fr:
        results = json.load(fr)
except: pass

oks = 0
for item in res:
    if item is None: continue
    results.append(item)
    oks += 1
print(f"Time: {time.time() - start_time:.2f}s", f"OK: {oks}/{len(inputs)}")

with open(path, "w") as fw:
    json.dump(results, fw, ensure_ascii=False, indent=4)