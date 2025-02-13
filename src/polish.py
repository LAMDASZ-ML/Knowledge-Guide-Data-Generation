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

REFINE = """给你一个包含若干法条的 JSON 字典，此字段是一个字典，Key 为推理过程中涉及的法律法条，Value 表示法律法条的具体内容。

{JSON}

法条的内容可能存在问题，请你将 Value 修正为 Key 对应的正确法条内容，并以 JSON 格式返回，不要附加其他内容或说明。
"""

CORRECT = """给你一个 JSON 格式的法律领域的问题及其答案。其中，instruction 字段指导如何回答问题，question 字段中包含问题，answer 字段中包含答案，reference 字段中包含法律法条的内容，reasoning 包含推理过程。

{JSON}

当前问题的推理过程与答案可能存在问题，请根据问题内容、法律法条内容，改进当前的推理过程与答案。
如果此问题的推理过程与答案无需改进，请直接输出原始 JSON 格式内容，否则请修改 reasoning 字段和 answer 字段的内容后，直接输出 JSON 格式内容。不要附加其他内容或说明。
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
    reply = reply.replace("'", '"')
    if reply.startswith("```json"): reply = reply[7: -3]
    if reply.startswith("```"): reply = reply[3: -3]
    return json.loads(reply.strip())

def toprob(item):
    ans = {}
    ans["instruction"] = item["instruction"]
    ans["reference"] = item["reference"]
    ans["question"] = item["question"]
    ans["reasoning"] = item["reasoning"]
    ans["answer"] = item["answer"]
    return ans

def solve(data):
    try:
        ref = REFINE.format(JSON=data["reference"])
        new_ref = tojson(generate(ref))
        if data["reference"] != new_ref:
            print("Diff =>", data["reference"], new_ref)
            data["reference_old"] = data["reference"]
            data["reference"] = new_ref
            cor = CORRECT.format(JSON=toprob(data))
            new_item = tojson(generate(cor))
            data["answer_old"] = data["answer"]
            data["reasoning_old"] = data["reasoning"]
            data["answer"] = new_item["answer"]
            data["reasoning"] = new_item["reasoning"]
        return data
    except Exception as err:
        print(err)
        return None

ori_path = f"./generation/GEN{'-' + args.suffix if args.suffix is not None else ''}.json"
tar_path = f"./generation/POL{'-' + args.suffix if args.suffix is not None else ''}.json"
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
for item in res:
    if item is None: continue
    results.append(item)
    oks += 1
print(f"Time: {time.time() - start_time:.2f}s", f"OK: {oks}/{len(inputs)}")

with open(tar_path, "w") as fw:
    json.dump(results, fw, ensure_ascii=False, indent=4)