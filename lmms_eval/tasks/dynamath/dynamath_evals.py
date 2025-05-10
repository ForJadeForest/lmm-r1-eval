import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yaml
from Levenshtein import distance
from loguru import logger as eval_logger
from openai import AzureOpenAI, OpenAI


def preprocess(str1):
    str1 = str(str1)
    if 0 <= str1.find("{") < str1.rfind("}"):
        str1 = str1[str1.find("{") : str1.rfind("}") + 1]
    str2 = str1.replace("\\", "")
    str2 = str2.replace("\\n", "\n")
    return str2


def transfer(str1):
    if "\u03c0" in str1:
        strs = str1.split("\u03c0")
        str1 = strs[0]
        return float(str1) * np.pi
    else:
        return float(str1)


def parse_answer(answer, answer_type="multiple choice"):
    if answer_type == "float":
        if answer.isdigit():
            return True, float(answer)
        else:
            parts = answer.split(" ")
            answer = parts[0]
            try:
                answer = transfer(answer)
                return True, answer
            except:
                return False, None
    elif answer_type == "multiple choice":
        if len(answer) == 1:
            return True, answer.upper()
        else:
            in_flag = [ch in answer.upper() for ch in "ABCDE"]
            if sum(in_flag) == 1:
                for ch in "ABCDE":
                    if ch in answer.upper():
                        return True, ch
            return False, None
    else:
        return True, answer


class DynaMathEvaluator:
    API_TYPE = os.getenv("API_TYPE", "openai")
    if API_TYPE == "openai":
        API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
        API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY")
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        }
    elif API_TYPE == "azure":
        API_URL = os.getenv("AZURE_ENDPOINT", "https://api.cognitive.microsoft.com/sts/v1.0/issueToken")
        API_KEY = os.getenv("AZURE_API_KEY", "YOUR_API_KEY")
        headers = {
            "api-key": API_KEY,
            "Content-Type": "application/json",
        }

    def __init__(self, api_key, gpt_model="gpt-3.5-turbo", quick_extract=False):
        self.api_key = self.API_KEY
        self.gpt_model = gpt_model
        print(f"gpt_model: {self.gpt_model}")
        self.quick_extract = quick_extract

    def _post_request(self, payload):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        response = requests.post(self.API_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()

    def get_chat_response(self, prompt, temperature=0, max_tokens=256, n=1, patience=10, sleep_time=0):
        messages = [
            {"role": "user", "content": prompt},
        ]
        payload = {"model": self.gpt_model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens, "n": n}

        while patience > 0:
            patience -= 1
            try:
                response = self._post_request(payload)
                if n == 1:
                    prediction = response["choices"][0]["message"]["content"].strip()
                    if prediction and prediction != "":
                        return prediction
                else:
                    prediction = [choice["message"]["content"].strip() for choice in response["choices"]]
                    if prediction and prediction[0] != "":
                        return prediction

            except Exception as e:
                # some model may output repetitive answer, which ChatGPT will throw an error.
                if "repetitive patterns" in str(e):
                    print(str(e))
                    print("Continue with empty answer")
                    return ""
                # some answer may contain some sensitive words, like 'test'
                if "sensitive" in str(e) or "400" in str(e):
                    print(str(e))
                    print("Continue with empty answer")
                    return "0"

                if "Rate limit" not in str(e):
                    eval_logger.error(e)

                if "Please reduce the length of the messages" in str(e):
                    eval_logger.error("!!Reduce prompt size")
                    # reduce input prompt and keep the tail
                    new_size = int(len(prompt) * 0.9)
                    new_start = len(prompt) - new_size
                    prompt = prompt[new_start:]
                    payload["messages"] = [
                        {"role": "user", "content": prompt},
                    ]

                if sleep_time > 0:
                    time.sleep(sleep_time)
        return ""

    def judge_answer(self, prediction, doc):
        result = {}
        pred = preprocess(prediction)
        succeed, short_answer = None, None
        result["gpt_extract_answer"] = ""
        try:
            dj = json.loads(pred, strict=False)
            short_answer = dj.get("short answer")
            assert short_answer is not None
            succeed, short_answer = parse_answer(short_answer, answer_type=doc["answer_type"])
            assert succeed
        except:
            # Failed to parse the JSON, use an auxiliary LLM to get the short answer
            if doc["answer_type"] == "multiple choice":
                inst = "Output the corresponing choice option, such as 'A', 'B', 'C', 'D', in a single line."
            elif doc["answer_type"] == "float":
                inst = "Output a three-digit floating-point number in a single line."
            else:
                inst = "Output a short answer in a single line. Any float numbers in the answer " "should be formatted as three-digit floating-point numbers."

            prompt = f"Free-form answer: {pred}\nInstruction: {inst}"
            response = pred
            succeed, short_answer = parse_answer(response, doc["answer_type"])

            if not succeed:
                response = self.get_chat_response(prompt)
                result["gpt_extract_answer"] = response
                succeed, short_answer = parse_answer(response, doc["answer_type"])

        if doc["answer_type"] == "float":
            if succeed:
                diff = float(short_answer) - float(doc["ground_truth"])
                result["extraction_prediction"] = short_answer
                if abs(diff) <= 0.001:
                    result["true_false"] = True
                else:
                    result["true_false"] = False
                return result
            else:
                result["extraction_prediction"] = None
                result["true_false"] = False
                return result
        elif doc["answer_type"] == "multiple choice":
            if succeed:
                result["extraction_prediction"] = short_answer
                result["true_false"] = short_answer == doc["ground_truth"]
                return result
            else:
                if doc["ground_truth"] in pred[:3].upper():
                    result["extraction_prediction"] = None
                    result["true_false"] = True
                else:
                    result["extraction_prediction"] = None
                    result["true_false"] = False
                return result
        else:
            if succeed:
                result["extraction_prediction"] = short_answer
                result["true_false"] = short_answer.lower() in doc["ground_truth"].lower()
                return result
            else:
                result["extraction_prediction"] = None
                result["true_false"] = short_answer.lower() in doc["ground_truth"].lower()
                return result

    def eval_results(self, results):
        results_df = pd.DataFrame(results)
        score_avg = {}
        # 1. 计算overall_accuracy
        score_avg["overall_accuracy"] = results_df["true_false"].mean()
        # 2. 分科目统计，统计各个科目的 overaccuracy
        subs = set(results_df["subject"])
        for sub in subs:
            sub_df = results_df[results_df["subject"] == sub]
            score_avg[f"Subject-{sub}"] = sub_df["true_false"].mean()

        # 3. 分知识水平统计，统计各个知识水平的 overaccuracy
        lvls = set(results_df["knowledge_level"])
        for lvl in lvls:
            knowledge_level_df = results_df[results_df["knowledge_level"] == lvl]
            score_avg[f"Level-{lvl}"] = knowledge_level_df["true_false"].mean()

        # 创建一个字典，表示平均情况
        score = {"Setting": "Average"}  # 添加一个标识列
        score.update(score_avg)  # 将平均分数添加到字典中

        # results_df to json
        results = results_df.to_dict(orient="records")
        print(score)
        return results, score


"""print(score):
    {'Setting': 'Average', 
    'overall_accuracy': 0.59375,
    'Subject-plane geometry': 0.7272727272727273,
    'Subject-algebra': 1.0,
    'Subject-statistics': 0.42857142857142855,
    'Subject-arithmetic': 0.8333333333333334,
    'Subject-analytic geometry': 0.3333333333333333,
    'Subject-solid geometry': 0.0,
    'Level-high school': 0.5294117647058824,
    'Level-elementary school': 0.6666666666666666
    }
"""
