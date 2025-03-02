import datetime
import json
import os
from collections import defaultdict
import requests
from loguru import logger as eval_logger
import time

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file

dir_name = os.path.dirname(os.path.abspath(__file__))

eval_type_dict = {
    "coarse perception": ["image scene and topic", "image style & quality", "image emotion"],
    "fine-grained perception": ["object counting", "recognition", "localization"],
    "instance reasoning": ["single-instance reasoning", "cross-instance attribute reasoning", "cross-instance relation reasoning"],
    "logical reasoning": ["code & sequence reasoning", "diagram reasoning", "common reasoning"],
    "science & technology": ["biology & chemistry & physics", "electronics & energy & mechanical eng.", "geography & earth science & agriculture"],
    "math": ["geometry", "numeric commonsense and calculation", "statistical reasoning"],
}


replace_prompt = " Please answer yes or no."

DEMO_PROMPT = """
Please read the following example. Then extract the answer from the model response and type it at the end of the prompt.

Hint: Please answer the question requiring an integer answer and provide the final value, e.g., 1, 2, 3, at the end.
Question: Which number is missing?

Model response: The number missing in the sequence is 14.

Extracted answer: 14

Hint: Please answer the question requiring a floating-point number with one decimal place and provide the final value, e.g., 1.2, 1.3, 1.4, at the end.
Question: What is the fraction of females facing the camera?

Model response: The fraction of females facing the camera is 0.6, which means that six out of ten females in the group are facing the camera.

Extracted answer: 0.6

Hint: Please answer the question requiring a floating-point number with two decimal places and provide the final value, e.g., 1.23, 1.34, 1.45, at the end.
Question: How much money does Luca need to buy a sour apple candy and a butterscotch candy? (Unit: $)

Model response: Luca needs $1.45 to buy a sour apple candy and a butterscotch candy.

Extracted answer: 1.45

Hint: Please answer the question requiring a Python list as an answer and provide the final list, e.g., [1, 2, 3], [1.2, 1.3, 1.4], at the end.
Question: Between which two years does the line  graph saw its maximum peak?

Model response: The line graph saw its maximum peak between 2007 and 2008.

Extracted answer: [2007, 2008]

Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.
Question: What fraction of the shape is blue?\nChoices:\n(A) 3/11\n(B) 8/11\n(C) 6/11\n(D) 3/5

Model response: The correct answer is (B) 8/11.

Extracted answer: B

Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.
Question: {question}

Model response: {response}

Extracted answer: 
"""


def mmstar_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def mmstar_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"].strip()
    if "pre_prompt" in lmms_eval_specific_kwargs and lmms_eval_specific_kwargs["pre_prompt"] != "":
        question = question.replace(replace_prompt, "")
        question = f"{lmms_eval_specific_kwargs['pre_prompt']}{question}"
    if "post_prompt" in lmms_eval_specific_kwargs and lmms_eval_specific_kwargs["post_prompt"] != "":
        question = question.replace(replace_prompt, "")
        question = f"{question}{lmms_eval_specific_kwargs['post_prompt']}"
    return question


def get_chat_response(prompt, temperature=0, max_tokens=256, n=1, patience=10000000, sleep_time=0,extra_parameters=None):
    API_KEY = os.getenv("OPENAI_API_KEY","xxx")
    API_URL = os.getenv("OPENAI_API_URL","https://api.openai.com/v1/engines/davinci-codex/completions")
    API_MODEL = os.getenv("OPENAI_API_MODEL","gpt-3.5-turbo")
    def _post_request(payload):
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        }
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()

    messages = [
        {"role": "user", "content": prompt},
    ]
    payload = {"model": API_MODEL, "messages": messages, "temperature": temperature, "max_tokens": max_tokens, "n": n}
    if extra_parameters:
        payload.update(extra_parameters)
    while patience > 0:
        patience -= 1
        try:
            response = _post_request(payload)
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

def exact_match(pred, gt,question):
    """Brought from MMStar"""
    answer = gt.lower().strip().replace("\n", " ")
    response = pred.lower().strip().replace("\n", " ")
    extract_prompt = DEMO_PROMPT.format(question=question, response=response)
    predict = get_chat_response(extract_prompt, temperature=0.6, max_tokens=8, n=1,extra_parameters={"guided_choice":["A","B","C","D","E","F","G","H","I","J"]})
    predict = predict.lower()
    try:
        if answer == predict[0]:
            return 1.0
        elif predict[0] == "(" and answer == predict[1]:
            return 1.0
        elif predict[0:7] == "option " and answer == predict[7]:
            return 1.0
        elif predict[0:14] == "the answer is " and answer == predict[14]:
            return 1.0
    except Exception as e:
        return 0.0
    return 0.0


def mmstar_process_results(doc, results):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name, value: metric value
    """
    pred = results[0]
    gt = doc["answer"]

    score = exact_match(pred, gt, doc["question"])
    category = doc["category"]
    l2_category = doc["l2_category"]
    return {category: {"question_id": doc["index"], "l2_category": l2_category, "score": score}, "average": {"question_id": doc["index"], "l2_category": l2_category, "score": score}}


def mmstar_aggregate_results(results):
    """
    Args:
        results: a list of values returned by process_results
    Returns:
        A score
    """
    l2_category_scores = defaultdict(list)
    for result in results:
        score = result["score"]
        l2_category = result["l2_category"]
        l2_category_scores[l2_category].append(score)

    l2_category_avg_score = {}
    for l2_category, scores in l2_category_scores.items():
        avg_score = sum(scores) / len(scores)
        l2_category_avg_score[l2_category] = avg_score
        eval_logger.info(f"{l2_category}: {avg_score:.2f}")

    avg_score = sum(l2_category_avg_score.values()) / len(l2_category_avg_score)
    return avg_score
