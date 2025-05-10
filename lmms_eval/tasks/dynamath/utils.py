import json
import os
import time
from pathlib import Path

import pandas as pd
import yaml
from loguru import logger as eval_logger

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file
from lmms_eval.tasks.dynamath.dynamath_evals import DynaMathEvaluator

dynamath_evaluator = DynaMathEvaluator(api_key=os.getenv("OPENAI_API_KEY", "YOUR_API_KEY"), gpt_model=os.getenv("MODEL_VERSION", "gpt-4o-mini"))


def get_timestamp():
    nowtime = time.strftime("%Y%m%d-%H%M", time.localtime(time.time()))
    return nowtime


with open(Path(__file__).parent / "dynamath.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

    config = yaml.safe_load("".join(safe_data))


def dynamath_doc_to_visual(doc):
    return [doc["decoded_image"].convert("RGB")]


def dynamath_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    GUIDE = """
## Answer Instruction Please provide an answer to the question outlined above. Your response should adhere \
to the following JSON format, which includes two keys: 'solution' and 'short answer'. The 'solution' key can contain \
detailed steps needed to solve the question, and the 'short answer' key should provide a concise response. {INST}

Example of expected JSON response format:

"""
    EXAMPLE = {"solution": "[Detailed step-by-step explanation]", "short answer": "[Concise Answer]"}
    TEXT_EXAMPLE = json.dumps(EXAMPLE, indent=4)
    query_prompt = f"## Question\n {doc['question']}"
    if doc["answer_type"] == "multiple choice":
        inst = "Provide the corresponing choice option in the 'short answer' key, such as 'A', 'B', 'C', or 'D'."
    elif doc["answer_type"] == "float":
        inst = "Format the answer as a three-digit floating-point number and provide it in the 'short answer' key."
    else:
        inst = "Float numbers in the answer should be formatted as three-digit floating-point numbers."
    if lmms_eval_specific_kwargs is not None:
        if lmms_eval_specific_kwargs.get("use_guide_prompt", False):
            query_prompt = query_prompt + GUIDE.format(INST=inst) + TEXT_EXAMPLE
    return query_prompt


# process 里面判断样本的answer和prediction是不是一致的
def dynamath_process_results(doc, results):
    prediction = results[0].strip()
    result = dynamath_evaluator.judge_answer(prediction, doc)
    return {
        "gpt_eval_score": {
            "prediction": prediction,
            "extraction_prediction": result["extraction_prediction"],
            "true_false": result["true_false"],
            "subject": doc["subject"],
            "knowledge_level": doc["knowledge_level"],
            "answer_type": doc["answer_type"],
            "answer": doc["ground_truth"] if "ground_truth" in doc else None,
            "question": doc["question"],
            "id": doc["id"],
            "varid": doc["varid"],
            "qid": doc["qid"],
            "gpt_extract_answer": result["gpt_extract_answer"],
            "index": doc["index"],
        }
    }


# aggregate里面计算总的指标
def dynamath_aggregate_results_eval(results, args):
    timestamp = get_timestamp()
    path = generate_submission_file(f"dynamath_{timestamp}_results.json", args)
    with open(path, "w") as f:
        json.dump(results, f, indent=4)

    # evaluation
    _, scores = dynamath_evaluator.eval_results(results)

    # save scores
    path = generate_submission_file(f"dymamath_{timestamp}_scores.json", args)
    with open(path, "w") as f:
        json.dump(scores, f, indent=4)

    eval_logger.info(f"Saved scores to {path}")
    if scores["overall_accuracy"] == 0:
        return None
    return scores["overall_accuracy"]
