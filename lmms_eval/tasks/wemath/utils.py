import json
import os
import time
from pathlib import Path

from loguru import logger as eval_logger

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file
from lmms_eval.tasks.wemath.wemath_evaluator import WeMathEvaluator

wemath_evaluator = WeMathEvaluator(api_key=os.getenv("OPENAI_API_KEY", "YOUR_API_KEY"), gpt_model=os.getenv("MODEL_VERSION", "gpt-4o-mini"))


def get_timestamp():
    nowtime = time.strftime("%Y%m%d-%H%M", time.localtime(time.time()))
    return nowtime


def wemath_doc_to_visual(doc):
    if doc["image_path"] is None:
        return []
    return [doc["image_path"]]


def wemath_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"]
    options = doc["option"]
    hint = "Now, we require you to solve a multiple-choice math question. Please briefly describe your thought process and provide the final answer(option)."

    requirements = """Regarding the format, please answer following the template below, and be sure to include two <> symbols:
<Thought process>: <<your thought process>> <Answer>: <<your option>>
    """.strip()

    prompt = f"Hint: {hint}\n"
    prompt += f"Question: {question}\n"

    prompt += f"Options:\n{options}\n"
    if lmms_eval_specific_kwargs is not None:
        if lmms_eval_specific_kwargs.get("use_requirements", False):
            prompt += f"{requirements}\n"

    return prompt


def wemath_process_results(doc, results):
    prediction = results[0].strip()
    result = wemath_evaluator.judge_answer(prediction, doc["option"], doc["answer"], doc["question"])
    return {
        "gpt_eval_score": {
            "prediction": prediction,
            "extraction_prediction": result["extraction_prediction"],
            "true_false": result["true_false"],
            "key": doc["key"],
            "knowledge concept": doc["knowledge concept"],
            "ID": doc["ID"],
        }
    }


def wemath_aggregate_results_eval(results, args):
    timestamp = get_timestamp()
    path = generate_submission_file(f"wemath_{timestamp}_results.json", args)
    with open(path, "w") as f:
        json.dump(results, f, indent=4)

    # gpt evaluation
    _, scores = wemath_evaluator.eval_results(results)

    # save scores
    path = generate_submission_file(f"wemath_{timestamp}_scores.json", args)
    with open(path, "w") as f:
        json.dump(scores, f, indent=4)

    eval_logger.info(f"Saved scores to {path}")
    if scores["overall_accuracy"] == 0:
        return None
    return scores["overall_accuracy"]
