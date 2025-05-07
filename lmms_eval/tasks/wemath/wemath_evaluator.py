import copy
import os
import string
import time

import pandas as pd
import requests
from loguru import logger as eval_logger

from lmms_eval.tasks.wemath.wemath_utils import wemath_accuracy, wemath_evaluate_models


def build_prompt_wemath(question, options, prediction):
    tmpl = (
        "You are an AI assistant who will help me to match "
        "an answer with several options of a single-choice question. "
        "You are provided with a question, several options, and an answer, "
        "and you need to find which option is most similar to the answer. "
        "If the meaning of all options are significantly different from the answer, output Z. "
        "Your should output a single uppercase character in A, B, C, D, E, F, G (if they are valid options), and Z. \n"
        "Example 1: \n"
        "Question: <start>\nWhat is the main object in image?\nOptions: A. teddy bear B. rabbit C. cat D. dog\n<end>\n"
        "Answer: <start>\na cute teddy bear\n<end>\nYour output: A\n"
        "Example 2: \n"
        "Question: <start>\nWhat is the main object in image?\nOptions: A. teddy bear B. rabbit C. cat D. dog\n<end>\n"
        "Answer: <start>\nSpider\n<end>\nYour output: Z\n"
        "Example 3: \n"
        "Question: <start>\n{}\nOptions: {}\n<end>\nAnswer: <start>\n{}\n<end>\nYour output: "
    )
    question = question.replace(
        ("Regarding the format, please answer following the template below, and be sure to include two <> symbols:\n" "<Thought process>: <<your thought process>> <Answer>: <<your option>>"),
        "",
    )
    return tmpl.format(question, options, prediction)


def build_choices(option_text):
    # Special case handling for known formats
    if option_text == "A. ①; ③; B. ②; ③; C. ②; ④; D. ①; ④; E. No correct answer":
        return {"A": "①; ③", "B": "②; ③", "C": "②; ④", "D": "①; ④", "E": "No correct answer"}

    ret = {}
    # Replace Chinese full-width period with English period
    options_list = option_text.replace("．", ".").split("; ")

    for option in options_list:
        # Only handle the first period (the one after the option letter)
        first_dot_idx = option.find(".")
        if first_dot_idx != -1:
            letter = option[:first_dot_idx].strip()
            content = option[first_dot_idx + 1 :].strip()

            if letter in "ABCDEFGH":
                # Replace special characters if needed
                ret[letter] = content
            else:
                raise ValueError(f"Invalid option letter: {letter}")
        else:
            raise ValueError(f"Invalid option format: {option}")
    return ret


def can_infer_option(answer, choices):
    verbose = os.environ.get("VERBOSE", 0)
    # Choices is a dictionary
    if "Failed to obtain answer via API" in answer:
        return False

    reject_to_answer = ["Sorry, I can't help with images of people yet.", "I can't process this file.", "I'm sorry, but without the image provided", "Cannot determine the answer"]
    for err in reject_to_answer:
        if err in answer:
            return "Z"

    def count_choice(splits, choices, prefix="", suffix=""):
        cnt = 0
        for c in choices:
            if prefix + c + suffix in splits:
                cnt += 1
        return cnt

    answer_mod = copy.copy(answer)
    chars = ".()[],:;!*#{}"
    for c in chars:
        answer_mod = answer_mod.replace(c, " ")

    splits = [x.strip() for x in answer_mod.split()]
    count = count_choice(splits, choices)

    if count == 1:
        for ch in choices:
            if "A" in splits and len(splits) > 3 and verbose:
                eval_logger.info(f"A might be a quantifier in the string: {answer}.")
                return False
            if ch in splits:
                return ch
    elif count == 0 and count_choice(splits, {"Z", ""}) == 1:
        return "Z"
    return False


def can_infer_text(answer, choices):
    answer = answer.lower()
    assert isinstance(choices, dict)
    for k in choices:
        assert k in string.ascii_uppercase
        choices[k] = str(choices[k]).lower()
    cands = []
    for k in choices:
        if choices[k] in answer:
            cands.append(k)
    if len(cands) == 1:
        return cands[0]
    return False


def can_infer(answer, choices):
    answer = str(answer)
    copt = can_infer_option(answer, choices)
    return copt if copt else can_infer_text(answer, choices)


class WeMathEvaluator:
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

    def get_chat_response(self, prompt, temperature=0, max_tokens=256, n=1, patience=10000000, sleep_time=0):
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

    def judge_answer(self, prediction, option_text, answer, question):
        choices = build_choices(option_text)
        extraction_prediction = can_infer(prediction, choices)
        result = {}
        if extraction_prediction:
            result["extraction_prediction"] = extraction_prediction
            result["true_false"] = extraction_prediction == answer
            return result
        retry = 0
        prompt = build_prompt_wemath(question, option_text, prediction)
        while retry < 5:
            extract_prediction = self.get_chat_response(prompt)
            extract_prediction = can_infer(extract_prediction, choices)
            if extract_prediction:
                result["extraction_prediction"] = extract_prediction
                result["true_false"] = extract_prediction == answer
                return result
            retry += 1
            eval_logger.info(f"Retry {retry} for response: {extract_prediction} with input: {prompt}")
        return result

    def eval_results(self, results):
        results_df = pd.DataFrame(results)
        overall_accuracy = results_df["true_false"].mean()
        accuracy_scores = wemath_evaluate_models(results_df)
        four_dim_scores = wemath_accuracy(results_df)
        combine_score = pd.DataFrame({**accuracy_scores, **four_dim_scores, "overall_accuracy": overall_accuracy})
        combine_score = combine_score.iloc[0].to_dict()
        # results_df to json
        results = results_df.to_dict(orient="records")
        return results, combine_score
