import base64
import json
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from typing import List, Optional, Tuple, Union

import requests
from accelerate import Accelerator, DistributedType
from openai import OpenAI
from PIL import Image
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

NUM_SECONDS_TO_SLEEP = 5

from loguru import logger

eval_logger = logger


API_URL = os.getenv("API_URL", "https://localhost:8000/v1")
API_KEY = os.getenv("API_KEY", "YOUR_API_KEY")


def extract_response(messages):
    response = ""
    messages = messages[2:]
    for m in messages:
        if isinstance(m["content"], str):
            response += m["content"]
        elif isinstance(m["content"], list):
            for content in m["content"]:
                if content["type"] == "text":
                    response += content["text"]
                elif content["type"] == "image_url":
                    response += "<IMAGE>"

    return response


def run_jupyter_code(cell_list, sandbox_url, upload_file_dict=None):
    response = requests.post(
        f"{sandbox_url}/run_jupyter",
        json={
            "cells": cell_list,
            "kernel": "python3",
            "files": upload_file_dict,
            "total_timeout": 600,
        },
    )
    output_cells = response.json().get("cells", [])
    if not output_cells:
        raise ValueError(f"No output cells returned from Jupyter execution. Cell List: {cell_list}")

    return output_cells


def parse_cell_output(cell_output: dict) -> dict:
    """Parse the output of a Jupyter cell."""
    if not cell_output:
        return {
            "text_output": "",
            "image_output": [],
        }
    stdout = cell_output.get("stdout", "")

    errors = cell_output.get("error", "")
    error_message = ""
    for e in errors:
        traceback = e.get("traceback", "")
        e_name = e.get("ename", "")
        e_value = e.get("evalue", "")
        error_message = f"Error: {e_name} - {e_value}\nTraceback: {traceback}"

    # show display output
    display_output = cell_output.get("display", [])
    display_text = ""
    display_image = []
    for cell_output_item in display_output:
        for key, value in cell_output_item.items():
            if key == "text/plain":
                display_text += value
            elif key == "image/png":
                display_image.append(f"data:image/png;base64,{value}")
            elif key == "image/jpeg":
                display_image.append(f"data:image/jpeg;base64,{value}")
            else:
                print(f"Unknown key: {key}")
    text_output = ""
    if stdout:
        text_output += f"stdout: {stdout}\n"
    if display_text:
        text_output += f"display text: {display_text}\n"
    if error_message:
        text_output += f"error: {error_message}\n"

    image_output = display_image
    return {
        "text_output": text_output.strip(),
        "image_output": image_output,
    }


def cell_output_to_content_list(cell_output: dict) -> list:
    text_output = cell_output["text_output"]
    image_output = cell_output["image_output"]
    content = [
        {"type": "text", "text": "<interpreter>" + text_output},
    ]
    for image in image_output:
        content.append({"type": "image_url", "image_url": {"url": image}})  # type: ignore
    content.append({"type": "text", "text": "</interpreter>"})
    return content


def get_tool_description() -> list:
    tool_description = [
        {
            "type": "function",
            "function": {
                "name": "run_python_code",
                "description": "Run the python code in jupyter environment.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "The code you need run",
                        }
                    },
                    "required": ["code"],
                },
            },
        }
    ]
    return tool_description

def get_tool_description_v2() -> list:
    tool_description = [
        {
            "type": "function",
            "function": {
                "name": "excute_python_code_in_jupyter",
                "description": """Execute Python code in a persistent Jupyter environment to solve a wide variety of problems. This powerful tool runs code and returns results and error information.

**Persistent Environment**: This is a stateful Jupyter notebook environment where:
- Variables and data structures persist between code executions
- Previously imported libraries remain available for reuse
- Functions and classes you define are remembered
- You can build upon previous computations step by step

Python code is incredibly versatile and can help you solve numerous types of problems:
1. **Mathematical & Scientific Computing**: Perform complex calculations, solve equations, statistical analysis, linear algebra operations using libraries like NumPy, SciPy, SymPy. If you are doing math question;
2. **Data Analysis & Visualization**: Process datasets, create charts and graphs, analyze trends using Pandas, Matplotlib, Plotly, Seaborn.
3. **Image Processing**: Load, manipulate, crop, rotate, enhance contrast, adjust brightness, apply filters, detect features using PIL. You can use img.show() to display results.
Don't limit your imagination - if there's a problem that can be solved computationally, Python likely has the tools to tackle it!
""",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "The Python code for a single Jupyter cell that you need to run",
                        }
                    },
                    "required": ["code"],
                },
            },
        }
    ]
    return tool_description

def get_tool_description_v3() -> list:
    tool_description = [
        {
            "type": "function",
            "function": {
                "name": "excute_python_code_in_jupyter",
                "description": "Execute Python code in a persistent Jupyter environment to solve a wide variety of problems. This powerful tool runs code and returns results and error information.\n\n**Persistent Environment**: This is a stateful Jupyter notebook environment where:\n- Variables and data structures persist between code executions\n- Previously imported libraries remain available for reuse\n- Functions and classes you define are remembered\n- You can build upon previous computations step by step\n- Commonly used packages such as matplotlib, scipy, pandas, and seaborn are already installed\n- You can get all output (including the image output) of jupyter cell. \n\nPython code is incredibly versatile and can help you solve numerous types of problems:\n1. **Mathematical & Scientific Computing**: Perform complex calculations, solve equations, statistical analysis, linear algebra operations using libraries like NumPy, SciPy, SymPy. If you are doing math question;\n2. **Data Analysis & Visualization**: Process datasets, create charts and graphs, analyze trends using Pandas, Matplotlib, Plotly, Seaborn.\n3. **Image Processing**: Load, manipulate, crop, rotate, enhance contrast, adjust brightness, apply filters, detect features using PIL. You can use img.show() to display results.",  # noqa: E501
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "The Python code for a single Jupyter cell that you need to run",
                        }
                    },
                    "required": ["code"],
                },
            },
        }
    ]
    return tool_description


def get_system_prompt() -> str:
    tool_description = get_tool_description_v3()
    system_message = f"""You are a helpful assistant.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{json.dumps(tool_description, ensure_ascii=False)}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>
"""

    return system_message


def get_upload_image_prompt(upload_img_paths: list | str) -> str:
    """Get the prompt for uploading an image."""
    if isinstance(upload_img_paths, str):
        upload_img_paths = [upload_img_paths]
    img_path_text = ""
    for i, img_path in enumerate(upload_img_paths):
        img_path_text += f'Picture {i} path: "{img_path}"\n'

    question_prefix = f"I have upload the following images:\n{img_path_text}\n\n"
    return question_prefix

def query_template(question: str, sandbox_image_path: list[str] | str, need_picture_text: bool = True):
    if isinstance(sandbox_image_path, str):
        sandbox_image_path = [sandbox_image_path]
    image_text = ''.join([f'Picture {i}: <image>\n' for i in range(len(sandbox_image_path))]) if need_picture_text else ""
    return f"""{get_upload_image_prompt(sandbox_image_path)}{image_text}
Now please answer the following question:
{question}
Please answer in the following format:
<think>...</think>
<answer>...</answer>""".strip()


def extract_code_from_response(response_text: str) -> str:
    """Extract code from the response."""
    if "<tool_call>" not in response_text or "</tool_call>" not in response_text:
        raise ValueError("Response does not contain a valid tool call.")

    code_json_str = response_text.split("<tool_call>")[1].split("</tool_call>")[0]
    try:
        code = json.loads(code_json_str)["arguments"]["code"]
    except (json.JSONDecodeError, KeyError):
        raise ValueError("Failed to extract code from the tool call.")
    return code


def call_tool(resposne: str, code_list: list, sandbox_url: str, upload_file_dict: dict) -> dict:
    """Call the tool with the extracted code."""

    try:
        code = extract_code_from_response(resposne)
    except Exception as e:
        return {"status": "error", "message": "Extract Code Error: " + str(e)}
    code_list.append(code)

    try:
        code_output = run_jupyter_code(code_list, sandbox_url, upload_file_dict=upload_file_dict)
    except Exception as e:
        return {"status": "error", "message": "Call Sandbox Error: " + str(e)}

    try:
        parsed_output = parse_cell_output(code_output[-1])
        code_output_text = cell_output_to_content_list(parsed_output)
    except Exception as e:
        return {"status": "error", "message": "Parse Cell Output Error: " + str(e)}

    return {
        "status": "success",
        "code_output_text": code_output_text,
        "code": code,
        "code_output": code_output,
        "code_list": code_list,
    }


@register_model("code_qwen2_5_vl")
class CodeQwen2_5_VL(lmms):
    def __init__(
        self,
        model_version: str = "Qwen2.5-VL-7B-Instruct",
        image_token: str = "<image>",  # Use to separate interleaved image and text
        system_prompt: str = "",  # Whether you want to add some special system prompt here
        sandbox_url: str = "",
        modality: str = "image",
        max_turn: int = 8,
        **kwargs,
    ) -> None:
        super().__init__()
        self.model_version = model_version
        self.image_token = image_token
        self.modality = modality
        self.sandbox_url = sandbox_url

        self.system_message = get_system_prompt()

        accelerator = Accelerator()
        print(accelerator.num_processes)
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.accelerator = accelerator
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes

        self.device = self.accelerator.device
        self.client = OpenAI(api_key=API_KEY, base_url=API_URL)
        print(API_KEY, API_URL)
        self.max_turn = max_turn

    def encode_image(self, image):
        if image.mode == "RGBA":
            image = image.convert("RGB")
        output_buffer = BytesIO()
        image.save(output_buffer, format="JPEG")
        byte_data = output_buffer.getvalue()
        base64_str = base64.b64encode(byte_data).decode("utf-8")
        return base64_str

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def get_image_size(self, image):
        # Create a BytesIO object to store the image bytes
        img_byte_array = BytesIO()

        # Save the image to the BytesIO object
        image.save(img_byte_array, format="PNG")

        # Get the size of the BytesIO object
        img_size = img_byte_array.tell()

        return img_size

    def request_completion(self, payload):
        response = self.client.chat.completions.create(**payload)
        return response.choices[0].message.content

    def retry_request(self, payload):
        MAX_RETRIES = 5
        response_text = ""
        for attempt in range(MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(**payload)
                response_text = response.choices[0].message.content
                break  # If successful, break out of the loop

            except Exception as e:
                error_msg = str(e)
                eval_logger.info(f"Attempt {attempt + 1}/{MAX_RETRIES} failed with error: {error_msg}")

                # On last attempt, log error and set empty response
                if attempt == MAX_RETRIES - 1:
                    eval_logger.error(f"All {MAX_RETRIES} attempts failed. Last error: {error_msg}")
                else:
                    time.sleep(NUM_SECONDS_TO_SLEEP)
        return response_text

    def process_request(self, model_name, messages, upload_image_dict, gen_kwargs):
        if gen_kwargs is None:
            gen_kwargs = {}

        code_list = []
        response_num = 0

        while response_num < self.max_turn:
            params = {
                "model": model_name,
                "messages": messages,
                **gen_kwargs,
            }
            response = self.client.chat.completions.create(**params)
            response_text = response.choices[0].message.content
            if "<tool_call>" in response_text and "</tool_call>" in response_text:
                tool_out = call_tool(response_text, code_list, self.sandbox_url, upload_image_dict)
                if tool_out["status"] == "error":
                    messages.extend(
                        [
                            {"role": "assistant", "content": response_text},
                            {
                                "role": "user",
                                "content": f"Error: {tool_out['message']}",
                            },
                        ]
                    )

                else:
                    code_list = tool_out["code_list"]
                    code_output_text = tool_out["code_output_text"]
                    messages.extend(
                        [
                            {"role": "assistant", "content": response_text},
                            {"role": "user", "content": code_output_text},
                        ]
                    )

                if response_num == self.max_turn - 2:
                    messages[-1]["content"].append(
                        {
                            "type": "text",
                            "text": "You have reached the maximum number of tool calls. Please provide a final response.",
                        }
                    )
            else:
                messages.append({"role": "assistant", "content": response_text})
                break
            response_num += 1
        return messages

    def _process_single_request(self, request_args):
        """Process a single request - helper method for multithreading"""
        """Instance(request_type='generate_until', arguments=('Please first conduct reasoning, and then answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.\nQuestion: In the diagram above, angle A is congruent to angle BED, and angle C is congruent to angle D. If the ratio of the length of AB to the length of EB is 5:1, and the area of the triangle BED is 5*a^2 + 10, what is the area of triangle ABC?\nChoices:\nA.5*a^2 + 10\nB.25*a^2 + 50\nC.25*a^2 + 100\nD.125*a^2 + 250\nE.cannot be determined', {'max_new_tokens': 4096, 'temperature': 0.0, 'repetition_penalty': 1.0, 'until': ['\n\n']}, <bound method ConfigurableTask.doc_to_visual of ConfigurableTask(task_name=mathverse_testmini,output_type=generate_until,num_fewshot=0,num_samples=3940)>, 3938, 'mathverse_testmini', 'testmini'), idx=0, metadata={'task': 'mathverse_testmini', 'doc_id': 3938, 'repeats': 1, 'split': 'testmini'}, resps=[], filtered_resps={}, task_name='mathverse_testmini', doc_id=3938, repeats=1, doc=None)
        """
        contexts, gen_kwargs, doc_to_visual, doc_id, task, split = request_args

        visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
        img_paths = []
        if None in visuals:
            visuals = []
            imgs = []
        else:
            visuals = self.flatten(visuals)
            imgs = []  # multiple images or frames for video
            for visual in visuals:
                if isinstance(visual, str) and (".mp4" in visual or ".avi" in visual or ".mov" in visual or ".flv" in visual or ".wmv" in visual):
                    raise NotImplementedError("Video input is not supported yet.")
                    frames = self.encode_video(visual, self.max_frames_num)
                    imgs.extend(frames)
                elif isinstance(visual, str) and (".jpg" in visual or ".jpeg" in visual or ".png" in visual or ".gif" in visual or ".bmp" in visual or ".tiff" in visual or ".webp" in visual):
                    img = self.encode_image(visual)
                    imgs.append(img)
                    img_paths.append(f"/mnt/data/{str(uuid.uuid4())}.jpg")
                elif isinstance(visual, Image.Image):
                    img = self.encode_image(visual)
                    imgs.append(img)
                    img_paths.append(f"/mnt/data/{str(uuid.uuid4())}.jpg")

        message = [{"role": "system", "content": self.system_message}]
        
        user_text = query_template(question=contexts, sandbox_image_path=img_paths)
        user_content = [{"type": "text", "text": user_text}]
        for i, img in enumerate(imgs):
            user_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}})
        message.append({"role": "user", "content": user_content})  # type: ignore

        gen_kwargs = {
            "temperature": 0.0,
            "max_tokens": 10240,
            "top_p": 0.9,
            "stop": ["<|im_end|>"],
            "extra_body": {
                "include_stop_str_in_output": True,
                "repetition_penalty": 1.05,
            },
        }
        upload_image_dict = {k: v for k, v in zip(img_paths, imgs)}
        output_messages = self.process_request(self.model_version, message, upload_image_dict=upload_image_dict, gen_kwargs=gen_kwargs)
        final_response = extract_response(output_messages)
        return final_response

    def generate_until(self, requests) -> List[str]:
        # Use multithreading for I/O-bound operations
        max_workers = min(32, len(requests))  # Limit concurrent requests
        request_args_list = [reg.args for reg in requests]

        results: List[str] = [""] * len(requests)  # Maintain order
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all requests
            future_to_index = {executor.submit(self._process_single_request, request_args): i for i, request_args in enumerate(request_args_list)}

            # Collect results as they complete
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result = future.result()
                    results[index] = result
                except Exception as e:
                    eval_logger.error(f"Request {index} failed with error: {e}")
                    results[index] = ""  # Default empty response for failed requests
                pbar.update(1)

        pbar.close()
        return results

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        assert False, "Not supported for claude"

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation for Claude")
