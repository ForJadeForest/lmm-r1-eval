import os
import re
from copy import deepcopy
from io import BytesIO
from multiprocessing import cpu_count
from typing import List, Optional, Tuple, Union

import numpy as np
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoConfig, AutoProcessor

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

try:
    from vllm import LLM, SamplingParams
except ImportError:
    vllm = None


def get_visual_content(visuals):
    visual_content = []
    for visual in visuals:
        if isinstance(visual, str) and (".mp4" in visual or ".avi" in visual or ".mov" in visual or ".flv" in visual or ".wmv" in visual):
            visual_content.append({"type": "video", "video": visual})
        elif isinstance(visual, Image.Image):
            visual_content.append({"type": "image", "image": visual})
    return visual_content


@register_model("qwen2_5_vl_vllm")
class Qwen2_5_VL_VLLM(lmms):
    def __init__(
        self,
        model_version: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.8,
        batch_size: int = 1,
        timeout: int = 60,
        max_images: int = 32,
        max_videos: int = 8,
        max_audios: int = 8,
        max_frame_num: int = 32,
        threads: int = 16,  # Threads to use for decoding visuals
        trust_remote_code: Optional[bool] = True,
        system_prompt: Optional[str] = "You are a helpful assistant.",
        extract_answer: Optional[bool] = False,
        place_visual_first: Optional[bool] = False,
        min_pixels: Optional[int] = 4 * 28 * 28,
        max_pixels: Optional[int] = 16384 * 28 * 28,
        **kwargs,
    ) -> None:
        super().__init__()
        # Manually set a image token for GPT4V so that we can search for it
        # and split the text and image
        # Here we just use the same token as llava for convenient
        self.model_version = model_version
        self.max_images = max_images
        self.max_frame_num = max_frame_num
        self.threads = threads
        self.place_visual_first = place_visual_first

        init_params = ["model_version", "tensor_parallel_size", "gpu_memory_utilization", "batch_size", "timeout", "max_images", "max_videos", "max_audios", "max_frame_num", "threads", "trust_remote_code", "place_visual_first"]

        # filter out the parameters already defined in __init__ to pass options to VLLM
        # this enables support for all VLLM Engine args:
        # https://github.com/vllm-project/vllm/blob/3147586ebdb36ceae653e9dceec8cf9922fe2c28/vllm/engine/arg_utils.py#L93
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in init_params}

        accelerator = Accelerator()
        self.client = LLM(
            model=self.model_version,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            limit_mm_per_prompt={"image": max_images, "video": max_videos, "audio": max_audios},
            trust_remote_code=trust_remote_code,
            **filtered_kwargs,
        )
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
        self.batch_size_per_gpu = int(batch_size)
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.system_prompt = system_prompt
        self.enable_extract_answer = extract_answer
        # if system_prompt is a path, read the file
        if os.path.exists(system_prompt):
            with open(system_prompt, "r") as f:
                self.system_prompt = f.read()
        else:
            self.system_prompt = system_prompt

        # Initialize processor for chat template application
        self.processor = AutoProcessor.from_pretrained(model_version)
        self.tokenizer = self.processor.tokenizer

    def extract_answer(self, response: str) -> str:
        """
        Extract the answer from the response with r1 format.
        <answer>xxx</answer>
        Extract xxx from the pattern.
        """
        if self.enable_extract_answer:
            answer_pattern = r"<answer>(.*?)</answer>"
            match = re.search(answer_pattern, response)
            if match:
                return match.group(1).strip()
        return response

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        batch_size = self.batch_size_per_gpu
        batched_requests = [requests[i : i + batch_size] for i in range(0, len(requests), batch_size)]

        for batch_requests in batched_requests:
            inputs = []
            for idx in tqdm(range(len(batch_requests)), disable=(self.rank != 0), desc="Processing batch"):
                contexts, gen_kwargs, doc_to_visual, doc_id, task, split = batch_requests[idx].arguments

                visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
                visuals = [] if None in visuals else self.flatten(visuals)

                # Create messages in the format expected by the model
                messages = [{"role": "system", "content": self.system_prompt}] if self.system_prompt else []
                messages.append({"role": "user", "content": []})
                # If is img, append type: image
                # If is video, append type: video
                # Arrange content based on place_image_first parameter

                visual_content = get_visual_content(visuals)
                if self.place_visual_first:
                    # Place images before text
                    messages[-1]["content"].extend(visual_content)
                    if contexts:
                        messages[-1]["content"].append({"type": "text", "text": contexts})
                else:
                    if contexts:
                        messages[-1]["content"].append({"type": "text", "text": contexts})
                    messages[-1]["content"].extend(visual_content)

                # Apply chat template to convert messages to prompt format
                prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                # Use process_vision_info to extract image data
                image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
                mm_data = {}
                if image_inputs is not None:
                    mm_data["image"] = image_inputs
                if video_inputs is not None:
                    mm_data["video"] = video_inputs
                mm_processor_kwargs = {
                    **video_kwargs,
                    "min_pixels": self.min_pixels,
                    "max_pixels": self.max_pixels,
                }
                inputs.append(
                    {
                        "prompt": prompt,
                        "multi_modal_data": mm_data,
                        "mm_processor_kwargs": mm_processor_kwargs,
                    }
                )

            # Debug print for the first message in batch
            if self.rank == 0 and len(inputs) > 0:
                demo_input = deepcopy(inputs[0])
                print(f"Demo input: {demo_input}")

            # Create sampling parameters
            sample_params = {}
            if "max_new_tokens" in gen_kwargs:
                sample_params["max_tokens"] = gen_kwargs.pop("max_new_tokens")
            if "temperature" in gen_kwargs:
                sample_params["temperature"] = gen_kwargs.pop("temperature")
            if "top_p" in gen_kwargs:
                sample_params["top_p"] = int(gen_kwargs.pop("top_p"))
            if "top_k" in gen_kwargs:
                sample_params["top_k"] = int(gen_kwargs.pop("top_k"))
            if "repetition_penalty" in gen_kwargs:
                sample_params["repetition_penalty"] = gen_kwargs.pop("repetition_penalty")

            sampling_params = SamplingParams(**sample_params)
            if self.rank == 0:
                print(f"Sampling params: {sampling_params}")

            # Call generate instead of chat
            model_outputs = self.client.generate(inputs, sampling_params=sampling_params)

            # Extract text from outputs
            response_text = [output.outputs[0].text for output in model_outputs]
            response_text = [self.extract_answer(text) for text in response_text]

            assert len(response_text) == len(batch_requests)
            res.extend(response_text)
            pbar.update(len(batch_requests))

        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        # TODO
        assert False, "GPT4V not support"

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")
