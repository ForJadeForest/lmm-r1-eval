# Standard library imports
import base64
import json
import os
import time
from copy import deepcopy
from io import BytesIO
import hashlib
import numpy as np
import requests as url_requests

# Related third-party imports
from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.state import AcceleratorState
from loguru import logger as eval_logger
from vllm import LLM, SamplingParams
from PIL import Image
from tqdm import tqdm

# Local application/library specific imports
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

# Conditional imports
try:
    from decord import VideoReader, cpu
except ImportError:
    eval_logger.warning("Decord is not installed. Video input will not be supported.")


@register_model("vllm")
class VLLM(lmms):
    def __init__(
        self,
        model_version: str = "Qwen/Qwen2-VL-2B-Instruct",
        modality: str = "image",
        max_frames_num: int = 10,
        batch_size:int = None,
        limit_img_num:int = 1,
        **kwargs,
    ) -> None:
        super().__init__()
        # Manually set a image token for GPT4V so that we can search for it
        # and split the text and image
        # Here we just use the same token as llava for convenient
        self.model_version = model_version
        self.modality = modality
        self.max_frames_num = max_frames_num
        self.image_token = "<image>"

        self.model = LLM(model=model_version,trust_remote_code=True,limit_mm_per_prompt={"image": limit_img_num},**kwargs)

        accelerator = Accelerator()
        assert accelerator.state.local_process_index == 0, "VLLM does not support distributed inference."
        assert accelerator.state.num_processes == 1, "VLLM does not support distributed inference."

    # Function to encode the image
    def encode_image(self, image: Image):
        output_buffer = BytesIO()
        image.save(output_buffer, format="PNG")
        byte_data = output_buffer.getvalue()
        base64_str = base64.b64encode(byte_data).decode("utf-8")
        return base64_str

    # Function to encode the video
    def encode_video(self, video_path, for_get_frames_num):
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frame_num = len(vr)
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, for_get_frames_num, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frames = vr.get_batch(frame_idx).asnumpy()

        base64_frames = []
        for frame in frames:
            img = Image.fromarray(frame)
            output_buffer = BytesIO()
            img.save(output_buffer, format="PNG")
            byte_data = output_buffer.getvalue()
            base64_str = base64.b64encode(byte_data).decode("utf-8")
            base64_frames.append(base64_str)

        return base64_frames

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests):
        # Prepare the batch requests data
        requests_data = []
        md5 = hashlib.md5()
        md5.update(str(requests).encode())
        file_path = os.getenv("HF_HOME", "~/.cache/huggingface") + f"/batchinput_{md5.hexdigest()}.jsonl"
        if os.path.exists(file_path):
            #load jsonl
            eval_logger.info(f"Loading batch input file cache from {file_path}")
            with open(file_path, "r") as file:
                for line in file.readlines():
                    requests_data.append(json.loads(line))
        else:
            pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Batch Preparing")
            for idx, (contexts, gen_kwargs, doc_to_visual, doc_id, task, split) in enumerate([reg.args for reg in requests]):
                visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
                visuals = self.flatten(visuals)
                imgs = []
                for visual in visuals:
                    if self.modality == "image":
                        img = self.encode_image(visual)
                        imgs.append(img)
                    elif self.modality == "video":
                        frames = self.encode_video(visual, self.max_frames_num)
                        imgs.extend(frames)

                content = []
                if self.image_token not in contexts:
                    content.append({"type": "text", "text": contexts})
                    for img in imgs:
                        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}})
                else:
                    contexts_split = contexts.split(self.image_token)
                    for idx, context in enumerate(contexts_split):
                        if idx < len(imgs):
                            content.append({"type": "text", "text": context})
                            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{imgs[idx]}"}})
                    if len(contexts_split) > len(imgs):
                        content.append({"type": "text", "text": contexts_split[-1]})

                messages = [{"role": "user", "content": content}]
                requests_data.append(messages)
                pbar.update(1)
            #save jsonl
            eval_logger.info(f"Saving batch input file to {file_path}")
            with open(file_path, "w") as file:
                for request_data in requests_data:
                    file.write(json.dumps(request_data) + "\n")


        batch_response = self.create_batch(requests_data)
        return batch_response
        

    def loglikelihood(self, requests):
        # TODO
        assert False, "VLLM not support"


    def create_batch(self, requests_data):
        sampling_params =SamplingParams(temperature=0,max_tokens=4096)
        responses = self.model.chat(requests_data,sampling_params=sampling_params)
        return [r.outputs[0].text for r in responses]


    def generate_until_multi_round(self, requests) -> list[str]:
        raise NotImplementedError("TODO: Implement multi-round generation for BatchGPT4")