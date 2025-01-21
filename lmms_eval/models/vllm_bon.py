# Standard library imports
import base64
from io import BytesIO
import numpy as np
import torch

# Related third-party imports
from accelerate import Accelerator
from loguru import logger as eval_logger
from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import PoolerConfig

import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from PIL import Image
from tqdm import tqdm
from lmms_eval.models.model_utils.qwen_vl.vision_process import process_vision_info
from lmms_eval.models.model_utils.qwen_vl.vision_process import smart_resize as qwenvl_smart_resize

# Local application/library specific imports
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

from transformers import AutoProcessor, AutoConfig

# Conditional imports
try:
    from decord import VideoReader, cpu
except ImportError:
    eval_logger.warning("Decord is not installed. Video input will not be supported.")


@ray.remote(num_gpus=1, num_cpus=2)
class LLMActor:
    def __init__(self, *args, **kwargs):
        # import os
        # import torch
        # Get the GPU IDs assigned to this actor by Ray
        # gpu_ids = ray.get_gpu_ids()
        # Set CUDA_VISIBLE_DEVICES to limit the GPUs visible to this process
        # os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(int(gpu_id)) for gpu_id in gpu_ids)
        # Set the default CUDA device
        # torch.cuda.set_device(0)  # Since only one GPU is visible, it's cuda:0
        # Initialize the LLM model
        self.llm = LLM(*args, **kwargs)  # Use cuda:0 since only one GPU is visibl

    def generate(self, *args, **kwargs):
        return self.llm.generate(*args, **kwargs)

    def chat(self, *args, **kwargs):
        return self.llm.chat(*args, **kwargs)

    def encode(self, *args, **kwargs):
        return self.llm.encode(*args, **kwargs)


@register_model("vllm_bon")
class VLLMForBoN(lmms):
    def __init__(
        self,
        model_version: str = "Qwen/Qwen2-VL-2B-Instruct",
        rm_version: str = "Qwen/Qwen2-VL-2B-Instruct",
        rm_head: str = "Qwen/Qwen2-VL-2B-Instruct/rm_head.pt",
        n: int = 8,
        mp: int = 1,
        modality: str = "image",
        max_frames_num: int = 10,
        batch_size: int = None,
        limit_img_num: int = 1,
        **kwargs,
    ) -> None:
        super().__init__()
        # Manually set a image token for GPT4V so that we can search for it
        # and split the text and image
        # Here we just use the same token as llava for convenient
        self.model_version = model_version
        self.modality = modality
        assert modality == "image", "VLLM only supports image modality for now."
        self.max_frames_num = max_frames_num
        self.image_token = "<image>"
        self.n = n
        config = AutoConfig.from_pretrained(model_version)
        self.model_type = config.model_type
        self.mp = mp

        assert self.n % self.mp == 0, "n must be divisible by mp"
        # Initialize Ray
        ray.init()
        # Create a placement group with one GPU and one CPU per bundle
        model_pg = placement_group(name="llm_pg", bundles=[{"GPU": 1, "CPU": 4} for _ in range(self.mp)], strategy="STRICT_PACK")
        rm_pg = placement_group(name="rm_pg", bundles=[{"GPU": 1, "CPU": 4}], strategy="STRICT_PACK")
        # Wait until the placement group is ready
        ray.get(model_pg.ready())
        ray.get(rm_pg.ready())

        self.models = []
        for i in range(self.mp):
            actor = LLMActor.options(scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=model_pg, placement_group_bundle_index=i)).remote(
                model=model_version, trust_remote_code=True, limit_mm_per_prompt={"image": limit_img_num}, **kwargs
            )
            self.models.append(actor)
        self.rm = LLMActor.options(scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=rm_pg, placement_group_bundle_index=0)).remote(
            model=rm_version, trust_remote_code=True, limit_mm_per_prompt={"image": limit_img_num}, task="reward", override_pooler_config=PoolerConfig(pooling_type="LAST"), **kwargs
        )
        self.rm_processor = AutoProcessor.from_pretrained(rm_version)
        self.rm_head = torch.load(rm_head, weights_only=False)
        accelerator = Accelerator()
        assert accelerator.state.local_process_index == 0, "VLLM does not support distributed inference."
        assert accelerator.state.num_processes == 1, "VLLM does not support distributed inference."

    def resize_image(self, image: Image):
        if self.model_type == "qwen2_vl":
            width, height = image.size
            resized_height, resized_width = qwenvl_smart_resize(height, width)
            image = image.resize((resized_width, resized_height))
        return image

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

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Batch Preparing")
        for idx, (contexts, gen_kwargs, doc_to_visual, doc_id, task, split) in enumerate([reg.args for reg in requests]):
            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = self.flatten(visuals)
            imgs = []
            for visual in visuals:
                visual = self.resize_image(visual)
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
                    content.append({"type": "image_url", "image_url": {'url':f"data:image/jpeg;base64,{img}"}})
            else:
                contexts_split = contexts.split(self.image_token)
                for idx, context in enumerate(contexts_split):
                    if idx < len(imgs):
                        content.append({"type": "text", "text": context})
                        content.append({"type": "image_url", "image_url": {'url':f"data:image/jpeg;base64,{imgs[idx]}"}})
                if len(contexts_split) > len(imgs):
                    content.append({"type": "text", "text": contexts_split[-1]})
            messages = [{"role": "user", "content": content}]
            requests_data.append(messages)
            pbar.update(1)

        batch_response = self.create_batch(requests_data)
        return batch_response

    def loglikelihood(self, requests):
        # TODO
        assert False, "VLLM not support"

    def create_batch(self, requests_data):
        n_per_model = self.n // self.mp
        sampling_params = SamplingParams(temperature=1, max_tokens=4096, n=n_per_model)
        dist_responses = []
        for i,model in enumerate(self.models):
            dist_responses.append(model.chat.remote(requests_data, sampling_params=sampling_params,use_tqdm=i==0))
        dist_responses = ray.get(dist_responses)
        responses = []
        for model_res in zip(*dist_responses):
            responses.append({"outputs": [ r for res in model_res for r in res.outputs]})
        rm_inputs = []
        for message, res in zip(requests_data, responses):
            img, _ = process_vision_info(message)
            for r in res['outputs']:
                completed_message = message + [{"role": "assistant", "content": [{"type": "text", "text": r.text}]}]
                prompt = self.rm_processor.apply_chat_template(completed_message, tokenize=False, add_generation_prompt=False)
                rm_inputs.append({"prompt": prompt, "multi_modal_data": {"image": img}})
        hidden_states = self.rm.encode.remote(rm_inputs)
        hidden_states = ray.get(hidden_states)
        hidden_states = [h.outputs.data for h in hidden_states]
        hidden_states = torch.stack(hidden_states)  # [sample_num*n, dim]
        hidden_states = hidden_states.view(len(requests_data), self.n, -1)  # [sample_num, n, dim]
        rewards = self.rm_head(hidden_states).squeeze(-1)  # [sample_num, n]
        best_indexes = torch.argmax(rewards, dim=-1)  # [sample_num]

        outputs = []
        for res, idx in zip(responses, best_indexes):
            outputs.append(res['outputs'][idx].text)

        return outputs

    def generate_until_multi_round(self, requests) -> list[str]:
        raise NotImplementedError("TODO: Implement multi-round generation for VLLM")
