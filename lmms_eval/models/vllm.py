# Standard library imports
import base64
from io import BytesIO
import numpy as np

# Related third-party imports
from accelerate import Accelerator
from loguru import logger as eval_logger
from vllm import LLM, SamplingParams
from PIL import Image
from tqdm import tqdm

from qwen_vl_utils import smart_resize as qwenvl_smart_resize
# Local application/library specific imports
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

from transformers import AutoConfig, AutoProcessor
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
        config = AutoConfig.from_pretrained(model_version)
        self.model_type = config.model_type

        self.model = LLM(model=model_version,trust_remote_code=True,limit_mm_per_prompt={"image": limit_img_num},**kwargs)
        self.processor = AutoProcessor.from_pretrained(model_version)
        accelerator = Accelerator()
        assert accelerator.state.local_process_index == 0, "VLLM does not support distributed inference."
        assert accelerator.state.num_processes == 1, "VLLM does not support distributed inference."

    def resize_image(self, image: Image):
        if self.model_type == "qwen2_vl":
            width, height = image.size
            resized_height, resized_width = qwenvl_smart_resize(height,width,max_pixels=1024*1024)
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
        all_imgs = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Batch Preparing")
        for idx, (contexts, gen_kwargs, doc_to_visual, doc_id, task, split) in enumerate([reg.args for reg in requests]):
            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = self.flatten(visuals)
            imgs = []
            for visual in visuals:
                visual = self.resize_image(visual)
                '''
                if self.modality == "image":
                    img = self.encode_image(visual)
                    imgs.append(img)
                elif self.modality == "video":
                    frames = self.encode_video(visual, self.max_frames_num)
                    imgs.extend(frames)
                '''
                imgs.append(visual)
            content = []
            if self.image_token not in contexts:
                for img in imgs:
                    content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,fake"}})
                content.append({"type": "text", "text": contexts})
            else:
                contexts_split = contexts.split(self.image_token)
                for idx, context in enumerate(contexts_split):
                    if idx < len(imgs):
                        content.append({"type": "text", "text": context})
                        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,fake"}})
                if len(contexts_split) > len(imgs):
                    content.append({"type": "text", "text": contexts_split[-1]})
            messages = [{"role": "user", "content": content}]
            requests_data.append(messages)
            all_imgs.append(imgs)
            pbar.update(1)
        pbar.close()

        batch_response = self.create_batch(requests_data,all_imgs)
        return batch_response
        

    def loglikelihood(self, requests):
        # TODO
        assert False, "VLLM not support"


    def create_batch(self, requests_data,all_imgs):
        sampling_params =SamplingParams(temperature=0,max_tokens=4096)
        prompts = self.processor.apply_chat_template(requests_data,tokenize=False, add_generation_prompt=True)
        responses = self.model.generate([{"prompt":p,"multi_modal_data":{"image":imgs}} for p,imgs in zip(prompts,all_imgs)],sampling_params=sampling_params)
        return [r.outputs[0].text for r in responses]


    def generate_until_multi_round(self, requests) -> list[str]:
        raise NotImplementedError("TODO: Implement multi-round generation for BatchGPT4")