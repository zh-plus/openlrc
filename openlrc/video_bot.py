import base64
from PIL import Image
import numpy as np
from decord import VideoReader, cpu
from openlrc.logger import logger
import anthropic
import os
import json

os.environ['OPENAI_API_KEY'] = "sk-OgF4d947838dcea5e04954eb578374b93715da50b196XXT9"

class videoBot:
    def __init__(self, video_path, text_path,num_frequence=5):
        self.video_path = video_path
        self.text_path = text_path
        self.num_frequence = num_frequence
    
    def encode_image_base64(self,img: Image.Image):
        from io import BytesIO
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def get_index(self, fps, max_frame):
        # 每 interval 秒采一帧，对应的帧间隔
        step = int(self.num_frequence * fps)
        frame_indices = list(range(0, max_frame + 1, step))
        return frame_indices

    def load_video_frames(self, video_path, bound=None):
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        max_frame = len(vr) - 1
        fps = float(vr.get_avg_fps())

        frame_indices = self.get_index(fps, max_frame)

        imgs = []
        for idx in frame_indices:
            img = Image.fromarray(vr[idx].asnumpy()).convert('RGB')
            imgs.append(img)
        return imgs 
    
    def inference(self):
        logger.info("start understanding")
        client = anthropic.Anthropic(
            base_url= "https://api.gptsapi.net",
            api_key=os.environ['OPENAI_API_KEY']
        )

        # 加载视频帧
        imgs = self.load_video_frames(self.video_path)

        # 构造内容块（含图像 + 提问）
        content_blocks = []
        for i, img in enumerate(imgs):
            base64_img = self.encode_image_base64(img)
            content_blocks.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": base64_img
                }
            })

        
        with open(self.text_path, "r", encoding="utf-8") as json_file:
            json_data = json.load(json_file)

        texts = [segment["text"] for segment in json_data.get("segments", [])]
        combined_text = " ".join(texts)


        description_text = f'''the following is the subtitle of the video: {combined_text}, 
        please give a detailed description of the video content based on the subtitle and the image.'''

        # 添加文字部分
        content_blocks.append({
            "type": "text",
            "text": description_text  # 将 JSON 文件中的内容添加到 text 字段
        })

        # Claude 请求
        response = client.messages.create(
            model="claude-3-sonnet-20240229",  # 或 claude-3-opus-20240229
            max_tokens=1000,
            temperature=0.5,
            messages=[{
                "role": "user",
                "content": content_blocks
            }]
        )

        output_text = response.content[0].text

        video_prefix = os.path.splitext(os.path.basename(self.video_path))[0]

        output_file_path = f"./tests/data/{video_prefix}_understanding.txt"

        with open(output_file_path, "w", encoding="utf-8") as file:
            file.write(output_text)

        logger.info(f"video understanding saved to {output_file_path}")
