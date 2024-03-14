from main import prepare_model, evaluation_epoch, prepare_tokenizer_and_collator, merge_datasets
from src.io.load_datasets import load_datasets
from src.io.args import parse_arguments, ListToArgsConstructor
from tqdm import tqdm
import os
from PIL import Image, ImageDraw, ImageFont
import json
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
import numpy as np
import random
import cv2
args = parse_arguments()

datasets = load_datasets(args, lambda x: x, split_langs=True)

base_folder = '/data2/users/amolina/OCR/control_netted/'
os.makedirs(base_folder, exist_ok=True)
splits = ['val', 'test', 'train']

captions = list(
    set(' '.join(x.strip().split(' ')[1:]) for x in\
               open('/data2/users/amolina/OCR/control_netted/descriptions.txt').readlines())
    )

width = 512
height = 224  # You can adjust the height as needed
font_size = 60

controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_canny", torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.safety_checker = lambda images, clip_input: (images, [False])
pipe.to('cuda')
font_file = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
font = ImageFont.truetype(font_file, font_size)

data_json = {}
for dataset in datasets:
    for split in splits:
        if dataset[split] is not None:
            dataset_name = dataset[split].name[:10]
            save_directory = os.path.join(base_folder, dataset_name, split)
            os.makedirs(save_directory, exist_ok=True)
            for idx in tqdm(range(len(dataset[split])), desc=f"Converting {dataset_name} dataset..."):

                filename = os.path.join(save_directory, f"{str(idx).zfill(6)}.png")
                filename_mask = os.path.join(save_directory, f"{str(idx).zfill(6)}_mask.png")

                sample = dataset[split][idx]['annotation']

                image = Image.new("RGB", (width, height), "black")

                draw = ImageDraw.Draw(image)
                _, _, text_width, text_height = draw.textbbox((0, 0), sample, font=font)
                text_position = ((width - text_width) // 2, (height - text_height) // 2)

                draw.text(text_position, sample, font=font, fill="white")

                image = cv2.Canny(np.array(image), 0, 128)
                canny_image = Image.fromarray(image.astype(np.uint8)).convert('RGB')

                caption = random.choice(captions)
                out_image = pipe(
                    caption + ' textless, speechless, no text visible', num_inference_steps=100, image=canny_image,
                    guidance_scale=10
                ).images[0].resize((width, height))

                canny_image.save(filename_mask)
                out_image.save(filename)
                data_json[filename.replace('/data2/users/amolina/OCR/control_netted/', '')] = [sample, caption]

                exit()

    json.dump(data_json, base_folder + 'dataset.json', indent=3)


