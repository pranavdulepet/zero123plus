import torch
import requests
from PIL import Image
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
import os

# Load the pipeline
pipeline = DiffusionPipeline.from_pretrained(
    "sudo-ai/zero123plus-v1.1", custom_pipeline="sudo-ai/zero123plus-pipeline",
    torch_dtype=torch.float16
)

#import json
#from pathlib import Path
#from pipeline_new import Zero123PlusPipeline
#from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, EulerAncestralDiscreteScheduler
#from transformers import CLIPTextModel, CLIPTokenizer, CLIPProcessor, CLIPVisionModel, CLIPImageProcessor
#import os 
#from PIL import Image

#base_path = Path("./zero123plus-v1.1")

#def load_config(file_path):
#    with open(file_path, "r") as file:
#        return json.load(file)

#vae_config = load_config(base_path / "vae/config.json")
#scheduler_config = load_config(base_path / "scheduler/scheduler_config.json")
#unet_config = load_config(base_path / "unet/config.json")

#unsupported_keys = ["clip_sample", "set_alpha_to_one", "skip_prk_steps", "_class_name", "_diffusers_version"]
#for key in unsupported_keys:
#    scheduler_config.pop(key, None)

#vae = AutoencoderKL(**vae_config)
#scheduler = EulerAncestralDiscreteScheduler(**scheduler_config)
#scheduler = DDPMScheduler(**scheduler_config)
#unet = UNet2DConditionModel(**unet_config)

#text_encoder = CLIPTextModel.from_pretrained(base_path / "text_encoder")
#tokenizer = CLIPTokenizer.from_pretrained(base_path / "tokenizer")
#clip_processor = CLIPImageProcessor.from_pretrained(base_path / "feature_extractor_clip")
#vae_processor = CLIPImageProcessor.from_pretrained(base_path / "feature_extractor_vae")
#vision_encoder = CLIPVisionModel.from_pretrained(base_path / "vision_encoder")

#pipeline = Zero123PlusPipeline(
#    vae=vae,
#    text_encoder=text_encoder,
#    tokenizer=tokenizer,
#    unet=unet,
#    scheduler=scheduler,
#    vision_encoder=vision_encoder, 
#    feature_extractor_clip=clip_processor, 
#    feature_extractor_vae=vae_processor,
#)

# Feel free to tune the scheduler
pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
    pipeline.scheduler.config, timestep_spacing='trailing'
)

pipeline.to('cuda:0')

# Run the pipeline
#cond = Image.open(requests.get("https://d.skis.ltd/nrp/sample-data/lysol.png", stream=True).raw)
#img_lst = ["architectures8", "architectures23", "architectures34", "birdanimal38", "city27", "human8"]
#for img in img_lst:
  # cond = Image.open(f"../synthetic_sdxl_images/{img}.png")
   #result = pipeline(cond, num_inference_steps=75).images[0]
   #result.show()
   #result.save(f"../zero_outputs/output_{img}.png")

print(os.getcwd())
cond = Image.open("./examples/birdanimal1.png")
result = pipeline(cond, num_inference_steps=75).images[0]
#result = pipeline(cond, num_inference_steps=75)
print("\n\nRESULT: ", result, "\n\n END.")
result.show()
result.save("./test_output_birdanimal1.png")
