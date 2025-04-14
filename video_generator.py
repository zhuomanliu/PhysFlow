import torch
from diffusers import CogVideoXImageToVideoPipeline
from diffusers.utils import export_to_video, load_image
import base64
import os
import random

## ========== PhysDreamer ============ ##
# data_dir = "model/phys_dreamer/hat"
# image_path = f"{data_dir}/images/frame_00001.png"
# prompt = "The hat is given a gentle tug."

# data_dir = "model/phys_dreamer/telephone"
# image_path = f"{data_dir}/images/frame_00001.png"
# prompt = "The telephone coil is given a gentle tug."

# data_dir = "model/phys_dreamer/carnation"
# image_path = f"{data_dir}/images/frame_00001.png"
# prompt = "The carnation is swaying in the wind."

# data_dir = "model/phys_dreamer/alocasia"
# image_path = f"{data_dir}/images/frame_00001.png"
# prompt = "The alocasia is swaying in the wind."

## ========== Others ============ ##
# data_dir = "model/nerf/plane_960"
# image_path = f"{data_dir}/images/frame_00018.png"
# prompt = "The plane propeller is spinning."

data_dir = "model/nerf/fox"
image_path = f"{data_dir}/images/0014.jpg"
prompt = "The fox is shaking its head."

# data_dir = "model/mip360/kitchen"
# image_path = f"{data_dir}/images/DSCF0656.JPG"
# prompt = "The Lego on the table is being squeezed by a downward force."

# data_dir = "model/co3d/sandcastle"
# image_path = f"{data_dir}/images/frame00000.png"
# prompt = "The sandcastle on the beach is collapsing."

# data_dir = "model/co3d/jam"
# image_path = f"{data_dir}/images/frame00000.png"
# prompt = "The jam on the toast is being spread."


negative_prompt = "The video is not of a high quality, it has a low resolution. The video contains camera transitions. Strange motion trajectory. Flickering, Blurriness, Face restore. Deformation, anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured"

frame_path = f"{data_dir}/images_generated"
os.makedirs(frame_path, exist_ok=True)

image = load_image(image=image_path)
pipe = CogVideoXImageToVideoPipeline.from_pretrained("THUDM/CogVideoX-5b-I2V",
    torch_dtype=torch.bfloat16
)

pipe.enable_sequential_cpu_offload()
pipe.vae.enable_tiling()
pipe.vae.enable_slicing()


frame_per_satge = 6
fps = 8
gs = 7
seed = random.randint(0, 2**8 - 1)
print('seed:', seed)

w, h = image.width, image.height
video = pipe(
    prompt=prompt,
    # negative_prompt=negative_prompt,
    image=image,
    num_videos_per_prompt=1,
    num_inference_steps=50,
    num_frames=frame_per_satge*fps+1,
    guidance_scale=gs,
    generator=torch.Generator(device="cuda").manual_seed(seed),
).frames[0]

video_resized = []
for i, vframe in enumerate(video):
    vframe = vframe.resize((w, h))
    video_resized.append(vframe)
    vframe.save(f'{frame_path}/frame_{i:05d}.png')

export_to_video(video_resized, f"{data_dir}/cogv_{seed}_gs{gs}.mp4", fps=fps)
