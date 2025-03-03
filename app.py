import os
import math
import gradio as gr
import numpy as np
import torch
import safetensors.torch as sf
import db_examples
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")
# Suppress specific PyTorch warnings
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.6,max_split_size_mb:128"
# Suppress Gradio warnings
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

from PIL import Image
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler, EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler
from diffusers.models.attention_processor import AttnProcessor2_0
from transformers import CLIPTextModel, CLIPTokenizer
from briarmbg import BriaRMBG
from enum import Enum
from torch.hub import download_url_to_file


# 'stablediffusionapi/realistic-vision-v51'
# 'runwayml/stable-diffusion-v1-5'
sd15_name = 'stablediffusionapi/realistic-vision-v51'
tokenizer = CLIPTokenizer.from_pretrained(sd15_name, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(sd15_name, subfolder="text_encoder")
vae = AutoencoderKL.from_pretrained(sd15_name, subfolder="vae")
rmbg = BriaRMBG.from_pretrained("briaai/RMBG-1.4")

# Create models directory if it doesn't exist
os.makedirs('./models', exist_ok=True)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
text_encoder = text_encoder.to(device=device, dtype=torch.float16)
vae = vae.to(device=device, dtype=torch.bfloat16)
rmbg = rmbg.to(device=device, dtype=torch.float32)

# SDP
vae.set_attn_processor(AttnProcessor2_0())

# Load both models - one for foreground only, one for foreground+background
# 1. Foreground-only model (8-channel UNet)
unet_fg = UNet2DConditionModel.from_pretrained(sd15_name, subfolder="unet")
with torch.no_grad():
    new_conv_in_fg = torch.nn.Conv2d(8, unet_fg.conv_in.out_channels, unet_fg.conv_in.kernel_size, unet_fg.conv_in.stride, unet_fg.conv_in.padding)
    new_conv_in_fg.weight.zero_()
    new_conv_in_fg.weight[:, :4, :, :].copy_(unet_fg.conv_in.weight)
    new_conv_in_fg.bias = unet_fg.conv_in.bias
    unet_fg.conv_in = new_conv_in_fg

unet_fg_original_forward = unet_fg.forward

def hooked_unet_fg_forward(sample, timestep, encoder_hidden_states, **kwargs):
    c_concat = kwargs['cross_attention_kwargs']['concat_conds'].to(sample)
    c_concat = torch.cat([c_concat] * (sample.shape[0] // c_concat.shape[0]), dim=0)
    new_sample = torch.cat([sample, c_concat], dim=1)
    kwargs['cross_attention_kwargs'] = {}
    return unet_fg_original_forward(new_sample, timestep, encoder_hidden_states, **kwargs)

unet_fg.forward = hooked_unet_fg_forward
unet_fg.set_attn_processor(AttnProcessor2_0())
unet_fg = unet_fg.to(device=device, dtype=torch.float16)

# Load foreground-only model weights
fg_model_path = './models/iclight_sd15_fc.safetensors'
if not os.path.exists(fg_model_path):
    download_url_to_file(url='https://huggingface.co/lllyasviel/ic-light/resolve/main/iclight_sd15_fc.safetensors', dst=fg_model_path)

sd_offset_fg = sf.load_file(fg_model_path)
sd_origin_fg = unet_fg.state_dict()

# Move tensors to the same device
sd_offset_fg = {k: v.to(device) for k, v in sd_offset_fg.items()}
sd_merged_fg = {k: sd_origin_fg[k] + sd_offset_fg[k] for k in sd_origin_fg.keys()}
unet_fg.load_state_dict(sd_merged_fg, strict=True)
del sd_offset_fg, sd_origin_fg, sd_merged_fg

# 2. Foreground+Background model (12-channel UNet)
unet_fgbg = UNet2DConditionModel.from_pretrained(sd15_name, subfolder="unet")
with torch.no_grad():
    new_conv_in_fgbg = torch.nn.Conv2d(12, unet_fgbg.conv_in.out_channels, unet_fgbg.conv_in.kernel_size, unet_fgbg.conv_in.stride, unet_fgbg.conv_in.padding)
    new_conv_in_fgbg.weight.zero_()
    new_conv_in_fgbg.weight[:, :4, :, :].copy_(unet_fgbg.conv_in.weight)
    new_conv_in_fgbg.bias = unet_fgbg.conv_in.bias
    unet_fgbg.conv_in = new_conv_in_fgbg

unet_fgbg_original_forward = unet_fgbg.forward

def hooked_unet_fgbg_forward(sample, timestep, encoder_hidden_states, **kwargs):
    c_concat = kwargs['cross_attention_kwargs']['concat_conds'].to(sample)
    c_concat = torch.cat([c_concat] * (sample.shape[0] // c_concat.shape[0]), dim=0)
    new_sample = torch.cat([sample, c_concat], dim=1)
    kwargs['cross_attention_kwargs'] = {}
    return unet_fgbg_original_forward(new_sample, timestep, encoder_hidden_states, **kwargs)

unet_fgbg.forward = hooked_unet_fgbg_forward
unet_fgbg.set_attn_processor(AttnProcessor2_0())
unet_fgbg = unet_fgbg.to(device=device, dtype=torch.float16)

# Load foreground+background model weights
fgbg_model_path = './models/iclight_sd15_fbc.safetensors'
if not os.path.exists(fgbg_model_path):
    download_url_to_file(url='https://huggingface.co/lllyasviel/ic-light/resolve/main/iclight_sd15_fbc.safetensors', dst=fgbg_model_path)

sd_offset_fgbg = sf.load_file(fgbg_model_path)
sd_origin_fgbg = unet_fgbg.state_dict()

# Move tensors to the same device
sd_offset_fgbg = {k: v.to(device) for k, v in sd_offset_fgbg.items()}
sd_merged_fgbg = {k: sd_origin_fgbg[k] + sd_offset_fgbg[k] for k in sd_origin_fgbg.keys()}
unet_fgbg.load_state_dict(sd_merged_fgbg, strict=True)
del sd_offset_fgbg, sd_origin_fgbg, sd_merged_fgbg

# SDP
unet_fg.set_attn_processor(AttnProcessor2_0())
unet_fgbg.set_attn_processor(AttnProcessor2_0())

# Samplers
ddim_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)

euler_a_scheduler = EulerAncestralDiscreteScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    steps_offset=1
)

dpmpp_2m_sde_karras_scheduler = DPMSolverMultistepScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    algorithm_type="sde-dpmsolver++",
    use_karras_sigmas=True,
    steps_offset=1
)

# Pipelines for foreground-only mode
t2i_pipe_fg = StableDiffusionPipeline(
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    unet=unet_fg,
    scheduler=dpmpp_2m_sde_karras_scheduler,
    safety_checker=None,
    requires_safety_checker=False,
    feature_extractor=None,
    image_encoder=None
)

i2i_pipe_fg = StableDiffusionImg2ImgPipeline(
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    unet=unet_fg,
    scheduler=dpmpp_2m_sde_karras_scheduler,
    safety_checker=None,
    requires_safety_checker=False,
    feature_extractor=None,
    image_encoder=None
)

# Pipelines for foreground+background mode
t2i_pipe_fgbg = StableDiffusionPipeline(
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    unet=unet_fgbg,
    scheduler=dpmpp_2m_sde_karras_scheduler,
    safety_checker=None,
    requires_safety_checker=False,
    feature_extractor=None,
    image_encoder=None
)

i2i_pipe_fgbg = StableDiffusionImg2ImgPipeline(
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    unet=unet_fgbg,
    scheduler=dpmpp_2m_sde_karras_scheduler,
    safety_checker=None,
    requires_safety_checker=False,
    feature_extractor=None,
    image_encoder=None
)

@torch.inference_mode()
def encode_prompt_inner(txt: str):
    max_length = tokenizer.model_max_length
    chunk_length = tokenizer.model_max_length - 2
    id_start = tokenizer.bos_token_id
    id_end = tokenizer.eos_token_id
    id_pad = id_end

    def pad(x, p, i):
        return x[:i] if len(x) >= i else x + [p] * (i - len(x))

    tokens = tokenizer(txt, truncation=False, add_special_tokens=False)["input_ids"]
    chunks = [[id_start] + tokens[i: i + chunk_length] + [id_end] for i in range(0, len(tokens), chunk_length)]
    chunks = [pad(ck, id_pad, max_length) for ck in chunks]

    token_ids = torch.tensor(chunks).to(device=device, dtype=torch.int64)
    conds = text_encoder(token_ids).last_hidden_state

    return conds


@torch.inference_mode()
def encode_prompt_pair(positive_prompt, negative_prompt):
    c = encode_prompt_inner(positive_prompt)
    uc = encode_prompt_inner(negative_prompt)

    c_len = float(len(c))
    uc_len = float(len(uc))
    max_count = max(c_len, uc_len)
    c_repeat = int(math.ceil(max_count / c_len))
    uc_repeat = int(math.ceil(max_count / uc_len))
    max_chunk = max(len(c), len(uc))

    c = torch.cat([c] * c_repeat, dim=0)[:max_chunk]
    uc = torch.cat([uc] * uc_repeat, dim=0)[:max_chunk]

    c = torch.cat([p[None, ...] for p in c], dim=1)
    uc = torch.cat([p[None, ...] for p in uc], dim=1)

    return c, uc


@torch.inference_mode()
def pytorch2numpy(imgs, quant=True):
    results = []
    for x in imgs:
        y = x.movedim(0, -1)

        if quant:
            y = y * 127.5 + 127.5
            y = y.detach().float().cpu().numpy().clip(0, 255).astype(np.uint8)
        else:
            y = y * 0.5 + 0.5
            y = y.detach().float().cpu().numpy().clip(0, 1).astype(np.float32)

        results.append(y)
    return results


@torch.inference_mode()
def numpy2pytorch(imgs):
    h = torch.from_numpy(np.stack(imgs, axis=0)).float() / 127.0 - 1.0  # so that 127 must be strictly 0.0
    h = h.movedim(-1, 1)
    return h


def resize_and_center_crop(image, target_width, target_height):
    pil_image = Image.fromarray(image)
    original_width, original_height = pil_image.size
    scale_factor = max(target_width / original_width, target_height / original_height)
    resized_width = int(round(original_width * scale_factor))
    resized_height = int(round(original_height * scale_factor))
    resized_image = pil_image.resize((resized_width, resized_height), Image.LANCZOS)
    left = (resized_width - target_width) / 2
    top = (resized_height - target_height) / 2
    right = (resized_width + target_width) / 2
    bottom = (resized_height + target_height) / 2
    cropped_image = resized_image.crop((left, top, right, bottom))
    return np.array(cropped_image)


def resize_without_crop(image, target_width, target_height):
    pil_image = Image.fromarray(image)
    resized_image = pil_image.resize((target_width, target_height), Image.LANCZOS)
    return np.array(resized_image)


@torch.inference_mode()
def run_rmbg(img, sigma=0.0):
    try:
        H, W, C = img.shape
        assert C == 3
        k = (256.0 / float(H * W)) ** 0.5
        feed = resize_without_crop(img, int(64 * round(W * k)), int(64 * round(H * k)))
        feed = numpy2pytorch([feed]).to(device=device, dtype=torch.float32)
        alpha = rmbg(feed)[0][0]
        alpha = torch.nn.functional.interpolate(alpha, size=(H, W), mode="bilinear")
        alpha = alpha.movedim(1, -1)[0]
        alpha = alpha.detach().float().cpu().numpy().clip(0, 1)
        result = 127 + (img.astype(np.float32) - 127 + sigma) * alpha
        return result.clip(0, 255).astype(np.uint8), alpha
    except Exception as e:
        print(f"Error in run_rmbg: {str(e)}")
        # Return the original image and a blank alpha mask
        return img, np.ones((img.shape[0], img.shape[1]), dtype=np.float32)


class BGSource(Enum):
    NONE = "None"
    LEFT = "Left Light"
    RIGHT = "Right Light"
    TOP = "Top Light"
    BOTTOM = "Bottom Light"
    UPLOAD = "Upload Background"
    UPLOAD_FLIP = "Use Flipped Background"
    GREY = "Ambient Grey"


@torch.inference_mode()
def process_fg_only(input_fg, prompt, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, lowres_denoise, bg_source):
    """Process with foreground only (from gradio_demo.py)"""
    try:
        bg_source = BGSource(bg_source)
        input_bg = None

        if bg_source == BGSource.NONE:
            # Create a black background when none is specified
            input_bg = np.zeros(shape=(image_height, image_width, 3), dtype=np.uint8)
        elif bg_source == BGSource.LEFT:
            gradient = np.linspace(255, 0, image_width)
            image = np.tile(gradient, (image_height, 1))
            input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
        elif bg_source == BGSource.RIGHT:
            gradient = np.linspace(0, 255, image_width)
            image = np.tile(gradient, (image_height, 1))
            input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
        elif bg_source == BGSource.TOP:
            gradient = np.linspace(255, 0, image_height)[:, None]
            image = np.tile(gradient, (1, image_width))
            input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
        elif bg_source == BGSource.BOTTOM:
            gradient = np.linspace(0, 255, image_height)[:, None]
            image = np.tile(gradient, (1, image_width))
            input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
        else:
            # Default to black background
            input_bg = np.zeros(shape=(image_height, image_width, 3), dtype=np.uint8)

        rng = torch.Generator(device=device).manual_seed(int(seed))

        fg = resize_and_center_crop(input_fg, image_width, image_height)
        
        # For the 8-channel UNet, we only need the foreground
        concat_conds = numpy2pytorch([fg]).to(device=vae.device, dtype=vae.dtype)
        concat_conds = vae.encode(concat_conds).latent_dist.mode() * vae.config.scaling_factor

        conds, unconds = encode_prompt_pair(positive_prompt=prompt + ', ' + a_prompt, negative_prompt=n_prompt)

        # Use foreground-only pipeline
        if input_bg is None or bg_source == BGSource.NONE:
            latents = t2i_pipe_fg(
                prompt_embeds=conds,
                negative_prompt_embeds=unconds,
                width=image_width,
                height=image_height,
                num_inference_steps=steps,
                num_images_per_prompt=num_samples,
                generator=rng,
                output_type='latent',
                guidance_scale=cfg,
                cross_attention_kwargs={'concat_conds': concat_conds},
            ).images.to(vae.dtype) / vae.config.scaling_factor
        else:
            try:
                bg = resize_and_center_crop(input_bg, image_width, image_height)
                bg_latent = numpy2pytorch([bg]).to(device=vae.device, dtype=vae.dtype)
                bg_latent = vae.encode(bg_latent).latent_dist.mode() * vae.config.scaling_factor
                latents = i2i_pipe_fg(
                    image=bg_latent,
                    strength=lowres_denoise,
                    prompt_embeds=conds,
                    negative_prompt_embeds=unconds,
                    width=image_width,
                    height=image_height,
                    num_inference_steps=int(round(steps / lowres_denoise)),
                    num_images_per_prompt=num_samples,
                    generator=rng,
                    output_type='latent',
                    guidance_scale=cfg,
                    cross_attention_kwargs={'concat_conds': concat_conds},
                ).images.to(vae.dtype) / vae.config.scaling_factor
            except Exception as e:
                print(f"Error using background: {str(e)}. Falling back to no background.")
                # Fall back to no background if there's an error
                latents = t2i_pipe_fg(
                    prompt_embeds=conds,
                    negative_prompt_embeds=unconds,
                    width=image_width,
                    height=image_height,
                    num_inference_steps=steps,
                    num_images_per_prompt=num_samples,
                    generator=rng,
                    output_type='latent',
                    guidance_scale=cfg,
                    cross_attention_kwargs={'concat_conds': concat_conds},
                ).images.to(vae.dtype) / vae.config.scaling_factor

        pixels = vae.decode(latents).sample
        pixels = pytorch2numpy(pixels)
        pixels = [resize_without_crop(
            image=p,
            target_width=int(round(image_width * highres_scale / 64.0) * 64),
            target_height=int(round(image_height * highres_scale / 64.0) * 64))
        for p in pixels]

        pixels = numpy2pytorch(pixels).to(device=vae.device, dtype=vae.dtype)
        latents = vae.encode(pixels).latent_dist.mode() * vae.config.scaling_factor
        latents = latents.to(device=unet_fg.device, dtype=unet_fg.dtype)

        image_height, image_width = latents.shape[2] * 8, latents.shape[3] * 8

        fg = resize_and_center_crop(input_fg, image_width, image_height)
        concat_conds = numpy2pytorch([fg]).to(device=vae.device, dtype=vae.dtype)
        concat_conds = vae.encode(concat_conds).latent_dist.mode() * vae.config.scaling_factor

        latents = i2i_pipe_fg(
            image=latents,
            strength=highres_denoise,
            prompt_embeds=conds,
            negative_prompt_embeds=unconds,
            width=image_width,
            height=image_height,
            num_inference_steps=int(round(steps / highres_denoise)),
            num_images_per_prompt=num_samples,
            generator=rng,
            output_type='latent',
            guidance_scale=cfg,
            cross_attention_kwargs={'concat_conds': concat_conds},
        ).images.to(vae.dtype) / vae.config.scaling_factor

        pixels = vae.decode(latents).sample

        return pytorch2numpy(pixels)
    except Exception as e:
        print(f"Error in process_fg_only function: {str(e)}")
        # Return a blank image as a fallback
        return [np.ones((image_height, image_width, 3), dtype=np.uint8) * 255]


@torch.inference_mode()
def process_fg_bg(input_fg, input_bg, prompt, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, bg_source):
    """Process with foreground and background (from gradio_demo_bg.py)"""
    try:
        bg_source = BGSource(bg_source)

        if bg_source == BGSource.UPLOAD:
            pass
        elif bg_source == BGSource.UPLOAD_FLIP:
            input_bg = np.fliplr(input_bg)
        elif bg_source == BGSource.GREY:
            input_bg = np.zeros(shape=(image_height, image_width, 3), dtype=np.uint8) + 64
        elif bg_source == BGSource.LEFT:
            gradient = np.linspace(224, 32, image_width)
            image = np.tile(gradient, (image_height, 1))
            input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
        elif bg_source == BGSource.RIGHT:
            gradient = np.linspace(32, 224, image_width)
            image = np.tile(gradient, (image_height, 1))
            input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
        elif bg_source == BGSource.TOP:
            gradient = np.linspace(224, 32, image_height)[:, None]
            image = np.tile(gradient, (1, image_width))
            input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
        elif bg_source == BGSource.BOTTOM:
            gradient = np.linspace(32, 224, image_height)[:, None]
            image = np.tile(gradient, (1, image_width))
            input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
        else:
            # Default to grey background
            input_bg = np.zeros(shape=(image_height, image_width, 3), dtype=np.uint8) + 64

        rng = torch.Generator(device=device).manual_seed(seed)

        fg = resize_and_center_crop(input_fg, image_width, image_height)
        bg = resize_and_center_crop(input_bg, image_width, image_height)
        concat_conds = numpy2pytorch([fg, bg]).to(device=vae.device, dtype=vae.dtype)
        concat_conds = vae.encode(concat_conds).latent_dist.mode() * vae.config.scaling_factor
        concat_conds = torch.cat([c[None, ...] for c in concat_conds], dim=1)

        conds, unconds = encode_prompt_pair(positive_prompt=prompt + ', ' + a_prompt, negative_prompt=n_prompt)

        latents = t2i_pipe_fgbg(
            prompt_embeds=conds,
            negative_prompt_embeds=unconds,
            width=image_width,
            height=image_height,
            num_inference_steps=steps,
            num_images_per_prompt=num_samples,
            generator=rng,
            output_type='latent',
            guidance_scale=cfg,
            cross_attention_kwargs={'concat_conds': concat_conds},
        ).images.to(vae.dtype) / vae.config.scaling_factor

        pixels = vae.decode(latents).sample
        pixels = pytorch2numpy(pixels)
        pixels = [resize_without_crop(
            image=p,
            target_width=int(round(image_width * highres_scale / 64.0) * 64),
            target_height=int(round(image_height * highres_scale / 64.0) * 64))
        for p in pixels]

        pixels = numpy2pytorch(pixels).to(device=vae.device, dtype=vae.dtype)
        latents = vae.encode(pixels).latent_dist.mode() * vae.config.scaling_factor
        latents = latents.to(device=unet_fgbg.device, dtype=unet_fgbg.dtype)

        image_height, image_width = latents.shape[2] * 8, latents.shape[3] * 8
        fg = resize_and_center_crop(input_fg, image_width, image_height)
        bg = resize_and_center_crop(input_bg, image_width, image_height)
        concat_conds = numpy2pytorch([fg, bg]).to(device=vae.device, dtype=vae.dtype)
        concat_conds = vae.encode(concat_conds).latent_dist.mode() * vae.config.scaling_factor
        concat_conds = torch.cat([c[None, ...] for c in concat_conds], dim=1)

        latents = i2i_pipe_fgbg(
            image=latents,
            strength=highres_denoise,
            prompt_embeds=conds,
            negative_prompt_embeds=unconds,
            width=image_width,
            height=image_height,
            num_inference_steps=int(round(steps / highres_denoise)),
            num_images_per_prompt=num_samples,
            generator=rng,
            output_type='latent',
            guidance_scale=cfg,
            cross_attention_kwargs={'concat_conds': concat_conds},
        ).images.to(vae.dtype) / vae.config.scaling_factor

        pixels = vae.decode(latents).sample
        pixels = pytorch2numpy(pixels, quant=True)

        return pixels, [fg, bg]
    except Exception as e:
        print(f"Error in process_fg_bg function: {str(e)}")
        # Return blank images as fallback
        blank = np.ones((image_height, image_width, 3), dtype=np.uint8) * 255
        return [blank], [blank, blank]


@torch.inference_mode()
def process_relight_fg_only(input_fg, prompt, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, lowres_denoise, bg_source):
    """Relight with foreground only (from gradio_demo.py)"""
    try:
        # Check if input_fg is None or empty
        if input_fg is None or not isinstance(input_fg, np.ndarray):
            raise ValueError("Please upload an image first")
        
        # Check if input_fg has the right shape and type
        if len(input_fg.shape) != 3 or input_fg.shape[2] != 3:
            raise ValueError("Input image must be a color image (RGB)")
        
        # Process the foreground image with background removal
        print("Removing background from foreground image...")
        input_fg, matting = run_rmbg(input_fg)
        
        # Process the image with the foreground-only pipeline
        print(f"Processing with settings: width={image_width}, height={image_height}, samples={num_samples}, steps={steps}, bg_source={bg_source}")
        results = process_fg_only(input_fg, prompt, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, lowres_denoise, bg_source)
        
        print("Processing complete!")
        return input_fg, results
    except Exception as e:
        print(f"Error in process_relight_fg_only: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Return the original image and a blank image as fallback
        if input_fg is not None and isinstance(input_fg, np.ndarray):
            blank = np.ones((image_height, image_width, 3), dtype=np.uint8) * 255
            return input_fg, [blank]
        else:
            # Create blank images with error message
            blank_fg = np.ones((400, 600, 3), dtype=np.uint8) * 255
            blank_result = np.ones((image_height, image_width, 3), dtype=np.uint8) * 255
            return blank_fg, [blank_result]


@torch.inference_mode()
def process_relight_fg_bg(input_fg, input_bg, prompt, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, bg_source):
    """Relight with foreground and background (from gradio_demo_bg.py)"""
    try:
        # Check if input_fg is None or empty
        if input_fg is None or not isinstance(input_fg, np.ndarray):
            raise ValueError("Please upload a foreground image first")
            
        # Check if input_fg has the right shape and type
        if len(input_fg.shape) != 3 or input_fg.shape[2] != 3:
            raise ValueError("Input foreground image must be a color image (RGB)")
            
        # For background upload options, check if input_bg is provided
        if bg_source in [BGSource.UPLOAD.value, BGSource.UPLOAD_FLIP.value] and (input_bg is None or not isinstance(input_bg, np.ndarray)):
            raise ValueError("Please upload a background image or select a different background source")
            
        print("Removing background from foreground image...")
        input_fg, matting = run_rmbg(input_fg)
        
        print(f"Processing with settings: width={image_width}, height={image_height}, samples={num_samples}, steps={steps}, bg_source={bg_source}")
        results, extra_images = process_fg_bg(input_fg, input_bg, prompt, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, bg_source)
        
        print("Processing complete!")
        return results + extra_images
    except Exception as e:
        print(f"Error in process_relight_fg_bg: {str(e)}")
        import traceback
        traceback.print_exc()
        # Return blank images as fallback
        blank = np.ones((image_height, image_width, 3), dtype=np.uint8) * 255
        return [blank, blank, blank]


@torch.inference_mode()
def process_normal(input_fg, input_bg, prompt, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, bg_source):
    """Process normal map (from gradio_demo_bg.py)"""
    try:
        # Check if input_fg is None or empty
        if input_fg is None or not isinstance(input_fg, np.ndarray):
            raise ValueError("Please upload a foreground image first")
            
        input_fg, matting = run_rmbg(input_fg, sigma=16)

        print('left ...')
        left = process_fg_bg(input_fg, input_bg, prompt, image_width, image_height, 1, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, BGSource.LEFT.value)[0][0]

        print('right ...')
        right = process_fg_bg(input_fg, input_bg, prompt, image_width, image_height, 1, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, BGSource.RIGHT.value)[0][0]

        print('bottom ...')
        bottom = process_fg_bg(input_fg, input_bg, prompt, image_width, image_height, 1, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, BGSource.BOTTOM.value)[0][0]

        print('top ...')
        top = process_fg_bg(input_fg, input_bg, prompt, image_width, image_height, 1, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, BGSource.TOP.value)[0][0]

        inner_results = [left * 2.0 - 1.0, right * 2.0 - 1.0, bottom * 2.0 - 1.0, top * 2.0 - 1.0]

        ambient = (left + right + bottom + top) / 4.0
        h, w, _ = ambient.shape
        matting = resize_and_center_crop((matting[..., 0] * 255.0).clip(0, 255).astype(np.uint8), w, h).astype(np.float32)[..., None] / 255.0

        def safa_divide(a, b):
            e = 1e-5
            return ((a + e) / (b + e)) - 1.0

        left = safa_divide(left, ambient)
        right = safa_divide(right, ambient)
        bottom = safa_divide(bottom, ambient)
        top = safa_divide(top, ambient)

        u = (right - left) * 0.5
        v = (top - bottom) * 0.5

        sigma = 10.0
        u = np.mean(u, axis=2)
        v = np.mean(v, axis=2)
        h = (1.0 - u ** 2.0 - v ** 2.0).clip(0, 1e5) ** (0.5 * sigma)
        z = np.zeros_like(h)

        normal = np.stack([u, v, h], axis=2)
        normal /= np.sum(normal ** 2.0, axis=2, keepdims=True) ** 0.5
        normal = normal * matting + np.stack([z, z, 1 - z], axis=2) * (1 - matting)

        results = [normal, left, right, bottom, top] + inner_results
        results = [(x * 127.5 + 127.5).clip(0, 255).astype(np.uint8) for x in results]
        return results
    except Exception as e:
        print(f"Error in process_normal: {str(e)}")
        import traceback
        traceback.print_exc()
        # Return blank images as fallback
        blank = np.ones((image_height, image_width, 3), dtype=np.uint8) * 255
        return [blank, blank, blank, blank, blank, blank, blank, blank, blank]

# Quick prompts from both apps
quick_prompts = [
    'sunshine from window',
    'neon light, city',
    'sunset over sea',
    'golden time',
    'sci-fi RGB glowing, cyberpunk',
    'natural lighting',
    'warm atmosphere, at home, bedroom',
    'magic lit',
    'evil, gothic, Yharnam',
    'light and shadow',
    'shadow from window',
    'soft studio lighting',
    'home atmosphere, cozy bedroom illumination',
    'neon, Wong Kar-wai, warm',
    'beautiful woman, cinematic lighting',
    'handsome man, cinematic lighting',
    'beautiful woman, natural lighting',
    'handsome man, natural lighting',
    'beautiful woman, neo punk lighting, cyberpunk',
    'handsome man, neo punk lighting, cyberpunk',
]
quick_prompts = [[x] for x in quick_prompts]

quick_subjects = [
    'beautiful woman, detailed face',
    'handsome man, detailed face',
]
quick_subjects = [[x] for x in quick_subjects]

# Custom CSS for funky theme
custom_css = """
:root {
    --main-bg-color: #1a0033;
    --secondary-bg-color: #330066;
    --accent-color: #ff00ff;
    --accent-color2: #00ffcc;
    --text-color: #f0f0ff;
    --border-color: #8800ff;
}

body {
    background: linear-gradient(135deg, var(--main-bg-color), var(--secondary-bg-color));
    color: var(--text-color);
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

.gradio-container {
    max-width: 100% !important;
    background: transparent !important;
}

.gr-button {
    background: linear-gradient(90deg, #ff00ff, #00ffcc) !important;
    border: none !important;
    color: black !important;
    font-weight: bold !important;
    font-size: 1.2em !important;
    transition: all 0.3s ease !important;
    transform: scale(1) !important;
    border-radius: 10px !important;
    box-shadow: 0 0 15px rgba(255, 0, 255, 0.5) !important;
}

.gr-button:hover {
    transform: scale(1.05) !important;
    box-shadow: 0 0 20px rgba(0, 255, 204, 0.8) !important;
    background: linear-gradient(90deg, #00ffcc, #ff00ff) !important;
}

.gr-form {
    border: 2px solid var(--border-color) !important;
    border-radius: 15px !important;
    box-shadow: 0 0 20px rgba(136, 0, 255, 0.3) !important;
    background-color: rgba(26, 0, 51, 0.7) !important;
    padding: 20px !important;
}

.gr-input, .gr-textarea, .gr-dropdown {
    background-color: rgba(51, 0, 102, 0.5) !important;
    border: 2px solid var(--border-color) !important;
    color: var(--text-color) !important;
    border-radius: 10px !important;
}

.gr-input:focus, .gr-textarea:focus {
    border-color: var(--accent-color) !important;
    box-shadow: 0 0 10px rgba(255, 0, 255, 0.5) !important;
}

.gr-slider {
    accent-color: var(--accent-color) !important;
}

.gr-slider-value {
    color: var(--accent-color) !important;
}

.gr-accordion {
    border: 2px solid var(--border-color) !important;
    border-radius: 10px !important;
    background-color: rgba(51, 0, 102, 0.3) !important;
}

.gr-radio {
    accent-color: var(--accent-color) !important;
}

h1, h2, h3 {
    color: #00ffcc !important;
    text-shadow: 0 0 10px rgba(0, 255, 204, 0.5) !important;
    font-weight: bold !important;
    text-align: center !important;
    font-size: 2.5em !important;
    letter-spacing: 2px !important;
    margin-bottom: 20px !important;
}

.gr-gallery {
    border: 2px solid var(--border-color) !important;
    border-radius: 15px !important;
    background-color: rgba(26, 0, 51, 0.5) !important;
    padding: 10px !important;
}

.gr-label {
    color: #00ffcc !important;
    font-weight: bold !important;
    text-shadow: 0 0 5px rgba(0, 255, 204, 0.3) !important;
}

.gr-box {
    border: 2px solid var(--border-color) !important;
    border-radius: 15px !important;
    background-color: rgba(26, 0, 51, 0.7) !important;
}

.gr-panel {
    border-radius: 15px !important;
    background-color: rgba(26, 0, 51, 0.7) !important;
}

.gr-image-viewer {
    border: 2px solid var(--accent-color) !important;
    border-radius: 15px !important;
    overflow: hidden !important;
}

.gr-gallery-item {
    border: 2px solid var(--border-color) !important;
    border-radius: 10px !important;
    overflow: hidden !important;
    transition: all 0.3s ease !important;
}

.gr-gallery-item:hover {
    transform: scale(1.03) !important;
    border-color: var(--accent-color) !important;
    box-shadow: 0 0 15px rgba(255, 0, 255, 0.5) !important;
}

.tabs {
    border: none !important;
    margin-top: 20px !important;
}

.tabitem {
    background-color: rgba(26, 0, 51, 0.7) !important;
    border: 2px solid var(--border-color) !important;
    border-radius: 15px !important;
    padding: 20px !important;
    margin-top: 10px !important;
}

.tab-nav {
    border-bottom: 2px solid var(--border-color) !important;
    margin-bottom: 20px !important;
}

.tab-nav * {
    background-color: rgba(26, 0, 51, 0.7) !important;
    border: 2px solid var(--border-color) !important;
    border-bottom: none !important;
    border-radius: 10px 10px 0 0 !important;
    color: var(--text-color) !important;
    font-weight: bold !important;
    padding: 10px 20px !important;
    margin-right: 5px !important;
    transition: all 0.3s ease !important;
}

.tab-nav *:hover {
    background: linear-gradient(90deg, rgba(255, 0, 255, 0.3), rgba(0, 255, 204, 0.3)) !important;
}

.tab-nav *.selected {
    background: linear-gradient(90deg, #ff00ff, #00ffcc) !important;
    color: black !important;
    border-bottom: none !important;
    transform: translateY(-5px) !important;
    box-shadow: 0 -5px 10px rgba(255, 0, 255, 0.3) !important;
}

.gr-dataset {
    border: 2px solid var(--border-color) !important;
    border-radius: 10px !important;
    background-color: rgba(51, 0, 102, 0.3) !important;
    padding: 10px !important;
    margin-bottom: 15px !important;
}

.gr-dataset-items {
    display: flex !important;
    flex-wrap: wrap !important;
    gap: 10px !important;
}

.gr-dataset-item {
    background: linear-gradient(45deg, rgba(255, 0, 255, 0.2), rgba(0, 255, 204, 0.2)) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 8px !important;
    padding: 8px 12px !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
}

.gr-dataset-item:hover {
    background: linear-gradient(45deg, rgba(255, 0, 255, 0.4), rgba(0, 255, 204, 0.4)) !important;
    transform: scale(1.05) !important;
    box-shadow: 0 0 10px rgba(255, 0, 255, 0.4) !important;
}

/* Add some animation */
@keyframes glow {
    0% { text-shadow: 0 0 10px rgba(0, 255, 204, 0.5); }
    50% { text-shadow: 0 0 20px rgba(255, 0, 255, 0.8); }
    100% { text-shadow: 0 0 10px rgba(0, 255, 204, 0.5); }
}

h1 {
    animation: glow 2s infinite alternate;
}

/* Make the main title extra funky */
#app-title {
    font-size: 1.5em !important;
    background: linear-gradient(90deg, #ff00ff, #00ffcc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 10px 0 !important;
    text-shadow: none !important;
}
"""

# Create the Gradio interface with tabs for different modes
block = gr.Blocks(css=custom_css).queue()
with block:
    with gr.Row():
        gr.Markdown("# ✨ IC-Light Ultimate Studio ✨", elem_id="app-title")
        
    with gr.Tabs() as tabs:
        # Tab 1: Foreground Only Mode (from gradio_demo.py)
        with gr.TabItem("✨ Foreground Only Mode ✨"):
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        input_fg_tab1 = gr.Image(source='upload', type="numpy", label="Upload Your Image", height=480)
                        output_bg_tab1 = gr.Image(type="numpy", label="Preprocessed Foreground", height=480)
                    prompt_tab1 = gr.Textbox(label="✨ Enter Your Magical Prompt ✨", placeholder="Describe the lighting you want...")
                    bg_source_tab1 = gr.Radio(choices=[e.value for e in BGSource if e not in [BGSource.UPLOAD, BGSource.UPLOAD_FLIP, BGSource.GREY]],
                                     value=BGSource.NONE.value,
                                     label="🌈 Lighting Direction 🌈", type='value')
                    
                    # Quick prompt datasets
                    example_quick_subjects_tab1 = gr.Dataset(samples=quick_subjects, label='🧙‍♂️ Magic Subject Suggestions 🧙‍♀️', samples_per_page=1000, components=[prompt_tab1])
                    example_quick_prompts_tab1 = gr.Dataset(samples=quick_prompts, label='💫 Funky Lighting Ideas 💫', samples_per_page=1000, components=[prompt_tab1])
                    
                    relight_button_tab1 = gr.Button(value="✨ TRANSFORM MY IMAGE ✨")

                    with gr.Group():
                        with gr.Row():
                            num_samples_tab1 = gr.Slider(label="Number of Magic Images ✨", minimum=1, maximum=12, value=1, step=1)
                            seed_tab1 = gr.Number(label="Magic Seed 🔮", value=12345, precision=0)

                        with gr.Row():
                            image_width_tab1 = gr.Slider(label="Image Width 📏", minimum=256, maximum=1024, value=512, step=64)
                            image_height_tab1 = gr.Slider(label="Image Height 📏", minimum=256, maximum=1024, value=640, step=64)

                    with gr.Accordion("🔮 Advanced Magic Options 🔮", open=False):
                        steps_tab1 = gr.Slider(label="Magic Steps ✨", minimum=1, maximum=100, value=25, step=1)
                        cfg_tab1 = gr.Slider(label="Magic Power Level 💪", minimum=1.0, maximum=32.0, value=2, step=0.01)
                        lowres_denoise_tab1 = gr.Slider(label="Initial Magic Strength 🧙‍♂️", minimum=0.1, maximum=1.0, value=0.9, step=0.01)
                        highres_scale_tab1 = gr.Slider(label="Magic Enlargement ✨", minimum=1.0, maximum=3.0, value=1.5, step=0.01)
                        highres_denoise_tab1 = gr.Slider(label="Final Magic Strength 🧙‍♀️", minimum=0.1, maximum=1.0, value=0.5, step=0.01)
                        a_prompt_tab1 = gr.Textbox(label="Magic Enhancer ✨", value='best quality')
                        n_prompt_tab1 = gr.Textbox(label="Magic Repellent 🛡️", value='lowres, bad anatomy, bad hands, cropped, worst quality')
                with gr.Column():
                    result_gallery_tab1 = gr.Gallery(height=832, object_fit='contain', label='✨ Your Magical Creations ✨')
            
            # Connect the components for tab 1
            ips_tab1 = [input_fg_tab1, prompt_tab1, image_width_tab1, image_height_tab1, num_samples_tab1, seed_tab1, steps_tab1, a_prompt_tab1, n_prompt_tab1, cfg_tab1, highres_scale_tab1, highres_denoise_tab1, lowres_denoise_tab1, bg_source_tab1]
            relight_button_tab1.click(fn=process_relight_fg_only, inputs=ips_tab1, outputs=[output_bg_tab1, result_gallery_tab1])
            example_quick_prompts_tab1.click(lambda x, y: ', '.join(y.split(', ')[:2] + [x[0]]), inputs=[example_quick_prompts_tab1, prompt_tab1], outputs=prompt_tab1, show_progress=False, queue=False)
            example_quick_subjects_tab1.click(lambda x: x[0], inputs=example_quick_subjects_tab1, outputs=prompt_tab1, show_progress=False, queue=False)

        # Tab 2: Foreground + Background Mode (from gradio_demo_bg.py)
        with gr.TabItem("🌟 Foreground + Background Mode 🌟"):
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        input_fg_tab2 = gr.Image(source='upload', type="numpy", label="🧙‍♂️ Foreground Magic ✨", height=480)
                        input_bg_tab2 = gr.Image(source='upload', type="numpy", label="🔮 Background Enchantment ✨", height=480)
                    prompt_tab2 = gr.Textbox(label="✨ Enter Your Magical Prompt ✨", placeholder="Describe the lighting you want...")
                    bg_source_tab2 = gr.Radio(choices=[e.value for e in BGSource],
                                     value=BGSource.UPLOAD.value,
                                     label="🌟 Background Source Selector 🌟", type='value')

                    example_prompts_tab2 = gr.Dataset(samples=quick_prompts, label='💫 Magical Prompt Ideas 💫', components=[prompt_tab2])
                    
                    if 'db_examples' in globals():
                        bg_gallery_tab2 = gr.Gallery(height=450, object_fit='contain', label='🎨 Background Inspiration Gallery 🎨', value=db_examples.bg_samples, columns=5, allow_preview=False)
                    
                    relight_button_tab2 = gr.Button(value="✨ TRANSFORM WITH MAGIC ✨")

                    with gr.Group():
                        with gr.Row():
                            num_samples_tab2 = gr.Slider(label="🖼️ Number of Magic Images ✨", minimum=1, maximum=12, value=1, step=1)
                            seed_tab2 = gr.Number(label="🔮 Magic Seed Number 🔮", value=12345, precision=0)
                        with gr.Row():
                            image_width_tab2 = gr.Slider(label="📏 Magic Width 📏", minimum=256, maximum=1024, value=512, step=64)
                            image_height_tab2 = gr.Slider(label="📐 Magic Height 📐", minimum=256, maximum=1024, value=640, step=64)

                    with gr.Accordion("🔮 Advanced Magic Options 🔮", open=False):
                        steps_tab2 = gr.Slider(label="✨ Magic Steps ✨", minimum=1, maximum=100, value=20, step=1)
                        cfg_tab2 = gr.Slider(label="💪 Magic Power Level 💪", minimum=1.0, maximum=32.0, value=7.0, step=0.01)
                        highres_scale_tab2 = gr.Slider(label="🔍 Magic Enlargement ✨", minimum=1.0, maximum=3.0, value=1.5, step=0.01)
                        highres_denoise_tab2 = gr.Slider(label="🧹 Magic Cleanup Power 🧹", minimum=0.1, maximum=0.9, value=0.5, step=0.01)
                        a_prompt_tab2 = gr.Textbox(label="✨ Magic Enhancer Words ✨", value='best quality')
                        n_prompt_tab2 = gr.Textbox(label="🛡️ Magic Repellent Words 🛡️",
                                          value='lowres, bad anatomy, bad hands, cropped, worst quality')
                with gr.Column():
                    result_gallery_tab2 = gr.Gallery(height=832, object_fit='contain', label='✨ Your Magical Creations ✨')
            
            # Connect the components for tab 2
            ips_tab2 = [input_fg_tab2, input_bg_tab2, prompt_tab2, image_width_tab2, image_height_tab2, num_samples_tab2, seed_tab2, steps_tab2, a_prompt_tab2, n_prompt_tab2, cfg_tab2, highres_scale_tab2, highres_denoise_tab2, bg_source_tab2]
            relight_button_tab2.click(fn=process_relight_fg_bg, inputs=ips_tab2, outputs=[result_gallery_tab2])
            example_prompts_tab2.click(lambda x: x[0], inputs=example_prompts_tab2, outputs=prompt_tab2, show_progress=False, queue=False)
            
            if 'db_examples' in globals():
                def bg_gallery_selected(gal, evt: gr.SelectData):
                    return gal[evt.index]['name']
                bg_gallery_tab2.select(bg_gallery_selected, inputs=bg_gallery_tab2, outputs=input_bg_tab2)

# Launch the app
block.launch(server_name='127.0.0.1', quiet=True, show_error=False)
