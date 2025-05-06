from celery import shared_task
from django.core.files.base import ContentFile
import os
import traceback

import numpy as np
import torch
from .models import VideoGenerationTask
from diffusers_helper.hunyuan import encode_prompt_conds, vae_decode, vae_encode
from diffusers_helper.utils import crop_or_pad_yield_mask, generate_timestamp, save_bcthw_as_mp4, resize_and_center_crop, soft_append_bcthw
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from diffusers_helper.memory import (
    DynamicSwapInstaller, get_cuda_free_memory_gb, unload_complete_models, 
    load_model_as_complete, move_model_to_device_with_memory_preservation,
    offload_model_from_device_for_memory_preservation
)
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.bucket_tools import find_nearest_bucket
from PIL import Image

text_encoder = None
text_encoder_2 = None
tokenizer = None
tokenizer_2 = None
vae = None
image_encoder = None
feature_extractor = None
transformer = None
high_vram = False

def initialize_models():
    global text_encoder, text_encoder_2, tokenizer, tokenizer_2
    global vae, image_encoder, feature_extractor, transformer, high_vram
    
    if text_encoder is not None:
        return
    
    try:
        from diffusers import AutoencoderKLHunyuanVideo
        from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer
        from transformers import SiglipImageProcessor, SiglipVisionModel
        from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked

        free_mem_gb = get_cuda_free_memory_gb(torch.device('cuda'))
        high_vram = free_mem_gb > 60
        
        text_encoder = LlamaModel.from_pretrained(
            "hunyuanvideo-community/HunyuanVideo", 
            subfolder='text_encoder', 
            torch_dtype=torch.float16
        ).cpu()
        
        text_encoder_2 = CLIPTextModel.from_pretrained(
            "hunyuanvideo-community/HunyuanVideo", 
            subfolder='text_encoder_2', 
            torch_dtype=torch.float16
        ).cpu()
        
        tokenizer = LlamaTokenizerFast.from_pretrained(
            "hunyuanvideo-community/HunyuanVideo", 
            subfolder='tokenizer'
        )
        
        tokenizer_2 = CLIPTokenizer.from_pretrained(
            "hunyuanvideo-community/HunyuanVideo", 
            subfolder='tokenizer_2'
        )
        
        vae = AutoencoderKLHunyuanVideo.from_pretrained(
            "hunyuanvideo-community/HunyuanVideo", 
            subfolder='vae', 
            torch_dtype=torch.float16
        ).cpu()
        
        feature_extractor = SiglipImageProcessor.from_pretrained(
            "lllyasviel/flux_redux_bfl", 
            subfolder='feature_extractor'
        )
        
        image_encoder = SiglipVisionModel.from_pretrained(
            "lllyasviel/flux_redux_bfl", 
            subfolder='image_encoder', 
            torch_dtype=torch.float16
        ).cpu()
        
        transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained(
            'lllyasviel/FramePack_F1_I2V_HY_20250503', 
            torch_dtype=torch.bfloat16
        ).cpu()

        # Optimization flags
        if not high_vram:
            vae.enable_slicing().enable_tiling()
            
        transformer.high_quality_fp32_output_for_inference = True
        transformer.to(dtype=torch.bfloat16)
        vae.to(dtype=torch.float16)
        image_encoder.to(dtype=torch.float16)
        text_encoder.to(dtype=torch.float16)
        text_encoder_2.to(dtype=torch.float16)

        # Device placement
        if not high_vram:
            DynamicSwapInstaller.install_model(transformer, device=torch.device('cuda'))
            DynamicSwapInstaller.install_model(text_encoder, device=torch.device('cuda'))
        else:
            text_encoder.to('cuda')
            text_encoder_2.to('cuda')
            image_encoder.to('cuda')
            vae.to('cuda')
            transformer.to('cuda')

    except Exception as e:
        print(f"Model initialization error: {str(e)}")
        traceback.print_exc()
        raise

@shared_task(bind=True)
def generate_video_task(self, task_id, input_image_path, params):
    task = VideoGenerationTask.objects.get(id=task_id)
    try:
        initialize_models()
        
        input_image = np.array(Image.open(input_image_path).convert('RGB'))
        H, W, C = input_image.shape
        height, width = find_nearest_bucket(H, W, resolution=640)
        input_image_np = resize_and_center_crop(input_image, target_width=width, target_height=height)
        input_image_pt = torch.from_numpy(input_image_np).float() / 127.5 - 1
        input_image_pt = input_image_pt.permute(2, 0, 1)[None, :, None]
        
        if not high_vram:
            load_model_as_complete(vae, target_device='cuda')
        start_latent = vae_encode(input_image_pt, vae)
        
        if not high_vram:
            load_model_as_complete(image_encoder, target_device='cuda')
        image_encoder_output = hf_clip_vision_encode(input_image_np, feature_extractor, image_encoder)
        image_encoder_last_hidden_state = image_encoder_output.last_hidden_state
        
        llama_vec, clip_l_pooler = encode_prompt_conds(
            params['prompt'], text_encoder, text_encoder_2, tokenizer, tokenizer_2
        )
        llama_vec_n, clip_l_pooler_n = encode_prompt_conds(
            params.get('n_prompt', ''), text_encoder, text_encoder_2, tokenizer, tokenizer_2
        )
        llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
        llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)
        
        total_latent_sections = int(max(round((params['total_second_length'] * 30) / (params['latent_window_size'] * 4)), 1))
        rnd = torch.Generator("cpu").manual_seed(params['seed'])
        history_latents = torch.zeros(
            size=(1, 16, 16 + 2 + 1, height // 8, width // 8), 
            dtype=torch.float32
        ).cpu()
        history_latents = torch.cat([history_latents, start_latent.to(history_latents)], dim=2)
        total_generated_latent_frames = 1
        outputs_folder = './outputs/'
        os.makedirs(outputs_folder, exist_ok=True)
        job_id = generate_timestamp()
        output_filename = os.path.join(outputs_folder, f'{job_id}.mp4')
        
        for section_index in range(total_latent_sections):
            if not high_vram:
                unload_complete_models()
                move_model_to_device_with_memory_preservation(
                    transformer, target_device='cuda', 
                    preserved_memory_gb=params['gpu_memory_preservation']
                )
            if params['use_teacache']:
                transformer.initialize_teacache(enable_teacache=True, num_steps=params['steps'])
            else:
                transformer.initialize_teacache(enable_teacache=False)
            
            indices = torch.arange(0, sum([1, 16, 2, 1, params['latent_window_size']])).unsqueeze(0)
            (clean_latent_indices_start, clean_latent_4x_indices, 
             clean_latent_2x_indices, clean_latent_1x_indices, 
             latent_indices) = indices.split([1, 16, 2, 1, params['latent_window_size']], dim=1)
            clean_latent_indices = torch.cat([clean_latent_indices_start, clean_latent_1x_indices], dim=1)
            clean_latents_4x, clean_latents_2x, clean_latents_1x = history_latents[
                :, :, -sum([16, 2, 1]):, :, :
            ].split([16, 2, 1], dim=2)
            clean_latents = torch.cat([start_latent.to(history_latents), clean_latents_1x], dim=2)
            generated_latents = sample_hunyuan(
                transformer=transformer,
                sampler='unipc',
                width=width,
                height=height,
                frames=params['latent_window_size'] * 4 - 3,
                real_guidance_scale=params['cfg'],
                distilled_guidance_scale=params['gs'],
                guidance_rescale=params['rs'],
                num_inference_steps=params['steps'],
                generator=rnd,
                prompt_embeds=llama_vec,
                prompt_embeds_mask=llama_attention_mask,
                prompt_poolers=clip_l_pooler,
                negative_prompt_embeds=llama_vec_n,
                negative_prompt_embeds_mask=llama_attention_mask_n,
                negative_prompt_poolers=clip_l_pooler_n,
                device='cuda',
                dtype=torch.bfloat16,
                image_embeddings=image_encoder_last_hidden_state,
                latent_indices=latent_indices,
                clean_latents=clean_latents,
                clean_latent_indices=clean_latent_indices,
                clean_latents_2x=clean_latents_2x,
                clean_latent_2x_indices=clean_latent_2x_indices,
                clean_latents_4x=clean_latents_4x,
                clean_latent_4x_indices=clean_latent_4x_indices,
                callback=None
            )
            total_generated_latent_frames += int(generated_latents.shape[2])
            history_latents = torch.cat([history_latents, generated_latents.to(history_latents)], dim=2)
            if not high_vram:
                offload_model_from_device_for_memory_preservation(
                    transformer, target_device='cuda', preserved_memory_gb=8
                )
                load_model_as_complete(vae, target_device='cuda')
            real_history_latents = history_latents[:, :, -total_generated_latent_frames:, :, :]
            if section_index == 0:
                history_pixels = vae_decode(real_history_latents, vae).cpu()
            else:
                section_latent_frames = params['latent_window_size'] * 2
                overlapped_frames = params['latent_window_size'] * 4 - 3
                current_pixels = vae_decode(
                    real_history_latents[:, :, -section_latent_frames:], vae
                ).cpu()
                history_pixels = soft_append_bcthw(history_pixels, current_pixels, overlapped_frames)
            save_bcthw_as_mp4(history_pixels, output_filename, fps=30, crf=params['mp4_crf'])
            if not high_vram:
                unload_complete_models()
                
        # Upload to storage
        with open(output_filename, 'rb') as f:
            content = ContentFile(f.read())
            task.video_key = f'results/{task.id}.mp4'
            task.status = 'completed'
            task.save()
        
        # Cleanup
        os.remove(output_filename)
        os.remove(input_image_path)
        
    except Exception as e:
        task.status = 'failed'
        task.save()
        traceback.print_exc()
        raise self.retry(exc=e, countdown=5, max_retries=1)