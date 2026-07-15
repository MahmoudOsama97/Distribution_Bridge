"""
Clean, import-safe copy of the StableUnCLIPImg2ImgEmbedsPipeline used throughout
this revision's generation scripts (Stage 1 embedding extraction, Stage 2a/2b
generation). Lifted from Distribution_Bridge/domainbed/DataAugDB.py (the second,
corrected definition in that file, ~lines 479-596), which itself subclasses
diffusers' StableUnCLIPImg2ImgPipeline to accept a raw CLIP image embedding
directly instead of an input image to re-encode.

Unlike the original file, nothing here executes at import time - callers must
call load_pipeline()/load_image_encoder() explicitly.
"""
import os
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from PIL import Image
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection
from diffusers import EulerDiscreteScheduler
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.pipelines.pipeline_utils import ImagePipelineOutput
from diffusers.pipelines.stable_diffusion.stable_unclip_image_normalizer import StableUnCLIPImageNormalizer
from diffusers.pipelines.stable_diffusion.pipeline_stable_unclip_img2img import StableUnCLIPImg2ImgPipeline

# stabilityai/stable-diffusion-2-1-unclip was removed/renamed upstream; this is the
# ungated mirror with an identical file layout (image_encoder/image_normalizer/unet/vae/...).
UNCLIP_REPO_ID = "diffusers/stable-diffusion-2-1-unclip-i2i-h"
OPEN_CLIP_MODEL_NAME = "ViT-H-14"
OPEN_CLIP_PRETRAINED = "laion2b_s32b_b79k"


class StableUnCLIPImg2ImgEmbedsPipeline(StableUnCLIPImg2ImgPipeline):
    """Image-to-image unCLIP pipeline conditioned on a precomputed CLIP image embedding
    (batch, 1024) instead of a PIL image. Only the embedding-conditioning path differs
    from the parent diffusers pipeline."""

    def __init__(
        self,
        feature_extractor: CLIPImageProcessor,
        image_encoder: CLIPVisionModelWithProjection,
        image_normalizer: StableUnCLIPImageNormalizer,
        image_noising_scheduler: KarrasDiffusionSchedulers,
        tokenizer: CLIPTokenizer,
        text_encoder: CLIPTextModel,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        vae: AutoencoderKL,
    ):
        super().__init__(
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
            image_normalizer=image_normalizer,
            image_noising_scheduler=image_noising_scheduler,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            unet=unet,
            scheduler=scheduler,
            vae=vae,
        )

    def _encode_image(
        self,
        image_embeds: torch.Tensor,
        device: torch.device,
        num_images_per_prompt: int,
        do_classifier_free_guidance: bool,
        noise_level: int,
        generator: Optional[torch.Generator],
    ):
        dtype = next(self.image_encoder.parameters()).dtype
        image_embeds = self.noise_image_embeddings(
            image_embeds=image_embeds.to(device=device, dtype=dtype),
            noise_level=noise_level,
            generator=generator,
        )
        image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
        if do_classifier_free_guidance:
            negative_prompt_image_embeds = torch.zeros_like(image_embeds)
            image_embeds = torch.cat([negative_prompt_image_embeds, image_embeds])
        return image_embeds

    def check_inputs(
        self,
        prompt,
        image_embeds,
        height,
        width,
        callback_steps,
        noise_level,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")
        if image_embeds is None:
            raise ValueError("`image_embeds` must be provided and cannot be None.")
        if not isinstance(image_embeds, torch.Tensor):
            raise TypeError(f"`image_embeds` has to be of type `torch.Tensor` but is {type(image_embeds)}")
        if noise_level < 0 or noise_level >= self.image_noising_scheduler.config.num_train_timesteps:
            raise ValueError(
                f"`noise_level` must be between 0 and {self.image_noising_scheduler.config.num_train_timesteps - 1}, inclusive."
            )

    @torch.no_grad()
    def __call__(
        self,
        image_embeds: torch.Tensor,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 20,
        guidance_scale: float = 10,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        noise_level: int = 0,
    ):
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        if prompt is None and prompt_embeds is None:
            prompt = [""] * image_embeds.shape[0]

        self.check_inputs(
            prompt=prompt, image_embeds=image_embeds, height=height, width=width,
            callback_steps=callback_steps, noise_level=noise_level,
            negative_prompt=negative_prompt, prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        if isinstance(prompt, str):
            prompt_batch_size = 1
        elif isinstance(prompt, list):
            prompt_batch_size = len(prompt)
        else:
            prompt_batch_size = prompt_embeds.shape[0]

        if image_embeds.shape[0] != prompt_batch_size:
            raise ValueError(
                f"The batch size of `image_embeds` ({image_embeds.shape[0]}) must match the batch size of the prompts ({prompt_batch_size})."
            )

        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0

        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        encoded_prompt_embeds, encoded_negative_prompt_embeds = self.encode_prompt(
            prompt=prompt, device=device, num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance, negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )
        final_prompt_embeds = (
            torch.cat([encoded_negative_prompt_embeds, encoded_prompt_embeds])
            if do_classifier_free_guidance else encoded_prompt_embeds
        )

        processed_image_embeds = self._encode_image(
            image_embeds=image_embeds, device=device, num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance, noise_level=noise_level, generator=generator,
        )

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        latents = self.prepare_latents(
            batch_size=prompt_batch_size * num_images_per_prompt,
            num_channels_latents=self.unet.config.in_channels,
            height=height, width=width, dtype=final_prompt_embeds.dtype,
            device=device, generator=generator, latents=latents,
        )

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        for i, t in enumerate(self.progress_bar(timesteps)):
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            noise_pred = self.unet(
                latent_model_input, t, encoder_hidden_states=final_prompt_embeds,
                class_labels=processed_image_embeds, cross_attention_kwargs=cross_attention_kwargs,
                return_dict=False,
            )[0]
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
            if callback is not None and i % callback_steps == 0:
                step_idx = i // getattr(self.scheduler, "order", 1)
                callback(step_idx, t, latents)

        if output_type != "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        else:
            image = latents
        image = self.image_processor.postprocess(image, output_type=output_type)

        self.maybe_free_model_hooks()
        if not return_dict:
            return (image,)
        return ImagePipelineOutput(images=image)


def load_image_encoder(device: str = "cuda", dtype: torch.dtype = torch.float16, offline: bool = True):
    """Load just the CLIP image encoder + feature extractor (no UNet/VAE/text encoder) -
    used for Stage 1 embedding extraction and Stage 4 eps_gen re-encoding, where we don't
    need the full diffusion pipeline. Guarantees embeddings live in exactly the same space
    the unCLIP UNet conditions on, since this is the same checkpoint load_pipeline() uses."""
    if offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        UNCLIP_REPO_ID, subfolder="image_encoder", torch_dtype=dtype
    ).to(device)
    image_encoder.eval()
    feature_extractor = CLIPImageProcessor.from_pretrained(UNCLIP_REPO_ID, subfolder="feature_extractor")
    return image_encoder, feature_extractor


def load_pipeline(device: str = "cuda", dtype: torch.dtype = torch.float16, offline: bool = True):
    """Load the unCLIP img2img-embeds pipeline plus its CLIP image encoder / feature
    extractor. Set offline=True (default) inside sbatch jobs to force HF_HUB_OFFLINE
    against a pre-populated $HF_HOME cache (see environment/setup_env.sh)."""
    if offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        UNCLIP_REPO_ID, subfolder="image_encoder", torch_dtype=dtype
    ).to(device)
    feature_extractor = CLIPImageProcessor.from_pretrained(UNCLIP_REPO_ID, subfolder="feature_extractor")

    pipe = StableUnCLIPImg2ImgEmbedsPipeline.from_pretrained(UNCLIP_REPO_ID, torch_dtype=dtype).to(device)
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)

    return pipe, image_encoder, feature_extractor


@torch.no_grad()
def encode_image(image: Image.Image, image_encoder, feature_extractor, device, dtype) -> torch.Tensor:
    """Return the unnormalized (batch=1, 1024) CLIP image embedding for a PIL image,
    matching the paper's convention (Supp. S1: unnormalized R^1024)."""
    processed = feature_extractor(images=image.convert("RGB"), return_tensors="pt").pixel_values
    processed = processed.to(device, dtype=dtype)
    return image_encoder(processed).image_embeds
