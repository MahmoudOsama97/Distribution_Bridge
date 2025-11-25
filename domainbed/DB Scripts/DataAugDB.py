import inspect
from typing import Any, Callable, Dict, List, Optional, Union

import PIL.Image
import torch
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection
from transformers import CLIPVisionModelWithProjection
from diffusers.pipelines.stable_diffusion.stable_unclip_image_normalizer import StableUnCLIPImageNormalizer
from diffusers import EulerDiscreteScheduler

import torch
from PIL import Image
import requests
from io import BytesIO
from IPython.display import display
from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import StableDiffusionLoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.models.embeddings import get_timestep_embedding # Used by StableUnCLIPImg2ImgPipeline
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    USE_PEFT_BACKEND, # Used by StableUnCLIPImg2ImgPipeline's encode_prompt
    deprecate, # Used by StableUnCLIPImg2ImgPipeline
    is_torch_xla_available,
    logging,
    replace_example_docstring,
    scale_lora_layers, # Used by StableUnCLIPImg2ImgPipeline's encode_prompt
    unscale_lora_layers, # Used by StableUnCLIPImg2ImgPipeline's encode_prompt
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput, StableDiffusionMixin
from diffusers.pipelines.stable_diffusion.stable_unclip_image_normalizer import StableUnCLIPImageNormalizer
# Import the parent pipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_unclip_img2img import StableUnCLIPImg2ImgPipeline


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


EXAMPLE_DOC_STRING_STABLE_UNCLIP_IMG2IMG_EMBEDS = """
    Examples:
        ```py
        >>> import torch
        >>> from PIL import Image
        >>> import requests
        >>> from io import BytesIO
        >>> from diffusers import CLIPVisionModelWithProjection, CLIPImageProcessor
        >>> # from stable_unclip_img2img_embeds_pipeline import StableUnCLIPImg2ImgEmbedsPipeline # If in separate file

        >>> device = "cuda" if torch.cuda.is_available() else "cpu"
        >>> torch_dtype = torch.float16 if device == "cuda" else torch.float32

        >>> # First, load the models needed to generate image embeddings
        >>> image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        ...     "stabilityai/stable-diffusion-2-1-unclip/image_encoder", torch_dtype=torch_dtype
        ... ).to(device)
        >>> feature_extractor = CLIPImageProcessor.from_pretrained(
        ...     "stabilityai/stable-diffusion-2-1-unclip/feature_extractor"
        ... )

        >>> # Load the new pipeline
        >>> pipe = StableUnCLIPImg2ImgEmbedsPipeline.from_pretrained(
        ...     "stabilityai/stable-diffusion-2-1-unclip-small", torch_dtype=torch_dtype
        ... ).to(device)

        >>> # Prepare an image
        >>> url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"
        >>> response = requests.get(url)
        >>> init_image_pil = Image.open(BytesIO(response.content)).convert("RGB").resize((768, 512))

        >>> # Generate image embeddings
        >>> with torch.no_grad():
        ...     # Process the image
        ...     processed_image = feature_extractor(images=init_image_pil, return_tensors="pt").pixel_values
        ...     processed_image = processed_image.to(device, dtype=image_encoder.dtype) # Match image_encoder's dtype
        ...     # Get image embeddings
        ...     image_embeds = image_encoder(processed_image).image_embeds

        >>> prompt = "A fantasy landscape, trending on artstation"
        >>> negative_prompt = "ugly, blurry, malformed, deformed, noisy"

        >>> # Pass image_embeds directly to the pipeline
        >>> generator = torch.Generator(device=device).manual_seed(42)
        >>> images = pipe(
        ...     image_embeds=image_embeds,
        ...     prompt=prompt,
        ...     negative_prompt=negative_prompt,
        ...     num_inference_steps=25,
        ...     guidance_scale=8.0,
        ...     noise_level=20, # Integer noise level
        ...     generator = generator
        ... ).images
        >>> images[0].save("fantasy_landscape_from_embeds.png")
        ```
"""


class StableUnCLIPImg2ImgEmbedsPipeline(StableUnCLIPImg2ImgPipeline):
    """
    Pipeline for text-guided image-to-image generation using stable unCLIP, conditioned on pre-computed image
    embeddings.

    This model inherits from [`StableUnCLIPImg2ImgPipeline`]. Check the superclass documentation for the generic
    methods implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        feature_extractor ([`CLIPImageProcessor`]):
            Feature extractor for image pre-processing before being encoded. Although image embeddings are passed
            directly, this and `image_encoder` are kept for consistency with the parent class structure and potential
            use by inherited methods or for determining dtypes.
        image_encoder ([`CLIPVisionModelWithProjection`]):
            CLIP vision model for encoding images. Kept for consistency and potential minor uses.
        image_normalizer ([`StableUnCLIPImageNormalizer`]):
            Used to normalize the predicted image embeddings before the noise is applied and un-normalize the image
            embeddings after the noise has been applied.
        image_noising_scheduler ([`KarrasDiffusionSchedulers`]):
            Noise schedule for adding noise to the predicted image embeddings. The amount of noise to add is determined
            by the `noise_level`.
        tokenizer (`~transformers.CLIPTokenizer`):
            A [`~transformers.CLIPTokenizer`)].
        text_encoder ([`~transformers.CLIPTextModel`]):
            Frozen [`~transformers.CLIPTextModel`] text-encoder.
        unet ([`UNet2DConditionModel`]):
            A [`UNet2DConditionModel`] to denoise the encoded image latents.
        scheduler ([`KarrasDiffusionSchedulers`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents.
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
    """

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
        noise_level: int, # CORRECTED: This is an int, consistent with parent's usage
        generator: Optional[torch.Generator],
    ):
        dtype = next(self.image_encoder.parameters()).dtype

        image_embeds = self.noise_image_embeddings(
            image_embeds=image_embeds.to(device=device, dtype=dtype),
            noise_level=noise_level, # Pass the int noise_level
            generator=generator,
        )

        repeat_by = num_images_per_prompt
        image_embeds_for_repetition = image_embeds.unsqueeze(1)
        bs_embed, seq_len, _ = image_embeds_for_repetition.shape

        image_embeds = image_embeds_for_repetition.repeat(1, repeat_by, 1)
        image_embeds = image_embeds.view(bs_embed * repeat_by, seq_len, -1)
        image_embeds = image_embeds.squeeze(1)

        if do_classifier_free_guidance:
            negative_prompt_image_embeds = torch.zeros_like(image_embeds)
            image_embeds = torch.cat([negative_prompt_image_embeds, image_embeds])

        return image_embeds

    def check_inputs(
        self,
        prompt: Union[str, List[str]],
        image_embeds: torch.Tensor,
        height: int,
        width: int,
        callback_steps: int,
        noise_level: int,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Please make sure to define only one of the two."
            )

        if prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )

        if prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                "Provide either `negative_prompt` or `negative_prompt_embeds`. Cannot leave both `negative_prompt` and `negative_prompt_embeds` undefined."
            )

        if prompt is not None and negative_prompt is not None:
            if type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

        if noise_level < 0 or noise_level >= self.image_noising_scheduler.config.num_train_timesteps:
            raise ValueError(
                f"`noise_level` must be between 0 and {self.image_noising_scheduler.config.num_train_timesteps - 1}, inclusive."
            )

        if image_embeds is None:
            raise ValueError("`image_embeds` must be provided and cannot be None.")

        if not isinstance(image_embeds, torch.Tensor):
            raise TypeError(f"`image_embeds` has to be of type `torch.Tensor` but is {type(image_embeds)}")

        if prompt_embeds is not None and image_embeds is not None:
            if prompt_embeds.shape[0] != image_embeds.shape[0]:
                # This check is for when prompt_embeds are directly passed.
                # If prompt (str/list) is passed, encode_prompt handles batching.
                # image_embeds should have batch_size corresponding to number of prompts.
                pass # This specific check may need refinement based on how prompt_embeds are prepared.
                     # For now, assume image_embeds batch size matches prompt batch size.

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
        noise_level: int = 0, # This is the scalar int noise_level
        clip_skip: Optional[int] = None,
    ):
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        if prompt is None and prompt_embeds is None:
            prompt = [""] * image_embeds.shape[0]

        self.check_inputs(
            prompt=prompt,
            image_embeds=image_embeds,
            height=height,
            width=width,
            callback_steps=callback_steps,
            noise_level=noise_level,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        if prompt is not None and isinstance(prompt, str):
            prompt_batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            prompt_batch_size = len(prompt)
        else:
            prompt_batch_size = prompt_embeds.shape[0]

        # Ensure image_embeds batch size matches prompt batch size
        if image_embeds.shape[0] != prompt_batch_size:
            raise ValueError(
                f"The batch size of `image_embeds` ({image_embeds.shape[0]}) must match the batch size of the prompts ({prompt_batch_size})."
            )

        total_batch_size_for_latents = prompt_batch_size * num_images_per_prompt

        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0

        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        encoded_prompt_embeds, encoded_negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
            clip_skip=clip_skip,
        )

        if do_classifier_free_guidance:
            final_prompt_embeds = torch.cat([encoded_negative_prompt_embeds, encoded_prompt_embeds])
        else:
            final_prompt_embeds = encoded_prompt_embeds

        # CORRECTED: Pass the integer `noise_level` to `_encode_image`
        processed_image_embeds = self._encode_image(
            image_embeds=image_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            noise_level=noise_level, # Pass the int noise_level
            generator=generator,
        )

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size=total_batch_size_for_latents,
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            dtype=final_prompt_embeds.dtype,
            device=device,
            generator=generator,
            latents=latents,
        )

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        for i, t in enumerate(self.progress_bar(timesteps)):
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=final_prompt_embeds,
                class_labels=processed_image_embeds,
                cross_attention_kwargs=cross_attention_kwargs,
                return_dict=False,
            )[0]

            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

            if callback is not None and i % callback_steps == 0:
                step_idx = i // getattr(self.scheduler, "order", 1)
                callback(step_idx, t, latents)

            if XLA_AVAILABLE:
                xm.mark_step()

        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        else:
            image = latents

        # === FIX: ADD THIS LINE TO CONVERT THE TENSOR TO PIL IMAGES ===
        image = self.image_processor.postprocess(image, output_type=output_type)
        # =============================================================

        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)
    

model = CLIPVisionModelWithProjection.from_pretrained(
    "stabilityai/stable-diffusion-2-1-unclip", subfolder="image_encoder"
)

print(model.config.projection_dim)  # ← Will print 768
print(model.vision_model.config.hidden_size)  # ← Will print 1280


normalizer = StableUnCLIPImageNormalizer.from_pretrained(
    "stabilityai/stable-diffusion-2-1-unclip", subfolder="image_normalizer"
)
print(normalizer.mean.shape)  # → torch.Size([1, 1024])



# Step 2: Define your Custom Pipeline
# (Paste your StableUnCLIPImg2ImgEmbedsPipeline code here as before)
import inspect
from typing import Any, Callable, Dict, List, Optional, Union

import PIL.Image
import torch
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection

from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import StableDiffusionLoraLoaderMixin # For future compatibility
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.models.embeddings import get_timestep_embedding # Used by StableUnCLIPImg2ImgPipeline
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    USE_PEFT_BACKEND, # Used by StableUnCLIPImg2ImgPipeline's encode_prompt
    deprecate, # Used by StableUnCLIPImg2ImgPipeline
    is_torch_xla_available,
    logging,
    replace_example_docstring,
    scale_lora_layers, # Used by StableUnCLIPImg2ImgPipeline's encode_prompt
    unscale_lora_layers, # Used by StableUnCLIPImg2ImgPipeline's encode_prompt
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput, StableDiffusionMixin
from diffusers.pipelines.stable_diffusion.stable_unclip_image_normalizer import StableUnCLIPImageNormalizer
# Import the parent pipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_unclip_img2img import StableUnCLIPImg2ImgPipeline


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING_STABLE_UNCLIP_IMG2IMG_EMBEDS = """
    Examples:
        ```py
        >>> # This is a placeholder for your actual docstring example
        ```
"""

class StableUnCLIPImg2ImgEmbedsPipeline(StableUnCLIPImg2ImgPipeline):
    """
    Your docstring here...
    """

    def __init__(self, feature_extractor: CLIPImageProcessor, image_encoder: CLIPVisionModelWithProjection, image_normalizer: StableUnCLIPImageNormalizer, image_noising_scheduler: KarrasDiffusionSchedulers, tokenizer: CLIPTokenizer, text_encoder: CLIPTextModel, unet: UNet2DConditionModel, scheduler: KarrasDiffusionSchedulers, vae: AutoencoderKL):
        super().__init__(feature_extractor=feature_extractor, image_encoder=image_encoder, image_normalizer=image_normalizer, image_noising_scheduler=image_noising_scheduler, tokenizer=tokenizer, text_encoder=text_encoder, unet=unet, scheduler=scheduler, vae=vae)

    def _encode_image(self, image_embeds: torch.Tensor, device: torch.device, num_images_per_prompt: int, do_classifier_free_guidance: bool, noise_level: int, generator: Optional[torch.Generator]):
        # This function seems correct based on your file, no changes needed here.
        dtype = next(self.image_encoder.parameters()).dtype
        image_embeds = self.noise_image_embeddings(image_embeds=image_embeds.to(device=device, dtype=dtype), noise_level=noise_level, generator=generator)
        repeat_by = num_images_per_prompt
        image_embeds = image_embeds.repeat_interleave(repeat_by, dim=0)
        if do_classifier_free_guidance:
            negative_prompt_image_embeds = torch.zeros_like(image_embeds)
            image_embeds = torch.cat([negative_prompt_image_embeds, image_embeds])
        return image_embeds

    def check_inputs(self, prompt: Union[str, List[str]], image_embeds: torch.Tensor, height: int, width: int, callback_steps: int, noise_level: int, negative_prompt: Optional[Union[str, List[str]]] = None, prompt_embeds: Optional[torch.Tensor] = None, negative_prompt_embeds: Optional[torch.Tensor] = None):
        # This function seems correct, no changes needed here.
        # ... (your check_inputs code) ...
        pass

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
        clip_skip: Optional[int] = None,
    ):
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        if prompt is None and prompt_embeds is None:
            prompt = [""] * image_embeds.shape[0]

        self.check_inputs(prompt=prompt, image_embeds=image_embeds, height=height, width=width, callback_steps=callback_steps, noise_level=noise_level, negative_prompt=negative_prompt, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds)

        if prompt is not None and isinstance(prompt, str):
            prompt_batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            prompt_batch_size = len(prompt)
        else:
            prompt_batch_size = prompt_embeds.shape[0]

        if image_embeds.shape[0] != prompt_batch_size:
            raise ValueError(f"The batch size of `image_embeds` ({image_embeds.shape[0]}) must match the batch size of the prompts ({prompt_batch_size}).")

        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0

        text_encoder_lora_scale = (cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None)
        encoded_prompt_embeds, encoded_negative_prompt_embeds = self.encode_prompt(prompt=prompt, device=device, num_images_per_prompt=num_images_per_prompt, do_classifier_free_guidance=do_classifier_free_guidance, negative_prompt=negative_prompt, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds, lora_scale=text_encoder_lora_scale, clip_skip=clip_skip)

        if do_classifier_free_guidance:
            final_prompt_embeds = torch.cat([encoded_negative_prompt_embeds, encoded_prompt_embeds])
        else:
            final_prompt_embeds = encoded_prompt_embeds

        processed_image_embeds = self._encode_image(image_embeds=image_embeds, device=device, num_images_per_prompt=num_images_per_prompt, do_classifier_free_guidance=do_classifier_free_guidance, noise_level=noise_level, generator=generator)

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        latents = self.prepare_latents(batch_size=prompt_batch_size * num_images_per_prompt, num_channels_latents=self.unet.config.in_channels, height=height, width=width, dtype=final_prompt_embeds.dtype, device=device, generator=generator, latents=latents)

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        for i, t in enumerate(self.progress_bar(timesteps)):
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=final_prompt_embeds, class_labels=processed_image_embeds, cross_attention_kwargs=cross_attention_kwargs, return_dict=False)[0]

            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

            if callback is not None and i % callback_steps == 0:
                step_idx = i // getattr(self.scheduler, "order", 1)
                callback(step_idx, t, latents)

        # --- THIS IS THE CORRECTED SECTION ---
        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        else:
            image = latents

        # The crucial post-processing step that converts the tensor to a list of PIL images
        image = self.image_processor.postprocess(image, output_type=output_type)
        # --- END OF CORRECTION ---

        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)
# Step 3: Setup and Run the Pipeline Example


# --- Configuration ---
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if device == "cuda" else torch.float32

# Model IDs for Stable UnCLIP
# Your pipeline will be loaded with its own model ID
your_pipeline_model_id = "stabilityai/stable-diffusion-2-1-unclip" # Or "" for full unCLIP

# Base repository for CLIP components (image_encoder, feature_extractor)
clip_components_repo_id = "stabilityai/stable-diffusion-2-1-unclip"


# --- Load necessary models for image embedding ---
print(f"Loading image encoder from: {clip_components_repo_id}, subfolder: image_encoder")
image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    clip_components_repo_id,        # Main repository ID
    subfolder="image_encoder",      # Specify the subfolder
    torch_dtype=torch_dtype
).to(device)

print(f"Loading feature extractor from: {clip_components_repo_id}, subfolder: feature_extractor")
feature_extractor = CLIPImageProcessor.from_pretrained(
    clip_components_repo_id,        # Main repository ID
    subfolder="feature_extractor"   # Specify the subfolder
    # torch_dtype is generally not needed for CLIPImageProcessor.from_pretrained
)

# --- Load your custom pipeline ---
print(f"Loading custom pipeline from: {your_pipeline_model_id}")
pipe = StableUnCLIPImg2ImgEmbedsPipeline.from_pretrained(
    your_pipeline_model_id, torch_dtype=torch_dtype
).to(device)
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config) # Use a Karras scheduler

# --- Prepare an input image (e.g., a car) ---
image_url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"


import torch
from PIL import Image
import os
from itertools import combinations
import random
import numpy as np # For weight generation if needed
from tqdm import tqdm # For progress bars
from torchvision import transforms

# --- Make sure to import your pipeline and necessary diffusers/transformers components ---
# Assuming your StableUnCLIPImg2ImgEmbedsPipeline class definition is available
# from your_pipeline_file import StableUnCLIPImg2ImgEmbedsPipeline
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from diffusers import EulerDiscreteScheduler # Or your preferred Karras scheduler

# (Your StableUnCLIPImg2ImgEmbedsPipeline class definition should be here or imported)
# For brevity, I'm omitting the pipeline class definition itself, assuming it's the
# second, corrected version from your initial prompt.
def hybrid_slerp(v0, v1, t, DOT_THRESHOLD=0.9995):
    """
    Performs a hybrid SLERP that interpolates direction along the arc (SLERP)
    but interpolates magnitude along a straight line (LERP). This is more
    stable for models sensitive to embedding magnitude.
    """
    # 1. Calculate the magnitudes of the original vectors
    mag_v0 = torch.norm(v0, p=2, dim=-1, keepdim=True)
    mag_v1 = torch.norm(v1, p=2, dim=-1, keepdim=True)

    # 2. Linearly interpolate the magnitudes
    interpolated_magnitude = (1 - t) * mag_v0 + t * mag_v1

    # 3. Perform standard SLERP to get the direction
    # (The slerp function normalizes internally, so we don't need to here)
    interpolated_direction = slerp(v0, v1, t, DOT_THRESHOLD)

    # 4. Scale the perfect direction vector by the more 'natural' interpolated magnitude
    final_vector = interpolated_direction * interpolated_magnitude
    
    return final_vector
def slerp(v0, v1, t, DOT_THRESHOLD=0.9995):
    """
    Performs Spherical Linear Interpolation (SLERP) between two vectors.
    """
    # Ensure vectors are normalized for correct geometric calculation
    v0_norm = v0 / torch.norm(v0, p=2, dim=-1, keepdim=True)
    v1_norm = v1 / torch.norm(v1, p=2, dim=-1, keepdim=True)
    
    # Calculate the dot product
    dot = torch.sum(v0_norm * v1_norm)
    
    # If the vectors are nearly collinear, fallback to linear interpolation and re-normalize
    if torch.abs(dot) > DOT_THRESHOLD:
        result = (1 - t) * v0_norm + t * v1_norm
        return result / torch.norm(result, p=2, dim=-1, keepdim=True)

    # Standard SLERP formula
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    
    if sin_theta == 0: # Safety check for identical vectors
        return v0_norm

    scale_v0 = torch.sin((1 - t) * theta) / sin_theta
    scale_v1 = torch.sin(t * theta) / sin_theta
    
    return scale_v0 * v0_norm + scale_v1 * v1_norm
# --- 1. CONFIGURATION ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
print(f"Using device: {DEVICE}, dtype: {TORCH_DTYPE}")

BASE_PIPELINE_MODEL_ID = "stabilityai/stable-diffusion-2-1-unclip"
CLIP_COMPONENTS_REPO_ID = "stabilityai/stable-diffusion-2-1-unclip"

# --- SOURCE DATA CONFIGURATION ---
SOURCE_DOMAIN_ROOT = "/home/mosama97/UBCO/research/PACS"  # Example: Update to your actual dataset root
# Adjusted OUTPUT_DIR for the new structure
OUTPUT_DIR_BASE = "./generated_Domains_lerp_two_Interpolated_PACS"
os.makedirs(OUTPUT_DIR_BASE, exist_ok=True)

# --- GENERATION PARAMETERS ---
NEGATIVE_PROMPT = "ugly, blurry, malformed, deformed, noisy, text, watermark, signature"
NUM_INFERENCE_STEPS = 40
GUIDANCE_SCALE = 10
NOISE_LEVEL = 10
SEED = 12345
# MODIFIED: Number of images per specific weight combination for a pair
N_IMAGES_PER_WEIGHT_COMBO = 10

# --- INTERPOLATION CONFIGURATION (MODIFIED) ---
# We only have one configuration: pairwise mixing with specific weights
INTERPOLATION_CONFIGS = [
    # {
    #     "name": "Triple_Way_LERP", "k": 3, "method": "lerp", # Standard 3-way linear interpolation
    #     "weights_sets": [
    #         [0.4, 0.3, 0.3], [0.3, 0.4, 0.3], [0.3, 0.3, 0.4],
    #     ]
    # },
    # {
    #     "name": "Pairwise_SLERP", "k": 2, "method": "slerp", # Pairwise spherical (arc) interpolation
    #     "weights_sets": [
    #         [0.6, 0.4], [0.5, 0.5], [0.4, 0.6],
    #     ]
    # },
    {
        "name": "Pairwise_LERP", "k": 2, "method": "lerp", # Pairwise linear (chord) interpolation for comparison
        "weights_sets": [
            [0.7, 0.3], [0.6, 0.4], [0.5, 0.5], [0.4, 0.6], [0.3, 0.7],
        ]
    },
]
MAX_PROTOTYPE_IMAGES_PER_CLASS_DOMAIN = 50

# --- Load Models (CLIP for embeddings, and your custom generation pipeline) ---
print("Loading CLIP vision model and feature extractor...")
image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    CLIP_COMPONENTS_REPO_ID, subfolder="image_encoder", torch_dtype=TORCH_DTYPE
).to(DEVICE)
feature_extractor = CLIPImageProcessor.from_pretrained(
    CLIP_COMPONENTS_REPO_ID, subfolder="feature_extractor"
)

print(f"Loading custom unCLIP pipeline from: {BASE_PIPELINE_MODEL_ID}...")
# Assuming StableUnCLIPImg2ImgEmbedsPipeline is defined in the same file or imported correctly
pipe = StableUnCLIPImg2ImgEmbedsPipeline.from_pretrained(
    BASE_PIPELINE_MODEL_ID, torch_dtype=TORCH_DTYPE
).to(DEVICE)
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
# pipe.enable_xformers_memory_efficient_attention() # If available

# --- HELPER FUNCTIONS --- (These remain the same as your original script)
def get_image_paths_from_dir(directory_path):
    supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
    image_files = []
    if not os.path.isdir(directory_path):
        return []
    for f_name in os.listdir(directory_path):
        if f_name.lower().endswith(supported_formats):
            image_files.append(os.path.join(directory_path, f_name))
    return image_files

@torch.no_grad()
def get_image_embedding(image_path, image_encoder_model, feature_extractor_model, device, dtype):
    if not os.path.exists(image_path):
        return None
    try:
        image = Image.open(image_path).convert("RGB")
        processed_image = feature_extractor_model(images=image, return_tensors="pt").pixel_values
        processed_image = processed_image.to(device, dtype=dtype)
        embeds = image_encoder_model(processed_image).image_embeds
        return embeds
    except Exception as e:
        print(f"Error generating embedding for {image_path}: {e}")
        return None

def discover_class_domain_data(root_dir):
    class_domain_data = {}
    if not os.path.isdir(root_dir):
        print(f"Error: Source domain root directory '{root_dir}' not found.")
        return class_domain_data
    print(f"Scanning for domains and classes in: {root_dir}")
    for domain_name in os.listdir(root_dir):
        domain_path = os.path.join(root_dir, domain_name)
        if not os.path.isdir(domain_path) or domain_name.startswith('.'):
            continue
        for class_name in os.listdir(domain_path):
            class_path = os.path.join(domain_path, class_name)
            if not os.path.isdir(class_path) or class_name.startswith('.'):
                continue
            image_files = get_image_paths_from_dir(class_path)
            if image_files:
                if class_name not in class_domain_data:
                    class_domain_data[class_name] = {}
                class_domain_data[class_name][domain_name] = image_files
    if not class_domain_data:
        print("No class/domain data found.")
    else:
        print("\n--- Discovered Data Summary ---")
        for cn, d_map in class_domain_data.items():
            d_counts = [f"{dn}({len(imgs)})" for dn, imgs in d_map.items()]
            print(f"Class '{cn}': Found in domains {', '.join(d_counts)}")
        print("--- End Summary ---\n")
    return class_domain_data

@torch.no_grad()
def calculate_prototypical_embeddings(class_data_map, image_encoder_model, fe_model, dev, dtype, max_images=100):
    print("\n--- Calculating Prototypical Embeddings ---")
    prototypical_embeds = {}
    for class_name, domains_map in tqdm(class_data_map.items(), desc="Processing Classes"):
        prototypical_embeds[class_name] = {}
        for domain_name, image_paths in tqdm(domains_map.items(), desc=f"  Domains for {class_name}", leave=False):
            embeddings_list = []
            selected_image_paths = image_paths[:max_images] if len(image_paths) > max_images else image_paths
            if not selected_image_paths:
                print(f"Warning: No images for {class_name} in {domain_name} to create prototype.")
                continue
            for img_path in selected_image_paths:
                embed = get_image_embedding(img_path, image_encoder_model, fe_model, dev, dtype)
                if embed is not None:
                    embeddings_list.append(embed)
            if embeddings_list:
                stacked_embeddings = torch.cat(embeddings_list, dim=0)
                mean_embedding = torch.mean(stacked_embeddings, dim=0, keepdim=True)
                prototypical_embeds[class_name][domain_name] = mean_embedding
            else:
                print(f"Warning: Could not generate any embeddings for {class_name} in {domain_name}.")
    print("--- Prototypical Embeddings Calculation Complete ---")
    return prototypical_embeds

# --- 2. PREPARE DATA & PROTOTYPES ---
class_domain_image_paths = discover_class_domain_data(SOURCE_DOMAIN_ROOT)
if not class_domain_image_paths:
    print(f"No data found in '{SOURCE_DOMAIN_ROOT}'. Exiting.")
    exit()

prototypes = calculate_prototypical_embeddings(
    class_domain_image_paths, image_encoder, feature_extractor, DEVICE, TORCH_DTYPE, MAX_PROTOTYPE_IMAGES_PER_CLASS_DOMAIN
)

# --- 3. MAIN GENERATION LOOP (MODIFIED) ---
main_generator = torch.Generator(device=DEVICE).manual_seed(SEED)
total_images_generated_count = 0


for class_name, domain_prototypes_map in tqdm(prototypes.items(), desc="Processing Classes"):
    print(f"\n===== Processing Class: '{class_name}' =====")

    available_domains_for_class = list(domain_prototypes_map.keys())
    if not available_domains_for_class:
        print(f"  No prototypical embeddings available for class '{class_name}'. Skipping.")
        continue

    # This outer loop iterates through your different interpolation STRATEGIES
    for interp_config in INTERPOLATION_CONFIGS:
        config_name = interp_config["name"]
        k_domains_to_mix = interp_config["k"]
        weights_sets_for_config = interp_config["weights_sets"]
        interp_method = interp_config.get("method", "lerp") # Default to 'lerp'

        print(f"  Applying Interpolation Strategy: '{config_name}' (mixing {k_domains_to_mix} domains)")
        # --- Add safety check for SLERP ---
        if interp_method == "slerp" and k_domains_to_mix != 2:
            print(f"    ERROR: SLERP method is only supported for k=2. Skipping config '{config_name}'.")
            continue
        if len(available_domains_for_class) < k_domains_to_mix:
            print(f"    Skipping (needs {k_domains_to_mix}, found {len(available_domains_for_class)}) for class '{class_name}'.")
            continue

        # This loop iterates through unique combinations of domains
        for domain_combination_tuple in tqdm(combinations(available_domains_for_class, k_domains_to_mix), desc=f"  Domain Combinations (k={k_domains_to_mix})", leave=False):
            
            current_proto_embeds_list = [domain_prototypes_map.get(dom_name) for dom_name in domain_combination_tuple]
            if any(embed is None for embed in current_proto_embeds_list):
                continue

            # --- CORRECTED & GENERALIZED FOLDER NAMING ---
            sorted_domain_names = sorted(list(domain_combination_tuple))
            source_domains_identifier_str = "_x_".join(sorted_domain_names) # e.g., "Art_x_Clipart_x_Product"
            synthetic_domain_folder_name = f"zSynDomain_M_{interp_method.upper()}_K{k_domains_to_mix}_{source_domains_identifier_str}"
            target_class_dir_for_pair = os.path.join(OUTPUT_DIR_BASE, synthetic_domain_folder_name, class_name)            

            # Resume Logic
            if os.path.isdir(target_class_dir_for_pair):
                if get_image_paths_from_dir(target_class_dir_for_pair):
                    print(f"      SKIPPING: Class '{class_name}' in domain '{synthetic_domain_folder_name}' already exists.")
                    continue
            
            os.makedirs(target_class_dir_for_pair, exist_ok=True)
            print(f"    Generating for Synthetic Domain: '{synthetic_domain_folder_name}', Class: '{class_name}'")

            # This loop iterates through the weight sets for the current strategy
            for weights_list in weights_sets_for_config:
                if len(weights_list) != k_domains_to_mix: continue
                if not np.isclose(sum(weights_list), 1.0): continue

                interpolated_embed = None # Initialize
                if interp_method == "slerp":
                    # We already know k=2 here
                    v0, v1 = current_proto_embeds_list[0], current_proto_embeds_list[1]
                    t = weights_list[1] # t is the weight of the second vector
                    interpolated_embed = hybrid_slerp(v0, v1, t)
                else: # Default to LERP for any k
                    interpolated_embed = torch.zeros_like(current_proto_embeds_list[0])
                    for i, weight_val in enumerate(weights_list):
                        interpolated_embed += weight_val * current_proto_embeds_list[i]

                if interpolated_embed is None: continue

                # GENERALIZED FILENAME WEIGHTS
                weights_identifier_str = "_".join([f"{w:.2f}p" for w in weights_list])

                # Use a simple, robust dynamic prompt
                clean_class_name = class_name.replace('_', ' ')
                dynamic_prompt = f"a high-quality photo of a {clean_class_name} in the style of {', '.join(domain_combination_tuple)}"

                # GENERALIZED LOG MESSAGE
                weights_log_str = " / ".join([f"{domain_combination_tuple[i]} {weights_list[i]*100:.0f}%" for i in range(k_domains_to_mix)])
                print(f"      Weights: {weights_log_str} -> Generating {N_IMAGES_PER_WEIGHT_COMBO} images")

                try:
                    with torch.no_grad():
                        # ... (pipeline call remains the same)
                        generated_images_list = pipe(
                            image_embeds=interpolated_embed,
                            prompt=dynamic_prompt,
                            negative_prompt=NEGATIVE_PROMPT,
                            num_inference_steps=NUM_INFERENCE_STEPS,
                            guidance_scale=GUIDANCE_SCALE,
                            noise_level=NOISE_LEVEL,
                            num_images_per_prompt=N_IMAGES_PER_WEIGHT_COMBO,
                            generator=main_generator,
                            output_type="pil",
                        ).images
                    
                    if not generated_images_list: continue

                    for img_idx, pil_image in enumerate(generated_images_list):
                        if pil_image:
                            base_filename_part = f"img_{img_idx:03d}_cls_{class_name}"
                            
                            # --- CORRECTED FILENAME ---
                            # Use the generalized weights_identifier_str created above
                            filename = f"{base_filename_part}_weights_{weights_identifier_str}.png"
                            
                            output_path = os.path.join(target_class_dir_for_pair, filename)
                            pil_image.save(output_path)
                            total_images_generated_count += 1
                        
                except Exception as e:
                    print(f"        ERROR during image generation for {class_name} with weights {weights_list}: {e}")
                    import traceback
                    traceback.print_exc()

print(f"\n--- Domain generation process complete. Total new images generated this run: {total_images_generated_count} ---")

