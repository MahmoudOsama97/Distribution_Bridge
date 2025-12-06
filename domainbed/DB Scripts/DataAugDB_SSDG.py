import inspect
from typing import Any, Callable, Dict, List, Optional, Union

import PIL.Image
import torch
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection
from transformers import CLIPVisionModelWithProjection
from diffusers.pipelines.stable_diffusion.stable_unclip_image_normalizer import StableUnCLIPImageNormalizer
from diffusers import EulerDiscreteScheduler
from PIL import ImageFile # Import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

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


# ==============================================================================
# === 1. CONFIGURATION =========================================================
# ==============================================================================

# --- METHOD SELECTION ---
# Choose which SSDG method to run. Options: "STYLE_JITTER", "STYLE_ARCHETYPE"
METHOD_CHOICE = "STYLE_JITTER"

# --- Model and Device ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
print(f"Using device: {DEVICE}, dtype: {TORCH_DTYPE}")

BASE_PIPELINE_MODEL_ID = "stabilityai/stable-diffusion-2-1-unclip"
CLIP_COMPONENTS_REPO_ID = "stabilityai/stable-diffusion-2-1-unclip"

# --- Path Configuration ---
# This is the single source directory. It should contain subfolders for each original domain.
SOURCE_DOMAIN_ROOT = "/home/mosama97/UBCO/research/PEFTFusion/data/PACS" # Example: PACS
# Base output directory where new folders will be created.
OUTPUT_DIR_BASE = "./generated_Domains_SSDG_PACS"
os.makedirs(OUTPUT_DIR_BASE, exist_ok=True)

# --- General Generation Parameters ---
NEGATIVE_PROMPT = "ugly, blurry, malformed, deformed, noisy, text, watermark, signature"
NUM_INFERENCE_STEPS = 40
GUIDANCE_SCALE = 10
NOISE_LEVEL = 10
SEED = 12345
N_IMAGES_PER_PROMPT = 15 # Number of images to generate per prototype

# --- SSDG Method-Specific Hyperparameters ---
# For METHOD_CHOICE = "STYLE_JITTER"
N_JITTERED_DOMAINS_PER_ORIGINAL = 2 # N: How many noisy variations to create per original domain
NOISE_STRENGTH = 0.15              # The magnitude of the random noise. Tune this value.

# For METHOD_CHOICE = "STYLE_ARCHETYPE"
# You must create this directory and place representative images inside (e.g., in subfolders 'sketch', 'cartoon').
ARCHETYPE_IMAGES_ROOT = "./style_archetypes"
# How many different archetype styles to mix with each original domain.
N_ARCHETYPE_DOMAINS_PER_ORIGINAL = 2
# The interpolation factor (t) for SLERP. 0.0 is pure original, 1.0 is pure archetype.
ARCHETYPE_INTERP_FACTOR = 0.6




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
def calculate_prototypical_embeddings(class_data_map, image_encoder_model, fe_model, dev, dtype):
    """Calculates a mean prototype embedding for each class/domain using ALL available images."""
    print("\n--- Calculating Prototypical Embeddings ---")
    prototypical_embeds = {}
    for class_name, domains_map in tqdm(class_data_map.items(), desc="Processing Classes"):
        prototypical_embeds[class_name] = {}
        for domain_name, image_paths in tqdm(domains_map.items(), desc=f"  Domains for {class_name}", leave=False):
            embeddings_list = []
            # Use all available image paths, no more max_images limit
            if not image_paths:
                print(f"Warning: No images for {class_name} in {domain_name} to create prototype.")
                continue
            for img_path in image_paths:
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

def generate_images_for_embed(pipe, embed, target_dir, class_name, identifier_str, log_str):
    """A shared function to run the generation pipeline and save images."""
    global total_images_generated_count, main_generator # Use global variables
    print(f"      {log_str} -> Generating {N_IMAGES_PER_PROMPT} images")
    
    # Resume Logic: Skip if the directory already has images
    if os.path.isdir(target_dir) and get_image_paths_from_dir(target_dir):
        print(f"      SKIPPING: Directory '{target_dir}' already contains images.")
        return
    os.makedirs(target_dir, exist_ok=True)
        
    try:
        with torch.no_grad():
            clean_class_name = class_name.replace('_', ' ')
            dynamic_prompt = f"a high-quality photo of a {clean_class_name}"
            
            generated_images_list = pipe(
                image_embeds=embed,
                prompt=dynamic_prompt,
                negative_prompt=NEGATIVE_PROMPT,
                num_inference_steps=NUM_INFERENCE_STEPS,
                guidance_scale=GUIDANCE_SCALE,
                noise_level=NOISE_LEVEL,
                num_images_per_prompt=N_IMAGES_PER_PROMPT,
                generator=main_generator,
            ).images
        
        if not generated_images_list: return

        for img_idx, pil_image in enumerate(generated_images_list):
            filename = f"img_{img_idx:03d}_cls_{class_name}_{identifier_str}.png"
            output_path = os.path.join(target_dir, filename)
            pil_image.save(output_path)
            total_images_generated_count += 1
            
    except Exception as e:
        print(f"        ERROR during image generation for {class_name} ({identifier_str}): {e}")

# ==============================================================================
# === 4. DATA PREPARATION ======================================================
# ==============================================================================

class_domain_image_paths = discover_class_domain_data(SOURCE_DOMAIN_ROOT)
if not class_domain_image_paths:
    print(f"No data found in '{SOURCE_DOMAIN_ROOT}'. Exiting.")
    exit()

source_prototypes = calculate_prototypical_embeddings(
    class_domain_image_paths, image_encoder, feature_extractor, DEVICE, TORCH_DTYPE
)

archetype_prototypes = None
if METHOD_CHOICE == "STYLE_ARCHETYPE":
    print("\n--- Preparing Style Archetype Prototypes ---")
    if not os.path.isdir(ARCHETYPE_IMAGES_ROOT):
        raise FileNotFoundError(f"METHOD_CHOICE is 'STYLE_ARCHETYPE' but the directory '{ARCHETYPE_IMAGES_ROOT}' was not found.")
    
    archetype_image_paths = discover_class_domain_data(ARCHETYPE_IMAGES_ROOT)
    # The "class" for an archetype is the style name itself (e.g., 'sketch')
    archetype_prototypes = calculate_prototypical_embeddings(
        archetype_image_paths, image_encoder, feature_extractor, DEVICE, TORCH_DTYPE
    )
    if not archetype_prototypes:
         raise ValueError(f"No archetype prototypes could be calculated from '{ARCHETYPE_IMAGES_ROOT}'.")

# ==============================================================================
# === 5. MAIN GENERATION LOOP ==================================================
# ==============================================================================

main_generator = torch.Generator(device=DEVICE).manual_seed(SEED)
total_images_generated_count = 0

print(f"\n===== STARTING SSDG GENERATION using method: {METHOD_CHOICE} =====")

for class_name, domain_prototypes_map in tqdm(source_prototypes.items(), desc="Processing Classes"):
    print(f"\n--- Processing Class: '{class_name}' ---")
    
    for original_domain_name, E_source in domain_prototypes_map.items():
        print(f"  Using source domain: '{original_domain_name}'")
        
        # --- METHOD 1: STYLE JITTER LOGIC ---
        if METHOD_CHOICE == "STYLE_JITTER":
            for i in range(N_JITTERED_DOMAINS_PER_ORIGINAL):
                noise = torch.randn_like(E_source) * NOISE_STRENGTH
                E_pseudo_noisy = E_source + noise
                E_pseudo_normalized = E_pseudo_noisy / torch.norm(E_pseudo_noisy, p=2, dim=-1, keepdim=True)
                
                synthetic_domain_folder_name = f"zSyn_{original_domain_name}_Jitter_{i+1}"
                target_dir = os.path.join(OUTPUT_DIR_BASE, synthetic_domain_folder_name, class_name)
                identifier_str = f"jitter{i+1}_str{NOISE_STRENGTH:.2f}"
                log_str = f"Jitter #{i+1} (Strength: {NOISE_STRENGTH})"
                
                generate_images_for_embed(pipe, E_pseudo_normalized, target_dir, class_name, identifier_str, log_str)

        # --- METHOD 2: STYLE ARCHETYPE LOGIC ---
        elif METHOD_CHOICE == "STYLE_ARCHETYPE":
            if not archetype_prototypes: continue

            available_archetypes = list(archetype_prototypes.keys())
            # If N is specified, take a random sample of N archetypes, otherwise use all
            if N_ARCHETYPE_DOMAINS_PER_ORIGINAL < len(available_archetypes):
                selected_archetypes = random.sample(available_archetypes, N_ARCHETYPE_DOMAINS_PER_ORIGINAL)
            else:
                selected_archetypes = available_archetypes
            
            for archetype_name in selected_archetypes:
                try:
                    # Archetype prototypes are class-agnostic, we get the prototype for the style "class"
                    E_archetype = next(iter(archetype_prototypes[archetype_name].values()))
                except (StopIteration, KeyError):
                    print(f"    Warning: No prototype found for archetype '{archetype_name}'. Skipping.")
                    continue

                E_pseudo_slerp = slerp(E_source, E_archetype, t=ARCHETYPE_INTERP_FACTOR)
                
                synthetic_domain_folder_name = f"zSyn_{original_domain_name}_x_{archetype_name}"
                target_dir = os.path.join(OUTPUT_DIR_BASE, synthetic_domain_folder_name, class_name)
                identifier_str = f"{archetype_name}_{ARCHETYPE_INTERP_FACTOR:.2f}t"
                log_str = f"SLERP with '{archetype_name}' (t={ARCHETYPE_INTERP_FACTOR})"

                generate_images_for_embed(pipe, E_pseudo_slerp, target_dir, class_name, identifier_str, log_str)

print(f"\n--- Domain generation process complete. Total new images generated this run: {total_images_generated_count} ---")