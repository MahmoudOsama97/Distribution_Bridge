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

        image = self.image_processor.postprocess(image, output_type=output_type)

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

    # Modified _encode_image to take pre-computed image_embeds
    def _encode_image(
        self,
        image_embeds: torch.Tensor, # These are RAW image_embeds from an external CLIP model
        device: torch.device,
        num_images_per_prompt: int,
        do_classifier_free_guidance: bool,
        noise_level: int,
        generator: Optional[torch.Generator],
    ):
        dtype = next(self.image_encoder.parameters()).dtype # Use pipeline's image_encoder for dtype
        image_embeds = image_embeds.to(device=device, dtype=dtype)

        image_embeds = self.image_normalizer.scale(image_embeds)

        image_embeds = self.noise_image_embeddings(
            image_embeds=image_embeds,
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

        if image_embeds is None:
            raise ValueError("`image_embeds` must be provided and cannot be None.")
        if not isinstance(image_embeds, torch.Tensor):
            raise TypeError(f"`image_embeds` must be a torch.Tensor but is {type(image_embeds)}")

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

        if prompt_embeds is not None:
            if image_embeds.shape[0] != prompt_embeds.shape[0]:
                raise ValueError(
                    f"The batch size of `image_embeds` ({image_embeds.shape[0]}) must match the batch size of `prompt_embeds` ({prompt_embeds.shape[0]}) when `prompt_embeds` are provided."
                )
    @torch.no_grad()
    def __call__(
        self,
        image_embeds: torch.Tensor,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 20,
        guidance_scale: float = 10.0,
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
            if isinstance(image_embeds, torch.Tensor):
                 prompt = [""] * image_embeds.shape[0]
            else:
                 prompt = ""

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
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if image_embeds.shape[0] != batch_size:
            raise ValueError(
                f"The batch size of `image_embeds` ({image_embeds.shape[0]}) must match the batch size "
                f"derived from `prompt` or `prompt_embeds` ({batch_size}). If using a single prompt string "
                f"for multiple image_embeds, please provide the prompt as a list of strings, "
                f"e.g., prompt=['my prompt'] * num_image_embeds."
            )

        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0

        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        final_prompt_embeds, final_negative_prompt_embeds = self.encode_prompt( # Renamed for clarity
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
            clip_skip=clip_skip,
        )
        if do_classifier_free_guidance:
            # Concatenate negative and positive prompt embeddings
            text_embeddings_for_unet = torch.cat([final_negative_prompt_embeds, final_prompt_embeds])
        else:
            text_embeddings_for_unet = final_prompt_embeds


        processed_image_embeds = self._encode_image(
            image_embeds=image_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            noise_level=noise_level,
            generator=generator,
        )

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            text_embeddings_for_unet.dtype, # Match dtype of text embeddings
            device,
            generator,
            latents,
        )

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=text_embeddings_for_unet,
                    class_labels=processed_image_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                ).sample

                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                if callback is not None and i % callback_steps == 0:
                    step_idx = i // getattr(self.scheduler, "order", 1)
                    callback(step_idx, t, latents)

                if XLA_AVAILABLE:
                    xm.mark_step()
                progress_bar.update()


        if not output_type == "latent":
            latents = latents / getattr(self.vae.config, "scaling_factor", 0.18215)
            image = self.vae.decode(latents, return_dict=False)[0]
        else:
            image = latents

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


# --- 1. CONFIGURATION ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
print(f"Using device: {DEVICE}, dtype: {TORCH_DTYPE}")

BASE_PIPELINE_MODEL_ID = "stabilityai/stable-diffusion-2-1-unclip"
CLIP_COMPONENTS_REPO_ID = "stabilityai/stable-diffusion-2-1-unclip"

# --- SOURCE DATA CONFIGURATION ---
SOURCE_DOMAIN_ROOT = "/home/mosama97/UBCO/research/PEFTFusion/data/office_home"  # Example: Update to your actual dataset root
# Adjusted OUTPUT_DIR for the new structure
OUTPUT_DIR_BASE = "./generated_Domains_Pairwise_Interpolated"
os.makedirs(OUTPUT_DIR_BASE, exist_ok=True)

# --- GENERATION PARAMETERS ---
NEGATIVE_PROMPT = "ugly, blurry, malformed, deformed, noisy, text, watermark, signature"
NUM_INFERENCE_STEPS = 40
GUIDANCE_SCALE = 10
NOISE_LEVEL = 10
SEED = 12345
# MODIFIED: Number of images per specific weight combination for a pair
N_IMAGES_PER_WEIGHT_COMBO = 15

# --- INTERPOLATION CONFIGURATION (MODIFIED) ---
# We only have one configuration: pairwise mixing with specific weights
INTERPOLATION_CONFIGS = [
    {
        "name": "Pairwise_Specific_Ratios", "k": 2,
        "weights_sets": [
            [0.7, 0.3], # 70% of first domain in pair, 30% of second
            [0.5, 0.5], # 50% of first domain in pair, 50% of second
            [0.3, 0.7]  # 30% of first domain in pair, 70% of second
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

    # This outer loop will only run once due to the modified INTERPOLATION_CONFIGS
    for interp_config in INTERPOLATION_CONFIGS: # No tqdm needed here if only one config
        # config_name_short = interp_config["name"] # e.g., "Pairwise_Specific_Ratios" (can be used in logs if needed)
        k_domains_to_mix = interp_config["k"]       # This will be 2
        weights_sets_for_config = interp_config["weights_sets"] # [[0.7,0.3], [0.5,0.5], [0.3,0.7]]

        if len(available_domains_for_class) < k_domains_to_mix:
            print(f"    Skipping (needs {k_domains_to_mix} domains, found {len(available_domains_for_class)}) for class '{class_name}'.")
            continue

        # This loop iterates through unique pairs of available domains, e.g., ('Art', 'Photo')
        for domain_combination_tuple in tqdm(combinations(available_domains_for_class, k_domains_to_mix), desc=f"  Domain Pairs for {class_name}", leave=False):
            # domain_combination_tuple is like ('DomainA', 'DomainB')

            current_proto_embeds_list = [domain_prototypes_map.get(dom_name) for dom_name in domain_combination_tuple]
            if any(embed is None for embed in current_proto_embeds_list):
                missing_domains = [dom_name for i, dom_name in enumerate(domain_combination_tuple) if current_proto_embeds_list[i] is None]
                print(f"      Warning: Skipping domain combination {domain_combination_tuple} for class '{class_name}' due to missing prototypes for: {', '.join(missing_domains)}.")
                continue

            # Create a consistent identifier for the PAIR of source domains for the directory name
            # Sorting ensures ('Art', 'Photo') and ('Photo', 'Art') map to the same directory
            sorted_domain_pair_names = sorted(list(domain_combination_tuple))
            source_domains_pair_identifier_str = f"{sorted_domain_pair_names[0]}_x_{sorted_domain_pair_names[1]}" # e.g., "Art_x_Photo"

            # Define the top-level synthetic domain folder for this PAIR
            synthetic_domain_folder_name = f"SynDomain_Pair_{source_domains_pair_identifier_str}"
            # Define the class-specific directory within this synthetic domain pair folder
            target_class_dir_for_pair = os.path.join(OUTPUT_DIR_BASE, synthetic_domain_folder_name, class_name)
            os.makedirs(target_class_dir_for_pair, exist_ok=True)

            print(f"    Generating for Synthetic Domain: '{synthetic_domain_folder_name}', Class: '{class_name}'")

            # This loop iterates through the three weight sets: [[0.7,0.3], [0.5,0.5], [0.3,0.7]]
            for weights_list in weights_sets_for_config:
                if len(weights_list) != k_domains_to_mix: # Should not happen with current config
                    print(f"      Warning: Weights list length mismatch. Skipping weights: {weights_list}")
                    continue
                if not np.isclose(sum(weights_list), 1.0): # Should not happen
                    print(f"      Warning: Weights {weights_list} do not sum to 1.0. Skipping.")
                    continue

                # Perform weighted interpolation
                # Embeddings in current_proto_embeds_list correspond to domain_combination_tuple's order
                interpolated_embed = torch.zeros_like(current_proto_embeds_list[0])
                for i, weight_val in enumerate(weights_list):
                    interpolated_embed += weight_val * current_proto_embeds_list[i]

                # String representation of weights for filenames
                weights_identifier_str = f"{weights_list[0]:.1f}p_{weights_list[1]:.1f}p" # e.g., "0.7p_0.3p"

                dynamic_prompt = "" # Using empty prompt as per original script

                print(f"      Weights: {domain_combination_tuple[0]} {weights_list[0]*100:.0f}% / {domain_combination_tuple[1]} {weights_list[1]*100:.0f}% -> Generating {N_IMAGES_PER_WEIGHT_COMBO} images")

                try:
                    with torch.no_grad():
                        output_pipeline = pipe(
                            image_embeds=interpolated_embed,
                            prompt=dynamic_prompt,
                            negative_prompt=NEGATIVE_PROMPT,
                            num_inference_steps=NUM_INFERENCE_STEPS,
                            guidance_scale=GUIDANCE_SCALE,
                            noise_level=NOISE_LEVEL,
                            num_images_per_prompt=N_IMAGES_PER_WEIGHT_COMBO, # Generate 15 images
                            generator=main_generator,
                            output_type="pil",
                        )
                    generated_images_list_from_pipe = output_pipeline.images
                    generated_items_from_pipe = output_pipeline.images # This is what comes from pipe(...).images

                    if generated_items_from_pipe is None:
                        print(f"          Error: Pipeline returned None for images for {synthetic_domain_folder_name}, Class: {class_name}, Weights: {weights_identifier_str}")
                        continue # Skip to the next weight_list or domain_combination

                    # Ensure generated_items_from_pipe is iterable (it should be a list of PILs or Tensors)
                    if not hasattr(generated_items_from_pipe, '__iter__'):
                        # If it's a single tensor (e.g. BxCxHxW), wrap it in a list for consistent processing
                        if isinstance(generated_items_from_pipe, torch.Tensor):
                            print(f"          Warning: Pipeline output 'images' was a single tensor. Wrapping in a list. Shape: {generated_items_from_pipe.shape}")
                            generated_items_from_pipe = [generated_items_from_pipe]
                        else:
                            print(f"          Error: Pipeline output 'images' is not iterable and not a tensor. Type: {type(generated_items_from_pipe)}. Skipping.")
                            continue


                    for img_idx, item_from_pipe in enumerate(generated_items_from_pipe):
                        pil_image_to_save = None  # Initialize to None

                        if isinstance(item_from_pipe, Image.Image):
                            pil_image_to_save = item_from_pipe
                        elif isinstance(item_from_pipe, torch.Tensor):
                            print(f"          Info: Item {img_idx} from pipeline is a Tensor. Attempting conversion. Shape: {item_from_pipe.shape}")
                            tensor_to_convert = item_from_pipe.cpu()

                            # Ensure tensor is in the correct shape (C, H, W)
                            # VAE output is typically B,C,H,W. If item_from_pipe is one image from batch, it's C,H,W
                            # If item_from_pipe is the whole batch tensor (B,C,H,W) because generated_items_from_pipe was a single tensor:
                            if tensor_to_convert.ndim == 4:
                                if tensor_to_convert.shape[0] == 1: # Single image in a batch
                                    tensor_to_convert = tensor_to_convert.squeeze(0)
                                else: # Multiple images in this single tensor item, process them individually
                                    print(f"          Info: Tensor item {img_idx} contains a batch of {tensor_to_convert.shape[0]} images. Processing them individually.")
                                    # This case requires a nested loop or adjustment, for now, we'll try to process the first.
                                    # A more robust solution would be to ensure generated_items_from_pipe is always a list of individual image tensors/PILs.
                                    # For now, let's assume if it's a 4D tensor here, it's one we should unbatch or take the first.
                                    # This part might need refinement based on exact pipeline output structure if it's a batched tensor here.
                                    # Safest is to assume if it's a 4D tensor here, it's a batch that wasn't properly split.
                                    # Let's try processing each image in this tensor batch:
                                    temp_pil_images = []
                                    for single_img_tensor in tensor_to_convert: # Iterate over batch dimension
                                        current_tensor_to_pil = single_img_tensor
                                        if current_tensor_to_pil.min() < 0.0: # Normalize if needed (e.g., -1 to 1 range)
                                            current_tensor_to_pil = (current_tensor_to_pil / 2 + 0.5)
                                        current_tensor_to_pil = current_tensor_to_pil.clamp(0, 1)
                                        try:
                                            temp_pil_images.append(transforms.ToPILImage()(current_tensor_to_pil))
                                        except Exception as conversion_e_batch:
                                            print(f"            Error converting sub-image in tensor item {img_idx} to PIL: {conversion_e_batch}")
                                    # For this loop structure, we'll just take the first successfully converted image if any
                                    if temp_pil_images:
                                        pil_image_to_save = temp_pil_images[0] # Or handle all temp_pil_images
                                        if len(temp_pil_images) > 1:
                                            print(f"            Warning: Processed first image of a batch of {len(temp_pil_images)} found in tensor item {img_idx}. Others ignored in this loop.")
                                    # This nested batch handling is a bit complex here; ideal is that items are already individual.
                                    # Let's simplify and assume if item_from_pipe is 4D, it's an error in upstream splitting.
                                    # Reverting to simpler logic: if tensor_to_convert is 4D, squeeze if B=1.
                                    if tensor_to_convert.ndim == 4 and tensor_to_convert.shape[0] == 1:
                                         tensor_to_convert = tensor_to_convert.squeeze(0)
                                    elif tensor_to_convert.ndim == 4 and tensor_to_convert.shape[0] > 1:
                                        print(f"          Error: Item {img_idx} is a batched tensor ({tensor_to_convert.shape}). Expected individual image tensor or PIL. Skipping this item.")
                                        continue


                            # At this point, tensor_to_convert should be 3D (C,H,W) or 2D (H,W for grayscale)
                            if tensor_to_convert.ndim == 2: # Grayscale H, W -> add channel dim
                                tensor_to_convert = tensor_to_convert.unsqueeze(0)

                            if tensor_to_convert.ndim != 3:
                                print(f"          Error: Tensor item {img_idx} has unexpected dimensions after processing: {tensor_to_convert.shape}. Skipping.")
                                continue

                            # Normalize if values suggest it's in -1 to 1 range (common for VAE outputs)
                            if tensor_to_convert.min() < 0.0:
                                tensor_to_convert = (tensor_to_convert / 2 + 0.5)
                            
                            tensor_to_convert = tensor_to_convert.clamp(0, 1) # Ensure 0-1 range

                            try:
                                pil_image_to_save = transforms.ToPILImage()(tensor_to_convert)
                            except Exception as conversion_e:
                                print(f"          Error converting tensor item {img_idx} to PIL: {conversion_e}")
                                print(f"          Tensor shape: {tensor_to_convert.shape}, dtype: {tensor_to_convert.dtype}, min: {tensor_to_convert.min()}, max: {tensor_to_convert.max()}")
                                continue # Skip this problematic item
                        else:
                            print(f"          Warning: Item {img_idx} from pipeline is of unexpected type: {type(item_from_pipe)}. Skipping.")
                            continue

                        # Now, pil_image_to_save is either a PIL.Image.Image or None
                        if pil_image_to_save: # This check is now safe
                            base_filename_part = f"img_{img_idx:03d}_cls_{class_name}"
                            # Ensure the current weights_list is used for the filename, corresponding to the outer loop
                            current_weights_identifier_str = f"{weights_list[0]:.1f}p_{weights_list[1]:.1f}p" # Re-derive or pass correctly
                            filename = f"{base_filename_part}_weights_{current_weights_identifier_str}.png"
                            output_path = os.path.join(target_class_dir_for_pair, filename)
                            try:
                                pil_image_to_save.save(output_path)
                                total_images_generated_count += 1
                            except Exception as save_e:
                                print(f"          Error saving image {output_path}: {save_e}")
                        else:
                             # This will catch cases where conversion failed or item was not image/tensor initially suitable for conversion
                             print(f"          Error: Image item {img_idx} could not be processed/converted for saving ({synthetic_domain_folder_name}/{class_name}, weights {weights_list}). Item was originally type: {type(item_from_pipe)}")

                    # This print statement was inside the loop, should be outside the img_idx loop if it's a summary for the weight set
                    # Original placement was: if generated_images_list_from_pipe and final_pil_image_to_save:
                    # Let's make it more accurate based on actual saves
                    # This part is tricky because `final_pil_image_to_save` is from the loop.
                    # A simple count check or a flag would be better.
                    # For now, removing this potentially misleading print. A count of saved images per batch can be added if needed.
                except Exception as e:
                    print(f"        ERROR during image generation or saving for {synthetic_domain_folder_name}, Class: {class_name}, Weights: {weights_identifier_str}: {e}")
                    import traceback
                    traceback.print_exc()
                    if "CUDA out of memory" in str(e) and DEVICE == "cuda":
                        print("        CUDA out of memory. Consider reducing N_IMAGES_PER_WEIGHT_COMBO or image resolution if possible.")
                        torch.cuda.empty_cache()

print(f"\n--- Domain generation process complete. Total unique images generated and saved: {total_images_generated_count} ---")