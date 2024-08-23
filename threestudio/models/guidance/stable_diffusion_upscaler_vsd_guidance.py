import random
from contextlib import contextmanager
from dataclasses import dataclass, field
import math
import time
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import (
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    StableDiffusionUpscalePipeline,
    UNet2DConditionModel,
)
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.loaders import AttnProcsLayers
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils import randn_tensor
from diffusers.models.embeddings import TimestepEmbedding
from diffusers.models.attention_processor import (
    AttnProcessor2_0,
    LoRAAttnProcessor2_0,
    LoRAXFormersAttnProcessor,
    XFormersAttnProcessor,
)

import threestudio
from threestudio.models.prompt_processors.base import PromptProcessorOutput
from threestudio.utils.base import BaseModule
from threestudio.utils.misc import C, cleanup, parse_version
from threestudio.utils.typing import *

import PIL
import numpy as np

import torch
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR

import torch
import torch.nn.functional as F
import random
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_distributed():
    torch.distributed.init_process_group(backend="nccl")
    torch.cuda.set_device(torch.distributed.get_rank())
    torch.manual_seed(0)

class EMA:
    def __init__(self, beta):
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model, ma_model):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_params, model_params, step_start_ema=2000):
        if self.step < step_start_ema:
            for ema_param, model_param in zip(ema_params, model_params):
                ema_param.data.copy_(model_param.data)
            self.step += 1
            return
        self.update_model_average(ema_params, model_params)
        self.step += 1

class ToWeightsDType(nn.Module):
    def __init__(self, module: nn.Module, dtype: torch.dtype):
        super().__init__()
        self.module = module
        self.dtype = dtype

    def forward(self, x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        return self.module(x).to(self.dtype)

@threestudio.register("stable-diffusion-upscaler-vsd-guidance")
class StableDiffusionUpscalerVSDGuidance(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        pretrained_model_name_or_path: str = "stabilityai/stable-diffusion-x4-upscaler"
        pretrained_model_name_or_path_lora: str = "stabilityai/stable-diffusion-x4-upscaler"
        enable_memory_efficient_attention: bool = False
        enable_sequential_cpu_offload: bool = False
        enable_attention_slicing: bool = False
        enable_channels_last_format: bool = False
        guidance_scale: float = 7.5
        guidance_scale_lora: float = 7.5
        grad_clip: Optional[
            Any
        ] = None  # field(default_factory=lambda: [0, 2.0, 8.0, 1000])
        half_precision_weights: bool = True
        view_dependent_prompting: bool = False

        min_step_percent: float = 0.02
        max_step_percent: float = 0.98
        anneal_start_step: Optional[int] = 5000
        anneal_end_step: Optional[int] = 25000
        random_timestep: bool = True
        anneal_strategy: str = "milestone"
        max_step_percent_annealed: float = 0.5

        step_ratio: float = 0.25
        num_inference_steps: int = 20

        guidance_type: str = 'vsd'

        camera_condition_type: str = 'c2w'
        use_img_loss: bool = False
        lora_cfg_training: bool = True

        t_min_shift_per_stage: float = 0.0
        t_max_shift_per_stage: float = 0.0
        cfg_shift_per_stage: float = 0.0

        lora_n_timestamp_samples: int = 1

    cfg: Config

    def configure(self) -> None:
        threestudio.info(f"Loading Stable Diffusion ...")
        setup_distributed()

        self.weights_dtype = (
            torch.float16 if self.cfg.half_precision_weights else torch.float32
        )

        pipe_kwargs = {
            "tokenizer": None,
            "safety_checker": None,
            "feature_extractor": None,
            "torch_dtype": self.weights_dtype,
        }

        pipe_kwargs = {
            "tokenizer": None,
            "safety_checker": None,
            "feature_extractor": None,
            "torch_dtype": self.weights_dtype,
        }

        @dataclass
        class SubModules:
            pipe: StableDiffusionUpscalePipeline
            pipe_lora: StableDiffusionUpscalePipeline

        pipe = StableDiffusionUpscalePipeline.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            **pipe_kwargs,
        )
        if (
            self.cfg.pretrained_model_name_or_path
            == self.cfg.pretrained_model_name_or_path_lora
        ):
            self.single_model = True
            pipe_lora = pipe
        
        else:
            self.single_model = False
            pipe_lora = StableDiffusionUpscalePipeline(
                self.cfg.pretrained_model_name_or_path_lora,
                **pipe_lora_kwargs,
            )
            del pipe_lora.vae
            cleanup()
            pipe_lora.vae = pipe.vae
        self.submodules = SubModules(pipe=pipe, pipe_lora=pipe_lora)

        if self.cfg.enable_memory_efficient_attention:
            if parse_version(torch.__version__) >= parse_version("2"):
                threestudio.info(
                    "PyTorch2.0 uses memory efficient attention by default."
                )
            elif not is_xformers_available():
                threestudio.warn(
                    "xformers is not available, memory efficient attention is not enabled."
                )
            else:
                self.pipe.enable_xformers_memory_efficient_attention()
                self.pipe_lora.enable_xformers_memory_efficient_attention()

        if self.cfg.enable_sequential_cpu_offload:
            self.pipe.enable_sequential_cpu_offload()
            self.pipe_lora.enable_sequential_cpu_offload()

        if self.cfg.enable_attention_slicing:
            self.pipe.enable_attention_slicing(1)
            self.pipe_lora.enable_attention_slicing(1)

        if self.cfg.enable_channels_last_format:
            self.pipe.unet.to(memory_format=torch.channels_last)
            self.pipe_lora.unet.to(memory_format=torch.channels_last)

        del self.pipe.text_encoder
        if not self.single_model:
            del self.pipe_lora.text_encoder
        cleanup()

        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.unet.parameters():
            p.requires_grad_(False)
        for p in self.unet_lora.parameters():
            p.requires_grad_(False)
    
        self.camera_embedding = ToWeightsDType(
            TimestepEmbedding(16, 1024), self.weights_dtype
        ).to(self.device)
        self.unet_lora.class_embedding = self.camera_embedding
        # set up LoRA layers
        print(self.unet_lora.attn_processors)
        lora_attn_procs = {}
        for name, attn_block in self.unet_lora.attn_processors.items():
            cross_attention_dim = (
                self.unet_lora.config.cross_attention_dim
            )
            if name.startswith("mid_block"):
                hidden_size = self.unet_lora.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.unet_lora.config.block_out_channels))[
                    block_id
                ]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.unet_lora.config.block_out_channels[block_id]
            lora_attn_procs[name] = LoRAAttnProcessor(
                hidden_size=hidden_size, cross_attention_dim=cross_attention_dim
            )

        self.unet_lora.set_attn_processor(lora_attn_procs)

        # Print the processor attribute of the attn1 module
        self.lora_layers = AttnProcsLayers(self.unet_lora.attn_processors).to(
            self.device
        )
        self.lora_layers._load_state_dict_pre_hooks.clear()
        self.lora_layers._state_dict_hooks.clear()
        

        self.lora_params = []
        for name, module in self.unet_lora.named_modules():
            if 'processor' in name and ('.down' in name or '.up' in name):
                self.lora_params.extend(module.parameters())

        # Initialize optimizer only for LoRA parameters
        self.optimizer = AdamW(self.lora_params, lr=1e-4, weight_decay=1e-2)
        
        # Initialize learning rate scheduler
        self.lr_scheduler = CosineAnnealingLR(self.optimizer, T_max=100000, eta_min=1e-6)
        
        # Initialize EMA
        self.ema = EMA(beta=0.9999)
        self.ema_params = [param.clone() for param in self.lora_params]
        
        # Initialize gradient scaler for mixed precision training
        self.scaler = GradScaler()

        self.scheduler = DDPMScheduler.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            subfolder="scheduler",
            torch_dtype=self.weights_dtype,
        )

        self.scheduler_lora = DDPMScheduler.from_pretrained(
            self.cfg.pretrained_model_name_or_path_lora,
            subfolder="scheduler",
            torch_dtype=self.weights_dtype,
        )

        self.scheduler_sample = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )
        self.scheduler_lora_sample = DPMSolverMultistepScheduler.from_config(
            self.pipe_lora.scheduler.config
        )

        self.pipe.scheduler = self.scheduler
        self.pipe_lora.scheduler = self.scheduler_lora

        self.low_res_scheduler = DDPMScheduler.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            subfolder="low_res_scheduler",
            torch_dtype=self.weights_dtype,
        )

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps

        self.alphas: Float[Tensor, "..."] = self.scheduler.alphas_cumprod.to(
            self.device
        )

        self.grad_clip_val: Optional[float] = None

        self.unet.to(self.device)
        if not self.single_model:
            self.unet_lora.to(self.device)
            print("LoRA model loaded")
        if self.unet == self.unet_lora:
            print("Single model loaded")
        self.unet_lora = DDP(unet_lora, device_ids=[torch.cuda.current_device()])

        self.scheduler_lora.config.prediction_type = "epsilon"
        self.scheduler.config.prediction_type = "epsilon"
        threestudio.info(f"Loaded Stable Diffusion x4 Upscaler!")

    @property
    def pipe(self):
        return self.submodules.pipe

    @property
    def pipe_lora(self):
        return self.submodules.pipe_lora

    @property
    def unet(self):
        return self.submodules.pipe.unet

    @property
    def unet_lora(self):
        return self.submodules.pipe_lora.unet

    @property
    def vae(self):
        return self.submodules.pipe.vae

    @property
    def vae_lora(self):
        return self.submodules.pipe_lora.vae

    def set_lora(self, name) -> None:
        cross_attention_dim = (
            self.unet_lora.config.cross_attention_dim
        )
        if name.startswith("mid_block"):
            hidden_size = self.unet_lora.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(self.unet_lora.config.block_out_channels))[
                block_id
            ]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = self.unet_lora.config.block_out_channels[block_id]
        lora_processor = LoRAAttnProcessor(
            hidden_size=hidden_size, cross_attention_dim=cross_attention_dim
        )
        result_list = name.split(".")
        try:
            block_index = int(result_list[1])
        except Exception as e:
            pass
        try:
            attn_index = int(result_list[3])
        except Exception as e:
            pass
        block_type = None
        if result_list[0] == "down_blocks":
            if result_list[-2] == "attn1":
                self.unet_lora.down_blocks[block_index].attentions[attn_index].transformer_blocks[0].attn1.to_q = lora_processor.to_q_lora
                self.unet_lora.down_blocks[block_index].attentions[attn_index].transformer_blocks[0].attn1.to_k = lora_processor.to_k_lora
                self.unet_lora.down_blocks[block_index].attentions[attn_index].transformer_blocks[0].attn1.to_v = lora_processor.to_v_lora
                self.unet_lora.down_blocks[block_index].attentions[attn_index].transformer_blocks[0].attn1.to_out = lora_processor.to_out_lora
            else:
                self.unet_lora.down_blocks[block_index].attentions[attn_index].transformer_blocks[0].attn2.to_q = lora_processor.to_q_lora
                self.unet_lora.down_blocks[block_index].attentions[attn_index].transformer_blocks[0].attn2.to_k = lora_processor.to_k_lora
                self.unet_lora.down_blocks[block_index].attentions[attn_index].transformer_blocks[0].attn2.to_v = lora_processor.to_v_lora
                self.unet_lora.down_blocks[block_index].attentions[attn_index].transformer_blocks[0].attn2.to_out = lora_processor.to_out_lora
        elif result_list[0] == "up_blocks":
            if result_list[-2] == "attn1":
                self.unet_lora.up_blocks[block_index].attentions[attn_index].transformer_blocks[0].attn1.to_q = lora_processor.to_q_lora
                self.unet_lora.up_blocks[block_index].attentions[attn_index].transformer_blocks[0].attn1.to_k = lora_processor.to_k_lora
                self.unet_lora.up_blocks[block_index].attentions[attn_index].transformer_blocks[0].attn1.to_v = lora_processor.to_v_lora
                self.unet_lora.up_blocks[block_index].attentions[attn_index].transformer_blocks[0].attn1.to_out = lora_processor.to_out_lora
            else:
                self.unet_lora.up_blocks[block_index].attentions[attn_index].transformer_blocks[0].attn2.to_q = lora_processor.to_q_lora
                self.unet_lora.up_blocks[block_index].attentions[attn_index].transformer_blocks[0].attn2.to_k = lora_processor.to_k_lora
                self.unet_lora.up_blocks[block_index].attentions[attn_index].transformer_blocks[0].attn2.to_v = lora_processor.to_v_lora
                self.unet_lora.up_blocks[block_index].attentions[attn_index].transformer_blocks[0].attn2.to_out = lora_processor.to_out_lora
        else:
            if result_list[-2] == "attn1":
                self.unet_lora.mid_block.attentions[0].transformer_blocks[0].attn1.to_q = lora_processor.to_q_lora
                self.unet_lora.mid_block.attentions[0].transformer_blocks[0].attn1.to_k = lora_processor.to_k_lora
                self.unet_lora.mid_block.attentions[0].transformer_blocks[0].attn1.to_v = lora_processor.to_v_lora
                self.unet_lora.mid_block.attentions[0].transformer_blocks[0].attn1.to_out = lora_processor.to_out_lora
            else:   
                self.unet_lora.mid_block.attentions[0].transformer_blocks[0].attn2.to_q = lora_processor.to_q_lora
                self.unet_lora.mid_block.attentions[0].transformer_blocks[0].attn2.to_k = lora_processor.to_k_lora
                self.unet_lora.mid_block.attentions[0].transformer_blocks[0].attn2.to_v = lora_processor.to_v_lora
                self.unet_lora.mid_block.attentions[0].transformer_blocks[0].attn2.to_out = lora_processor.to_out_lora

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def _sample(
        self,
        pipe: StableDiffusionUpscalePipeline,
        sample_scheduler: DPMSolverMultistepScheduler,
        text_embeddings: Float[Tensor, "BB N Nf"],
        num_inference_steps: int,
        guidance_scale: float,
        num_images_per_prompt: int = 1,
        height: Optional[int] = None,
        width: Optional[int] = None,
        class_labels: Optional[Float[Tensor, "BB 16"]] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    ) -> Float[Tensor, "B H W 3"]:
        vae_scale_factor = 2 ** (len(pipe.vae.config.block_out_channels) - 1)
        height = height or pipe.unet.config.sample_size * vae_scale_factor
        width = width or pipe.unet.config.sample_size * vae_scale_factor
        batch_size = text_embeddings.shape[0] // 2
        device = self.device

        sample_scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = sample_scheduler.timesteps
        num_channels_latents = pipe.unet.config.in_channels

        latents = pipe.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            self.weights_dtype,
            device,
            generator,
        )

        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = sample_scheduler.scale_model_input(
                latent_model_input, t
            )

            if class_labels is None:
                with self.disable_unet_class_embedding(pipe.unet) as unet:
                    noise_pred = unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=text_embeddings.to(self.weights_dtype),
                        cross_attention_kwargs=cross_attention_kwargs,
                    ).sample
            else:
                noise_pred = pipe.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=text_embeddings.to(self.weights_dtype),
                    class_labels=class_labels,
                    cross_attention_kwargs=cross_attention_kwargs,
                ).sample

            noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

            # compute the previous noisy sample x_t -> x_t-1
            latents = sample_scheduler.step(noise_pred, t, latents).prev_sample
        
        latents = 1 / pipe.vae.config.scaling_factor * latents
        images = pipe.vae.decode(latents).sample
        images = (images / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        images = images.permute(0, 2, 3, 1).float()
        return images

    def timestep_annealing(self, stage_step: int):
        # not part of Updateable, needs to be run manually in each training step

        if self.cfg.anneal_strategy == "none":
            self.min_step = int(self.num_train_timesteps * self.cfg.min_step_percent)
            self.max_step = int(self.num_train_timesteps * self.cfg.max_step_percent)
            return

        if (self.cfg.anneal_start_step is not None and stage_step >= self.cfg.anneal_start_step):

            if self.cfg.anneal_strategy == "milestone":
                self.max_step = int(self.num_train_timesteps * self.cfg.max_step_percent_annealed)
                return
            
            if self.cfg.anneal_strategy == "sqrt":
                max_step_percent_annealed = self.cfg.max_step_percent - (self.cfg.max_step_percent - self.cfg.min_step_percent) * math.sqrt(
                    (stage_step - self.cfg.anneal_start_step)
                    / (self.cfg.anneal_end_step - self.cfg.anneal_start_step)
                )
                self.max_step = int(self.num_train_timesteps * max_step_percent_annealed)
                self.min_step = self.max_step #non-stochastic, monotonically decreasing t
                return
            
            if self.cfg.anneal_strategy == "linear":
                step_fraction = 1.0-(stage_step - self.cfg.anneal_start_step) / (self.cfg.anneal_end_step - self.cfg.anneal_start_step)
                max_step_percent_annealed = step_fraction*(self.cfg.max_step_percent-self.cfg.min_step_percent)+self.cfg.min_step_percent
                self.max_step = int(self.num_train_timesteps * max_step_percent_annealed)
                self.min_step = self.max_step #non-stochastic, monotonically decreasing t
                return
            
            if self.cfg.anneal_strategy == 'discrete':
                self.cfg.random_timestep = False
                self.step_fraction = (stage_step - self.cfg.anneal_start_step) / (self.cfg.anneal_end_step - self.cfg.anneal_start_step)
                self.min_step = int(self.num_train_timesteps * self.cfg.min_step_percent)
                self.max_step = int(self.num_train_timesteps * self.cfg.max_step_percent)
                return

            raise ValueError(
                f"Unknown anneal strategy {self.cfg.anneal_strategy}, should be one of 'milestone', 'sqrt', 'none'"
            )

    def sample(
        self,
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        seed: int = 0,
        **kwargs,
    ) -> Float[Tensor, "N H W 3"]:
        text_embeddings_vd = prompt_utils.get_text_embeddings(
            elevation,
            azimuth,
            camera_distances,
            view_dependent_prompting=self.cfg.view_dependent_prompting,
        )
        cross_attention_kwargs = {"scale": 0.0} if self.single_model else None
        generator = torch.Generator(device=self.device).manual_seed(seed)

        return self._sample(
            pipe=self.pipe,
            sample_scheduler=self.scheduler_sample,
            text_embeddings=text_embeddings_vd,
            num_inference_steps=25,
            guidance_scale=self.cfg.guidance_scale,
            cross_attention_kwargs=cross_attention_kwargs,
            generator=generator,
        )

    def sample_lora(
        self,
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        mvp_mtx: Float[Tensor, "B 4 4"],
        c2w: Float[Tensor, "B 4 4"],
        seed: int = 0,
        **kwargs,
    ) -> Float[Tensor, "N H W 3"]:
        # input text embeddings, view-independent
        text_embeddings = prompt_utils.get_text_embeddings(
            elevation, azimuth, camera_distances, view_dependent_prompting=False
        )
        if self.cfg.camera_condition_type == "extrinsics":
            camera_condition = c2w
        elif self.cfg.camera_condition_type == "mvp":
            camera_condition = mvp_mtx
        else:
            warnings.warn(
                f"Unknown camera_condition_type {self.cfg.camera_condition_type}, using c2w."
            )
            camera_condition = c2w
        indices = torch.randint(0, camera_condition.shape[0], (image.shape[0],), device=self.device)
        camera_condition = camera_condition[indices]

        B = elevation.shape[0]
        camera_condition_cfg = torch.cat(
            [
                camera_condition.view(B, -1),
                torch.zeros_like(camera_condition.view(B, -1)),
            ],
            dim=0,
        )

        generator = torch.Generator(device=self.device).manual_seed(seed)
        return self._sample(
            sample_scheduler=self.scheduler_lora_sample,
            pipe=self.pipe_lora,
            text_embeddings=text_embeddings,
            num_inference_steps=25,
            guidance_scale=self.cfg.guidance_scale_lora,
            class_labels=camera_condition_cfg,
            cross_attention_kwargs={"scale": 1.0},
            generator=generator,
        )
    
    @torch.cuda.amp.autocast(enabled=False)
    def forward_unet(
        self,
        unet: UNet2DConditionModel,
        latents: Float[Tensor, "..."],
        t: Float[Tensor, "..."],
        encoder_hidden_states: Float[Tensor, "..."],
        class_labels: Optional[Float[Tensor, "B 16"]] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Float[Tensor, "..."]:
        input_dtype = latents.dtype
        return unet(
            latents.to(self.weights_dtype),
            t.to(self.weights_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
            class_labels=class_labels,  
        ).sample.to(input_dtype)
    
    def upcast_vae(self):
        dtype = self.vae.dtype
        self.vae.to(dtype=torch.float32)

        use_torch_2_0_or_xformers = isinstance(
            self.vae.decoder.mid_block.attentions[0].processor,
            (
                AttnProcessor2_0,
                XFormersAttnProcessor,
                LoRAXFormersAttnProcessor,
                LoRAAttnProcessor2_0,
            ),
        )
        # if xformers or torch_2_0 is used attention block does not need
        # to be in float32 which can save lots of memory
        if use_torch_2_0_or_xformers:
            self.vae.post_quant_conv.to(dtype)
            self.vae.decoder.conv_in.to(dtype)
            self.vae.decoder.mid_block.to(dtype)

        self.vae_lora.to(dtype=torch.float32)
        use_torch_2_0_or_xformers = isinstance(
            self.vae_lora.decoder.mid_block.attentions[0].processor,
            (
                AttnProcessor2_0,
                XFormersAttnProcessor,
                LoRAXFormersAttnProcessor,
                LoRAAttnProcessor2_0,
            ),
        )

        if use_torch_2_0_or_xformers:
            self.vae_lora.post_quant_conv.to(dtype)
            self.vae_lora.decoder.conv_in.to(dtype)
            self.vae_lora.decoder.mid_block.to(dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def encode_images(
        self, imgs: Float[Tensor, "B 3 H W"]
    ) -> Float[Tensor, "B 4 H/4 W/4"]:
        if self.vae.dtype == torch.float16:
            self.upcast_vae()

        device = imgs.device  # Get the device of the input tensor
        self.vae.to(device)  # Move the VAE model to the same device as the input

        input_dtype = imgs.dtype
        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor
        return latents.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    # @torch.no_grad()
    def decode_latents(
        self, latents: Float[Tensor, "B 4 H/4 W/4"],
    ) -> Float[Tensor, "B 3 H W"]:
        if self.vae.dtype==torch.float16:
            self.upcast_vae()

        input_dtype = latents.dtype
        latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)

        # latents = F.interpolate(latents, (latent_height, latent_width), mode="bilinear", align_corners=False)
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents, return_dict=False)[0]
        image = (image * 0.5 + 0.5).clamp(0, 1)
        return image.to(input_dtype)

    @contextmanager
    def disable_unet_class_embedding(self, unet: UNet2DConditionModel):
        class_embedding = unet.class_embedding
        try:
            unet.class_embedding = None
            yield unet
        finally:
            unet.class_embedding = class_embedding

    def compute_grad_vsd(
        self,
        latents: Float[Tensor, "B 4 64 64"],
        image: Float[Tensor, "B 3 256 256"],
        text_embeddings_vd: Float[Tensor, "BB 77 768"],
        text_embeddings: Float[Tensor, "BB 77 768"],
        camera_condition: Float[Tensor, "B 4 4"],
    ):
        B = latents.shape[0]

        with torch.no_grad():
            # random timestamp
            if self.cfg.random_timestep:
                t = torch.randint(
                    self.min_step,
                    self.max_step + 1,
                    [B],
                    dtype=torch.long,
                    device=self.device,
                )
            else:
                t = torch.full([B], self.max_step, dtype=torch.long, device=self.device)

            last_timestep = t[0].detach().cpu().numpy()
            self.last_timestep = torch.tensor(last_timestep)

            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy, image], dim=1)
            latent_model_input = torch.cat([latent_model_input] * 2)
            with self.disable_unet_class_embedding(self.unet) as unet:
                cross_attention_kwargs = {"scale": 0.0} if self.single_model else None
                noise_pred_pretrain = self.forward_unet(
                    unet,
                    latent_model_input,
                    torch.cat([t] * 2),
                    encoder_hidden_states=text_embeddings_vd.to(self.weights_dtype),
                    cross_attention_kwargs=cross_attention_kwargs,
                )
            # use view-independent text embeddings in LoRA
            text_embeddings_cond, _ = text_embeddings.chunk(2)
            noise_pred_est = self.forward_unet(
                self.unet_lora,
                latent_model_input,
                torch.cat([t] * 2),
                encoder_hidden_states=torch.cat([text_embeddings_cond] * 2),
                class_labels=torch.cat(
                    [
                        camera_condition.view(B, -1),
                        torch.zeros_like(camera_condition.view(B, -1)),
                    ],
                    dim=0,
                ),
                cross_attention_kwargs={"scale": 1.0},
            )

        (
            noise_pred_pretrain_text,
            noise_pred_pretrain_uncond,
        ) = noise_pred_pretrain.chunk(2)

        # NOTE: guidance scale definition here is aligned with diffusers, but different from other guidance
        noise_pred_pretrain = noise_pred_pretrain_uncond + self.cfg.guidance_scale * (
            noise_pred_pretrain_text - noise_pred_pretrain_uncond
        )

        # TODO: more general cases
        assert self.scheduler.config.prediction_type == "epsilon"
        if self.scheduler_lora.config.prediction_type == "v_prediction":
            alphas_cumprod = self.scheduler_lora.alphas_cumprod.to(
                device=latents_noisy.device, dtype=latents_noisy.dtype
            )
            alpha_t = alphas_cumprod[t] ** 0.5
            sigma_t = (1 - alphas_cumprod[t]) ** 0.5

            noise_pred_est = latent_model_input * torch.cat([sigma_t] * 2, dim=0).view(
                -1, 1, 1, 1
            ) + noise_pred_est * torch.cat([alpha_t] * 2, dim=0).view(-1, 1, 1, 1)

        (
            noise_pred_est_camera,
            noise_pred_est_uncond,
        ) = noise_pred_est.chunk(2)

        # NOTE: guidance scale definition here is aligned with diffusers, but different from other guidance
        noise_pred_est = noise_pred_est_uncond + self.cfg.guidance_scale * (
            noise_pred_est_camera - noise_pred_est_uncond
        )

        w = (1 - self.alphas[t]).view(-1, 1, 1, 1)

        grad = w * (noise_pred_pretrain - noise_pred_est)

        alpha = (self.alphas[t] ** 0.5).view(-1, 1, 1, 1)
        sigma = ((1 - self.alphas[t]) ** 0.5).view(-1, 1, 1, 1)
        # image-space SDS proposed in HiFA: https://hifa-team.github.io/HiFA-site/
        if self.cfg.use_img_loss:
            latents_denoised_pretrain = (
                latents_noisy - sigma * noise_pred_pretrain
            ) / alpha
            latents_denoised_est = (latents_noisy - sigma * noise_pred_est) / alpha
            image_denoised_pretrain = self.decode_latents(latents_denoised_pretrain)
            image_denoised_est = self.decode_latents(latents_denoised_est)
            grad_img = (
                w * (image_denoised_est - image_denoised_pretrain) * alpha / sigma
            )
        else:
            grad_img = None
        grad_img = None
        return grad, grad_img

    def train_lora(
        self,
        image: Float[Tensor, "B 3 256 256"],
        latents: Float[Tensor, "B 4 64 64"],
        text_embeddings: Float[Tensor, "BB 77 768"],
        camera_condition: Float[Tensor, "B 4 4"],
    ):
        B = latents.shape[0]
        latents = latents.detach().repeat(self.cfg.lora_n_timestamp_samples, 1, 1, 1)
        
        # Set model to training mode
        self.unet_lora.train()
        
        # Ensure only LoRA parameters are set to require gradients
        for param in self.unet_lora.parameters():
            param.requires_grad_(False)
        for param in self.lora_params:
            param.requires_grad_(True)
        torch.cuda.empty_cache()
        # Use autocast for mixed precision training
        with torch.cuda.amp.autocast(enabled=True):
            t = torch.randint(
                int(self.num_train_timesteps * 0.0),
                int(self.num_train_timesteps * 1.0),
                [B * self.cfg.lora_n_timestamp_samples],
                dtype=torch.long,
                device=self.device,
            )

            noise = torch.randn_like(latents)
            noisy_latents = self.scheduler_lora.add_noise(latents, noise, t)
            if self.scheduler_lora.config.prediction_type == "epsilon":
                target = noise
            elif self.scheduler_lora.config.prediction_type == "v_prediction":
                target = self.scheduler_lora.get_velocity(latents, noise, t)
            else:
                raise ValueError(
                    f"Unknown prediction type {self.scheduler_lora.config.prediction_type}"
                )
            latent_model_input = torch.cat([noisy_latents, image], dim=1)
            text_embeddings_cond, _ = text_embeddings.chunk(2)
            if self.cfg.lora_cfg_training and random.random() < 0.1:
                camera_condition = torch.zeros_like(camera_condition)
            print(torch.cuda.memory_summary(device=self.device, abbreviated=True))
            noise_pred = self.forward_unet(
                self.unet_lora,
                latent_model_input,
                t,
                encoder_hidden_states=text_embeddings_cond.repeat(
                    self.cfg.lora_n_timestamp_samples, 1, 1
                ),
                class_labels=camera_condition.view(B, -1).repeat(
                    self.cfg.lora_n_timestamp_samples, 1
                ),
                cross_attention_kwargs={"scale": 1.0},
            )
            loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")

        # Backward pass with gradient scaling
        self.scaler.scale(loss).backward()
        
        # Update weights with gradient clipping
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.lora_params, max_norm=1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        # Update learning rate
        self.lr_scheduler.step()
        
        # Update EMA model
        self.ema.step_ema(self.ema_params, self.lora_params)
        
        # Zero gradients
        self.optimizer.zero_grad()

        return loss.item()

    def forward(
        self,
        render: Float[Tensor, "B H W C"],
        image: Float[Tensor, "B H W C"],
        prompt_utils: PromptProcessorOutput, 
        mvp_mtx: Float[Tensor, "B 4 4"],
        c2w: Float[Tensor, "B 4 4"],
        noise_level=20,
    ):
        render = render.permute(0,3,1,2)
        image = image.permute(0,3,1,2)
        image = image * 2.0 - 1.0
        latents = render

        # dummy variables
        elevation = torch.Tensor([0]).to(device=self.device)
        azimuth = torch.Tensor([0]).to(device=self.device)
        camera_distances = torch.Tensor([1]).to(device=self.device)

        # 2. Encode prompt into view dependent text embeddings (equivalent to prompt_embeds)
        text_embeddings_vd = prompt_utils.get_text_embeddings(
            elevation,
            azimuth,
            camera_distances,
            view_dependent_prompting=self.cfg.view_dependent_prompting,
        )
        text_embeddings_vd = text_embeddings_vd.unsqueeze(1) #[2,1,77,1024]
        text_embeddings_vd  = torch.cat([text_embeddings_vd]*image.shape[0],dim=1) # [2,B,77,1024]
        text_embeddings_vd = text_embeddings_vd.flatten(0,1) # [2B,77,1024]
        text_embeddings = prompt_utils.get_text_embeddings(
            elevation, azimuth, camera_distances, view_dependent_prompting=False
        )
        text_embeddings = text_embeddings.unsqueeze(1) #[2,1,77,1024]
        text_embeddings  = torch.cat([text_embeddings]*image.shape[0],dim=1) # [2,B,77,1024]
        text_embeddings = text_embeddings.flatten(0,1) # [2B,77,1024]

        # 4. Add noise to LR image
        noise_level = torch.Tensor([noise_level]).to(dtype=torch.long, device=self.device)
        noise = randn_tensor(image.shape, device=self.device, dtype=text_embeddings.dtype) # draw gaussian noise sample
        image = self.low_res_scheduler.add_noise(image, noise, noise_level)
        noise_level  = torch.cat([noise_level]*image.shape[0]) # [BB]

        # 5. Check that sizes of lr image and latents match
        assert image.shape[2:] == latents.shape[2:], f"height/width mismatch! {image.shape, latents.shape}"
        num_channels_image = image.shape[1]
        num_channels_latents = self.vae.config.latent_channels

        if num_channels_latents + num_channels_image != self.unet.config.in_channels:
            raise ValueError(
                f"Incorrect configuration settings! The config of `pipeline.unet`: {self.unet.config} expects"
                f" {self.unet.config.in_channels} but received `num_channels_latents`: {num_channels_latents} +"
                f" `num_channels_image`: {num_channels_image} "
                f" = {num_channels_latents+num_channels_image}. Please verify the config of"
                " `pipeline.unet` or your `image` input."
            )

        if self.cfg.camera_condition_type == "extrinsics":
            camera_condition = c2w
        elif self.cfg.camera_condition_type == "mvp":
            camera_condition = mvp_mtx
        else:
            warnings.warn(
                f"Unknown camera_condition_type {self.cfg.camera_condition_type}, using c2w."
            )
            camera_condition = c2w
        indices = torch.randint(0, camera_condition.shape[0], (image.shape[0],), device=self.device)
        camera_condition = camera_condition[indices]

        grad, grad_img = self.compute_grad_vsd(
            latents, image, text_embeddings_vd, text_embeddings, camera_condition
        )
        if self.grad_clip_val is not None:
            grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)


        assert self.cfg.guidance_type in ['vsd']
        batch_size = latents.shape[0]
        target = (latents - grad).detach()
        loss_vsd = 0.5 * F.mse_loss(latents, target, reduction="sum") / batch_size
        loss_lora = self.train_lora(image, latents, text_embeddings, camera_condition)

        loss_dict = {
            "loss_vsd": loss_vsd,
            "loss_lora": loss_lora,
            "grad_norm": grad.norm(),
            "min_step": self.min_step,
            "max_step": self.max_step,
        }
        if self.cfg.use_img_loss:
            grad_img = torch.nan_to_num(grad_img)
            if self.grad_clip_val is not None:
                grad_img = grad_img.clamp(-self.grad_clip_val, self.grad_clip_val)
            target_img = (rgb_BCHW_512 - grad_img).detach()
            loss_vsd_img = (
                0.5 * F.mse_loss(rgb_BCHW_512, target_img, reduction="sum") / batch_size
            )
            loss_dict["loss_vsd_img"] = loss_vsd_img

        return loss_dict

    
