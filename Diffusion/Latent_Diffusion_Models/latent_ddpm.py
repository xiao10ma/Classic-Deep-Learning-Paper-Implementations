import torch
import torch.nn as nn

class DiffusionWrapper(nn.Module):
    def __init__(self, diffusion_model: UNetModel):
        super().__init__()
        self.diffusion_model = diffusion_model

        def forward(self, x: torch.Tensor, time_steps: torch.Tensor, context: torch.Tensor):
            return self.diffusion_model(x, time_steps, context)
        
class LatentDiffusion(nn.Module):
    # 1. AutoEncoder
    # 2. UNet with attention
    # 3. CLIP embeddings generator
    model: DiffusionWrapper
    first_stage_model: Autoencoder
    cond_stage_model: CLIPTextEmbedder
    def __init__(self,
                 unet_model: UNetModel,
                 autoencoder: Autoencoder,
                 clip_embedder: CLIPTextEmbedder,
                 latent_scaling_factor: float,
                 n_steps: int,
                 linear_start: float,
                 linear_end: float):
        super().__init__()
        self.model = DiffusionWrapper(unet_model)

        self.first_stage_model = autoencoder
        self.latent_scaling_factor = latent_scaling_factor

        self.cond_stage_model = clip_embedder
        self.n_steps = n_steps

        beta = torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_steps, dtype=torch.float64) ** 2
        self.beta = nn.Parameter(beta.to(torch.float32), requires_grad=False)

        alpha = 1. - beta
        alpha_bar = torch.cumprod(alpha, dim=0)
        self.alpha_bar = nn.Parameter(alpha_bar.to(torch.float32), requires_grad=False)

    @property
    def device(self):
        return next(iter(self.model.parameters())).device
    
    def get_text_conditioning(self, prompts: list[str]):
        return self.cond_stage_model(prompts)
    
    def autoencoder_encode(self, image: torch.Tensor):
        return self.latent_scaling_factor * self.first_stage_model.encode(image).sample()
    
    def autoencoder_decode(self, z: torch.tensor):
        return self.first_stage_model.decode(z / self.latent_scaling_factor)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, context: torch.Tensor)
        return self.model(x, t, context)