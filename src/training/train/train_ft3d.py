import torch
from src.training.models.diffusion3d import build_diffusion_3d
from src.training.models.inflation import map_state_dict_2d_to_3d


def inflate_and_load(model_3d, state_dict_2d):
    mapped, missing = map_state_dict_2d_to_3d(state_dict_2d, model_3d.state_dict())
    model_3d.load_state_dict(mapped, strict=False)
    return missing


def train_step(model, latents, optimizer):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    timesteps = torch.randint(0, 1000, (latents.shape[0],), device=latents.device)
    noise = torch.randn_like(latents)
    noisy_latents = latents + noise
    pred_noise = model(noisy_latents, timesteps)
    loss = torch.mean((pred_noise - noise) ** 2)
    loss.backward()
    optimizer.step()
    return {"loss": float(loss.detach().cpu())}


@torch.no_grad()
def eval_step(model, latents):
    model.eval()
    timesteps = torch.randint(0, 1000, (latents.shape[0],), device=latents.device)
    noise = torch.randn_like(latents)
    noisy_latents = latents + noise
    pred_noise = model(noisy_latents, timesteps)
    loss = torch.mean((pred_noise - noise) ** 2)
    return {"loss": float(loss.detach().cpu())}


def build_model(config):
    return build_diffusion_3d(config)


def main():
    raise NotImplementedError("3D fine-tuning entrypoint not implemented.")


if __name__ == "__main__":
    main()
