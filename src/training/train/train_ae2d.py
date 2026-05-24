import torch
from src.training.models.autoencoder2d import build_autoencoder_2d


def train_step(model, batch, optimizer):
    model.train()
    optimizer.zero_grad(set_to_none=True)

    outputs = model(batch)
    if isinstance(outputs, (list, tuple)) and len(outputs) >= 3:
        recon, z_mu, z_sigma = outputs[:3]
    else:
        raise ValueError("AutoencoderKL output is unexpected.")

    recon_loss = torch.mean(torch.abs(recon - batch))
    kl_loss = torch.mean(0.5 * (z_mu.pow(2) + z_sigma.pow(2) - 1.0 - torch.log(z_sigma.pow(2) + 1.0e-6)))
    loss = recon_loss + (1.0e-6 * kl_loss)
    loss.backward()
    optimizer.step()

    return {
        "loss": float(loss.detach().cpu()),
        "recon_l1": float(recon_loss.detach().cpu()),
        "kl": float(kl_loss.detach().cpu()),
    }


@torch.no_grad()
def eval_step(model, batch):
    model.eval()
    outputs = model(batch)
    if isinstance(outputs, (list, tuple)) and len(outputs) >= 3:
        recon, z_mu, z_sigma = outputs[:3]
    else:
        raise ValueError("AutoencoderKL output is unexpected.")

    recon_loss = torch.mean(torch.abs(recon - batch))
    kl_loss = torch.mean(0.5 * (z_mu.pow(2) + z_sigma.pow(2) - 1.0 - torch.log(z_sigma.pow(2) + 1.0e-6)))
    loss = recon_loss + (1.0e-6 * kl_loss)
    return {
        "loss": float(loss.detach().cpu()),
        "recon_l1": float(recon_loss.detach().cpu()),
        "kl": float(kl_loss.detach().cpu()),
    }


def build_model(config):
    return build_autoencoder_2d(config)


def main():
    raise NotImplementedError("2D AutoencoderKL training entrypoint not implemented.")


if __name__ == "__main__":
    main()
