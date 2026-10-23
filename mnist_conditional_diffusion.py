"""Class-conditional diffusion on MNIST — CSE/MB620 course project.

Ask the model to write a digit: sample(label=3) -> an image of a 3.

What is built here (vs downloaded):
  * a compact U-Net denoiser with horizontal skip connections (course ch. 8)
  * sinusoidal timestep embedding + nn.Embedding for the digit label
  * classifier-free guidance (labels dropped 10% of the time in training)
  * the DDPM training loop and sampler (course ch. 10)
Downloaded: only the dataset. Reference repo for comparison:
  https://github.com/TeaPearce/Conditional_Diffusion_MNIST

Usage:
  python mnist_conditional_diffusion.py --dataset mnist  --epochs 30   # Colab GPU
  python mnist_conditional_diffusion.py --dataset digits --epochs 4    # offline smoke test
  python mnist_conditional_diffusion.py --sample-only --label 3        # after training
"""
import argparse, math, os, time
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------- data
def load_data(name, img_size, subset=None):
    if name == "mnist":
        from torchvision import datasets, transforms
        tf = transforms.Compose([transforms.Resize(img_size), transforms.ToTensor()])
        ds = datasets.MNIST("./data", train=True, download=True, transform=tf)
        xs = torch.stack([ds[i][0] for i in range(subset or len(ds))])
        ys = torch.tensor([ds[i][1] for i in range(subset or len(ds))])
    elif name == "digits":  # offline: sklearn's 8x8 digits, no network needed
        from sklearn.datasets import load_digits
        d = load_digits()
        xs = torch.tensor(d.images, dtype=torch.float32).unsqueeze(1) / 16.0
        xs = F.interpolate(xs, size=img_size, mode="bilinear", align_corners=False)
        ys = torch.tensor(d.target)
        if subset: xs, ys = xs[:subset], ys[:subset]
    else:
        raise ValueError(name)
    return xs * 2 - 1, ys  # scale to [-1, 1]

# ---------------------------------------------------------------- schedule
def make_schedule(T, device):
    betas = torch.linspace(1e-4, 0.02, T, device=device)
    alphas = 1.0 - betas
    abar = torch.cumprod(alphas, dim=0)
    abar_prev = F.pad(abar[:-1], (1, 0), value=1.0)
    return dict(betas=betas, alphas=alphas, abar=abar,
                sqrt_abar=torch.sqrt(abar),
                sqrt_1m_abar=torch.sqrt(1 - abar),
                sqrt_recip_alphas=torch.sqrt(1.0 / alphas),
                posterior_var=betas * (1 - abar_prev) / (1 - abar))

def forward_diffusion(x0, t, sch):
    noise = torch.randn_like(x0)
    a = sch["sqrt_abar"][t].view(-1, 1, 1, 1)
    b = sch["sqrt_1m_abar"][t].view(-1, 1, 1, 1)
    return a * x0 + b * noise, noise

# ---------------------------------------------------------------- model
class SinEmb(nn.Module):
    def __init__(self, dim):
        super().__init__(); self.dim = dim
    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t.float().unsqueeze(-1) * freqs
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

class Block(nn.Module):
    """Conv block that also ingests the (time + label) embedding."""
    def __init__(self, cin, cout, emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.emb = nn.Linear(emb_dim, cout)
        self.norm1 = nn.GroupNorm(8, cout)
        self.norm2 = nn.GroupNorm(8, cout)
    def forward(self, x, e):
        h = F.silu(self.norm1(self.conv1(x)))
        h = h + self.emb(e).unsqueeze(-1).unsqueeze(-1)
        return F.silu(self.norm2(self.conv2(h)))

class CondUNet(nn.Module):
    """Small U-Net: 2 downs, bottleneck, 2 ups with skip concatenation.
    Label 10 is the learned 'no label' token for classifier-free guidance."""
    def __init__(self, base=32, emb_dim=64):
        super().__init__()
        self.t_emb = SinEmb(emb_dim)
        self.y_emb = nn.Embedding(11, emb_dim)          # 10 digits + null
        self.inc   = Block(1, base, emb_dim)
        self.down1 = Block(base, base * 2, emb_dim)
        self.down2 = Block(base * 2, base * 2, emb_dim)
        self.up1   = Block(base * 4, base, emb_dim)     # cat skip -> 4x
        self.up2   = Block(base * 2, base, emb_dim)     # cat skip -> 2x
        self.out   = nn.Conv2d(base, 1, 1)
        self.pool  = nn.MaxPool2d(2)
    def forward(self, x, t, y):
        e = self.t_emb(t) + self.y_emb(y)
        s1 = self.inc(x, e)                               # H
        s2 = self.down1(self.pool(s1), e)                 # H/2
        b  = self.down2(self.pool(s2), e)                 # H/4
        u1 = F.interpolate(b, scale_factor=2, mode="nearest")
        u1 = self.up1(torch.cat([u1, s2], dim=1), e)      # skip 1
        u2 = F.interpolate(u1, scale_factor=2, mode="nearest")
        u2 = self.up2(torch.cat([u2, s1], dim=1), e)      # skip 2
        return self.out(u2)

# ---------------------------------------------------------------- train
def train(model, xs, ys, sch, epochs, batch, lr, device, p_uncond=0.1):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    T = len(sch["betas"]); n = len(xs); first = last = None
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            x0, y = xs[idx].to(device), ys[idx].to(device)
            y = torch.where(torch.rand(len(y), device=device) < p_uncond,
                            torch.full_like(y, 10), y)   # drop label -> null token
            t = torch.randint(0, T, (len(x0),), device=device)
            x_t, noise = forward_diffusion(x0, t, sch)
            loss = F.mse_loss(model(x_t, t, y), noise)
            opt.zero_grad(); loss.backward(); opt.step()
            if first is None: first = loss.item()
            last = loss.item()
        print(f"epoch {ep + 1}/{epochs}  loss {last:.4f}")
    return first, last

# ---------------------------------------------------------------- sample
@torch.no_grad()
def sample(model, sch, labels, img_size, device, guidance=2.0):
    model.eval()
    T = len(sch["betas"]); n = len(labels)
    x = torch.randn(n, 1, img_size, img_size, device=device)
    y = torch.tensor(labels, device=device)
    null = torch.full_like(y, 10)
    for i in range(T - 1, -1, -1):
        t = torch.full((n,), i, dtype=torch.long, device=device)
        eps = (1 + guidance) * model(x, t, y) - guidance * model(x, t, null)
        bt = sch["betas"][i]; sa = sch["sqrt_recip_alphas"][i]
        mean = sa * (x - bt * eps / sch["sqrt_1m_abar"][i])
        x = mean if i == 0 else mean + torch.sqrt(sch["posterior_var"][i]) * torch.randn_like(x)
    return x.clamp(-1, 1)

def save_grid(imgs, labels, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(imgs); cols = min(n, 10); rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.2, rows * 1.4))
    axes = [axes] if n == 1 else list(axes.flat)
    for ax in axes: ax.axis("off")
    for ax, im, lb in zip(axes, imgs, labels):
        ax.imshow(im.squeeze().cpu(), cmap="gray"); ax.set_title(str(lb), fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
    print("saved", path)

# ---------------------------------------------------------------- main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mnist", choices=["mnist", "digits"])
    ap.add_argument("--img-size", type=int, default=28)
    ap.add_argument("--timesteps", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--subset", type=int, default=None)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--guidance", type=float, default=2.0)
    ap.add_argument("--label", type=int, default=None, help="sample this digit only")
    ap.add_argument("--sample-only", action="store_true")
    ap.add_argument("--ckpt", default="cond_diffusion.pt")
    ap.add_argument("--out", default="samples.png")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    sch = make_schedule(args.timesteps, device)
    model = CondUNet(base=args.base).to(device)
    print("device:", device, "| params:",
          sum(p.numel() for p in model.parameters()))

    if not args.sample_only:
        xs, ys = load_data(args.dataset, args.img_size, args.subset)
        print("data:", tuple(xs.shape))
        t0 = time.time()
        first, last = train(model, xs, ys, sch, args.epochs, args.batch, args.lr, device)
        print(f"loss {first:.3f} -> {last:.3f}  ({time.time() - t0:.0f}s)")
        torch.save(model.state_dict(), args.ckpt); print("saved", args.ckpt)
    else:
        model.load_state_dict(torch.load(args.ckpt, map_location=device))

    labels = [args.label] * 10 if args.label is not None else list(range(10)) * 2
    imgs = sample(model, sch, labels, args.img_size, device, args.guidance)
    save_grid(imgs, labels, args.out)
