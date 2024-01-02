
from torch import nn
import math

import torch
import torchvision
import matplotlib.pyplot as plt

import wandb

import torchvision.datasets as datasets 

import torch.nn.functional as F

def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)


# Define beta schedule


from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np

IMG_SIZE = 64
BATCH_SIZE = 240


import os
import torch
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset


class StanfordCars(torch.utils.data.Dataset):
    def __init__(self, root_path, split='train', transform=None):
        self.root_path = root_path
        self.transform = transform
        self.split = split
        self._load_images()

    def _load_images(self):
        split_path = os.path.join(self.root_path, f'cars_{self.split}/cars_{self.split}')
        if not os.path.exists(split_path):
            raise ValueError(f"Path not found: {split_path}")

        self.images = [os.path.join(split_path, file) for file in os.listdir(split_path) if file.endswith(('.jpg', '.jpeg', '.png'))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_file = self.images[index]
        image = Image.open(image_file).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image
    

def load_transformed_dataset():
    data_transforms = [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), # Scales data into [0,1]
        transforms.Lambda(lambda t: (t * 2) - 1) # Scale between [-1, 1]
    ]
    data_transform = transforms.Compose(data_transforms)
    train = StanfordCars(root_path="/home/brett/Desktop/tutorials/diff/stanfordCars/",transform=data_transform)
    test = StanfordCars(root_path="/home/brett/Desktop/tutorials/diff/stanfordCars/", transform=data_transform, split='test')
    return torch.utils.data.ConcatDataset([train, test])

def load_transformed_dataset_f1s(root_dir="/home/brett/Desktop/tutorials/diff/f1s", img_size=224):
    # Define the transformations
    data_transforms = [
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), # Scales data into [0,1]
        transforms.Lambda(lambda t: (t * 2) - 1) # Scale between [-1, 1]
    ]
    data_transform = transforms.Compose(data_transforms)

    # Load datasets
    train_dir = os.path.join(root_dir, '')  # Adjust subdirectories if needed
    # test_dir = os.path.join(root_dir, 'test')    # Adjust subdirectories if needed

    train_dataset = datasets.ImageFolder(train_dir, transform=data_transform)
    # test_dataset = datasets.ImageFolder(test_dir, transform=data_transform)

    # Concatenate train and test datasets
    # combined_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])

    return train_dataset

# Example of using the function

import os
import numpy as np


def save_tensor_image(image, epoch_number="00", filename="output_image"):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])

    # Take first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :]
    
    # Apply reverse transformations and save the image
    pil_image = reverse_transforms(image)
    save_path = os.path.join(os.getcwd(), f"{filename}_epoch_{str(epoch_number).zfill(2)}.png")
    pil_image.save(save_path)
    print(f"Image saved at: {save_path}")




data = load_transformed_dataset()
dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)


dataset_f1s = load_transformed_dataset_f1s()
f1_dataloader = DataLoader(dataset_f1s, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)


class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp =  nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(2*in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU()

    def forward(self, x, t, ):
        # First Conv
        h = self.bnorm1(self.relu(self.conv1(x)))
        # Time embedding
        time_emb = self.relu(self.time_mlp(t))
        # Extend last 2 dimensions
        time_emb = time_emb[(..., ) + (None, ) * 2]
        # Add time channel
        h = h + time_emb
        # Second Conv
        h = self.bnorm2(self.relu(self.conv2(h)))
        # Down or Upsample
        return self.transform(h)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class SimpleUnet(nn.Module):
    """
    A simplified variant of the Unet architecture.
    """
    def __init__(self):
        super().__init__()
        image_channels = 3
        down_channels = (64, 128, 512, 1023, 1600)
        up_channels = (1600, 1023, 512, 128, 64)
        out_dim = 3
        time_emb_dim = 32

        # Time embedding
        self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(time_emb_dim),
                nn.Linear(time_emb_dim, time_emb_dim),
                nn.ReLU()
            )

        # Initial projection
        self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)

        # Downsample
        self.downs = nn.ModuleList([Block(down_channels[i], down_channels[i+1], \
                                    time_emb_dim) \
                    for i in range(len(down_channels)-1)])
        # Upsample
        self.ups = nn.ModuleList([Block(up_channels[i], up_channels[i+1], \
                                        time_emb_dim, up=True) \
                    for i in range(len(up_channels)-1)])

        # Edit: Corrected a bug found by Jakub C (see YouTube comment)
        self.output = nn.Conv2d(up_channels[-1], out_dim, 1)

    def forward(self, x, timestep):
        # Embedd time
        t = self.time_mlp(timestep)
        # Initial conv
        x = self.conv0(x)
        # Unet
        residual_inputs = []
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)
        for up in self.ups:
            residual_x = residual_inputs.pop()
            # Add residual x as additional channels
            x = torch.cat((x, residual_x), dim=1)
            x = up(x, t)
        return self.output(x)

model = SimpleUnet()
print("Num params: ", sum(p.numel() for p in model.parameters()))


def get_loss(model, x_0, t):
    x_noisy, noise = forward_diffusion_sample(x_0, t, device)
    noise_pred = model(x_noisy, t)
    return F.l1_loss(noise, noise_pred)

T = 300
betas = linear_beta_schedule(timesteps=T)

# Pre-calculate different terms for closed form
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)


def get_index_from_list(vals, t, x_shape):

    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def forward_diffusion_sample(x_0, t, device="cpu"):

    noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x_0.shape
    )
    # mean + variance
    return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)


@torch.no_grad()
def sample_timestep(x, t):
    """
    Calls the model to predict the noise in the image and returns
    the denoised image.
    Applies noise to this image, if we are not in the last step yet.
    """
    betas_t = get_index_from_list(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)

    # Call model (current image - noise prediction)
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )
    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)

    if t == 0:
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise





def sample_plot_image(epoch):
    img_size = IMG_SIZE
    img = torch.randn((1, 3, img_size, img_size), device=device)

    for i in range(T-1, -1, -1):  # Loop from T-1 to 0
        t = torch.full((1,), i, device=device, dtype=torch.long)
        img = sample_timestep(img, t)

    # img now contains the last image in the sequence
    img = torch.clamp(img, -1.0, 1.0)
    filename = f"stanfordCarsTr_epoch_{epoch}"
    save_tensor_image(img.cpu(), epoch_number=epoch, filename=filename)

    # Return the path of the saved image
    save_path = os.path.join(os.getcwd(), f"{filename}_epoch_{str(epoch).zfill(2)}.png")
    return save_path

from torch.optim import Adam

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"

model.to(device)
optimizer = Adam(model.parameters(), lr=0.001)
epochs = 1000 
tr_f1 = True   

# model_file = "/home/brett/Desktop/tutorials/diff/sc_models/model_epoch_95.pt"  # Replace with your model file path
model_file = "/home/brett/Desktop/tutorials/diff/sc_models/model_epoch_190.pt"  # Replace with your model file path

# Create directory for saving models if it doesn't exist
models_dir = "./sc_models" if not tr_f1 else "./f1_models"
os.makedirs(models_dir, exist_ok=True)

# Load model state if a model file exists
if os.path.isfile(model_file):
    model.load_state_dict(torch.load(model_file))
    print(f"Loaded model state from {model_file}")

d_loader = dataloader if not tr_f1 else f1_dataloader

# Initialize a new run
wandb.init(project="diff_models", entity="byyoung3")

best_loss = float('inf')

for epoch in range(epochs):
    cumulative_loss = 0.0
    num_steps = 0

    for step, batch in enumerate(d_loader):
        optimizer.zero_grad()

        t = torch.randint(0, T, (BATCH_SIZE,), device=device).long()
        loss = get_loss(model, batch, t)
        loss.backward()
        optimizer.step()

        cumulative_loss += loss.item()
        num_steps += 1

        if epoch % 10 == 0 and step == 0:
            print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()} ")

            image_path = sample_plot_image(epoch)
            wandb.log({"samples": [wandb.Image(image_path, caption=f"Epoch {epoch}")]}, step=epoch)

    # Log cumulative loss at the end of each epoch
    average_loss = cumulative_loss / num_steps
    wandb.log({"cumulative_loss": cumulative_loss, "average_loss": average_loss}, step=epoch)

    # Save model if this epoch's loss is the best so far
    if average_loss < best_loss:
        best_loss = average_loss
        model_save_path = os.path.join(models_dir, f"best_model.pt")
        torch.save(model.state_dict(), model_save_path)
        print(f"Saved best model state with loss {best_loss} to {model_save_path}")
