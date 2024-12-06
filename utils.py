import torchvision.utils as vutils
from matplotlib import pyplot as plt
import torch
import torchvision.transforms as transforms
from PIL import Image
import os

def sample_and_save(vae, captions, epoch, device, output_dir="samples"):
    """
    Samples images from the BaselineConvVAE and saves them.
    
    Args:
        vae: The trained BaselineConvVAE model.
        captions: A batch of caption embeddings from the dataset.
        epoch: Current epoch number.
        device: Device (CPU or GPU) on which the model runs.
        output_dir: Directory to save the sampled images.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    batch_size = captions.size(0)
    latent_dim = vae.latent_dim
    
    z = torch.randn(batch_size, latent_dim).to(device)

    random_caption_idx = torch.randint(0, captions.size(0), (1,)).item()
    selected_caption = captions[random_caption_idx].unsqueeze(0).repeat(batch_size, 1).to(device)

    vae.eval()
    with torch.no_grad():
        generated_images = vae.decode(z, selected_caption)
        
    generated_images = (generated_images.clamp(0, 1) * 255).to(torch.uint8)

    image_path = os.path.join(output_dir, f"sample_epoch_{epoch}.png")
    pil_image = transforms.ToPILImage()(generated_images[0]) 
    pil_image.save(image_path)  
