import torchvision.utils as vutils
from matplotlib import pyplot as plt

def sample_and_save(cvae, captions, epoch, device, output_dir="samples"):
    """
    Samples an image from the CVAE and saves it.
    
    Args:
        cvae: The trained CVAE model.
        captions: A batch of caption embeddings from the dataset.
        epoch: Current epoch number.
        device: Device (CPU or GPU) on which the model runs.
        output_dir: Directory to save the sampled images.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    batch_size = captions.size(0)
    latent_dim = cvae.latent_dim
    z = torch.randn(batch_size, latent_dim).to(device)

    # Conditioned latent 
    random_caption_idx = torch.randint(0, captions.size(0), (1,)).item()
    selected_caption = captions[random_caption_idx].unsqueeze(0).repeat(batch_size, 1)  
    z_cond = torch.cat([z, selected_caption], dim=1)

    cvae.eval()
    with torch.no_grad():
        generated_images = cvae.decoder(z_cond, is_train=False)
        
    generated_images = (generated_images + 1) / 2
    image_path = os.path.join(output_dir, f"sample_epoch_{epoch}.png")
    vutils.save_image(generated_images[0], image_path)

    # plt.imshow(generated_images[0].cpu().permute(1, 2, 0).numpy())
    # plt.axis('off')
    # plt.title(f"Sampled Image at Epoch {epoch}")
    # plt.show()