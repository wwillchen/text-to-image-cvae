import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import os

import CocoDataset
from utils import sample_and_save


dataset = CocoDataset(
    root_dir='/path/to/coco/images',
    annotation_file='/path/to/coco/annotations/captions_train2017.json',
    transform=transform,
    fraction=0.3
)

dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)

latent_dim = 128
text_embedding_dim = 768  # BERT embedding size
encoder = Encoder(latent_dim=latent_dim, text_embedding_dim=text_embedding_dim)
decoder = Decoder(latent_dim=latent_dim, label_dim=text_embedding_dim)
cvae = ConvCVAE(encoder, decoder, text_vocab_size=None, text_embedding_dim=text_embedding_dim, latent_dim=latent_dim)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cvae = cvae.to(device)
optimizer = optim.Adam(cvae.parameters(), lr=1e-4)
num_epochs = 10

cvae.train()
log_file = "training_log.txt"
os.makedirs(os.path.dirname(log_file), exist_ok=True)

with open(log_file, "w") as f: 
    f.write("Epoch, Total Loss, Latent Loss, Reconstruction Loss\n") 

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        latent_loss_epoch = 0.0
        reconstr_loss_epoch = 0.0
        captions_batch = None  

        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            images = batch['image'].to(device)  # Shape: [batch_size, 3, 64, 64]
            captions = batch['caption'].to(device)  # Shape: [batch_size, 768]

            if captions_batch is None:
                captions_batch = captions

            optimizer.zero_grad()

            outputs = cvae((images, captions), is_train=True)
            loss = outputs['loss']
            latent_loss = outputs['latent_loss'].mean()
            reconstr_loss = outputs['reconstr_loss'].mean()

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            latent_loss_epoch += latent_loss.item()
            reconstr_loss_epoch += reconstr_loss.item()

        avg_loss = epoch_loss / len(dataloader)
        avg_latent_loss = latent_loss_epoch / len(dataloader)
        avg_reconstr_loss = reconstr_loss_epoch / len(dataloader)

        f.write(f"{epoch + 1}, {avg_loss:.4f}, {avg_latent_loss:.4f}, {avg_reconstr_loss:.4f}\n")
        f.flush()  

        print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {avg_loss:.4f}, Latent Loss: {avg_latent_loss:.4f}, Reconstruction Loss: {avg_reconstr_loss:.4f}")

        # Save a sampled image every other epoch
        if (epoch + 1) % 2 == 0:
            sample_and_save(cvae, captions_batch, epoch + 1, device)