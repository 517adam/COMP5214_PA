#  COMP 6211D & ELEC 6910T , Assignment 3
#
# This is the main training file for the vanilla GAN part of the assignment.
#
# Usage:
# ======
#    To train with the default hyperparamters (saves results to checkpoints_vanilla/ and samples_vanilla/):
#       python vanilla_gan.py

import os
import argparse
import warnings
import numpy as np
warnings.filterwarnings("ignore")

# Torch imports
import torch
import torch.nn as nn
import torch.optim as optim

# Local imports
from data_loader import get_emoji_loader
from models import CycleGenerator, DCDiscriminator
from vanilla_utils import create_dir, create_model, checkpoint, sample_noise, save_samples
import matplotlib.pyplot as plt
SEED = 11

# Set the random seed manually for reproducibility.
np.random.seed(SEED)
torch.manual_seed(SEED)

# --- Modify this plotting function ---
def save_loss_plot(iterations, G_losses, D_real_losses, D_fake_losses, D_total_losses, epoch, output_dir):
    """Saves plots of the training losses."""

    plt.figure(figsize=(12, 6)) # Slightly wider figure for more curves
    plt.plot(iterations, G_losses, label='Generator Loss', alpha=0.8)
    plt.plot(iterations, D_total_losses, label='Discriminator Total Loss', alpha=0.8)
    plt.plot(iterations, D_real_losses, label='Discriminator Real Loss', linestyle='--', alpha=0.6) # Uncommented and styled
    plt.plot(iterations, D_fake_losses, label='Discriminator Fake Loss', linestyle=':', alpha=0.6) # Uncommented and styled

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'GAN Training Losses up to Epoch {epoch}')
    plt.legend()
    plt.grid(True)
    plt.ylim(bottom=0) # Often useful to start y-axis at 0 for losses
    # Consider adding plt.ylim(top=...) if losses become very large and obscure details near 0

    # Save the combined plot with epoch number
    loss_plot_path = os.path.join(output_dir, f'loss_plot_epoch_{epoch:04d}.png')
    try:
        plt.savefig(loss_plot_path)
        plt.close() # Close the figure to free memory
        # print(f'Loss plot saved to {loss_plot_path}') # Optional: uncomment to print save message
    except Exception as e:
        print(f"Error saving plot: {e}")

    # --- Remove the optional individual plots section if you don't need them ---
    # --- End of plotting function ---


def train(train_loader, opts, device):
    
    G, D = create_model(opts)
    
    G.to(device)
    D.to(device)
    
    g_optimizer = optim.Adam(G.parameters(), opts.lr, [opts.beta1, opts.beta2])
    d_optimizer = optim.Adam(D.parameters(), opts.lr, [opts.beta1, opts.beta2])
    
    fixed_noise = sample_noise(16, opts.noise_size).to(device)
    
    iteration = 1
    
    # mse_loss = torch.nn.MSELoss() # Can be used, but direct formula is clearer for LSGAN
    # bce_loss = torch.nn.BCELoss() # No longer needed
    total_train_iters = opts.num_epochs * len(train_loader)
    
    # Track losses for plotting
    G_losses = []
    D_real_losses = []
    D_fake_losses = []
    D_total_losses = []
    iterations_list = []
    
    for epoch in range(opts.num_epochs):
        epoch_G_losses = []
        epoch_D_real_losses = []
        epoch_D_fake_losses = []
        epoch_D_total_losses = []
        
        for batch_idx, batch in enumerate(train_loader):
            real_images = batch[0].to(device)

            ################################################
            ###         TRAIN THE DISCRIMINATOR         ####
            ################################################

            d_optimizer.zero_grad()

            # 1. Compute the discriminator loss on real images using LSGAN loss
            D_real = D(real_images)
            # D_real_loss = bce_loss(D_real, torch.ones_like(D_real)) # Old BCE loss
            D_real_loss = 0.5 * torch.mean((D_real - 1)**2) # LSGAN loss for real images

            # 2. Sample noise
            noise = sample_noise(real_images.size(0), opts.noise_size).to(device)

            # 3. Generate fake images from the noise
            fake_images = G(noise)

            # 4. Compute the discriminator loss on the fake images using LSGAN loss
            D_fake = D(fake_images.detach())
            # D_fake_loss = bce_loss(D_fake, torch.zeros_like(D_fake)) # Old BCE loss
            D_fake_loss = 0.5 * torch.mean((D_fake)**2) # LSGAN loss for fake images

            # 5. Compute the total discriminator loss
            D_total_loss = D_real_loss + D_fake_loss

            D_total_loss.backward()
            d_optimizer.step()

            ###########################################
            ###          TRAIN THE GENERATOR        ###
            ###########################################
            for _ in range(2):
             g_optimizer.zero_grad()

            # 1. Sample noise
             noise = sample_noise(real_images.size(0), opts.noise_size).to(device)

            # 2. Generate fake images from the noise
             fake_images = G(noise)

            # 3. Compute the generator loss (tries to' fool D)
             D_fake_for_G = D(fake_images)
            # G_loss = bce_loss(D_fake_for_G, torch.ones_like(D_fake_for_G))
             G_loss = torch.mean((D_fake_for_G - 1)**2) 
             G_loss.backward()
             g_optimizer.step()
            
            # Track losses
            epoch_G_losses.append(G_loss.item())
            epoch_D_real_losses.append(D_real_loss.item())
            epoch_D_fake_losses.append(D_fake_loss.item())
            epoch_D_total_losses.append(D_total_loss.item())

            # Print the log info
            current_iter = epoch * len(train_loader) + batch_idx + 1
            if current_iter % opts.log_step == 0:
                 print('Epoch [{:4d}/{:4d}] Iter [{:4d}/{:4d}] | D_real_loss: {:6.4f} | D_fake_loss: {:6.4f} | G_loss: {:6.4f}'.format(
                       epoch+1, opts.num_epochs, current_iter, total_train_iters, D_real_loss.item(), D_fake_loss.item(), G_loss.item()))


            # Save the generated samples
            if current_iter % opts.sample_every == 0:
                save_samples(G, fixed_noise, current_iter, opts)

            # Save the model parameters
            if current_iter % opts.checkpoint_every == 0:
                checkpoint(current_iter, G, D, opts)

        # At the end of each epoch, record average losses
        if epoch_G_losses: # Avoid division by zero if epoch has no batches
            avg_G_loss = sum(epoch_G_losses) / len(epoch_G_losses)
            avg_D_real_loss = sum(epoch_D_real_losses) / len(epoch_D_real_losses)
            avg_D_fake_loss = sum(epoch_D_fake_losses) / len(epoch_D_fake_losses)
            avg_D_total_loss = sum(epoch_D_total_losses) / len(epoch_D_total_losses)

            G_losses.append(avg_G_loss)
            D_real_losses.append(avg_D_real_loss)
            D_fake_losses.append(avg_D_fake_loss)
            D_total_losses.append(avg_D_total_loss)
            iterations_list.append(epoch + 1) # Append epoch number

            # Print epoch-wise average loss
            print('--- Epoch [{:4d}/{:4d}] Avg Losses | D_real: {:6.4f} | D_fake: {:6.4f} | G: {:6.4f} ---'.format(
                   epoch+1, opts.num_epochs, avg_D_real_loss, avg_D_fake_loss, avg_G_loss))

            # --- Plot losses every 10 epochs ---
            if (epoch + 1) % 10 == 0:
                save_loss_plot(iterations_list, G_losses, D_real_losses, D_fake_losses, D_total_losses, epoch + 1, opts.sample_dir)
        else:
             print(f"--- Epoch [{epoch+1}/{opts.num_epochs}] had no batches, skipping loss calculation. ---")
    
    
    
def main(opts):
    """Loads the data, creates checkpoint and sample directories, and starts the training loop.
    """

    # Create a dataloader for the training images
    train_loader, _ = get_emoji_loader(opts.emoji, opts)

    # Create checkpoint and sample directories
    create_dir(opts.checkpoint_dir)
    create_dir(opts.sample_dir)
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    train(train_loader, opts, device)


def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Model hyper-parameters
    parser.add_argument('--image_size', type=int, default=32, help='The side length N to convert images to NxN.')
    parser.add_argument('--conv_dim', type=int, default=32)
    parser.add_argument('--noise_size', type=int, default=100)

    # Training hyper-parameters
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=16, help='The number of images in a batch.')
    parser.add_argument('--num_workers', type=int, default=0, help='The number of threads to use for the DataLoader.')
    parser.add_argument('--lr', type=float, default=0.0003, help='The learning rate (default 0.0003)')
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)

    # Data sources
    parser.add_argument('--emoji', type=str, default='Apple', choices=['Apple', 'Facebook', 'Windows'], help='Choose the type of emojis to generate.')

    # Directories and checkpoint/sample iterations
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints_vanilla')
    parser.add_argument('--sample_dir', type=str, default='./samples_vanilla')
    parser.add_argument('--log_step', type=int , default=10)
    parser.add_argument('--sample_every', type=int , default=200)
    parser.add_argument('--checkpoint_every', type=int , default=400)

    return parser
 

if __name__ == '__main__':

    parser = create_parser()
    opts = parser.parse_args()

    batch_size = opts.batch_size

    print(opts)
    main(opts)

