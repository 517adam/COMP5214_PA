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

SEED = 11

# Set the random seed manually for reproducibility.
np.random.seed(SEED)
torch.manual_seed(SEED)

def train(train_loader, opts, device):
    
    G, D = create_model(opts)
    
    G.to(device)
    D.to(device)
    
    g_optimizer = optim.Adam(G.parameters(), 1.6*opts.lr, [opts.beta1, opts.beta2])
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
    iterations = []
    
    for epoch in range(opts.num_epochs):
        epoch_G_losses = []
        epoch_D_real_losses = []
        epoch_D_fake_losses = []
        epoch_D_total_losses = []
        
        for batch in train_loader:
            real_images = batch[0].to(device)

            ################################################
            ###         TRAIN THE DISCRIMINATOR         ####
            ################################################

            d_optimizer.zero_grad()

            # 1. Compute the discriminator loss on real images using BCE loss
            D_real = D(real_images)
            D_real_loss = nn.BCELoss()(D_real, torch.ones_like(D_real))

            # 2. Sample noise
            noise = sample_noise(real_images.size(0), opts.noise_size).to(device)

            # 3. Generate fake images from the noise
            fake_images = G(noise)

            # 4. Compute the discriminator loss on the fake images using BCE loss
            D_fake = D(fake_images.detach())
            D_fake_loss = nn.BCELoss()(D_fake, torch.zeros_like(D_fake))

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

                # 3. Compute the generator loss (tries to fool D)
                D_fake_for_G = D(fake_images)
                G_loss = nn.BCELoss()(D_fake_for_G, torch.ones_like(D_fake_for_G))
                G_loss.backward()
                g_optimizer.step()
            
            # Track losses
            epoch_G_losses.append(G_loss.item())
            epoch_D_real_losses.append(D_real_loss.item())
            epoch_D_fake_losses.append(D_fake_loss.item())
            epoch_D_total_losses.append(D_total_loss.item())

            # Print the log info
            if iteration % opts.log_step == 0:
                print('Iteration [{:4d}/{:4d}] | D_real_loss: {:6.4f} | D_fake_loss: {:6.4f} | G_loss: {:6.4f}'.format(
                       iteration, total_train_iters, D_real_loss.item(), D_fake_loss.item(), G_loss.item()))

            # Save the generated samples
            if iteration % opts.sample_every == 0:
                save_samples(G, fixed_noise, iteration, opts)

            # Save the model parameters
            if iteration % opts.checkpoint_every == 0:
                checkpoint(iteration, G, D, opts)

            iteration += 1
        
        # At the end of each epoch, record average losses
        G_losses.append(sum(epoch_G_losses) / len(epoch_G_losses))
        D_real_losses.append(sum(epoch_D_real_losses) / len(epoch_D_real_losses))
        D_fake_losses.append(sum(epoch_D_fake_losses) / len(epoch_D_fake_losses))
        D_total_losses.append(sum(epoch_D_total_losses) / len(epoch_D_total_losses))
        iterations.append(epoch + 1)
        
        # Print epoch-wise loss
        print('Epoch [{:4d}/{:4d}] | D_real_loss: {:6.4f} | D_fake_loss: {:6.4f} | G_loss: {:6.4f}'.format(
               epoch+1, opts.num_epochs, D_real_losses[-1], D_fake_losses[-1], G_losses[-1]))
    
    # After training, plot the losses
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 8))
        plt.plot(iterations, G_losses, label='Generator Loss')
        plt.plot(iterations, D_real_losses, label='Discriminator Real Loss')
        plt.plot(iterations, D_fake_losses, label='Discriminator Fake Loss')
        plt.plot(iterations, D_total_losses, label='Discriminator Total Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('GAN Training Losses')
        plt.legend()
        plt.grid(True)
        
        # Save the plot
        loss_plot_path = os.path.join(opts.sample_dir, 'loss_plot.png')
        plt.savefig(loss_plot_path)
        print(f'Loss plot saved to {loss_plot_path}')
        
        # Create individual plots for better visualization
        plt.figure(figsize=(10, 8))
        plt.plot(iterations, G_losses, 'r-')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Generator Loss')
        plt.grid(True)
        plt.savefig(os.path.join(opts.sample_dir, 'g_loss_plot.png'))
        
        plt.figure(figsize=(10, 8))
        plt.plot(iterations, D_total_losses, 'b-')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Discriminator Loss')
        plt.grid(True)
        plt.savefig(os.path.join(opts.sample_dir, 'd_loss_plot.png'))
        
    except ImportError:
        print("Matplotlib not found. Skipping loss plot generation.")
    
    
    
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

