#  COMP 6211D & ELEC 6910T , Assignment 3
#
# This is the main training file for the CycleGAN part of the assignment.
#
# Usage:
# ======
#    To train with the default hyperparamters (saves results to samples_cyclegan/):
#       python cycle_gan.py
#
#    For optional experimentation:
#    -----------------------------
#    If you have a powerful computer (ideally with a GPU), then you can obtain better results by
#    increasing the number of filters used in the generator and/or discriminator, as follows:
#      python cycle_gan.py --g_conv_dim=64 --d_conv_dim=64

import os
import argparse
import warnings
warnings.filterwarnings("ignore")

# Torch imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Numpy & Scipy imports
import numpy as np
import scipy
import scipy.misc
import imageio
# Local imports
from data_loader import get_emoji_loader
from models import CycleGenerator, DCDiscriminator
from cycle_utils import create_dir, create_model, checkpoint, save_samples


SEED = 11
# Set the random seed manually for reproducibility.
np.random.seed(SEED)
torch.manual_seed(SEED)


def training_loop(dataloader_X, dataloader_Y, test_dataloader_X, test_dataloader_Y, device, opts):
    """Runs the training loop.
        * Saves checkpoint every opts.checkpoint_every iterations
        * Saves generated samples every opts.sample_every iterations
    """

    # Create generators and discriminators
    G_XtoY, G_YtoX, D_X, D_Y = create_model(opts)
    G_XtoY.to(device)
    G_YtoX.to(device)
    D_X.to(device)
    D_Y.to(device)

    g_params = list(G_XtoY.parameters()) + list(G_YtoX.parameters())  # Get generator parameters
    d_params = list(D_X.parameters()) + list(D_Y.parameters())  # Get discriminator parameters

    # Create optimizers for the generators and discriminators
    g_optimizer = optim.Adam(g_params, opts.lr, [opts.beta1, opts.beta2])
    d_optimizer = optim.Adam(d_params, opts.lr, [opts.beta1, opts.beta2])

    iter_X = iter(dataloader_X)
    iter_Y = iter(dataloader_Y)

    test_iter_X = iter(test_dataloader_X)
    test_iter_Y = iter(test_dataloader_Y)

    # Get some fixed data from domains X and Y for sampling. These are images that are held
    # constant throughout training, that allow us to inspect the model's performance.
    fixed_X = next(test_iter_X)[0].to(device)
    fixed_Y = next(test_iter_Y)[0].to(device)

    iter_per_epoch = min(len(dataloader_X), len(dataloader_Y))
    
    # Lists to track losses for plotting
    d_real_losses = []
    d_fake_losses = []
    g_losses = []
    g_cycle_losses = []
    g_adv_losses = []
    iterations_list = []
    
    lambda_cycle = 3  # cycle‐consistency weight

    for iteration in range(1, opts.train_iters+1):

        # Reset data_iter for each epoch
        if iteration % iter_per_epoch == 0:
            iter_X = iter(dataloader_X)
            iter_Y = iter(dataloader_Y)
            
        images_X = next(iter_X)[0].to(device)
        images_Y = next(iter_Y)[0].to(device)

        # ============================================
        #            TRAIN THE DISCRIMINATORS
        # ============================================

        # (1) Real images - LSGAN loss = (D(x) - 1)²
        d_optimizer.zero_grad()
        out_real_X = D_X(images_X)
        out_real_Y = D_Y(images_Y)
        # LSGAN: use MSE with target=1 for real images
        D_X_loss = 0.5 * torch.mean((out_real_X - 1)**2)
        D_Y_loss = 0.5 * torch.mean((out_real_Y - 1)**2)
        d_real_loss = D_X_loss + D_Y_loss
        d_real_loss.backward()
        d_optimizer.step()

        # (2) Fake images - LSGAN loss = (D(G(y))²
        d_optimizer.zero_grad()
        fake_X = G_YtoX(images_Y)
        fake_Y = G_XtoY(images_X)
        out_fake_X = D_X(fake_X.detach())
        out_fake_Y = D_Y(fake_Y.detach())
        # LSGAN: use MSE with target=0 for fake images
        D_X_fake_loss = 0.5 * torch.mean(out_fake_X**2)
        D_Y_fake_loss = 0.5 * torch.mean(out_fake_Y**2)
        d_fake_loss = D_X_fake_loss + D_Y_fake_loss
        d_fake_loss.backward()
        d_optimizer.step()

        # =========================================
        #            TRAIN THE GENERATORS
        # =========================================

        # (3) Y → X → Y cycle
        g_optimizer.zero_grad()
        fake_X = G_YtoX(images_Y)
        out_fake_X = D_X(fake_X)
        # LSGAN: use MSE with target=1 for generator
        g_adv_X = torch.mean((out_fake_X - 1)**2)

        rec_Y = G_XtoY(fake_X)
        # --- Change cycle consistency loss to MSE (L2 Loss) ---
        cycle_Y = torch.mean(torch.square(rec_Y - images_Y)) # MSE Loss
        # --- End of change ---
        # Keep the weighted sum using lambda_cycle
        g_loss_X = g_adv_X + lambda_cycle * cycle_Y

        g_loss_X.backward()
        g_optimizer.step()

        # (4) X → Y → X cycle
        g_optimizer.zero_grad()
        fake_Y = G_XtoY(images_X)
        out_fake_Y = D_Y(fake_Y)
        # LSGAN: use MSE with target=1 for generator
        g_adv_Y = torch.mean((out_fake_Y - 1)**2)
        rec_X = G_YtoX(fake_Y)
        # --- Change cycle consistency loss to MSE (L2 Loss) ---
        # cycle_X = F.l1_loss(rec_X, images_X) # Original L1 Loss
        cycle_X = torch.mean(torch.square(rec_X - images_X)) # MSE Loss
        # --- End of change ---
        # Keep the weighted sum using lambda_cycle
        g_loss_Y = g_adv_Y + lambda_cycle * cycle_X
        g_loss_Y.backward()
        g_optimizer.step()

        # total losses for logging
        g_loss = g_loss_X + g_loss_Y
        g_cycle_loss = cycle_X + cycle_Y
        g_adv_loss = g_adv_X + g_adv_Y
        
        # Track losses for plotting
        if iteration % 10 == 0:  # Save every 10 iterations to save memory
            d_real_losses.append(d_real_loss.item())
            d_fake_losses.append(d_fake_loss.item())
            g_losses.append(g_loss.item())
            g_cycle_losses.append(g_cycle_loss.item())
            g_adv_losses.append(g_adv_loss.item())
            iterations_list.append(iteration)

        # Print the log info
        if iteration % opts.log_step == 0:
            print('Iteration [{:5d}/{:5d}] | d_real_loss: {:6.4f} | d_fake_loss: {:6.4f} | '
                  'g_adv_loss: {:6.4f} | g_cycle_loss: {:6.4f} | g_loss: {:6.4f}'.format(
                    iteration, opts.train_iters, d_real_loss.item(), d_fake_loss.item(),
                    g_adv_loss.item(), g_cycle_loss.item(), g_loss.item()))

        # Save the generated samples
        if iteration % opts.sample_every == 0:
            save_samples(iteration, fixed_Y, fixed_X, G_YtoX, G_XtoY, opts)

        # Save the model parameters
        if iteration % opts.checkpoint_every == 0:
            checkpoint(iteration, G_XtoY, G_YtoX, D_X, D_Y, opts)
            
            # Also save loss plots at checkpoints
            try:
                import matplotlib
                matplotlib.use('Agg')  # Use non-interactive backend
                import matplotlib.pyplot as plt
                
                # Combined loss plot
                plt.figure(figsize=(12, 8))
                plt.plot(iterations_list, d_real_losses, label='D Real Loss')
                plt.plot(iterations_list, d_fake_losses, label='D Fake Loss')
                plt.plot(iterations_list, g_losses, label='G Total Loss')
                plt.plot(iterations_list, g_cycle_losses, label='G Cycle Loss')
                plt.plot(iterations_list, g_adv_losses, label='G Adversarial Loss')
                plt.xlabel('Iterations')
                plt.ylabel('Loss')
                plt.title('CycleGAN Training Losses')
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(opts.sample_dir, f'losses_{iteration}.png'))
                plt.close()
                
                # Individual loss plots
                plt.figure(figsize=(10, 6))
                plt.plot(iterations_list, d_real_losses, 'b-', label='D Real Loss')
                plt.plot(iterations_list, d_fake_losses, 'r-', label='D Fake Loss')
                plt.xlabel('Iterations')
                plt.ylabel('Loss')
                plt.title('Discriminator Losses')
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(opts.sample_dir, f'd_losses_{iteration}.png'))
                plt.close()
                
                plt.figure(figsize=(10, 6))
                plt.plot(iterations_list, g_losses, 'g-')
                plt.xlabel('Iterations')
                plt.ylabel('Loss')
                plt.title('Generator Total Loss')
                plt.grid(True)
                plt.savefig(os.path.join(opts.sample_dir, f'g_total_loss_{iteration}.png'))
                plt.close()
                
                plt.figure(figsize=(10, 6))
                plt.plot(iterations_list, g_cycle_losses, 'c-', label='Cycle Loss')
                plt.plot(iterations_list, g_adv_losses, 'm-', label='Adversarial Loss')
                plt.xlabel('Iterations')
                plt.ylabel('Loss')
                plt.title('Generator Component Losses')
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(opts.sample_dir, f'g_component_losses_{iteration}.png'))
                plt.close()
                
                print(f"Loss plots saved at iteration {iteration}")
            except ImportError:
                print("Matplotlib not found. Skipping loss plot generation.")


def main(opts):
    """Loads the data, creates checkpoint and sample directories, and starts the training loop.
    """

    # Create train and test dataloaders for images from the two domains X and Y
    dataloader_X, test_dataloader_X = get_emoji_loader(emoji_type=opts.X, opts=opts)
    dataloader_Y, test_dataloader_Y = get_emoji_loader(emoji_type=opts.Y, opts=opts)

    # Create checkpoint and sample directories
    create_dir(opts.checkpoint_dir)
    create_dir(opts.sample_dir)
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        

    # Start training
    training_loop(dataloader_X, dataloader_Y, test_dataloader_X, test_dataloader_Y, 
                  device, opts)


def print_opts(opts):
    """Prints the values of all command-line arguments.
    """
    print('=' * 80)
    print('Opts'.center(80))
    print('-' * 80)
    for key in opts.__dict__:
        if opts.__dict__[key]:
            print('{:>30}: {:<30}'.format(key, opts.__dict__[key]).center(80))
    print('=' * 80)


def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Model hyper-parameters
    parser.add_argument('--image_size', type=int, default=32, help='The side length N to convert images to NxN.')
    parser.add_argument('--g_conv_dim', type=int, default=32)
    parser.add_argument('--d_conv_dim', type=int, default=32)
    parser.add_argument('--init_zero_weights', action='store_true', default=False, help='Choose whether to initialize the generator conv weights to 0 (implements the identity function).')

    # Training hyper-parameters
    parser.add_argument('--train_iters', type=int, default=100000, help='The number of training iterations to run (you can Ctrl-C out earlier if you want).')
    parser.add_argument('--batch_size', type=int, default=16, help='The number of images in a batch.')
    parser.add_argument('--num_workers', type=int, default=0, help='The number of threads to use for the DataLoader.')
    parser.add_argument('--lr', type=float, default=0.0003, help='The learning rate (default 0.0003)')
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)

    # Data sources
    parser.add_argument('--X', type=str, default='Apple', choices=['Apple', 'Windows'], help='Choose the type of images for domain X.')
    parser.add_argument('--Y', type=str, default='Windows', choices=['Apple', 'Windows'], help='Choose the type of images for domain Y.')

    # Saving directories and checkpoint/sample iterations
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_cyclegan')
    parser.add_argument('--sample_dir', type=str, default='samples_cyclegan')
    parser.add_argument('--log_step', type=int , default=10)
    parser.add_argument('--sample_every', type=int , default=100)
    parser.add_argument('--checkpoint_every', type=int , default=800)

    return parser


if __name__ == '__main__':

    parser = create_parser()
    opts = parser.parse_args()

    print_opts(opts)
    main(opts)

