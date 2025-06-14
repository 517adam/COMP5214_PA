import os
import torch
import scipy
import scipy.misc
import numpy as np
from models import DCGenerator, DCDiscriminator
import imageio
SEED = 11

# Set the random seed manually for reproducibility.
torch.manual_seed(SEED)

def create_dir(directory):
    """Creates a directory if it does not already exist.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def create_model(opts):
    """Builds the generators and discriminators.
    """
    G = DCGenerator(noise_size=opts.noise_size, conv_dim=opts.conv_dim)
    D = DCDiscriminator(conv_dim=opts.conv_dim)

    return G, D

def checkpoint(iteration, G, D, opts):
    """Saves the parameters of the generator G and discriminator D.
    """
    ckpt_path = os.path.join(opts.checkpoint_dir, 'ckpt_{:06d}.pth.tar'.format(iteration))
    torch.save({'G': G.state_dict(),
                'D': D.state_dict(),
                'iter': iteration}, 
               ckpt_path)

def create_image_grid(array, ncols=None):
    num_images, channels, cell_h, cell_w = array.shape

    if not ncols:
        ncols = int(np.sqrt(num_images))
    nrows = int(np.floor(num_images / float(ncols)))
    result = np.zeros((cell_h*nrows, cell_w*ncols, channels), dtype=array.dtype)
    for i in range(0, nrows):
        for j in range(0, ncols):
            result[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w, :] = array[i*ncols+j].transpose(1, 2, 0)

    if channels == 1:
        result = result.squeeze()
    return result

def save_samples(G, fixed_noise, iteration, opts):
    generated_images = G(fixed_noise).cpu().data.numpy()
    # create H×W×C grid in float32 in [–1,1]
    grid = create_image_grid(generated_images)

    # convert floats [–1,1] → uint8 [0,255]
    grid = (grid + 1.0) / 2.0             # now in [0,1]
    grid = (grid * 255.0).clip(0, 255)    # [0,255]
    grid = grid.astype(np.uint8)

    path = os.path.join(opts.sample_dir, f'sample-{iteration:06d}.png')
    imageio.imwrite(path, grid)
    print(f'Saved {path}')
    

def sample_noise(batch_size, dim):
    """
    Generate a PyTorch Variable of uniform random noise.

    Input:
    - batch_size: Integer giving the batch size of noise to generate.
    - dim: Integer giving the dimension of noise to generate.

    Output:
    - A PyTorch Variable of shape (batch_size, dim, 1, 1) containing uniform
      random noise in the range (-1, 1).
    """
    noise = torch.rand(batch_size, dim) * 2 - 1
    return noise.view(batch_size, dim, 1, 1)

