import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset
from tqdm.auto import tqdm
import argparse
import wandb
import os
import yaml


# other files
from utils import *
from models import *

from diffusion.diffusion_sampler import *
from diffusion.denoiser import *

import matplotlib.pyplot as plt

def main():
    np.random.seed(1)
    torch.manual_seed(1)
    parser = argparse.ArgumentParser(description='Training script for setting various parameters.')
    parser.add_argument('--config', type=str, default='./configs/train_phase_two_config.yml', help='Path to the configuration YAML file.')

    args = parser.parse_args()
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
        

    num_epochs = config['num_epochs']
    num_train_demos = config['num_train_demos']
    pred_horizon = config['pred_horizon']
    obs_horizon = config['obs_horizon']
    action_horizon = config['action_horizon']
    batch_size = config['batch_size']
    dataset_path_dir = config['dataset_path_dir']
    models_save_dir = config['models_save_dir']



    if config["wandb"]:
        wandb.init(
            project="world_model_train_phase_two",
            config=config,
            name='world_model_train_phase_two'
        )
    else:
        print("warning: wandb flag set to False")
        

    resize_scale = 96


    if not os.path.exists(models_save_dir):
        os.makedirs(models_save_dir)

    dataset_list = []
    combined_stats = []
    num_datasets = 0
    dataset_name = {} # mapping for domain filename
    all_data_stats = {'agent_pos': {'min': np.array([2.0407837e-04, 1.0189312e+00], dtype=np.float32), 'max': np.array([509.08173, 509.43417], dtype=np.float32)}, 'action': {'min': np.array([0., 0.], dtype=np.float32), 'max': np.array([511., 511.], dtype=np.float32)}}

    for entry in sorted(os.listdir(dataset_path_dir)):
        if not (entry[-5:] == '.zarr'):
            continue
        full_path = os.path.join(dataset_path_dir, entry)

        domain_filename = entry.split(".")[0]
        dataset_name[num_datasets] = domain_filename        

        # create dataset from file
        dataset = TrainDataset(
            dataset_path=full_path,
            pred_horizon=pred_horizon,
            obs_horizon=obs_horizon,
            action_horizon=action_horizon,
            id = num_datasets,
            num_demos = num_train_demos,
            resize_scale = resize_scale,
            stats = all_data_stats
        )
        num_datasets += 1
        # save training data statistics (min, max) for each dim
        stats = dataset.stats
        dataset_list.append(dataset)
        combined_stats.append(stats)

    combined_dataset = ConcatDataset(dataset_list)

    # create dataloader
    dataloader = torch.utils.data.DataLoader(
        combined_dataset,
        batch_size=batch_size,
        num_workers=4,
        shuffle=True,
        # accelerate cpu-gpu transfer
        pin_memory=True,
        # don't kill worker process afte each epoch
        persistent_workers=True
    )

    nets = nn.ModuleDict({})

    @dataclass
    class SigmaDistributionConfig:
        loc = -1.2 
        scale = 1.2
        sigma_min = 2e-3
        sigma_max = 20

    sigma_distribution_config = SigmaDistributionConfig()
    
    @dataclass
    class InnerModelConfig:
        img_channels = 3
        num_steps_conditioning = 4
        cond_channels = 256

        depths = [2,2,2,2]
        channels = [96, 96, 96, 96]

        attn_depths = [0, 0, 1, 1]
        num_actions = 4
        is_upsampler = None  # set by Denoiser
    
    inner_model_config = InnerModelConfig()
        
        
        
    @dataclass
    class DenoiserConfig:
        inner_model: InnerModelConfig
        sigma_data: float = 0.5
        sigma_offset_noise: float = 0.1
        noise_previous_obs: bool = True
        upsampling_factor: Optional[int] = None
        
    denoiser_config = DenoiserConfig(inner_model = inner_model_config)
    
    @dataclass
    class DiffusionSamplerConfig:
        num_steps_denoising: int
        sigma_min: float = 2e-3
        sigma_max: float = 5
        rho: int = 7
        order: int = 1
        s_churn: float = 0
        s_tmin: float = 0
        s_tmax: float = float("inf")
        s_noise: float = 1
        s_cond: float = 0 
        
        
    diffusion_sampler_config = DiffusionSamplerConfig(num_steps_denoising=3)
    
    nets["denoiser"] = Denoiser(denoiser_config)
    nets = nets.to(device)
    nets["denoiser"].setup_training(sigma_distribution_config)
    eval_sampler = DiffusionSampler(nets["denoiser"], diffusion_sampler_config)
    
    model_state_dict = torch.load(config['phase_one_checkpoint'])
    nets["denoiser"].load_state_dict(model_state_dict)

    # Standard ADAM optimizer
    # Note that EMA parameters are not optimized
    optimizer = torch.optim.AdamW(
        params=nets.parameters(),
        lr=0.0001)

  
  
    with tqdm(range(1, num_epochs+1), desc='Epoch') as tglobal:
        # unique_ids = torch.arange(num_datasets).cpu()
        # epoch loop
        for epoch_idx in tglobal:
            if config['wandb']:
                wandb.log({'epoch': epoch_idx})    
            epoch_loss = list()
            # batch loop
            with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
                for nbatch in tepoch:
                    
                    loss, metrics = nets["denoiser"](nbatch, device) 

                    # optimize
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                
                    # logging
                    loss_cpu = loss.item()
                    if config['wandb']:
                        wandb.log({'loss': loss_cpu, 'epoch': epoch_idx})
                    epoch_loss.append(loss_cpu)
                    tepoch.set_postfix(loss=loss_cpu)
            tglobal.set_postfix(loss=np.mean(epoch_loss))

            # save and eval upon request
            if (epoch_idx % 1 ==0):

                # create new checkpoint
                checkpoint_dir = '{}/checkpoint_epoch_{}'.format(models_save_dir, epoch_idx)

                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                save(nets, checkpoint_dir)
                
                # pred_image, _ = eval_sampler.sample(nbatch['image'][:, :4].to(device), nbatch['action'][:, :4].to(device))

                # for i in range(pred_image.shape[0]):
                #     plt.figure()
                #     plt.imshow(torch.moveaxis(torch.squeeze(pred_image[i]), 0, -1).detach().cpu().numpy())
                #     plt.savefig(f"pred_image_{i}.png")
                    
                #     plt.figure()
                #     plt.imshow(torch.moveaxis(torch.squeeze(nbatch['image'][:, 4][i]), 0, -1).detach().numpy())
                #     plt.savefig(f"ground_truth_image_{i}.png")

                    
if __name__ == "__main__":
    main()