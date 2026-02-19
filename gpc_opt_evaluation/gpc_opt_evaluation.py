import numpy as np
import torch
import argparse
import wandb
import yaml
from eval_baseline import eval_baseline

def main():
    np.random.seed(1)
    torch.manual_seed(1)
    parser = argparse.ArgumentParser(description='Training script for setting various parameters.')
    parser.add_argument('--config', type=str, default='./configs/gpc_opt_evaluation_config.yml', help='Path to the configuration YAML file.')

    args = parser.parse_args()
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
        

    if config["wandb"]:
        wandb.init(
            project="gpc_opt_evaluation",
            config=config,
            name="gpc_opt_evaluation"
        )
    else:
        print("warning: wandb flag set to False")
        

    eval_baseline(config, config['policy_checkpoint'])

                
if __name__ == "__main__":
    main()