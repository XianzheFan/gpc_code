import numpy as np
import torch
import torch.nn as nn
import collections
from diffusers.training_utils import EMAModel
from tqdm.auto import tqdm
from skvideo.io import vwrite
import os
import argparse
import json
from utils import *
from pusht_env import *
from models import *
import matplotlib.pyplot as plt
from diffusion.diffusion_sampler import *
from diffusion.denoiser import *
import torchvision.models as models

debug = False


class RewardPredictor(nn.Module):
    def __init__(self):
        super(RewardPredictor, self).__init__()
        # Load the pretrained ResNet18 model
        self.resnet18 = models.resnet18(pretrained=False)
        
        # Remove the final fully connected layer (classifier)
        self.resnet18 = nn.Sequential(*list(self.resnet18.children())[:-1])
        
        # Define the MLP for (x, y, theta) regression
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # Output: (x, y, theta)
        )

    def forward(self, x):
        # Extract features using ResNet18
        features = self.resnet18(x)
        
        # Predict (x, y, theta) using MLP
        pose = self.mlp(features)
        
        return pose

def transform_vertices_torch(px: torch.Tensor, py: torch.Tensor, ptheta: torch.Tensor, 
                           vertices1: torch.Tensor, vertices2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Create rotation matrix for counter-clockwise rotation
    rotation_matrix = torch.vstack([
        torch.stack((torch.cos(ptheta), -torch.sin(ptheta))),
        torch.stack((torch.sin(ptheta), torch.cos(ptheta)))
    ])

    # Perform rotation
    new_vertices1 = vertices1 @ rotation_matrix
    new_vertices2 = vertices2 @ rotation_matrix
    
    # Perform translation
    translation = torch.stack([px, py])
    new_vertices1 = new_vertices1 + translation
    new_vertices2 = new_vertices2 + translation
    
    return new_vertices1, new_vertices2

def estimate_reward_torch(block_pose: torch.Tensor, target_pose: torch.Tensor) -> torch.Tensor:
    # Convert initial vertices to torch tensors
    vertices1 = torch.tensor([[-10.0, 2.5], [10.0, 2.5], [10.0, 7.5], [-10.0, 7.5]], dtype=torch.float32, device=device) * 6.0
    vertices2 = torch.tensor([[-2.5, 2.5], [-2.5, -12.5], [2.5, -12.5], [2.5, 2.5]], dtype=torch.float32, device=device) * 6.0

    # Transform vertices for both block and goal
    block_verts1, block_verts2 = transform_vertices_torch(
        block_pose[0], block_pose[1], block_pose[2], 
        vertices1, vertices2
    )
    goal_verts1, goal_verts2 = transform_vertices_torch(
        target_pose[0], target_pose[1], target_pose[2], 
        vertices1, vertices2
    )

    # Concatenate vertices
    block_verts = torch.cat([block_verts1, block_verts2], dim=0)
    goal_verts = torch.cat([goal_verts1, goal_verts2], dim=0)

    # Calculate distances and reward
    dist_sum = torch.norm(block_verts - goal_verts, dim=1).sum()
    reward = 0.01 * dist_sum

    return reward


def eval_baseline(config, models_save_dir):

    dynamics_stats = {'agent_pos': {'min': np.array([2.0407837e-04, 1.0189312e+00], dtype=np.float32), 'max': np.array([509.08173, 509.43417], dtype=np.float32)}, 'action': {'min': np.array([0., 0.], dtype=np.float32), 'max': np.array([511., 511.], dtype=np.float32)}}
    domain18_stats = {'agent_pos': {'min': np.array([9.897889, 9.63592 ], dtype=np.float32), 'max': np.array([499.517  , 499.00488], dtype=np.float32)}, 'action': {'min': np.array([2., 2.], dtype=np.float32), 'max': np.array([511., 511.], dtype=np.float32)}}

    num_diffusion_iters = config['num_diffusion_iters']
    pred_horizon = config['pred_horizon']
    obs_horizon = config['obs_horizon']
    action_horizon = config['action_horizon']
    output_dir = config['output_dir']
    resize_scale = config["resize_scale"]


    nets = nn.ModuleDict({})

    vision_encoder = get_resnet()
    vision_encoder = replace_bn_with_gn(vision_encoder)


    nets['vision_encoder'] = vision_encoder

    vision_feature_dim = 512
    lowdim_obs_dim = 2
    obs_dim = vision_feature_dim + lowdim_obs_dim
    action_dim = 2

    invariant = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim*obs_horizon
    )
    nets['invariant'] = invariant 

    nets = nets.to(device)

    ema = EMAModel(
        parameters=nets.parameters(),
        power=0.75)

        
    for model_name, model in nets.items():
        model_path = os.path.join(models_save_dir, f"{model_name}.pth")
        model_state_dict = torch.load(model_path)
        model.load_state_dict(model_state_dict)

    ema_nets = nets
    ema_path = os.path.join(models_save_dir, f"ema_nets.pth")
    model_state_dict = torch.load(ema_path)
    ema.load_state_dict(model_state_dict)
    ema.copy_to(ema_nets.parameters())
    

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
    
    
    model_state_dict = torch.load(config['world_model_checkpoint'])
    nets["denoiser"].load_state_dict(model_state_dict)
    eval_sampler = DiffusionSampler(nets["denoiser"], diffusion_sampler_config)

    print("All models have been loaded successfully.")
    reward_predictor_unnormalized_xy = RewardPredictor().to(device)
    reward_predictor_cossin_angle = RewardPredictor().to(device)

    reward_predictor_unnormalized_xy.load_state_dict(torch.load(config['reward_predictor_xy_checkpoint']))
    reward_predictor_unnormalized_xy.eval()
    
    reward_predictor_cossin_angle.load_state_dict(torch.load(config['reward_predictor_angle_checkpoint']))
    reward_predictor_cossin_angle.eval()
    

    # (num_domains, num_tests)
    scores = [] 
    json_dict = dict()
    
    env_j_scores = []
    env_seed = 100000   # first test seed


    with open("./domains_yaml/{}.yml".format('push_t'), 'r') as stream:
        data_loaded = yaml.safe_load(stream)        
    env_id = data_loaded["domain_id"]

    json_dict["domain_{}".format(env_id)] = []

    print("\nEval Diff Policy on Domain #{}:".format(env_id))

    start_number_test = 0
    end_number_test = start_number_test + 100
    env_seed = env_seed + start_number_test

    for test_index in range(start_number_test, end_number_test):
        noise_scheduler = create_injected_noise(num_diffusion_iters)

        max_steps = config["max_steps"]
        env = PushTImageEnv(domain_filename='push_t', resize_scale=resize_scale)

        env.seed(env_seed)
        domain18_stats = {'agent_pos': {'min': np.array([9.897889, 9.63592 ], dtype=np.float32), 'max': np.array([499.517  , 499.00488], dtype=np.float32)}, 'action': {'min': np.array([2., 2.], dtype=np.float32), 'max': np.array([511., 511.], dtype=np.float32)}, 'train_block_position': {'min': np.array([8.597838e+01, 9.022391e+01, 1.947196e-04], dtype=np.float32), 'max': np.array([423.86835 , 426.0116  ,   6.283044], dtype=np.float32)}, 'train_target_positon': {'min': np.array([1.00122409e+02, 1.00698269e+02, 1.06752418e-04]), 'max': np.array([410.93441879, 410.32839602,   6.2766682 ])}, 'train_future_block_position': {'min': np.array([8.597838e+01, 9.022391e+01, 1.947196e-04], dtype=np.float32), 'max': np.array([423.86835 , 426.0116  ,   6.283044], dtype=np.float32)}, 'train_future_agent_pose': {'min': np.array([9.897889, 9.63592 ], dtype=np.float32), 'max': np.array([499.517  , 499.00488], dtype=np.float32)}}

        # get first observation
        obs, info = env.reset()
        aa = obs['goal_pose']

        target_pose = torch.tensor(aa, dtype=torch.float32).to(device)
        target_pose[2] = target_pose[2] % (2 * np.pi)

        # keep a queue of last 2 steps of observations
        obs_deque = collections.deque(
            [obs] * obs_horizon, maxlen=obs_horizon)
        # save visualization and rewards
        imgs = [env.render(mode='rgb_array')]
        pred_video_imgs = [env.render(mode='rgb_array')]
        
        rewards = list()
        done = False
        step_idx = 0
        
        transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.uint8, scale=True),
            v2.Resize(96),
            v2.ToDtype(torch.float32, scale=True),
        ])   
        
        
        tqdm._instances.clear()
        
        last_obs_gt = []
        all_action_save = []
        
        original_reward = []
        optimized_reward = []
        start_registration_loss_lst = []
        end_registration_loss_lst = []
        gt_start_registration_loss_lst = []
        gt_end_registration_loss_lst = []
        with tqdm(total=max_steps, desc="Eval Trial #{}".format(test_index)) as pbar:
            while not done:
                B = 1
                # stack the last obs_horizon number of observations
                images = np.stack([x['image'] for x in obs_deque])
                agent_poses = np.stack([x['agent_pos'] for x in obs_deque])

                # normalize observation
                nagent_poses = normalize_data(agent_poses, stats=domain18_stats['agent_pos'])
                nagent_poses_dynamics = normalize_data(agent_poses, stats=dynamics_stats['agent_pos'])

                # device transfer
                nimages = torch.from_numpy(images).to(device, dtype=torch.float32)
                # (2,3,96,96)
                nagent_poses = torch.from_numpy(nagent_poses).to(device, dtype=torch.float32)
                nagent_poses_dynamics = torch.from_numpy(nagent_poses_dynamics).to(device, dtype=torch.float32)

                # (2,2)                

                num_trial = 1
                # infer action
                with torch.no_grad():
                    # get image features
                    #image_features = ema_nets["vision_encoder"](nimages)

                    # encoder vision features
                    image_features = ema_nets["vision_encoder"](nimages)

                    image_features = image_features.squeeze()
                    # concat with low-dim observations
                    obs_features = torch.cat([image_features, nagent_poses], dim=-1)

                    # reshape observation to (B,obs_horizon*obs_dim)
                    obs_cond = obs_features.unsqueeze(0).flatten(start_dim=1)

                    # initialize action from Guassian noise
                    noisy_action = torch.randn((num_trial, pred_horizon, action_dim), device=device)
                    naction = noisy_action

                    # init scheduler
                    noise_scheduler.set_timesteps(num_diffusion_iters)

                    for k in noise_scheduler.timesteps:
                        # predict noise
                        noise_pred = ema_nets["invariant"](
                            sample=naction,
                            timestep=k,
                            global_cond=obs_cond.repeat(num_trial, 1)
                        )                   

                        # inverse diffusion step (remove noise)
                        naction = noise_scheduler.step(
                            model_output=noise_pred,
                            timestep=k,
                            sample=naction
                        ).prev_sample
                
                    # unnormalize action
                    naction = naction.detach().to('cpu').numpy()
                    # (B, pred_horizon, action_dim)
                    # naction = naction[0]
                    action_pred = unnormalize_data(naction, stats=domain18_stats['action'])

                    # only take action_horizon number of actions
                    start = obs_horizon - 1
                    end = start + action_horizon
                    action_all = action_pred[:, start:,:]
                    # (action_horizon, action_dim)


                if  len(last_obs_gt) > 0:
                    final_state_loss_trials = []
                    optimized_action = []
                    for action in action_all:
                        action = np.expand_dims(action, axis=0)
                        
                        naction = normalize_data(action, stats=dynamics_stats['action'])
                        action_before_gradient = copy.deepcopy(naction)
                        
                        ori_loss = 0
                        
                        action_tensor_ori = torch.tensor(naction).to(device, dtype=torch.float32)  
                        gradient_total = 25
                        save_start_lst = []

                        for gradient_step in range(gradient_total):
    
                            pred_imgs = last_obs_gt
                            
                            all_action_save_tensor = torch.tensor(normalize_data(all_action_save, stats=domain18_stats['action'])).to(device)
                            action_tensor = torch.tensor(naction).to(device, dtype=torch.float32)  
                            action_tensor.requires_grad = True
                            pred_imgs_tensor = torch.from_numpy(pred_imgs).to(device)
                            
                            for i in range(action_tensor.shape[1]):

                                all_action_save_tensor = torch.concatenate((all_action_save_tensor, action_tensor[:,i]), axis=0)
                        
                                denoiser_input_img = pred_imgs_tensor[:,-4:]
                                denoiser_input_action =torch.unsqueeze(all_action_save_tensor[-4:], 0)

                                if gradient_step == 0:
                                    pred_image, _, random_start = eval_sampler.sample(denoiser_input_img[:, :4], denoiser_input_action[:, :4], True)
                                    save_start_lst.append(random_start)
                                else:

                                    pred_image, _ = eval_sampler.sample(denoiser_input_img[:, :4], denoiser_input_action[:, :4], False, save_start_lst[i])
        
                                pred_image = torch.unsqueeze(pred_image, 1)
                                pred_imgs_tensor = torch.concatenate((pred_imgs_tensor, pred_image), axis=1)


                            last_pred_image = pred_imgs_tensor[0, -1]

                            unnormalized_xy = reward_predictor_unnormalized_xy(torch.unsqueeze(last_pred_image, 0))[0]
                            cossin_angle = reward_predictor_cossin_angle(torch.unsqueeze(last_pred_image, 0))[0]
                            
                            cossin_angle = cossin_angle/torch.sqrt(cossin_angle[0]*cossin_angle[0] + cossin_angle[1]*cossin_angle[1])
                            block_angle = torch.atan2(cossin_angle[1], cossin_angle[0]) % (2 * torch.pi)

                            block_pose = torch.stack((unnormalized_xy[0], unnormalized_xy[1], block_angle))

                            reward = estimate_reward_torch(block_pose, target_pose)
                            reward.backward()
                            #print(reward)

                            
                            if gradient_step!= gradient_total-1:
                                gradient = action_tensor.grad
                                action_tensor = action_tensor - 0.005 * gradient

                                #print(gradient)
                                naction = action_tensor.detach().cpu().numpy()
                            
                                #print(action_tensor)
                                
                            if gradient_step == 0:
                                start_registration_loss = reward.detach().cpu().numpy()
                            if gradient_step == gradient_total-1:
                                end_registration_loss = reward.detach().cpu().numpy()
                            
                        optimized_action.append(action_tensor)
                        final_state_loss_trials.append(end_registration_loss)
                    
                    
                    final_state_loss_trials = np.squeeze(np.array(final_state_loss_trials))
                    pick_index = np.argsort(final_state_loss_trials)[0]
                    
                    action_tensor_pick = optimized_action[pick_index]
                    action_pick = action_tensor_pick.detach().cpu().numpy()
                    action_pick = unnormalize_data(action_pick, stats=dynamics_stats['action'])   
                    action_pick = action_pick[0][:9]
                    

                    action_before_gradient = unnormalize_data(action_before_gradient, stats=dynamics_stats['action'])   
                    action_before_gradient = action_before_gradient[0][:9]
                    
                    if debug:
                        plt.figure()
                        plt.imshow(((torch.moveaxis(torch.squeeze(last_pred_image), 0, -1).detach().cpu().numpy())*255).astype('uint8'))
                        plt.savefig(f"pred_image_{step_idx+9}.png")

                else:

                    action_pick = action_all[0][:9]
                    action_before_gradient = action_pick
                    start_registration_loss='a'
                    end_registration_loss='a'
                        
                            
                last_obs_gt = []
                if debug:
                    env_copy = PushTImageEnv(domain_filename='push_t', resize_scale=resize_scale, pretrained=config["use_pretrained"])
                    env_copy.seed(env_seed)
                    env_copy.reset()
                    env_copy.block.angle = env.block.angle 
                    env_copy.agent.position = env.agent.position
                    env_copy.agent.velocity = env.agent.velocity
                    env_copy.block.position = env.block.position
                    env_copy.goal_pose = env.goal_pose
                    
                    for i in range(len(action_before_gradient)):

                        # stepping env
                        obs_copy, reward_copy, done_copy, _, info_copy = env_copy.step(action_before_gradient[i])

                    info_copy['block_pose'][2] = info_copy['block_pose'][2] % (2*np.pi)
                    gt_original_registration_loss = estimate_reward_torch(torch.tensor(info_copy['block_pose'], device=device, dtype=torch.float32), target_pose)
                    gt_original_registration_loss = gt_original_registration_loss.detach().cpu().numpy()

                for i in range(len(action_pick)):

                    # stepping env
                    obs, reward, done, _, info = env.step(action_pick[i])
                    # save observations
                    obs_deque.append(obs)
                    # and reward/vis
                    rewards.append(reward)
                    imgs.append(env.render(mode='rgb_array'))
                    
                    last_obs_gt.append(np.expand_dims(transform(imgs[-1]).numpy(), axis=0))
                    all_action_save.append(action_pick[i])
                    
                    # update progress bar
                    step_idx += 1
                    pbar.update(1)
                    pbar.set_postfix({"current": reward, "max": max(rewards)})
                    if step_idx > max_steps:
                        done = True
                    if done:
                        break
                if debug:
                    plt.figure()
                    plt.imshow(imgs[-1])
                    plt.savefig(f"gt_pred_image_{step_idx}.png")
                
                    info['block_pose'][2] = info['block_pose'][2] % (2*np.pi)
                    gt_optimized_registration_loss = estimate_reward_torch(torch.tensor(info['block_pose'], device=device, dtype=torch.float32), target_pose)
                    gt_optimized_registration_loss = gt_optimized_registration_loss.detach().cpu().numpy()
                    print("optimized action reward")
                    print(reward)
                    
                    print("original action reward")
                    print(reward_copy)
                    
                    
                    print("optimized registration loss")
                    print(end_registration_loss)
                    
                    print("original registration loss")
                    print(start_registration_loss)
                    
                    
                    print("gt optimized registration loss")
                    print(gt_optimized_registration_loss)
                    
                    print("gt original registration loss")
                    print(gt_original_registration_loss)

                    print("gt block pose")
                    print(info['block_pose'][0], info['block_pose'][1], np.cos(info['block_pose'][2]), np.sin(info['block_pose'][2]))
                    print("predicted block pose")
                    if step_idx>10:
                        print(block_pose[0], block_pose[1], cossin_angle[0], cossin_angle[1])

                    original_reward.append(reward_copy)
                    optimized_reward.append(reward)
                    
                    start_registration_loss_lst.append(start_registration_loss)
                    end_registration_loss_lst.append(end_registration_loss)
                
                
                    gt_start_registration_loss_lst.append(gt_original_registration_loss)
                    gt_end_registration_loss_lst.append(gt_optimized_registration_loss)
                
                last_obs_gt = np.array(last_obs_gt)
                last_obs_gt = np.transpose(last_obs_gt, (1, 0, 2, 3, 4))
                last_obs_gt = np.tile(last_obs_gt, (1, 1, 1, 1, 1))
                if debug:
                    vwrite(os.path.join(output_dir, "baseline_single_dp_on_domain_{}_test_{}.mp4".format(env_id, test_index)), imgs)

                

        
        env_seed += 1
        env_j_scores.append(max(rewards))
        # save the visualization of the first few demos
        #vwrite(os.path.join(output_dir, "baseline_single_dp_on_domain_{}_test_{}.mp4".format(env_id, test_index)), imgs)
        #vwrite(os.path.join(output_dir, "pred_baseline_single_dp_on_domain_{}_test_{}.mp4".format(env_id, test_index)), pred_video_imgs)
        if debug:
            print("optimized action reward")
            print(optimized_reward)
            
            print("original action reward")
            print(original_reward)
            
            
            print("optimized registration loss")
            print(end_registration_loss_lst)
            
            print("original registration loss")
            print(start_registration_loss_lst)
            
            print("gt optimized registration loss")
            print(gt_end_registration_loss_lst)
            
            print("gt original registration loss")
            print(gt_start_registration_loss_lst)

        
        np.save(f'corrected_sampling_based_testing_no_simulation_planning_receding_result_from_index_f{start_number_test}.npy', np.array(env_j_scores))

    print("Single DP on Domain #{} Avg Score: {}".format(env_id, np.mean(env_j_scores)))

############################ Save Result  ############################ 
    scores.append(env_j_scores)    

    np.save(f'corrected_sampling_based_testing_no_simulation_planning_receding_result_from_index_f{start_number_test}.npy', np.array(scores))


    print("Eval done!")
    return scores




