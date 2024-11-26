if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import tqdm
import numpy as np
import shutil
from equi_diffpo.workspace.base_workspace import BaseWorkspace
from equi_diffpo.policy.base_image_policy import BaseImagePolicy
from equi_diffpo.dataset.base_dataset import BaseImageDataset
from equi_diffpo.env_runner.base_image_runner import BaseImageRunner
from equi_diffpo.common.checkpoint_util import TopKCheckpointManager
from equi_diffpo.common.json_logger import JsonLogger
from equi_diffpo.common.pytorch_util import dict_apply, optimizer_to
from equi_diffpo.model.diffusion.ema_model import EMAModel
from equi_diffpo.model.common.lr_scheduler import get_scheduler

OmegaConf.register_new_resolver("eval", eval, replace=True)


# Generate perturbed observation using FGSM
# This version of PGD only use the final predicted action to compute MSE loss,
# which is used to perturb the input image(s)
def attack_policy_FGSM_final_pred_only(batch, policy, key, eps=8/255, device='cuda:0'):

    # sample trajectory from training set, and evaluate difference
    batch = copy.deepcopy(batch)
    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
    # the clean input image
    obs_dict = batch['obs']
    obs_dict[key] = obs_dict[key].clone().detach().cuda()
    # ground truth action
    gt_action = batch['action'].clone().detach().cuda()

    # actovate img gradient for attack
    obs_dict[key].requires_grad = True

    # model inference
    result = policy.predict_action(obs_dict)
    pred_action = result['action_pred']
    
    criterion = torch.nn.MSELoss()
    loss = criterion(pred_action, gt_action)

    grad = torch.autograd.grad(loss, obs_dict[key], retain_graph=False, create_graph=False)[0]

    perturbed_obs = obs_dict[key] + eps * grad.sign()
    delta = torch.clamp(
        perturbed_obs - obs_dict[key],
        min = -eps, max = eps
    )
    # clip to 0-1
    perturbed_obs = torch.clamp(obs_dict[key] + delta, min=0, max=1)

    return perturbed_obs


# Generate perturbed observation using FGSM
# This version of PGD use the final step MSE and randomly sampled intermediate step MSE,
# in order to avoid the vanishing gradient problem
def attack_policy_FGSM_all_steps(batch, policy, key, eps=8/255, attack_steps=3, weight=0.1, device='cuda:0'):

    # sample trajectory from training set, and evaluate difference
    batch = copy.deepcopy(batch)
    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
    # the clean input image
    obs_dict = batch['obs']
    obs_dict[key] = obs_dict[key].clone().detach().cuda()
    # ground truth action
    gt_action = batch['action'].clone().detach().cuda()

    # actovate img gradient for attack
    obs_dict[key].requires_grad = True

    # model inference
    result = policy.predict_action_for_attack(obs_dict, attack_steps)
    
    criterion = torch.nn.MSELoss()
    
    # Final step MSE
    pred_action = result['action_pred']
    loss = criterion(pred_action, gt_action)

    # Intermediate steps MSE
    intermediate_preds = result['intermediate_preds']
    for pred in intermediate_preds:
        loss += weight * criterion(pred, gt_action)

    grad = torch.autograd.grad(loss, obs_dict[key], retain_graph=False, create_graph=False)[0]

    perturbed_obs = obs_dict[key] + eps * grad.sign()
    delta = torch.clamp(
        perturbed_obs - obs_dict[key],
        min = -eps, max = eps
    )
    # clip to 0-1
    perturbed_obs = torch.clamp(obs_dict[key] + delta, min=0, max=1)

    return perturbed_obs


# Generate perturbed observation using PGD
# This version of PGD only use the final predicted action to compute MSE loss,
# which is used to perturb the input image(s)
def attack_policy_PGD_final_pred_only(batch, policy, steps, key, eps=8/255, device='cuda:0'):
    
    # sample trajectory from training set, and evaluate difference
    batch = copy.deepcopy(batch)
    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
    # the clean input image
    obs_dict = batch['obs']
    obs_dict[key] = obs_dict[key].clone().detach().cuda()
    # ground truth action
    gt_action = batch['action'].clone().detach().cuda()

    # random start
    obs_dict[key] = obs_dict[key] + \
        torch.empty_like(obs_dict[key]).uniform_(-eps, eps)
    obs_dict[key] = torch.clamp(obs_dict[key], min=0, max=1).detach()

    criterion = torch.nn.MSELoss()

    # attack loop
    for i in range(steps):

        obs_dict[key].requires_grad = True

        # model inference
        result = policy.predict_action(obs_dict)
        pred_action = result['action_pred']
        
        loss = criterion(pred_action, gt_action)

        grad = torch.autograd.grad(loss, obs_dict[key], retain_graph=False, create_graph=False)[0]

        perturbed_obs = obs_dict[key] + eps * grad.sign()
        delta = torch.clamp(
            perturbed_obs - obs_dict[key],
            min = -eps, max = eps
        )
        # clip to 0-1
        perturbed_obs = torch.clamp(obs_dict[key] + delta, min=0, max=1)

        obs_dict[key] = perturbed_obs
    
    return obs_dict[key]


# Generate perturbed observation using PGD
# This version of PGD use the final step MSE and randomly sampled intermediate step MSE,
# in order to avoid the vanishing gradient problem
def attack_policy_PGD_all_steps(batch, policy, steps, key, eps=8/255, attack_steps=3, weight=0.1, device='cuda:0'):

    # sample trajectory from training set, and evaluate difference
    batch = copy.deepcopy(batch)
    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
    # the clean input image
    obs_dict = batch['obs']
    obs_dict[key] = obs_dict[key].clone().detach().cuda()
    # ground truth action
    gt_action = batch['action'].clone().detach().cuda()

    # random start
    obs_dict[key] = obs_dict[key] + \
        torch.empty_like(obs_dict[key]).uniform_(-eps, eps)
    obs_dict[key] = torch.clamp(obs_dict[key], min=0, max=1).detach()

    criterion = torch.nn.MSELoss()

    # attack loop
    for i in range(steps):

        obs_dict[key].requires_grad = True

        # model inference
        result = policy.predict_action_for_attack(obs_dict, attack_steps)

        # Final step MSE
        pred_action = result['action_pred']
        loss = criterion(pred_action, gt_action)

        # Intermediate steps MSE
        intermediate_preds = result['intermediate_preds']
        for pred in intermediate_preds:
            loss += weight * criterion(pred, gt_action)

        grad = torch.autograd.grad(loss, obs_dict[key], retain_graph=False, create_graph=False)[0]

        perturbed_obs = obs_dict[key] + eps * grad.sign()
        delta = torch.clamp(
            perturbed_obs - obs_dict[key],
            min = -eps, max = eps
        )
        # clip to 0-1
        perturbed_obs = torch.clamp(obs_dict[key] + delta, min=0, max=1)

        obs_dict[key] = perturbed_obs
    
    return obs_dict[key]