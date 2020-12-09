import math
import gym
from enum import IntEnum
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding
import gym_minigrid
import argparse
import time
import datetime
import torch
import torch_ac
import tensorboardX
import sys
sys.path.insert(0,'/root/server/Semantic-Loss/complex_constraints/curriculum learning/automatic-curriculum/')
import utils
import auto_curri as ac
from model import ACModel
import numpy as np
def preprocess_images(images, device=None):
    # Bug of Pytorch: very slow if not first converted to numpy array
    images = np.array(images)
    return torch.tensor(images, device=device, dtype=torch.float)

def get_obss_preprocessor():
    obs_space = {"image": [7,7,3]}

    def preprocess_obss(obss, device=None):
        return torch_ac.DictList({
            "image": preprocess_images([obs["image"] for obs in obss], device=device)
        })

    return obs_space, preprocess_obss
class parser:
    def __init__(self):
        self.env=None#'MiniGrid-Unlock-v0'#'MiniGrid-BlockedUnlockPickup-v0'
        self.curriculum='empty'
        self.model=None
        self.seed=1
        self.log_interval=1
        self.save_interval=10
        self.procs=1
        self.frames=10**7
        self.epochs=4
        self.batch_size=256
        self.frames_per_proc=128
        self.discount=0.99
        self.lr=0.001
        self.gae_lambda=0.95
        self.entropy_coef=0.01
        self.value_loss_coef=0.5
        self.max_grad_norm=0.5
        self.adam_eps=1e-8
        self.clip_eps=0.2
        self.lpe="Linreg"
        self.lpe_alpha=0.1
        self.lpe_K=10
        self.acp="MR"
        self.acp_MR_K=10
        self.acp_MR_power=6
        self.acp_MR_pot_prop=0.5
        self.acp_MR_att_pred=0.2
        self.acp_MR_att_succ=0.05
        self.a2d="Prop"
        self.a2d_eps=0.1
        self.a2d_tau=4e-4
args = parser()
args2 = parser()
args2.curriculum='fetch'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import torch_ac


# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def init_params(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)

device = torch.device("cuda")
class ACModel(nn.Module, torch_ac.ACModel):
    def __init__(self, obs_space, action_space):
        super().__init__()

        # Define image embedding
        self.image_embedding_size = 64
        self.image_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(2, 2)),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(2, 2)),
            nn.ReLU(),
        )

        # Define actor's model
        self.actor = nn.Sequential(
            nn.Linear(16, 64),
            nn.Tanh(),
            nn.Linear(64, action_space.n)
        )

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(16, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Initialize parameters correctly
        self.apply(init_params)

    def forward(self, obs):
        x = obs.image.transpose(1, 3).transpose(2, 3).to(device)
        x = self.image_conv(x)
        x = x.reshape(x.shape[0], -1)
        
        embedding = x
        x = self.actor(embedding)
        dist = Categorical(logits=F.log_softmax(x, dim=1))
        
        x = self.critic(embedding)
        value = x.squeeze(1)

        return dist, value
config_hash = utils.save_config_in_table(args, "config_rl")

# Set run dir

name = args.env or args.curriculum
date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
default_model_name = f"{name}_seed{args.seed}_{config_hash}_{date}"

model_name = args.model or default_model_name
model_dir = utils.get_model_dir(model_name)

# Load loggers and Tensorboard writer

txt_logger = utils.get_txt_logger(model_dir)
csv_file, csv_logger = utils.get_csv_logger(model_dir)
tb_writer = tensorboardX.SummaryWriter(model_dir)

# Log command and all script arguments
txt_logger.info("{}\n".format(" ".join(sys.argv)))
txt_logger.info("{}\n".format(args))
txt_logger.info("Config hash: {}\n".format(config_hash))
# Set seed for all randomness sources

utils.seed(args.seed)

# Set device

device = torch.device("cuda")
txt_logger.info(f"Device: {device}\n")

# Load environments

if args.env is not None:
    envs=[gym_minigrid.envs.EmptyEnv5x5(),gym_minigrid.envs.EmptyRandomEnv5x5(),gym_minigrid.envs.EmptyEnv6x6(),gym_minigrid.envs.EmptyRandomEnv6x6(),gym_minigrid.envs.EmptyEnv(),gym_minigrid.envs.EmptyEnv16x16()]

    #envs=[DoorKeyEnv5x5(),DoorKeyEnv6x6(),DoorKeyEnv8x8(),DoorKeyEnv10x10(),DoorKeyEnv12x12(),DoorKeyEnv14x14(),DoorKeyEnv16x16()]


elif args.curriculum is not None:
    # Load curriculum
    G, env_ids, init_min_returns, init_max_returns = utils.get_curriculum(args.curriculum)

    # Make distribution computer
    compute_dist = ac.make_dist_computer(
                        len(env_ids), args.lpe, args.lpe_alpha, args.lpe_K,
                        args.acp, G, init_min_returns, init_max_returns, args.acp_MR_K, args.acp_MR_power,
                        args.acp_MR_pot_prop, args.acp_MR_att_pred, args.acp_MR_att_succ,
                        args.a2d, args.a2d_eps, args.a2d_tau)

    # Make polymorph environments
    penv_head = ac.PolyEnvHead(args.procs, len(env_ids), compute_dist)
    envs = []
    for i in range(args.procs):
        seed = args.seed + 10000 * i
        envs.append(ac.PolyEnv(utils.make_envs_from_curriculum(env_ids, seed), penv_head.remotes[i], seed))
txt_logger.info("Environments loaded\n")

# Load training status

try:
    status = utils.get_status(model_dir)
except OSError:
    status = {"num_frames": 0, "update": 0}
txt_logger.info("Training status loaded\n")

# Load observations preprocessor

obs_space, preprocess_obss = utils.get_obss_preprocessor(envs[0].observation_space)
txt_logger.info("Observations preprocessor loaded")

# Load model

acmodel = ACModel(obs_space, envs[0].action_space)
if "model_state" in status:
    acmodel.load_state_dict(status["model_state"])
#acmodel.to(device)
txt_logger.info("Model loaded\n")
txt_logger.info("{}\n".format(acmodel))
# Load algo

algo = torch_ac.PPOAlgo(envs, acmodel, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                        args.entropy_coef, args.value_loss_coef, args.max_grad_norm, 1,
                        args.adam_eps, args.clip_eps, args.epochs, args.batch_size, preprocess_obss,
                        utils.reshape_reward)
if "optimizer_state" in status:
    algo.optimizer.load_state_dict(status["optimizer_state"])
txt_logger.info("Optimizer loaded\n")

# Train model

num_frames = status["num_frames"]
update = status["update"]
start_time = time.time()


envs=[gym_minigrid.envs.EmptyEnv(),gym_minigrid.envs.EmptyEnv16x16()]
env=gym_minigrid.envs.EmptyEnv()
obs_space, preprocess_obss=get_obss_preprocessor()
visualise=True
steps=[]
rewards=[]
avgrewards=[]

for i in range(0,1000000):
    print(i)
    exps, logs1 = algo.collect_experiences()
    logs2 = algo.update_parameters(exps)
    logs = {**logs1, **logs2}
    if args.curriculum is not None:
        penv_head.update_dist()
 #   print(i,logs['return_per_episode'],penv_head.dist)
    #print(i,logs_2['return_per_episode'],penv_head2.dist)
    if i%10==0:
        observations = envs[-1].reset().copy()
        rewardcnt=0
        for t in range(10000):

            action =algo.acmodel(preprocess_obss([observations]))[0].sample()

            # Step (this will plot if visualise is True)
            observations,reward, done, _ = envs[-1].step(action)
            rewardcnt+=reward
            #print(t,done,action,reward)
            #print("[{}] reward={} done={} \n".format(t, reward, done))

            if reward:
                #rewarding_frame = observations['image'].copy()
                #rewarding_frame[:40] *= np.array([0, 1, 0])
                print("[{}] Got a rewaaaard! {:.1f}".format(t, reward))
            elif done:
                print("[{}] Finished with nothing... Reset".format(t))
                print(rewardcnt)
                break
        steps.append(t)
        rewards.append(rewardcnt)
        np.save('rewardsFetch.npy',rewards)
    #https://github.com/Feryal/craft-env/blob/master/envs[-1].py        