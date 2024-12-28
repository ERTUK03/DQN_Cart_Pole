from engine import train
from memory import ReplayMemory
from DQN import DQN
from utils import load_config
import gymnasium as gym
import torch

config = load_config()

BATCH_SIZE = config['BATCH_SIZE']
GAMMA = config['GAMMA']
EPS_START = config['EPS_START']
EPS_END = config['EPS_END']
EPS_DECAY = config['EPS_DECAY']
TAU = config['TAU']
LR = config['LR']
EPISODES = config['EPISODES']
MEMORY_SIZE = config['MEMORY_SIZE']
MODEL_NAME = config['MODEL_NAME']
TASK_NAME = config['TASK_NAME']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = gym.make(TASK_NAME, render_mode="rgb_array")

actions = env.action_space.n
inputs = env.observation_space.shape[0]

policy_net = DQN(inputs, actions)
target_net = DQN(inputs, actions)

criterion = torch.nn.SmoothL1Loss()
optimizer = torch.optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)

memory = ReplayMemory(MEMORY_SIZE)

train(policy_net, target_net, criterion, optimizer, env,
      memory, EPISODES, device, EPS_END, EPS_DECAY,
      EPS_START, BATCH_SIZE, GAMMA, TAU, MODEL_NAME)
