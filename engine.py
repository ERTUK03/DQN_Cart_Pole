from plot_durations import plot_durations
from itertools import count
import torch
import math
import random

def train(policy_net, target_net, criterion, optimizer,
          env, memory, EPISODES, device, EPS_END, EPS_DECAY,
          EPS_START, BATCH_SIZE, GAMMA, TAU, model_name):
  steps_done = 0
  action = None
  episode_durations = []

  for i_episode in range(EPISODES):
      state, info = env.reset()
      state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
      for t in count():
          sample = random.random()
          eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
          steps_done += 1
          if sample > eps_threshold:
              with torch.no_grad():
                  action = policy_net(state).max(1).indices.view(1, 1)
          else:
              action = torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

          observation, reward, terminated, truncated, _ = env.step(action.item())
          done = terminated or truncated

          if terminated:
              next_state = None
          else:
              next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

          reward = torch.tensor([reward], device=device)

          memory.push(state, action, next_state, reward)

          state = next_state

          if len(memory) >= BATCH_SIZE:

            transitions = memory.sample(BATCH_SIZE)

            states, actions, next_states, rewards = zip(*transitions)

            non_final_mask = torch.tensor(tuple([True if x is not None else False for x in next_states]), device=device, dtype=torch.bool)

            non_final_next_states = torch.cat([s for s in next_states if s is not None])

            state_batch = torch.cat(states)
            action_batch = torch.cat(actions)
            reward_batch = torch.cat(rewards)

            state_actions = policy_net(state_batch).gather(1, action_batch)

            next_state_values = torch.zeros(BATCH_SIZE, device=device)
            with torch.no_grad():
                next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values

            expected_state_actions = (next_state_values * GAMMA) + reward_batch

            loss = criterion(state_actions, expected_state_actions.unsqueeze(1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
            optimizer.step()

          target_net_state_dict = target_net.state_dict()
          policy_net_state_dict = policy_net.state_dict()
          for key in policy_net_state_dict:
              target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
          target_net.load_state_dict(target_net_state_dict)

          if done:
              episode_durations.append(t + 1)
              plot_durations(episode_durations)
              break

  plot_durations(episode_durations, show_result=True)
  plt.ioff()
  plt.show()

  torch.save({
              'model_state_dict': policy_net.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              }, f'{model_name}.pth')
