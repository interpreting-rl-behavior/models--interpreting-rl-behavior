import numpy as np
import pandas as pd
from collections import deque
from torch.utils.tensorboard import SummaryWriter
import time

class Logger(object):

    def __init__(self, n_envs, logdir):
        self.start_time = time.time()
        self.n_envs = n_envs
        self.logdir = logdir

        self.episode_rewards = []
        self.episode_values = []
        for _ in range(n_envs):
            self.episode_rewards.append([])
            self.episode_values.append([])
        self.episode_len_buffer = deque(maxlen = 40)
        self.episode_reward_buffer = deque(maxlen = 40)

        self.policy_loss_buffer = deque(maxlen = 40)
        self.episode_value_buffer = deque(maxlen = 40)
        self.value_loss_buffer = deque(maxlen = 40)
        self.entropy_loss_buffer = deque(maxlen = 40)

        #valid
        self.episode_rewards_v = []
        for _ in range(n_envs):
            self.episode_rewards_v.append([])
        self.episode_len_buffer_v = deque(maxlen = 40)
        self.episode_reward_buffer_v = deque(maxlen = 40)

        self.log = pd.DataFrame(columns = ['timesteps', 'wall_time', 'num_episodes', 'num_episodes_val',
                               'max_episode_rewards', 'mean_episode_rewards','min_episode_rewards',
                               'max_episode_len', 'mean_episode_len', 'min_episode_len',
                               'mean_episode_value', 'mean_policy_loss', 'mean_value_loss', 'mean_entropy_loss',
                               'val_max_episode_rewards', 'val_mean_episode_rewards', 'val_min_episode_rewards',
                               'val_max_episode_len', 'val_mean_episode_len', 'val_min_episode_len'])
        self.writer = SummaryWriter(logdir)
        self.timesteps = 0
        self.num_episodes = 0
        self.num_episodes_val = 0

    def feed(self, rew_batch, done_batch, value_batch, losses_summary, rew_batch_v=None, done_batch_v=None):
        steps = rew_batch.shape[0]
        rew_batch = rew_batch.T
        done_batch = done_batch.T
        value_batch = value_batch.T

        valid = rew_batch_v is not None and done_batch_v is not None
        if valid:
            rew_batch_v = rew_batch_v.T
            done_batch_v = done_batch_v.T

        for i in range(self.n_envs):
            for j in range(steps):
                self.episode_rewards[i].append(rew_batch[i][j])
                self.episode_values[i].append(value_batch[i][j])
                if valid:
                    self.episode_rewards_v[i].append(rew_batch_v[i][j])
                if done_batch[i][j]:
                    self.episode_len_buffer.append(len(self.episode_rewards[i]))
                    self.episode_reward_buffer.append(np.sum(self.episode_rewards[i]))
                    self.episode_value_buffer.append(np.mean(self.episode_values[i]))
                    self.episode_rewards[i] = []
                    self.episode_values[i]  = []
                    self.num_episodes += 1
                if valid:
                    if done_batch_v[i][j]:
                        self.episode_len_buffer_v.append(len(self.episode_rewards_v[i]))
                        self.episode_reward_buffer_v.append(np.sum(self.episode_rewards_v[i]))
                        self.episode_rewards_v[i] = []
                        self.num_episodes_val += 1

        self.policy_loss_buffer.append(losses_summary['Loss/pi'])
        self.value_loss_buffer.append(losses_summary['Loss/v'])
        self.entropy_loss_buffer.append(losses_summary['Loss/entropy'])

        self.timesteps += (self.n_envs * steps)

    def write_summary(self, summary):
        for key, value in summary.items():
            self.writer.add_scalar(key, value, self.timesteps)

    def dump(self):
        wall_time = time.time() - self.start_time
        if self.num_episodes > 0:
            episode_statistics = self._get_episode_statistics()
            episode_statistics_list = list(episode_statistics.values())
            for key, value in episode_statistics.items():
                self.writer.add_scalar(key, value, self.timesteps)
        else:
            # Set None for all columns except for the 4 listed below
            episode_statistics_list = [None] * (len(self.log.columns) - 4)
        log = [self.timesteps] + [wall_time] + [self.num_episodes] + [self.num_episodes_val] + episode_statistics_list
        self.log.loc[len(self.log)] = log

        # TODO: logger to append, not write!
        with open(self.logdir + '/log.csv', 'w') as f:
            self.log.to_csv(f, index = False)
        print(self.log.loc[len(self.log)-1])

    def _get_episode_statistics(self):
        episode_statistics = {}
        episode_statistics['Rewards/max_episodes']  = self._safe_array_op(self.episode_reward_buffer, np.max)
        episode_statistics['Rewards/mean_episodes'] = self._safe_array_op(self.episode_reward_buffer, np.mean)
        episode_statistics['Rewards/min_episodes']  = self._safe_array_op(self.episode_reward_buffer, np.min)
        episode_statistics['Len/max_episodes']  = self._safe_array_op(self.episode_len_buffer, np.max)
        episode_statistics['Len/mean_episodes'] = self._safe_array_op(self.episode_len_buffer, np.mean)
        episode_statistics['Len/min_episodes']  = self._safe_array_op(self.episode_len_buffer, np.min)

        episode_statistics['Values/mean_episodes'] = self._safe_array_op(self.episode_value_buffer, np.mean)
        episode_statistics['Loss/value'] = self._safe_array_op(self.value_loss_buffer, np.mean)
        episode_statistics['Loss/policy'] = self._safe_array_op(self.policy_loss_buffer, np.mean)
        episode_statistics['Loss/entropy'] = self._safe_array_op(self.entropy_loss_buffer, np.mean)

        # valid
        episode_statistics['[Valid] Rewards/max_episodes'] = self._safe_array_op(self.episode_reward_buffer_v, np.max)
        episode_statistics['[Valid] Rewards/mean_episodes'] = self._safe_array_op(self.episode_reward_buffer_v, np.mean)
        episode_statistics['[Valid] Rewards/min_episodes'] = self._safe_array_op(self.episode_reward_buffer_v, np.min)
        episode_statistics['[Valid] Len/max_episodes'] = self._safe_array_op(self.episode_len_buffer_v, np.max)
        episode_statistics['[Valid] Len/mean_episodes'] = self._safe_array_op(self.episode_len_buffer_v, np.mean)
        episode_statistics['[Valid] Len/min_episodes'] = self._safe_array_op(self.episode_len_buffer_v, np.min)
        return episode_statistics

    @staticmethod
    def _safe_array_op(arr, op):
        try:
            return op(arr)
        except ValueError:
            return np.nan