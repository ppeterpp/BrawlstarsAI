import brawlstar_game
from brawlstar_utils import CONFIG
import gym
# import numpy as np
import torch
import math
import time


class BrawlEnv(gym.Env):
    def __init__(self, watcher, time_step=6, render=True):
        self.max_score = 20000
        self.game_state = brawlstar_game.Gamestate(render)
        # self.action_space = gym.spaces.Discrete(3)
        self.time_step = time_step  # RNN单次接收几帧数据输入
        self.peak_score = 0
        self.watcher = watcher
        self.mask_false = torch.zeros(1)
        mask_true = torch.ones(1)
        self.mask = [mask_true for i in range(CONFIG['max_player_num'])]

    def step(self, action: str):
        """每次step()将与游戏交互time_step帧，某帧done=True则整体为True"""
        self.HP = []
        self.charge = []
        self.game_state.frame_step(action)
        obsv, entity_obsv, mask, done_, info = \
            self.watcher.hidden_obsv, self.watcher.entity_obsv, self.watcher.mask, self.watcher.done, self.watcher.info
        print('time:', info[0])
        print('action: ', action)
        print('done:', done_)
        self.HP.append(info[1])
        self.charge.append(info[2])
        mask = self._mask_process(mask)
        reward = self._info_process(info, action)
        # assert type(obsv) == '<class \'list\'>'
        for i in range(self.time_step - 1):
            self.game_state.frame_step(action.split('-')[0] + '-.')
            observation, entity_observation, mask_, done, info = \
                self.watcher.hidden_obsv, self.watcher.entity_obsv, self.watcher.mask, self.watcher.done, self.watcher.info
            print('time:', info[0])
            print('done:', done)
            self.last_supercharged = info[3]
            self.HP.append(info[1])
            self.charge.append(info[2])
            r = self._info_process(info, action)
            for j in range(len(obsv)):
                obsv[j] = torch.cat((obsv[j], observation[j]), dim=0)
            mask = torch.cat([mask, self._mask_process(mask_)], dim=0)
            entity_obsv = torch.cat((entity_obsv, entity_observation), dim=0)
            if done:
                done_ = done
                # self.peak_score = self.game_state.peak_score  # 每局游戏结束记录最高分
            reward += r
        self._history_info_process()
        # if self.game_state.score >= 3000: r += 6
        # if self.game_state.score >= 6000: r += 6
        # if self.game_state.score >= self.max_score:
        #     # print('______win:%s_____' % self.game_state.score)
        #     self.reset()
        #     r = 12
        #     done_ = True
        obsv.append(entity_obsv)
        obsv.append(mask)
        return obsv, reward/self.time_step, done_, {}

    def reset(self):
        self.HP = []
        self.charge = []
        action = '.-.'  # left: still, right: none
        self.game_state.initialize()
        obsv, entity_observation = self.watcher.hidden_obsv, self.watcher.entity_obsv
        obsv, entity_obsv, mask, info = \
            self.watcher.hidden_obsv, self.watcher.entity_obsv, self.watcher.mask, self.watcher.info
        self.HP.append(info[1])
        self.charge.append(info[2])
        mask = self._mask_process(mask)
        mask_ = mask
        # r_ = self._info_process(info, action, True)
        observation = obsv
        entity_observation = entity_obsv
        # assert type(obsv) == '<class \'list\'>'
        for i in range(self.time_step - 1):
            # observation, entity_observation, d_ = self.game_state.frame_step(action)
            self.last_supercharged = info[3]
            self.HP.append(info[1])
            self.charge.append(info[2])
            for j in range(len(obsv)):
                obsv[j] = torch.cat((obsv[j], observation[j]), dim=0)
            mask = torch.cat([mask, mask_], dim=0)
            entity_obsv = torch.cat((entity_obsv, entity_observation), dim=0)
        self._history_info_process()
        obsv.append(entity_obsv)
        obsv.append(mask)
        # for i in range(6):
        #     self.game_state.frame_step(action)
        return obsv

    def _info_process(self, info: list, action: str, reset: bool = False):
        """
        entity_obsv includes:
        nums: time, HP, enemy HP,
        bool: super-charged flag, kill flag, killed flag,
        coordinates: position(x, y), enemy position(x,y)
        """
        if not reset:
            action = action.split('-')
            [time_, hp, charge, supercharged, kill, killed] = info
            # discount factor = f(time)
            # print('x:', x, 'y:', y)
            # print('ex:', ex, 'ey:', ey)
            # distance = math.sqrt((x-ex)**2 + (y-ey)**2)
            # print('distance:', distance)
            delta_hp = (hp - self.last_HP) / CONFIG['HP_max']
            delta_charge = charge - self.last_charge / 1000
            attack_usage = 0
            if action[1] == 'o':
                attack_usage = 1 if delta_charge > 0 else -0.5
            super_usage = 0
            if action[1] == 'p':
                super_usage = -0.2  # -1/5
                if self.last_supercharged:
                    super_usage = 1 if delta_charge > 0 else 0  # 5/5
            # reward = 1.42-distance + supercharged + 15*kill - 8*killed + delta_hp + 2*attack_usage + super_usage
            reward = 1*kill - 0.67*killed + 0.17*delta_hp + 0.33*attack_usage + 0.33*super_usage  # (6-4+1+2+1)
        else:
            reward = 0
        print('reward:', reward, time.asctime(time.localtime(time.time())))
        return reward
        # if use super in vain, punish
        # check enemy-HP variation if attack
        # need to store info of last round to achieve the above
        # need to eliminate the highest&lowest HP in the time-series to avoid abnormal value
        # TODO: how to design discount factor using time info
        # is reward needed to be projected to 0~1?

    def _history_info_process(self):
        if len(self.HP) >= 6:
            self.HP.sort()
            self.HP.pop(0); self.HP.pop()
        if len(self.charge) >= 6:
            self.charge.sort()
            self.charge.pop(0); self.charge.pop()
        # try:
        self.last_HP = sum(self.HP) / len(self.HP)
        self.last_charge = sum(self.charge) / len(self.charge)
        # except ZeroDivisionError:
        #     self.last_HP = 5700
        #     self.last_enemy_HP = 10000

    def _mask_process(self, mask):
        mask_bool = self.mask.copy()
        for i in range(CONFIG['max_player_num']-mask):
            mask_bool.pop(0)
            mask_bool.append(self.mask_false)
        return torch.cat(mask_bool).unsqueeze(0).bool()  # add time-step dim in the front

    def render(self, mode='human'):
        pass
