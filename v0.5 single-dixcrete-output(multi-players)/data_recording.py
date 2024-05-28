import sys
import time
import torch
from multi_thread import *
import game_wrapped
from brawlstar_utils import CONFIG
import inspect
import ctypes
import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


if not os.path.exists('pretrain_history'):
    os.mkdir('pretrain_history')
time_step = CONFIG['time_step']
store_size = 1000  # data store size(>= training batch)
store_count = 0
save_flag = False  # turned on when game starts


def async_raise(tid, exctype):
    """raises the exception, performs cleanup if needed"""
    tid = ctypes.c_long(tid)
    if not inspect.isclass(exctype):
        exctype = type(exctype)
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))
    if res == 0:
        raise ValueError("invalid thread id")
    elif res != 1:
        # """if it returns a number greater than one, you're in trouble,
        # and you should call it again with exc=NULL to revert the effect"""
        ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
        raise SystemError("PyThreadState_SetAsyncExc failed")


def stop_thread(thread):
    async_raise(thread.ident, SystemExit)


def get_obsv(time_step, watcher):
    [s1_tmp, s2_tmp, s3_tmp] = watcher.hidden_obsv
    s4_tmp = watcher.entity_obsv
    s5_tmp = env._mask_process(watcher.mask)
    info = watcher.info.copy()
    info_tmp = torch.tensor(info).unsqueeze(0)  # add an time-step dim
    for i in range(time_step-1):
        [s1_i, s2_i, s3_i] = watcher.hidden_obsv
        s4_i = watcher.entity_obsv
        s5_i = env._mask_process(watcher.mask)
        info = watcher.info.copy()
        info_i = torch.tensor(info).unsqueeze(0)
        s1_tmp = torch.cat([s1_tmp, s1_i], dim=0)
        s2_tmp = torch.cat([s2_tmp, s2_i], dim=0)
        s3_tmp = torch.cat([s3_tmp, s3_i], dim=0)
        s4_tmp = torch.cat([s4_tmp, s4_i], dim=0)
        s5_tmp = torch.cat([s5_tmp, s5_i], dim=0)
        info_tmp = torch.cat([info_tmp, info_i], dim=0)
    return s1_tmp, s2_tmp, s3_tmp, s4_tmp, s5_tmp, info_tmp  # time-step * feature dims


"""
(s, a, s_, r)其中r由s_计算得到
"""
store_s1 = torch.zeros((store_size, time_step, 128, 40, 80))   # s
store_s2 = torch.zeros((store_size, time_step, 256, 20, 40))
store_s3 = torch.zeros((store_size, time_step, 512, 10, 20))
store_s4 = torch.zeros((store_size, time_step, 6, 6))
store_s5 = torch.zeros((store_size, time_step, 6))
store_info = torch.zeros((store_size, time_step, 6))

store_s1_ = torch.zeros((store_size, time_step, 128, 40, 80))  # s_
store_s2_ = torch.zeros((store_size, time_step, 256, 20, 40))
store_s3_ = torch.zeros((store_size, time_step, 512, 10, 20))
store_s4_ = torch.zeros((store_size, time_step, 6, 6))
store_s5_ = torch.zeros((store_size, time_step, 6))
store_info_ = torch.zeros((store_size, time_step, 6))
store_action = torch.zeros((store_size, 1))                    # a

env = game_wrapped.BrawlEnv(None)
camera = CameraThread()
camera_assist = CameraAssistThread(camera)
keyboardthread = KeyboardThread(name='keyboardthread')
yolothread = YoloThread(camera, camera_assist, name='yolothread')
camera.start()
camera_assist.start()
time.sleep(1)
keyboardthread.start()
yolothread.start()
time.sleep(2)
s1, s2, s3, s4, s5, info = get_obsv(time_step, yolothread)
while store_count < store_size:
    # start = time.time()
    # while (time.time()-start) < 120:
    time.sleep(0.3)
    print(CONFIG['action_mix'][keyboardthread.get_command()], save_flag, '{}/{}'.format(store_count, store_size))
    print(yolothread.info)
    if 1 <= yolothread.info[0] < 100:
        save_flag = True
    elif 119 <= yolothread.info[0] <= 120:
        save_flag = False
    # TODO s -> s_ -> s的衔接问题
    if s1 is None:
        s1, s2, s3, s4, s5, info = get_obsv(time_step, yolothread)
    if save_flag:
        a = keyboardthread.get_command()
        s1_, s2_, s3_, s4_, s5_, info_ = get_obsv(time_step, yolothread)
        store_s1[store_count], store_s2[store_count] = s1, s2
        store_s3[store_count], store_s4[store_count] = s3, s4
        store_s1_[store_count], store_s2_[store_count] = s1_, s2_
        store_s3_[store_count], store_s4_[store_count] = s3_, s4_
        store_info[store_count], store_info_[store_count] = info, info_
        store_action[store_count] = a

        store_count += 1
        s1, s2, s3, s4, info = s1_, s2_, s3_, s4_, info_
    else:
        s1 = s2 = s3 = s4 = info = None
stop_thread(keyboardthread)
stop_thread(yolothread)
stop_thread(camera_assist)
stop_thread(camera)
print('---------saving data---------')
save = {'s1': store_s1, 's2': store_s2, 's3': store_s3, 's4': store_s4,
        's1_': store_s1_, 's2_': store_s2_, 's3_': store_s3_, 's4_': store_s4_,
        's5': store_s5, 's5_': store_s5_, 'info': store_info, 'info_': store_info_,
        'a': store_action}
torch.save(save, 'pretrain_history/data.pth')
print('store_s1/store_s1_:', store_s1.shape)
print('store_s2/store_s2_:', store_s2.shape)
print('store_s3/store_s3_:', store_s3.shape)
print('store_s4/store_s4_:', store_s4.shape)
print('store_s5/store_s5_:', store_s5.shape)
print('store_info/store_info_:', store_info.shape)
print('store_action:', store_action.shape)
