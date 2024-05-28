CONFIG = {}

CONFIG['network_c_out'] = 8  # convolution output channel
CONFIG['network_l_out'] = 128  # linear encoder output dimension
CONFIG['network_input_size'] = CONFIG['network_c_out']*(40*80 + 20*40 + 10*20) + CONFIG['network_l_out']  # lstm input-size
CONFIG['action_dim_l'] = 9  # left hand options
CONFIG['action_dim_r'] = 3  # right hand options
CONFIG['action_dim_mix'] = CONFIG['action_dim_l'] * CONFIG['action_dim_r']
CONFIG['time_step'] = 2  # the net takes 'time_step' frames of data each time
CONFIG['device'] = 'cuda'
CONFIG['render'] = False

CONFIG['store_size'] = 150  # buffer size，即经验池能存多少条数据
CONFIG['decline'] = 0.9  # 衰减系数
CONFIG['learn_time'] = 0  # 学习次数
CONFIG['update_time'] = 20  # 隔多少个学习次数更新一次网络
CONFIG['gama'] = 0.9  # Q值计算时的折扣因子
CONFIG['b_size'] = 128  # batch size，一次从经验池中抽取多少数据进行学习

CONFIG['action_left'] = ['w', 'a', 's', 'd', 'wa', 'wd', 'as', 'sd', '.']
CONFIG['action_right'] = ['o', 'p', '.']  # ['attack', 'super', 'none']
CONFIG['action_mix'] = ['w-o', 'w-p', 'w-.', 'a-o', 'a-p', 'a-.', 's-o', 's-p', 's-.',
                        'd-o', 'd-p', 'd-.', 'wa-o', 'wa-p', 'wa-.', 'wd-o', 'wd-p', 'wd-.',
                        'as-o', 'as-p', 'as-.', 'sd-o', 'sd-p', 'sd-.', '.-o', '.-p', '.-.']

CONFIG['time_max'] = 120
CONFIG['HP_max'] = 3500
CONFIG['enemy_HP_max'] = 10000
CONFIG['max_player_num'] = 6
