# from realtime_detection import Gamestate_yolo, get_hyper_parameters
import time
import pydirectinput
import win32gui
# import sys
# sys.path.append('../yolov5-master')
# sys.path.append('../yolov5-master/demo')
# import realtime_detection


class Gamestate:
    def __init__(self, render=False):
        # self.render = render
        # P = realtime_detection.get_hyper_parameters()
        # P['weights'] = '../yolov5-master/runs/train/exp/weights/best.pt'
        # self.gamestate_yolo = realtime_detection.Gamestate_yolo(P)
        self.action_left = ['w', 'a', 's', 'd', 'wa', 'wd', 'sa', 'sd', '.']
        self.action_right = ['o', 'p', '.']  # ['attack', 'super', 'none']
        self.action_mix = ['w-o', 'w-p', 'w-.', 'a-o', 'a-p', 'a-.', 's-o', 's-p', 's-.',
                           'd-o', 'd-p', 'd-.', 'wa-o', 'wa-p', 'wa-.', 'wd-o', 'wd-p', 'wd-.',
                           'sa-o', 'sa-p', 'sa-.', 'sd-o', 'sd-p', 'sd-.', '.-o', '.-p', '.-.']
        self.find_window = win32gui.FindWindow(0, 'MuMu模拟器12')
        self.action_l = ''
        print('pydirectinput.typewrite should work under:', '\n', 'Admin mode', '\n', 'without minimizing target window')

    def send_command_l(self, command: str, gap: float = 0):
        win32gui.SetForegroundWindow(self.find_window)
        last_command = list(self.action_l)
        if len(command) > 0:
            command = list(command)
            to_press = list(set(command)-set(last_command))
            to_release = list(set(last_command)-set(command))
            for i in to_press:
                pydirectinput.keyDown(i)
            for i in to_release:
                pydirectinput.keyUp(i)
            time.sleep(gap)
        else:
            for i in last_command:
                pydirectinput.keyUp(i)

    def send_command_r(self, command: str, gap: float):
        command = list(command)
        win32gui.SetForegroundWindow(self.find_window)
        for i in command:
            pydirectinput.keyDown(i)
        time.sleep(gap)
        for i in command:
            pydirectinput.keyUp(i)
        # pydirectinput.typewrite(command, gap)  # every character causes 'gap' secs pause

    def initialize(self):  # start another round
        time.sleep(4)  # Wait for game to end
        self.send_command_r('e', 0.0)
        time.sleep(2.5)
        self.send_command_r('e', 0.0)
        time.sleep(4)  # Wait for game to begin, need to further check the time gap
        # for i in range(30):
        #     self.keyboard.press_key(B)
        #     time.sleep(0.1)
        #     self.keyboard.release_key(B)
        # hidden_obsv, entity_obsv, done = self.gamestate_yolo.frame(self.render)
        # return hidden_obsv, entity_obsv

    def frame_step(self, action):
        # print(action)
        # assert action[0] in self.action_left
        # assert action[1] in self.action_right
        action = action.split('-')
        a0, a1 = self.action_left.index(action[0]), self.action_right.index(action[1])
        al = self.action_left[a0] if a0 <= 7 else ''
        ar = self.action_right[a1] if a1 <= 1 else ''
        self.send_command_l(al, 0.2)  # left
        self.send_command_r(ar, 0.0)  # right
        self.action_l = al
        # time.sleep(0.3)
        # hidden_obsv, entity_obsv, done = self.gamestate_yolo.frame(self.render)
        # print(done)
        # return hidden_obsv, entity_obsv, done


if __name__ == '__main__':
    gamestate = Gamestate()
    start = time.time()
    gamestate.initialize()
    print('%ss'%(int(time.time()-start)))
    # action_mix = []
    # for i in gamestate.action_left:
    #     for j in gamestate.action_right:
    #         action_mix.append(i+'-'+j)
    # print(action_mix)
