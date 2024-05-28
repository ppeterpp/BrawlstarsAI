import threading
import win32gui
import sys
sys.path.append('../yolov5-master(multi-players)')
sys.path.append('../yolov5-master(multi-players)/demo')
import realtime_detection
from pynput.keyboard import Controller, Key, Listener
from pynput import keyboard


class YoloThread(threading.Thread):
    """every daemon thread would be killed when main thread ends"""
    def __init__(self, camera, camera_assist, name=None, render=False, daemon=True):
        threading.Thread.__init__(self, name=name, daemon=daemon)
        self.render = render
        P = realtime_detection.get_hyper_parameters()
        P['weights'] = '../yolov5-master(multi-players)/runs/train/exp/weights/best.pt'
        self.gamestate_yolo = realtime_detection.Gamestate_yolo(P, camera, camera_assist)
        self.find_window = win32gui.FindWindow(0, 'MuMu模拟器12')
        # self.i = 0

    def run(self):
        while True:
            # print(threading.current_thread().name + ' test', i)
            win32gui.SetForegroundWindow(self.find_window)
            self.hidden_obsv, self.entity_obsv, self.mask, self.done, self.info = self.gamestate_yolo.frame(self.render)
            # self.i += 1
            # print(self.i)
            # time.sleep(1)


class KeyboardThread(threading.Thread):
    def __init__(self, name=None, daemon=True):
        threading.Thread.__init__(self, name=name, target=self.start_listen, daemon=daemon)
        self.w_press = self.a_press = self.s_press = self.d_press = self.o_press = self.p_press = False
        self.action_mix = ['w-o', 'w-p', 'w-.', 'a-o', 'a-p', 'a-.', 's-o', 's-p', 's-.',
                           'd-o', 'd-p', 'd-.', 'wa-o', 'wa-p', 'wa-.', 'wd-o', 'wd-p', 'wd-.',
                           'sa-o', 'sa-p', 'sa-.', 'sd-o', 'sd-p', 'sd-.', '.-o', '.-p', '.-.']

    def get_key_name(self, key):
        return key.char if isinstance(key, keyboard.KeyCode) else str(key)

    def on_press(self, key):
        # global w_press, a_press, s_press, d_press, o_press, p_press, curr_time
        key_name = self.get_key_name(key)
        if key_name in ['w', 'a', 's', 'd', 'o', 'p']:
            exec('self.%s_press = True' % key_name, globals(), locals())

    def on_release(self, key):
        # global w_press, a_press, s_press, d_press, o_press, p_press
        key_name = self.get_key_name(key)
        if key_name in ['w', 'a', 's', 'd', 'o', 'p']:
            exec('self.%s_press = False' % key_name, globals(), locals())

    def start_listen(self):
        with Listener(on_press=self.on_press, on_release=self.on_release) as listener:
            listener.join()

    def get_command(self):
        l1 = l2 = ''
        r = '.'
        if any([self.w_press, self.s_press]):
            l1 = 'w' if self.w_press else 's'
        if any([self.a_press, self.d_press]):
            l2 = 'a' if self.a_press else 'd'
        l = l1 + l2 if len(l1 + l2) > 0 else '.'
        if any([self.o_press, self.p_press]):
            r = 'o' if self.o_press else 'p'
        return self.action_mix.index(l + '-' + r)


class CameraThread(threading.Thread):
    def __init__(self, name=None, daemon=True):
        threading.Thread.__init__(self, name=name, daemon=daemon)
        import win32gui, win32ui, win32con
        from PIL import Image
        import numpy as np
        import cv2
        self.win32ui, self.win32con, self.Image, self.np, self.cv2 = win32ui, win32con, Image, np, cv2
        self.img_grab = np.asarray(self.grab_screen('MuMu模拟器12'))  # PIL.Image object
        self.img = cv2.cvtColor(self.img_grab, cv2.COLOR_BGR2RGB)  # transform to cv2 object

    def run(self):
        while True:
            self.img_grab = self.np.asarray(self.grab_screen('MuMu模拟器12'))
            self.img = self.cv2.cvtColor(self.img_grab, self.cv2.COLOR_BGR2RGB)

    def grab_screen(self, window_name):
        # 获取后台窗口的句柄，注意后台窗口不能最小化
        hWnd = win32gui.FindWindow(0, window_name)  # 窗口的类名可以用Visual Studio的SPY++工具获取
        # hWnd = win32gui.GetDesktopWindow()
        # 获取句柄窗口的大小信息
        left, top, right, bot = win32gui.GetWindowRect(hWnd)
        # width = right - left
        width = 2160
        # height = bot - top
        height = 1080+42
        # 返回句柄窗口的设备环境，覆盖整个窗口，包括非客户区，标题栏，菜单，边框
        hWndDC = win32gui.GetWindowDC(hWnd)
        # 创建设备描述表
        mfcDC = self.win32ui.CreateDCFromHandle(hWndDC)
        # 创建内存设备描述表
        saveDC = mfcDC.CreateCompatibleDC()
        # 创建位图对象准备保存图片
        saveBitMap = self.win32ui.CreateBitmap()
        # 为bitmap开辟存储空间
        saveBitMap.CreateCompatibleBitmap(mfcDC, width, height)
        # 将截图保存到saveBitMap中
        saveDC.SelectObject(saveBitMap)
        # 保存bitmap到内存设备描述表
        saveDC.BitBlt((0, -42), (width, height), mfcDC, (0, 0), self.win32con.SRCCOPY)

        bmpinfo = saveBitMap.GetInfo()
        bmpstr = saveBitMap.GetBitmapBits(True)
        # 生成图像
        im_PIL = self.Image.frombuffer('RGB', (bmpinfo['bmWidth'], bmpinfo['bmHeight']), bmpstr, 'raw', 'BGRX')
        # im_PIL = Image.frombuffer('RGB', (bmpinfo['bmWidth'], bmpinfo['bmHeight']), bmpstr)
        # im_PIL =Image.frombytes('RGB',(bmpinfo['bmWidth'],bmpinfo['bmHeight']),bmpstr)

        # box = (8, 31, 968, 511)  # 960*480
        # box = (8, 31, 1504, 779)  # 1496*748
        box = (0, 0, 2160, 1080)
        im2 = im_PIL.crop(box)
        # im2 = im_PIL
        # im2.save('./dd2d.jpg')

        mfcDC.DeleteDC()
        saveDC.DeleteDC()
        win32gui.ReleaseDC(hWnd, hWndDC)
        win32gui.DeleteObject(saveBitMap.GetHandle())
        return im2


class CameraAssistThread(threading.Thread):
    def __init__(self, camera, name=None, daemon=True):
        threading.Thread.__init__(self, name=name, daemon=daemon)
        import numpy as np
        import cv2
        from PIL import Image
        import pytesseract as tess
        self.np, self.cv2, self.Image, self.tess = np, cv2, Image, tess
        import re
        self.img = None
        self.camera = camera
        self.pattern = re.compile(r'\d{1,2}:\d{2}')

    def run(self):
        while True:
            self.img = self.camera.img
            time_ = self.extract_charge(self.img[35:80, 1034:1134], 'white')
            self.cv2.bitwise_not(time_, time_)
            text = self.tess.image_to_string(self.Image.fromarray(time_))
            try:
                self.time = max(self.pattern.findall(text), key=len)
            except ValueError:
                self.time = 'UNKNOWN'
            charge_ = self.cv2.cvtColor(self.extract_charge(self.img[773:902, 1507:1636]), self.cv2.COLOR_RGB2GRAY)
            self.charge = self.cv2.countNonZero(charge_)

    def extract_charge(self, src, mode='charge'):
        """extract areas that are almost white"""
        hsv = self.cv2.cvtColor(src, self.cv2.COLOR_BGR2HSV)  # BGR -> HSV
        # for the following two params consider CSDN blog:
        # 《Python Opencv cv2提取图像中某种特定颜色区域（例如黑字白纸背景下的红色公章提取），并将纯色背景透明化》
        if mode == 'white':
            low_hsv = self.np.array([0, 0, 221])  # hmin, smin, vmin
            high_hsv = self.np.array([180, 60, 255])  # hmax, smax, vmax
        else:
            low_hsv = self.np.array([19, 151, 192])  # hmin, smin, vmin
            high_hsv = self.np.array([23, 252, 255])  # hmax, smax, vmax
        mask = self.cv2.inRange(hsv, lowerb=low_hsv, upperb=high_hsv)  # 提取掩膜
        # 黑色背景转透明部分
        mask_contrary = mask.copy()
        mask_contrary[mask_contrary == 0] = 1
        mask_contrary[mask_contrary == 255] = 0  # 把黑色背景转白色
        # mask_bool = mask_contrary.astype(bool)
        mask_img = self.cv2.add(src, self.np.zeros(self.np.shape(src), dtype=self.np.uint8), mask=mask)
        return mask_img


if __name__ == '__main__':
    import time
    def xl():
        global yolothread
        print('current:', yolothread.i)

    yolothread = YoloThread('yolothread')
    yolothread.start()
    # for i in range(5):
    #     print(threading.current_thread().name + ' main', i)
    #     print(yolothread.name + ' is alive ', yolothread.is_alive())
    #     time.sleep(1)
    while True:
        input()
        xl()
