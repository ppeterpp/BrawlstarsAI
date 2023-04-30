# Brawlstars-AI

### Brawlstars official:  https://bs.qq.com/index.html

### Requirements:

```
cd v0.5.1 single-dixcrete-output(multi-players)
pip install -r requirements.txt
cd yolov5-master(multi-players)
pip install -r requirements.txt
```

### Before Starting

- (Optional)[easyocr](https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/craft_mlt_25k.zip)解压.pth权重到C:\Users\Administrator\.EasyOCR\model

- (Optional)[yolov5](https://ultralytics.com/assets/Arial.ttf)字体文件放到C:\Users\Administrator\AppData\Roaming\Ultralytics
- 将根目录Tesseract-OCR添加到环境变量

### Yolo Training

如果想为自己的游戏训练yolo，先在“yolo数据集制作”下准备数据集（附README，下同）；再在“yolov5-master(multi-players)”下训练（参考commands.txt）

### (Optional) Behavior Cloning Pretrain

```
cd v0.5 single-dixcrete-output(multi-players)
python data_recording.py  # -> pretrain_history
python pretrain.py  # -> history
```

在data_recording.py开始游戏后，自己操作智能体。也可跳过预训练。

### Reinforcement Learning

```
python DQN_brawlstar.py
```
