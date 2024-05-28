### V0.0：初始版本

- doodle_jump_game、DQN_doodlejump
  本程序来自doodle jump的DQN实现
- brawlstar_game
  通过win32gui定位模拟器窗口，使用pyautogui模拟键盘输出以控制模拟器
  此版使用了yolov5-master版本
- game_wrapped
  将游戏标准化为Gym风格的RL训练环境，此版本直接调用
  此版本每一次step与游戏交互n次，将n步的s1~s4分别按时间步cat起来，同时在_entity_obsv_process中计算reward，以及done，一并返回（其中s4为长度10的游戏属性值向量，为使用lstm处理时序数据，将多个时间步的内容存储在s4中，即s4具体为：[[时间步1的值向量],[时间步2],[时间步3],...]，s1~s3为YOLO的中间状态，与s4同理保存多个时间步的内容）
  （这样一来，比方说第一个s4保存的是时间步1~3的内容，第二个s4则是4~6的内容，和“第二个s4保存1~6的内容”是不一样的，后者我认为是更科学的，但是由于特征很占内存，处理起来计算量不小，所以很难实现）
- models
  神经网络定义，作为DQN的深度学习部分
  此版本中，卷积层编码hidden特征（yolo中间状态），全连接层编码属性特征，将这些特征首尾相连得到一个时间步；将多个时间步按“时间步维”cat到一起，lstm处理，取lstm最后时刻输出，分别经过两个全连接层（输出维度对应左/右手指令维度），Soft-max激活得到输出
- DQN_brawlstar
  DQN强化学习主程序
  与DQN_doodlejump大同小异，仅针对神经网络输入特征的异构性，设计了多个经验回放池分别存储单一结构，使用时再分别取出

### V0.1：加入多线程及模拟学习预训练

- networks
  即上版本models，由于将左右手的动作集取笛卡尔积合并为一个集合，因此输出层仅使用一个全连接结构
- brawlstar_game
  由于动作集合并，这里的具体实现也有相应的改动
- brawlstar_utils
  为方便统一管理，将一些超参数集中起来
- **新增multi_thread**
  包含不同功能的线程化实现，各个线程可以被主程序调用，同时、独立地运行
  此版本实现了：YoloThread循环获取游戏画面并更新处理结果、KeyboardThread监听键盘
- game_wrapped
  相应地，game_wrapped变为直接访问yolo线程获取实时特征
- **新增data_recording** -> pretrain history/data.pth
  在人为操作游戏时同时捕获窗口，保存特征与监听键盘（使用并行线程）。此版本中：
  obsv为s1+s2+s3(yolo中间状态)+s4(属性)
  s4为[time, HP, enemyHP, super_charged, kill, killed, my_x, my_y, enemy_x, enemy_y]
  time：格式转换min:sec -> secs，并归一化
  HP：归一化
  supercharged, kill, killed：bool -> int
  my_x：角色坐标，除以画面宽度归一化
  针对s1~s4存储对应的下一时刻信息s1\_~s4_，加上action，全部打包
- **新增pretrain**
  使用data_recording生成的数据填充经验回放池，进行DQN式的网络更新
  此版本中：将s4, s4_, a取出，计算得到reward，获得完整的经验回放池，其后的内容与DQN_brawkstar主程序类似
- DQN_brawlstar
  增加yolo线程的加载、进入主循环前开启线程（线程为daemon守护类型，在此主程序运行完毕时线程会随之关闭）

### V0.2：在神经网络中加入注意力机制

- data_recording
  由于使用线程数较多，在完成数据采集、进行保存时，使用stop_thread关闭线程，减少cpu资源占用，缩短大容量预训练数据的保存时间。游戏特征的结构则和上版本相同
- pretrain
  若有多个预训练数据文件，可以使用load_all读取后拼接。但是由于电脑运行内存的限制，且游戏特征的数据占用较大，因而将buffer size上限设置为1000条
  在进行模型权重迭代更新的时候，新增**训练过程loss数据保存**，可用于绘制图像撰写论文，同时在下一次训练读取权重时一并读取，实现完整的断点续训
- networks
  上一版本的遗留程序，不作使用
- **新增nn_module**
  从Go Bigger官方程序的深度学习引擎ding中摘出，为transformer提供全连接层及normalization
- **新增transformer**
  注意力层定义，直接来自Go Bigger官方的ding。使用其可以直接获得包含多层注意力及mlp的计算模块。此版本的属性特征仍为长度10的定长值向量，不使用mask
- **新增networks_transformer**
  新的神经网络定义，在networks的基础上加入注意力层，将网络的输出层linear_o和输入部分属性值向量的编码层linear4改为注意力层（此处的注意力层来自**transformer**）;
  此外，输出层计算方式**新增Dueling DQN方式**，与原DQN方式的对比也出现在论文中
- game_wrapped
  在奖励值计算_entity_obsv_process中，对计算公式中的权重进行归一化
- DQN_brawlstar
  **新增Double DQN计算方式**
  新增matplotlib实时绘制奖励值曲线（答辩演示用）

### V0.3：尝试新的游戏特征提取（最终效果不好，但是开拓思路）（此版本有用到一些copy、clone方法）

- **新增feature_camera**
  线程化编写，使用CameraThread实时获取并保存游戏画面截图；使用ColorThread对游戏画面进行hsv空间阈值分割后，经过resnet101变为特征向量，以此作为一种新的游戏特征。（如游戏中敌方的攻击贴图都是红色的，经过hsv阈值分割提取红色后，滤除画面中的其他元素，使agent注意到攻击；如将画面与上一帧的画面逐像素做差，还可以进一步过滤不相关元素，毕竟敌方的攻击（子弹）是快速移动的）
  此版本以resnet101输出的特征向量代替原YOLO中间状态，大幅缩小每一条游戏特征数据的体积，但是似乎是信息量不如之前，因而效果不好
- game_wrapped
  与之前版本相比，使用resnet提取特征替换YOLO中间状态；在reward计算完成时将属性值向量进行转化，变为画面中一个角色对应一条向量，使用one-hot区分敌我，为下阶段加入友方做准备（由于此时仍然在1v1下进行，角色数量固定，因此这一部分特征的维度仍然是固定的）
- DQN_brawlstar
  相应地新增新线程camera、rgbthread及resnet的实例化及开启
- networks_transformer
  新增resnet定义
  由于使用resnet特征替换了YOLO中间状态特征，使用全连接层作为resnet特征的输入层，其余不变
- data_recording
  在预训练数据录制时不再记录保存YOLO中间状态，转而保存resnet特征向量，大幅缩减单条游戏数据的内存使用量（具体程序改起来要考虑各种矩阵维度，还是较为复杂）
- pretrain
  由于上面game_wrapped对属性值向量维度进行了更改，在此处取出预训练数据（需补充计算reward）的时候，需要将维度转换回原来的形式，再使用game_wrapped的reward计算功能，因而新增加了get_reward0函数。但是来回转换维度不如在录制数据集时就按原维度保存，因而将这两次多余的转换取出，在此处仍使用原函数get_reward（具体操作为，在data_recording中将game_wrapped计算reward前后两种维度的数据都保存了下来，分别用于pretrain中reward的补充计算及用于DQN训练）
- 注：在该版本中设计了如game_wrapped所述的属性特征维度转化（按角色类别独热码划分特征），但该版本最终将属性特征维度做如下转换：
  长度10属性值向量 -> 长度13，将首位的时间(s)转为长度4的独热码，以表征游戏进度（如进行到15s为初期，则为0001）

### V0.4：尝试独立神经网络输出层，分别输出左右手指令（如果左右手指令不离散化，则难以取笛卡尔积合并，尝试将两者独立输出）

- networks_transformer
  简单将神经网络lstm部分输出经过两个独立结构（均为注意力层+全连接层），输出左右手指令即可。由于同样使用上一版本的resnet特征，网络其余部分不变。**独立两个输出头关键在于损失函数**，具体见pretrain
- pretrain
  神经网络的输出部分彼此独立，而共享前半部分权重，为同时对整个结构进行训练，查询资料后做如下处理：
  对两个输出分别独立计算损失，后**将两个损失直接相加**（可以各0.5权重等），对所得的**总损失值直接backward**。根据官方说法，两个输出结构的loss在相加时，各自计算图会合并起来，得以正常反向传播训练
- DQN_brawlstar
  决策阶段，网络输出两个独立结果，对应左右手指令，将两个指令按之前版本格式合并即可（其实就是字符串拼接）；
  训练阶段与pretrain同理，只不过预训练效果不好，因此在这里只验证游戏表现（效果也不好），跳过训练

### V0.5：加入友方

- transformer
  该版本加入友方后，属性值特征变为长度可变的矩阵（具体组织形式可参考论文）：画面中的每个角色信息对应一条属性值向量，各个向量拼接为矩阵。由于画面中出现角色数量可变，为使数据维度对齐方便存储在经验回放池中，将属性值矩阵使用mask-padding到最大，即画面中未出现的角色，如未出现的为己方，则使用上回合的数据替代；如为友方或敌方，则用-∞向量mask替代。
  此程序72行（forward部分）的mask用法，官方实现有bug，在本版本自行修正（并在github中提出且被采纳）。源程序的mask为sequence mask功能，而此处是要将输入矩阵使用全负无穷的行补齐，原理上不相同，在程序上使用了transpose实现
- networks_transformer
  由于0.3~0.4版本效果不好，此版本基于0.2修改加入友方（使用YOLO中间状态）。在网络的前向传播部分，除了原s1~s4特征，增加一个mask输入，在s4输入编码层使用
- multi_thread
  **新增yolov5-master(multi-players)（根目录）**，该版本yolo训练时增加“友方”类别，去除了“时间”类别（以往版本中目标检测会将时间元素框出并在yolo线程中完成时间信息提取，但时间元素在游戏画面中位置是固定的，采用固定座标分割效果相同）。
  此版本的YoloThread调用了新的yolov5模型；
  **CameraThread**来自0.3版本的feature_camera（后者在此版本中删除），实时获取更新画面截图。此版本的YoloThread自身不具备窗口部分功能，而从CameraThread中获取游戏画面，功能上进一步并行化（实际处理帧率可能没有太大提升）；
  **新增CameraAssistThread**，其同样从CameraThread获得游戏画面，通过固定坐标裁剪获得时间元素和大招充能元素的图像，对前者进行OCR文字信息提取（来自以往版本的YoloThread）获得时间信息；对后者通过HSV阈值分割，将黄色充能环之外的像素滤除，再计算黄色像素点数量获得大招充能进度。对于一些长距离攻击，可能在屏幕外击中目标/目标被击杀消失，因而以往版本通过角色血量变化来表征是否击中，不如直接通过充能环的变化来代替，更加直观。
- game_wrapped
  0.3版本game_wrapped中，为将输入神经网络的属性特征按角色区分而改变其维度，导致预训练补充计算reward时需将维度转换回来。0.3版本的data_recording遂同时保存转换前后的属性特征，免去维度逆转换的麻烦。此版本中属性特征也按角色划分，是在YoloThread中完成的，因而与0.3版本类似，在YoloThread中将两种维度保存下来，一种为用于神经网络的entity_obsv，一种为用于reward计算的info
- data_recording
  在录制预训练数据时同时保存上述的entity_obsv(原s4)及info，同时保存mask
- DQN_brawlstar & pretrain
  除了pretrain使用info进行reward补充计算、两个程序为mask新增回放池外，其余较0.3版本无太大变化

### V0.5.1：适配新模拟器

- Brawlstar更新，此后必须在新版本MuMu模拟器12运行
- brawlstar_game
  pyautogui在MuMu模拟器12被屏蔽，整体替换为pydirectinput，两个库下使用到的函数名完全一致
- 注意新模拟器的窗口名称变化
  在brawlstar_game中修改win32gui.FindWindow()；在multi_thread/realtime_detection中修改grab_screen()

