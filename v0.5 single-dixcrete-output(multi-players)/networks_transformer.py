import torch.nn as nn
import torch
from transformer import Transformer


class MultiModalEncoder(nn.Module):
    # Classification head, i.e. x(b,c1,20,20) to x(b,c2)
    def __init__(self, c_out, l_out, input_size, action_dim, device='cpu',
                 hidden_size=64, num_layers=1, k=1, s=1, p=None, g=1):  # channel_out, kernel, stride, padding, groups
        super(MultiModalEncoder, self).__init__()
        self.aap = nn.AdaptiveAvgPool2d(8)  # to x(b,c1,1,1)
        self.device = device
        self.conv1 = nn.Conv2d(128, c_out, k, s)#, autopad(k, p), groups=g)  # to x(b,c2,1,1)
        self.conv2 = nn.Conv2d(256, c_out, k, s)#, autopad(k, p), groups=g)
        self.conv3 = nn.Conv2d(512, c_out, k, s)#, autopad(k, p), groups=g)
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
        self.flat = nn.Flatten()
        self.attention = Transformer(
            input_dim=6, hidden_dim=128, output_dim=l_out, activation=nn.ReLU()
        )
        self.linear_o = Transformer(
            input_dim=64, hidden_dim=128, output_dim=128, activation=nn.ReLU()
        )
        self.linear_o_value = nn.Linear(128, 1)  # 此处的output_size即为类别数
        self.linear_o_advantage = nn.Linear(128, action_dim)
        # self.activate = nn.Softmax(dim=1)
        self.mls = nn.MSELoss()
        # self.mls = nn.CrossEntropyLoss()
        # self.opt = torch.optim.Adam(self.parameters(), lr=0.1)
        self.opt = torch.optim.SGD(self.parameters(), lr=0.001)
        self.dyna_lr = torch.optim.lr_scheduler.MultiStepLR(optimizer=self.opt, milestones=[6000+6400, 8000+6400], gamma=0.1)

    def forward(self, xlist):
        x1, x2, x3, x4, x5 = xlist[0], xlist[1], xlist[2], xlist[3], xlist[4]
        # x1, x2, x3:                                       batch * time-stepS * feature-map direction
        o = []
        for i in range(x1.shape[1]):  # range(time_step)
            idx = torch.tensor([i]).to(self.device)
            # nn.Flatten默认数据为(batch,d1,d2,d3,...), 默认从d1开始flatten
            # torch.index_select(x1, 1, torch.tensor([i])): batch * time-step=i * feature-map dimensions
            # after squeeze(1):                             batch * feature-map dimensions(CHW for example)
            # after conv & flat:                            batch * input-size(flatten)
            # after unsqueeze(1):                           batch * time-step=i * input-size
            # after torch.cat:                              batch * time-step=i * input-size(concatenated)
            x1_i = self.flat(self.conv1(torch.index_select(x1, 1, idx).squeeze(1))).unsqueeze(1)
            x2_i = self.flat(self.conv2(torch.index_select(x2, 1, idx).squeeze(1))).unsqueeze(1)
            x3_i = self.flat(self.conv3(torch.index_select(x3, 1, idx).squeeze(1))).unsqueeze(1)
            x4_i_ = torch.index_select(x4, 1, idx).squeeze(1)
            x5_i_ = torch.index_select(x5, 1, idx).bool().squeeze(1)
            x4_i = self.attention(x=x4_i_, mask=x5_i_).mean(dim=1).unsqueeze(1)
            o.append(torch.cat([x1_i, x2_i, x3_i, x4_i], dim=2))
        o = torch.cat([j for j in o], dim=1)              # batch * time-stepS(concatenated) * input-size
        o, (h_n, h_c) = self.rnn(o, None)
        o = o[:, -1, :]

        # solution 1/2: Dueling DQN
        value = self.linear_o_value(self.linear_o(o.unsqueeze(1)).squeeze(1))
        advantage = self.linear_o_advantage(self.linear_o(o.unsqueeze(1)).squeeze(1))
        avg_advantage = torch.mean(input=advantage, dim=-1, keepdim=True)
        q_values = value + (advantage - avg_advantage)
        return q_values

        # solution 2/2: DQN
        # o = self.linear_o_advantage(self.linear_o(o.unsqueeze(1)).squeeze(1))
        # return o


if __name__ == '__main__':
    conv_out_channel = 16
    l_out = 128
    batch = 1
    time_step = 6
    input_size = 16*40*80 + 16*20*40 + 16*10*20 + 128
    net = MultiModalEncoder(conv_out_channel, l_out, input_size, 9, 3)
    x1 = torch.ones((batch,time_step,128,40,80))
    x2 = torch.ones((batch,time_step,256,20,40))
    x3 = torch.ones((batch,time_step,512,10,20))
    x4 = torch.ones((batch,time_step,10))
    out = net([x1,x2,x3,x4])
    print(out)
    print(torch.argmax(out).data.item())
