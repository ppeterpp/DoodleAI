import random
import torch
import torch.nn as nn
import numpy as np
import gym
import os
import game_wrapped

from tqdm import tqdm
import matplotlib.pyplot as plt


class RNN(nn.Module):
    def __init__(self, obsv_dim, action_dim, hidden_size=64, num_layers=1):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(
            input_size=obsv_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
        self.out = nn.Linear(64, action_dim)
        self.mls = nn.MSELoss()
        # self.mls = nn.CrossEntropyLoss()
        # self.opt = torch.optim.Adam(self.parameters(), lr=0.01)
        self.opt = torch.optim.SGD(self.parameters(), lr=0.01)
        # self.dyna_lr = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.opt, gamma=0.9)
        self.dyna_lr = torch.optim.lr_scheduler.MultiStepLR(optimizer=self.opt, milestones=[6000, 10000], gamma=0.1)
        # 学习率: 0.01 -> 0.001 -> 0.0001

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        r_out, (h_n, h_c) = self.rnn(x, None)  # None represents zero initial hidden state
        # choose r_out at the last time step
        out = self.out(r_out[:, -1, :])
        return out


obsv_dim = 26  # 每一帧包含的特征长度
action_dim = 3
time_step = 24  # RNN一次读取几帧游戏数据
device = 'cuda'  # 'cpu'
render = True
# render = False

env = game_wrapped.DoodleEnv(render=render, time_step=time_step, fps=90)
env = env.unwrapped  # gym对游戏进行了封装，有些变量变得不可访问，这里使游戏内部的变量可以访问

# 两个网络，进行延迟更新，即不能每次动作都更新网络，
# 要隔一段时间进行一次更新，否则奖励累积可能爆炸
net = RNN(obsv_dim, action_dim).to(device)
net2 = RNN(obsv_dim, action_dim).to(device)

store_count = 0  # 经验池，将智能体碰到的各种情况都记录下来
store_size = 6000  # buffer size，即经验池能存2000条数据
decline = 0.9  # 衰减系数
learn_time = 0  # 学习次数
update_time = 20  # 隔多少个学习次数更新一次网络
gama = 0.9  # Q值计算时的折扣因子
b_size = 3000  # batch size，一次从经验池中抽取多少数据进行学习
iter = 6000  # 游戏进行轮次
start_study = False
loss = 0
loss_history = []
score_history = []
score_history_ = []  # for periodical performance
iter_history = []

if not os.path.exists('history'):
    os.mkdir('history')

if os.path.exists('history/model_params.pkl'):# and False:
    print('---------loading trained model---------')
    net.load_state_dict(torch.load('history/model_params.pkl'))
    net2.load_state_dict(torch.load('history/model_params.pkl'))
    learn_time = int(open('history/learn_time.txt', 'r').read())
    loss_history = np.load('history/loss_history.npy').tolist()
    score_history = np.load('history/score_history.npy').tolist()
    iter_history = np.load('history/iter_history.npy').tolist()

# 初始化buffer 列中储存 s, a, s_, r
# 即s状态下进行操作a会跳转至s_，得到reward
# cartpole中state(observation)长度为4，action、reward长度为1
# 因此经验池中每条数据的长度为obsv_dim+1+obsv_dim+1
store_s = np.zeros((store_size, time_step, obsv_dim))
store_s_ = np.zeros((store_size, time_step, obsv_dim))
store_action = np.zeros((store_size, 1))
store_reward = np.zeros((store_size, 1))

pbar = tqdm(total=iter)
for i in range(iter):
    s = env.reset()  # 初始化游戏env，s即为state (time_step * obsv_dim)
    while True:
        if random.randint(0, 100) < 100*(decline**(learn_time*0.1)):
            # 随着学习次数(learn_time)的增加，乘上decline也越小(decline<1)
            # 即越到后期，操作越不可能是随机生成
            a = random.randint(0, 2)
        else:
            # 与上面随机生成action相对，此处采用神经网络输出选择action
            # out为[左走累计奖励, 右走累计奖励]
            # 累计奖励即此刻到游戏结束所有奖励之和
            # 每次反向传播之前都有梯度归零，所以后面的detach()可以不用
            # np.newaxis为输入增加了一个维度，即batch_size，此处batch_size为1
            out = net(torch.Tensor(s[np.newaxis, :, :]).to(device)).detach()
            # torch.max返回的是元组(max, index)
            # 此处argmax则只返回最大值的索引，是个tensor，要转化后提取
            a = torch.argmax(out).data.item()  # 索引值012正好对应三种action
        s_, r, done, info = env.step(a)  # s_即observation，info没什么用
        # 奖励r的设计最难也最重要
        # reward的范围在0~1，防止累积值过大

        # 此处添加回放池数据，由于长度为2000，第2001条将覆盖第1条数据
        # print(s.shape, a, s_.shape, r)
        # print(type(s), type(a), s_.shape, type(r))
        store_s[store_count % store_size] = s
        store_action[store_count % store_size] = a
        store_s_[store_count % store_size] = s_
        store_reward[store_count % store_size] = r
        store_count += 1
        s = s_

        if store_count > store_size:  # 判断条件意为回放池已满(可以进行训练了)

            if learn_time % update_time == 0:
                # 每经过一个update_time，更新net2
                # 将net的权重提取、加载到net2
                # net则是实时更新的，如后面
                net2.load_state_dict(net.state_dict())

                if not isinstance(loss, int):
                    loss_history.append(torch.Tensor.cpu(loss).detach())  # detach()将去除所带梯度
                    if len(score_history) > 20:
                        score_history_.append(env.peak_score)
                    score_history.append(env.peak_score)
                    iter_history.append(learn_time)

                if learn_time < 1600:
                    pass
                elif learn_time % (update_time * 20) == 0 and len(score_history) > 80 and len(score_history_) != 0:
                    # print(len(score_history_))
                    if os.path.exists('history/model_params.pkl'):
                        os.remove('history/model_params.pkl')
                        print('---------权重保存数据更新---------')
                    else:
                        # pass
                        print('---------权重数据进行保存---------')
                    torch.save(net2.state_dict(), 'history/model_params.pkl')
                    # 训练时有类似epsilon greedy的效果，即随着learn_time增加，也即模型愈发完善，随机action的几率会降低
                    # 所以在保存模型权重的时候也要保存这个量，在加载时一并加载，否则每次重新运行初期都会大量随机动作
                    with open('history/learn_time.txt', 'w') as file:
                        file.write(str(learn_time))
                    np.save('history/loss_history.npy', loss_history)
                    np.save('history/score_history.npy', score_history)
                    np.save('history/iter_history.npy', iter_history)
                    if sum(score_history_)/len(score_history_) > sum(score_history[-160:-80])/len(score_history[-160:-80]):  # 如何save best
                        print('---------model got evolved---------')
                        if not os.path.exists('history_best'):
                            os.mkdir('history_best')
                        if os.path.exists('history_best/model_params.pkl'):
                            os.remove('history_best/model_params.pkl')
                        else:
                            pass
                        torch.save(net2.state_dict(), 'history_best/model_params.pkl')
                        # 训练时有类似epsilon greedy的效果，即随着learn_time增加，也即模型愈发完善，随机action的几率会降低
                        # 所以在保存模型权重的时候也要保存这个量，在加载时一并加载，否则每次重新运行初期都会大量随机动作
                        with open('history_best/learn_time.txt', 'w') as file:
                            file.write(str(learn_time))
                        np.save('history_best/loss_history.npy', loss_history)
                        np.save('history_best/score_history.npy', score_history)
                        np.save('history_best/iter_history.npy', iter_history)
                        plt.subplot(121)
                        plt.plot(iter_history, loss_history)
                        plt.subplot(122)
                        plt.plot(iter_history, score_history)
                        plt.savefig('history_best/train.png')
                        plt.close()
                    score_history_ = []

            # 随机从某处开始，连续抽取b_size(1000)条数据用于神经网络的训练
            index = random.randint(0, store_size - b_size -1)
            b_s  = torch.Tensor(store_s[index:index + b_size]).to(device)  # batch * time_step * input_dim
            b_a  = torch.Tensor(store_action[index:index + b_size]).long().to(device)  # b * t * 1
            b_s_ = torch.Tensor(store_s_[index:index + b_size]).to(device)  # b * t * i
            b_r  = torch.Tensor(store_reward[index:index + b_size]).to(device)  # b * t * 1

            # 抽取出的b_a为前面探索时的action记录，某些是随机选取的，某些是参考网络做出的
            # 此处的b_s有1000条，送入net生成1000对输出，每一对都是对向左向右两种操作的判断
            # 即[[r1_1,r1_2], [r2_1,r2_2], ...]共1000对，这一整条数据具有"两个维度"
            # gather中的1就是说，对第二个维度进行操作，b_a则是说对第二个维度每一对数据选取哪一个
            # 例如b_a是[0,1...]即第一次向左，第二次向右
            # 则gather将对应提取并整合出[r1_1,r2_2...]
            q = net(b_s).gather(1, b_a)
            
            # detach()是截断梯度流，反向传播时有用
            # 注意此处计算q_next用的是net2，是有一个滞后效果的，因为net2是隔一段时间才会更新一次
            # net代入b_s_的输出同样为[[r1_1,r1_2], [r2_1,r2_2], ...]，这是对下一个state做不同action对应累积reward的估计
            # 之前的话是依照历史action筛选历史reward，此处是直接根据net做出reward选择，所以此处直接选择每一对中最大的reward
            # max返回的是(max, index)元组，我们要的是值；max(1)的1与上面的gather的1类似，是对第二个维度操作
            q_next = net2(b_s_).detach().max(1)[0].reshape(b_size, 1)  # rechape把1*1000转置成1000*1，才能相加
            
            # Q value计算公式
            tq = b_r + gama * q_next

            loss = net.mls(q, tq)
            net.opt.zero_grad()
            loss.backward()
            net.opt.step()
            if i % 60 == 0:
                net.dyna_lr.step()  # 更新学习率

            learn_time += 1
            if not start_study:
                print('start study')
                start_study = True
                break

        if done:
            break
        # env.render()
    if not isinstance(loss, int):
        pbar.set_description("loss: %s" % loss)
    pbar.update(1)
    if learn_time >= 17200 and not render:
        break

plt.subplot(121)
plt.plot(iter_history, loss_history)
plt.subplot(122)
plt.plot(iter_history, score_history)
plt.savefig('history/train.png')
plt.show()
pbar.close()
