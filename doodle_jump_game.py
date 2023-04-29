import sys
import random
import pygame
from doodle_jump_utils import CONST

pygame.init()
FPSCLOCK = pygame.time.Clock()
SCREEN = pygame.display.set_mode((CONST['SCREEN_WIDTH'], CONST['SCREEN_HEIGHT']))
pygame.display.set_caption('Doodle Jump')


class GameState:
    def __init__(self, render=True, fps=75):
        global CONST
        self.ifrender = render
        self.fps = fps

        self.background = pygame.transform.scale(pygame.image.load("assets/background.png").convert(), (CONST['SCREEN_WIDTH'], CONST['SCREEN_HEIGHT']))
        self.doodle = pygame.transform.scale(pygame.image.load('assets/doodle.png'), (CONST['DOODLE_WIDTH'], CONST['DOODLE_HEIGHT']))  # pygame.image.load("?").convert_alpha()
        self.pedal = pygame.transform.scale(pygame.image.load("assets/pedal.png").convert_alpha(), (CONST['PEDAL_WIDTH'], CONST['PEDAL_HEIGHT']))
        SCREEN.blit(self.background, (0, 0))
        self.initialize()

    def initialize(self):
        self.score = CONST['SCREEN_HEIGHT'] - 100  # 用高度作为得分，此为初始高度
        # self.last_score = CONST['SCREEN_HEIGHT'] - 100  # 记录上一帧的得分
        self.peak_score_ = CONST['SCREEN_HEIGHT'] - 100
        # reward为两帧score之差，这样设置使初始化后的reward为0
        self.x = 100
        self.y = 100
        self.dy = 0
        self.pedal_x = []
        self.pedal_y = []
        # 踏板间距与踏板数量相关，后续通过减少踏板数增加难度时，要考虑间距与跳起高度的关系
        self.p_gap = int(CONST['SCREEN_HEIGHT'] / CONST['PEDAL_NUM'])
        # assert(self.p_gap*2 < ?)  # 踩踏板的跳起高度为?
        for i in range(CONST['PEDAL_NUM']-1, -1, -1):
            a = self.p_gap*i
            b = self.p_gap*(i+1)
            self.pedal_y.append(random.randint(a, (a+b)/2))
            self.pedal_x.append(random.randint(0, CONST['SCREEN_WIDTH']-CONST['PEDAL_WIDTH']))

    def frame_step(self, action):
        self.done = False

        # If you are not using other event functions in your game,
        # you should call pygame.event.pump() to allow pygame to handle internal actions
        pygame.event.pump()

        assert (action in [0, 1, 2]), 'Action is not available!'
        if action == 1:  # 0误操作，1向左，2向右
            self.x -= 4.2
            self.dx = -1  # horizontal moving direction
        elif action == 2:
            self.x += 4.2
            self.dx = 1
        else:
            self.dx = 0

        self.dy += 0.2
        self.y += self.dy
        self.score -= self.dy

        if self.x < 0 and action == 1:
            self.x = CONST['SCREEN_WIDTH']
        elif self.x + CONST['DOODLE_WIDTH'] > CONST['SCREEN_WIDTH']\
                and action == 2:
            self.x = - CONST['DOODLE_WIDTH']

        if self.y < CONST['DOODLE_Y_LIMIT']:
            for i in range(len(self.pedal_y)):
                self.y = CONST['DOODLE_Y_LIMIT']
                self.pedal_y[i] -= self.dy
                # 更新踏板
                if self.pedal_y[i] > CONST['SCREEN_HEIGHT']:
                    self.pedal_y.pop(i)
                    self.pedal_x.pop(i)
                    # self.pedal_y.append(random.randint(0, self.p_gap))
                    self.pedal_y.append(random.randint(int(self.pedal_y[-1] - self.p_gap*1.05),
                                                       int(self.pedal_y[-1] - self.p_gap)))
                    self.pedal_x.append(random.randint(0, CONST['SCREEN_WIDTH']-CONST['PEDAL_WIDTH']))

        for i in range(CONST['PEDAL_NUM']):
            if ((self.x + 50 > self.pedal_x[i]) and
                    (self.x + 20 < self.pedal_x[i] + CONST['PEDAL_WIDTH']) and
                    (self.y + CONST['DOODLE_HEIGHT'] > self.pedal_y[i]) and
                    (self.y + CONST['DOODLE_HEIGHT'] < self.pedal_y[i] + CONST['PEDAL_HEIGHT']) and
                    (self.dy > 0)):
                self.dy = -10
        if self.ifrender:
            self.render()

        if self.y > CONST['SCREEN_HEIGHT']:
            self.done = True
            self.peak_score = self.peak_score_
            self.initialize()  # 注意初始化的时候会重置很多变量
            # print(self.peak_score)

        if self.score > self.peak_score_:
            self.peak_score_ = self.score
        # 注意归一化
        observation = []
        # 所有踏板左上角的坐标
        for i in range(len(self.pedal_y)):
            observation.append(round(self.pedal_x[i] / CONST['SCREEN_WIDTH'], 4))
            observation.append(round(self.pedal_y[i] / CONST['SCREEN_HEIGHT'], 4))
            observation.append(round((self.pedal_x[i] + CONST['PEDAL_WIDTH']) / CONST['SCREEN_WIDTH'], 4))
            observation.append(round((self.pedal_y[i] + CONST['PEDAL_HEIGHT']) / CONST['SCREEN_HEIGHT'], 4))
        # # 屏幕最下方踏板的右下角坐标
        # observation.insert(2, round((self.pedal_x[0] + CONST['PEDAL_WIDTH']) / CONST['SCREEN_WIDTH'], 4))
        # observation.insert(3, round((self.pedal_y[0] + CONST['PEDAL_HEIGHT']) / CONST['SCREEN_HEIGHT'], 4))
        # 角色坐标
        observation.append(round(self.x / CONST['SCREEN_WIDTH'], 4))
        observation.append(round(self.y / CONST['SCREEN_HEIGHT'], 4))
        observation.append(round((self.x + CONST['DOODLE_WIDTH']) / CONST['SCREEN_WIDTH'], 4))
        observation.append(round((self.y + CONST['DOODLE_HEIGHT']) / CONST['SCREEN_HEIGHT'], 4))
        # 角色x, y方向速度，正负表示方向
        observation.append(self.dx)
        observation.append(round(self.dy / 10, 4))
        done = self.done
        # reward = max(int(self.score - self.last_score) / 10, -1)  # reward最低限制在-1
        reward = max(int(-self.dy) / 10, -1)  # reward最低限制在-1
        # self.last_score = self.score
        # print(reward)
        assert(-1 <= reward <= 1), '%s'%(reward)
        # print(observation, reward, done)
        return observation, reward, done

    def render(self):
        SCREEN.blit(self.background, (0, 0))
        SCREEN.blit(self.doodle, (self.x, self.y))
        for i in range(CONST['PEDAL_NUM']):
            SCREEN.blit(self.pedal, (self.pedal_x[i], self.pedal_y[i]))
        pygame.display.update()
        # pygame.display.flip()
        # pygame.display.update()
        FPSCLOCK.tick(self.fps)


if __name__ == '__main__':
    game_state = GameState(render=True)
    action = 0
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    action = 1
                if event.key == pygame.K_RIGHT:
                    action = 2
            elif event.type == pygame.KEYUP:
                action = 0
        game_state.frame_step(action)
