from enum import Enum
import os
import random
from qnn import QNN
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
import numpy as np
from queue import deque # D is basically a deque
CAPACITY = 1000000 # below this threshold, the deque keeps accepting; above this, old ones will be exiled
BATCH_SIZE = 100

from typing import List


class MoveType(Enum):
    STAY = 0
    MOVEUP = 1
    MOVEDOWN = 2


class State:
    def __init__(self, paddleY, ballX, ballY, ballVX, ballVY):
        self.paddleY = paddleY
        self.ballX = ballX
        self.ballY = ballY
        self.ballVX = ballVX
        self.ballVY = ballVY
    def tensorize(self):
        return torch.FloatTensor([
            self.paddleY,
            self.ballX,
            self.ballY,
            self.ballVX,
            self.ballVY,
        ])

class Entry():
    def __init__(self, oldState: State, action: MoveType, reward: float, newState: State):
        self.oldState = oldState
        self.action = action
        self.reward = reward
        self.newState = newState

def push(D: deque, entry: Entry):
    if len(D) == CAPACITY:
        D.popleft() # exile an old observation
    D.append(entry)
def getMiniBatch(D: deque) -> List[Entry]:
    sampled = random.choices(D, k=BATCH_SIZE)
    return sampled

class OutcomeType(Enum):
    CONT = 0
    LEFTWIN = 1
    RIGHTWIN = 2

class Ball:
    def __init__(self):
        self.x = 0.5
        self.y = 0.5
        self.vx = 0.03
        self.vy = 0.01

    def move(self, game) -> OutcomeType:
        self.x += self.vx
        self.y += self.vy
        outcome = self._bounce(game)
        return outcome

    def _augmentVelocityX(self):
        if self.vx > 0 and self.vx < 0.03:
            self.vx = 0.03
        if self.vx < 0 and self.vx > -0.03:
            self.vx = -0.03

    def _bounce(self, game) -> OutcomeType:
        if self.y < 0:
            self.y = -self.y
            self.vy = -self.vy
        elif self.y > 1:
            self.y = 2 - self.y
            self.vy = -self.vy
        if self.x < 0 or self.x > 1:
            agentInvolved = game.leftAgent
            possibleOutcome = OutcomeType.RIGHTWIN
            if self.x > 1:
                agentInvolved = game.rightAgent
                possibleOutcome = OutcomeType.LEFTWIN
            if isinstance(agentInvolved, WallAgent):
                self.x = 2 * agentInvolved.paddleX - self.x
                self.vx = -self.vx
                return OutcomeType.CONT
            if agentInvolved.canCapture(self):
                self.x = 2 * agentInvolved.paddleX - self.x
                u, v = random.uniform(-0.015, 0.015)\
                    , random.uniform(-0.03, 0.03)
                self.vx += u
                self.vy += v
                self._augmentVelocityX()
                agentInvolved.rewardValue = 1
                return OutcomeType.CONT
            agentInvolved.rewardValue = -1
            return possibleOutcome
        return OutcomeType.CONT

class Agent():
    def __init__(self, isLeft: bool):
        self.paddleHeight = 0.2
        self.paddleY = 0.4
        self.speed = 0.04
        self.paddleX = int(not isLeft)
        self.rewardValue = 0

    def _moveUp(self):
        if self.paddleY >= self.speed:
            self.paddleY -= self.speed
    def _moveDown(self):
        if self.paddleY + self.paddleHeight + self.speed <= 1:
            self.paddleY += self.speed

    def move(self, mt: MoveType):
        if mt == MoveType.MOVEUP:
            self._moveUp()
        elif mt == MoveType.MOVEDOWN:
            self._moveDown()
    def moveSmartly(self, *args, **kwargs) -> MoveType:
        return MoveType.STAY

    def getActions(self) -> list:
        moves = [MoveType.STAY]
        if self.paddleY >= self.speed:
            moves.append(MoveType.MOVEUP)
        if self.paddleY + self.speed + self.paddleHeight <= 1:
            moves.append(MoveType.MOVEDOWN)
        return moves

    def reward(self):
        rwd = self.rewardValue
        self.rewardValue = 0
        return rwd

    def canCapture(self, ball: Ball) -> bool:
        return ball.y >= self.paddleY \
               and ball.y <= self.paddleY + self.paddleHeight

class WallAgent(Agent):
    def __init__(self, isLeft: bool):
        super().__init__(isLeft)
        self.paddleHeight = 1
        self.paddleY = 0
    def getActions(self) -> list:
        return [MoveType.STAY]
    def moveSmartly(self) -> MoveType:
        return MoveType.STAY

class HardCodeAgent(Agent):
    def __init__(self, isLeft: bool):
        super().__init__(isLeft)
        self.speed = 0.02

    def moveSmartly(self, game) -> MoveType:
        ball = game.ball
        paddleCenter = self.paddleY + 0.5 * self.paddleHeight
        if ball.y > paddleCenter:
            return MoveType.MOVEDOWN
        if ball.y < paddleCenter:
            return MoveType.MOVEUP
        return MoveType.STAY

class TrainedAgent(Agent):
    def __init__(self, isLeft: bool, epsilon: float, gamma: float, learningRate: float):
        super().__init__(isLeft)
        self.qnn = QNN() # a neural net (hopefully deep) to compute Q value
        for p in self.qnn.parameters():
            if p.dim() >= 2:
                nn.init.xavier_normal(p)
        self.epsilon = epsilon
        self.gamma = gamma
        self.optimizer = optim.SGD(self.qnn.parameters(), lr=learningRate)

    def moveSmartly(self, game) -> MoveType:
        actions = self.getActions()
        r = random.uniform(0, 1)
        if r < self.epsilon:
            return random.choice(actions)
        snapshot = game.makeState().tensorize()
        snapshot = Variable(torch.unsqueeze(snapshot, dim=0))
        out = self.qnn.forward(snapshot)  # out is a 3-D vector
        _, maxAction = out.max(1)
        maxAction = int(maxAction.data[0])
        return MoveType(maxAction)

class Game:
    def __init__(self, leftAgent: Agent, rightAgent: Agent, ball: Ball):
        self.leftAgent = leftAgent
        self.rightAgent = rightAgent
        self.ball = ball

    def makeState(self) -> State:
        return State(
            self.rightAgent.paddleY,
            self.ball.x,
            self.ball.y,
            self.ball.vx,
            self.ball.vy,
        ) # the returned tuple is basically, x_t

    def play(self, D: deque, performance: List[float]) -> OutcomeType:
        outcome = OutcomeType.CONT
        consecutive = 0
        while outcome == OutcomeType.CONT:
            # retrieve s_t
            oldState = self.makeState()
            mleft = self.leftAgent.moveSmartly()
            mright = self.rightAgent.moveSmartly(game=self)
            self.leftAgent.move(mleft)
            self.rightAgent.move(mright)
            outcome = self.ball.move(game=self)
            if self.rightAgent.rewardValue > 0:
                consecutive += 1
            # now we have s_(t+1)
            reward = self.rightAgent.reward()
            newState = self.makeState()
            if outcome != OutcomeType.CONT:
                newState = None # indicate a terminal state
            if isinstance(self.rightAgent, TrainedAgent):
                entry = Entry(oldState, mright, reward, newState)
                push(D, entry)
                sampledEntries = getMiniBatch(D)
                # build ground truth y_i
                # TODO: vectorize
                nonterm = []
                for sampledEntry in sampledEntries:
                    if sampledEntry.newState is not None:
                        nonterm.append(torch.unsqueeze(sampledEntry.newState.tensorize(), dim=0))
                nonterm = torch.cat(nonterm, dim=0)
                nonterm = self.rightAgent.qnn.forward(Variable(nonterm)) # n * 3 vector
                nonterm, _ = nonterm.max(1)
                nonterm = iter(nonterm)
                targets = np.zeros(BATCH_SIZE)
                for idx, sampledEntry in enumerate(sampledEntries):
                    if sampledEntry.newState is None:
                        targets[idx] = -1
                    else:
                        targets[idx] = sampledEntry.reward + self.rightAgent.gamma * next(nonterm)

                # compute Q-value guess
                sampledOldStates = map(lambda ent: torch.unsqueeze(ent.oldState.tensorize(), dim=0), sampledEntries)
                sampledOldStates = Variable(torch.cat(sampledOldStates, dim=0))
                out = self.rightAgent.qnn.forward(sampledOldStates) # out is a BATCH_SIZE * 3 tensor
                # convert actions into a LongTensor
                sampledActions = map(lambda ent: ent.action.value, sampledEntries)
                guesses = out.gather(dim=1, index=Variable(torch.LongTensor([[x] for x in sampledActions])))
                # calculate loss
                targets = Variable(torch.from_numpy(targets).float())
                criterion = nn.MSELoss()
                loss = criterion.forward(guesses, targets)
                # TODO: perform gradient descent
                self.rightAgent.qnn.zero_grad()
                loss.backward() # can backward work here?
                self.rightAgent.optimizer.step()
        print("Consecutive: ", consecutive)
        performance.append(consecutive)
        return outcome

def train(gamma: float, learningRate: float, epsilon: float, epoch) -> List:
    leftAgent = WallAgent(True)
    rightAgent = TrainedAgent(True, epsilon, gamma, learningRate)
    if os.path.exists("./memory.pkl"):
        mem = open("./memory.pkl", "rb")
        rightAgent.qnn.load_state_dict(torch.load(mem))
    print(list(rightAgent.qnn.parameters()))
    ball = Ball()
    game = Game(leftAgent, rightAgent, ball)
    # replay memory
    D = deque()
    performance = []
    for epk in range(epoch):
        if (epk + 1) % 10 == 0:
            print("EPOCH: ", epk)
        game.play(D, performance)
        game.ball = Ball()
        game.rightAgent.paddleY = 0.4 # reset
    print(list(rightAgent.qnn.parameters()))
    with open("./memory.pkl", "wb") as mem:
        torch.save(rightAgent.qnn.state_dict(), mem)
    return performance

if __name__ == '__main__':
    performance = train(gamma=0.9, learningRate=0.1, epsilon=0.05, epoch=5000)
    print("Mean: ", np.mean(performance))
    print("Std Dev: ", np.std(performance))
    for pc in range(10):
        print("Percentile (" + str(pc * 10) + "): ", np.percentile(performance, pc * 10))
