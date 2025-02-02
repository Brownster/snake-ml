import pygame
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import math

# ---------------------------
# Helper Function for Direction Mapping
# ---------------------------
# Action indices: 0 = turn right, 1 = straight, 2 = turn left.
def get_new_direction(current, action_index):
    mapping = {
        "UP":    {0: "RIGHT", 1: "UP",    2: "LEFT"},
        "RIGHT": {0: "DOWN",  1: "RIGHT", 2: "UP"},
        "DOWN":  {0: "LEFT",  1: "DOWN",  2: "RIGHT"},
        "LEFT":  {0: "UP",    1: "LEFT",  2: "DOWN"}
    }
    return mapping[current][action_index]

# ---------------------------
# Neural Network for Snake AI
# ---------------------------
class SnakeAI(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SnakeAI, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = self.linear3(x)
        return x

# ---------------------------
# Snake Game Environment for AI Training
# ---------------------------
class SnakeGameAI:
    def __init__(self, width=600, height=400, block_size=20):
        self.width = width
        self.height = height
        self.block_size = block_size
        
        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption('Snake AI Training')
        self.clock = pygame.time.Clock()
        
        self.reset()
        
    def reset(self):
        # Initialize snake in the middle of the screen
        self.snake = [[self.width // 2, self.height // 2]]
        self.direction = "RIGHT"
        self.food = self._spawn_food()
        self.score = 0
        self.frame_iteration = 0
        return self.get_state()
        
    def _spawn_food(self):
        while True:
            x = random.randrange(0, self.width, self.block_size)
            y = random.randrange(0, self.height, self.block_size)
            if [x, y] not in self.snake:
                return [x, y]
    
    def get_state(self):
        head = self.snake[0]
        
        # Define points around the head (for collision checks)
        point_l = [head[0] - self.block_size, head[1]]
        point_r = [head[0] + self.block_size, head[1]]
        point_u = [head[0], head[1] - self.block_size]
        point_d = [head[0], head[1] + self.block_size]
        
        # Current direction booleans
        dir_l = self.direction == "LEFT"
        dir_r = self.direction == "RIGHT"
        dir_u = self.direction == "UP"
        dir_d = self.direction == "DOWN"
        
        state = [
            # Danger straight ahead
            (dir_r and self._is_collision(point_r)) or
            (dir_l and self._is_collision(point_l)) or
            (dir_u and self._is_collision(point_u)) or
            (dir_d and self._is_collision(point_d)),
            
            # Danger right
            (dir_u and self._is_collision(point_r)) or
            (dir_d and self._is_collision(point_l)) or
            (dir_l and self._is_collision(point_u)) or
            (dir_r and self._is_collision(point_d)),
            
            # Danger left
            (dir_d and self._is_collision(point_r)) or
            (dir_u and self._is_collision(point_l)) or
            (dir_r and self._is_collision(point_u)) or
            (dir_l and self._is_collision(point_d)),
            
            # Current move direction
            dir_l, dir_r, dir_u, dir_d,
            
            # Food location relative to head
            self.food[0] < head[0],  # food is left
            self.food[0] > head[0],  # food is right
            self.food[1] < head[1],  # food is up
            self.food[1] > head[1]   # food is down
        ]
        
        return np.array(state, dtype=int)
    
    def _is_collision(self, point):
        # Check for wall collision
        if (point[0] >= self.width or point[0] < 0 or 
            point[1] >= self.height or point[1] < 0):
            return True
        # Check for collision with itself
        if point in self.snake[1:]:
            return True
        return False
    
    def step(self, action):
        self.frame_iteration += 1
        reward = 0
        game_over = False
        
        # Convert one-hot action into an index (0 = right turn, 1 = straight, 2 = left turn)
        action_index = np.argmax(action)
        # Update the snake's direction using the helper function
        self.direction = get_new_direction(self.direction, action_index)
        
        # Move snake: compute new head position based on updated direction
        head = self.snake[0].copy()
        if self.direction == "RIGHT":
            head[0] += self.block_size
        elif self.direction == "LEFT":
            head[0] -= self.block_size
        elif self.direction == "DOWN":
            head[1] += self.block_size
        elif self.direction == "UP":
            head[1] -= self.block_size
            
        # Check for collisions
        if self._is_collision(head):
            game_over = True
            reward = -10
            return reward, game_over, self.score
        
        self.snake.insert(0, head)
        
        # Check if food is eaten
        if head == self.food:
            self.score += 1
            reward = 10
            self.food = self._spawn_food()
        else:
            self.snake.pop()
        
        # Timeout penalty to discourage endless wandering
        if self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10
            
        return reward, game_over, self.score
    
    def render(self):
        self.screen.fill((0, 0, 0))
        
        # Draw snake (head and body)
        for i, segment in enumerate(self.snake):
            color = (0, 128, 128) if i == 0 else (0, 150, 150)
            pygame.draw.rect(self.screen, color, 
                             pygame.Rect(segment[0], segment[1], 
                                         self.block_size, self.block_size))
        
        # Draw food
        pygame.draw.rect(self.screen, (255, 0, 0), 
                         pygame.Rect(self.food[0], self.food[1], 
                                     self.block_size, self.block_size))
        
        pygame.display.flip()

# ---------------------------
# Q-Learning Trainer with Vectorized Training Step
# ---------------------------
class QTrainer:
    def __init__(self, model, lr, gamma):
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        
    def train_step(self, state, action, reward, next_state, game_over):
        # Convert data to tensors
        state      = torch.tensor(np.array(state), dtype=torch.float)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float)
        action     = torch.tensor(np.array(action), dtype=torch.long)
        reward     = torch.tensor(np.array(reward), dtype=torch.float)
        done       = torch.tensor(np.array(game_over), dtype=torch.float)
        
        if len(state.shape) == 1:
            state      = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action     = torch.unsqueeze(action, 0)
            reward     = torch.unsqueeze(reward, 0)
            done       = torch.unsqueeze(done, 0)
            
        # Get predictions for current state (batch, num_actions)
        pred = self.model(state)
        
        # Compute Q-values for the next state in a vectorized manner
        with torch.no_grad():
            q_next = self.model(next_state)
            max_q_next, _ = torch.max(q_next, dim=1)
            
        # Compute target Q values: for terminal states, future reward is 0.
        q_target = reward + (1 - done) * self.gamma * max_q_next
        
        # Gather predicted Q-values corresponding to the taken actions
        batch_indices = torch.arange(action.shape[0])
        action_indices = torch.argmax(action, dim=1)
        
        # Clone predictions and update only the Q-value for the chosen action
        target = pred.clone()
        target[batch_indices, action_indices] = q_target
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()

# ---------------------------
# Training Loop with Experience Replay & Periodic Rendering
# ---------------------------
def train():
    # Training hyperparameters
    MAX_MEMORY = 100_000
    BATCH_SIZE = 1000
    LR = 0.001
    GAMMA = 0.9
    
    # Initialize game, model, trainer, and replay memory
    game = SnakeGameAI()
    model = SnakeAI(11, 256, 3)  # 11 input states, 256 hidden neurons, 3 actions
    trainer = QTrainer(model, lr=LR, gamma=GAMMA)
    memory = deque(maxlen=MAX_MEMORY)
    
    episodes = 1000
    for episode in range(episodes):
        state = game.reset()
        total_reward = 0
        
        while True:
            # Process pygame events to allow clean exit
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
            
            state_old = state
            
            # Epsilon-greedy: linearly decaying epsilon
            epsilon = 1 - episode / episodes
            if random.random() < epsilon:
                move_index = random.randint(0, 2)
            else:
                state_tensor = torch.tensor(state, dtype=torch.float)
                prediction = model(state_tensor)
                move_index = torch.argmax(prediction).item()
            final_move = [0, 0, 0]
            final_move[move_index] = 1
            
            # Execute move and get feedback
            reward, game_over, score = game.step(final_move)
            state_new = game.get_state()
            
            # Short-term training on current move
            trainer.train_step(state_old, final_move, reward, state_new, game_over)
            
            # Save experience into replay memory
            memory.append((state_old, final_move, reward, state_new, game_over))
            
            # Train on a random sample from memory if enough samples are available
            if len(memory) > BATCH_SIZE:
                mini_sample = random.sample(memory, BATCH_SIZE)
                states, actions, rewards, next_states, dones = zip(*mini_sample)
                trainer.train_step(states, actions, rewards, next_states, dones)
            
            state = state_new
            total_reward += reward
            
            # Render only every 10 episodes for performance improvements
            if episode % 10 == 0:
                game.render()
                game.clock.tick(30)
            else:
                game.clock.tick(300)
            
            if game_over:
                print(f'Episode {episode}, Score: {score}, Total Reward: {total_reward}')
                break
            
        # Save model periodically
        if episode % 100 == 0:
            torch.save(model.state_dict(), f'snake_model_{episode}.pth')

if __name__ == '__main__':
    train()
