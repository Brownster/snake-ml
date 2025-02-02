import pygame
import random
import time
import math
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# ---------------------------
# Helper: Direction Mapping
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
# Environment: Enhanced SnakeGameAI
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

        # Trap-related variables
        self.trap = None
        self.trap_warning = False
        self.last_trap_time = time.time()
        self.warning_start = 0

        return self.get_state()

    def _spawn_food(self):
        # Define food types with properties
        FOOD_TYPES = {
            'regular': {'color': (255, 0, 0), 'points': 10, 'chance': 70},
            'special': {'color': (255, 215, 0), 'points': 20, 'chance': 20},
            'rare':    {'color': (0, 0, 255), 'points': 50, 'chance': 10}
        }
        # Choose food type based on chance
        rand = random.randint(1, 100)
        cumulative = 0
        food_type = 'regular'
        for t, props in FOOD_TYPES.items():
            cumulative += props['chance']
            if rand <= cumulative:
                food_type = t
                break

        # Spawn food at a random position not occupied by the snake (or trap)
        while True:
            x = random.randrange(0, self.width, self.block_size)
            y = random.randrange(0, self.height, self.block_size)
            if [x, y] not in self.snake and (self.trap is None or [x, y] != self.trap):
                return {
                    'position': [x, y],
                    'type': food_type,
                    'points': FOOD_TYPES[food_type]['points'],
                    'spawn_time': time.time()
                }

    def _spawn_trap(self):
        # Spawn a trap at a random location not occupied by the snake or the current food
        while True:
            x = random.randrange(0, self.width, self.block_size)
            y = random.randrange(0, self.height, self.block_size)
            if [x, y] not in self.snake and (self.food is None or [x, y] != self.food['position']):
                return [x, y]

    def get_state(self):
        head = self.snake[0]

        # Points adjacent to the head (for collision/danger checks)
        point_l = [head[0] - self.block_size, head[1]]
        point_r = [head[0] + self.block_size, head[1]]
        point_u = [head[0], head[1] - self.block_size]
        point_d = [head[0], head[1] + self.block_size]

        # Current direction booleans
        dir_l = self.direction == "LEFT"
        dir_r = self.direction == "RIGHT"
        dir_u = self.direction == "UP"
        dir_d = self.direction == "DOWN"

        # Basic danger flags (if a collision would occur moving straight/right/left)
        danger_straight = ((dir_r and self._is_collision(point_r)) or
                           (dir_l and self._is_collision(point_l)) or
                           (dir_u and self._is_collision(point_u)) or
                           (dir_d and self._is_collision(point_d)))

        danger_right = ((dir_u and self._is_collision(point_r)) or
                        (dir_d and self._is_collision(point_l)) or
                        (dir_l and self._is_collision(point_u)) or
                        (dir_r and self._is_collision(point_d)))

        danger_left = ((dir_d and self._is_collision(point_r)) or
                       (dir_u and self._is_collision(point_l)) or
                       (dir_r and self._is_collision(point_u)) or
                       (dir_l and self._is_collision(point_d)))

        # Food relative location flags
        food_left = self.food['position'][0] < head[0]
        food_right = self.food['position'][0] > head[0]
        food_up = self.food['position'][1] < head[1]
        food_down = self.food['position'][1] > head[1]

        # Food type one-hot encoding
        food_type_regular = 1 if self.food['type'] == 'regular' else 0
        food_type_special = 1 if self.food['type'] == 'special' else 0
        food_type_rare = 1 if self.food['type'] == 'rare' else 0

        # Trap-related information
        trap_exists = 1 if self.trap is not None else 0
        if self.trap is not None:
            trap_left = 1 if self.trap[0] < head[0] else 0
            trap_right = 1 if self.trap[0] > head[0] else 0
            trap_up = 1 if self.trap[1] < head[1] else 0
            trap_down = 1 if self.trap[1] > head[1] else 0
        else:
            trap_left = trap_right = trap_up = trap_down = 0

        # Trap warning flag (active during the 2-second warning phase)
        trap_warning_flag = 1 if self.trap_warning else 0

        # Assemble state vector (20 features)
        state = [
            int(danger_straight),
            int(danger_right),
            int(danger_left),
            int(dir_l),
            int(dir_r),
            int(dir_u),
            int(dir_d),
            int(food_left),
            int(food_right),
            int(food_up),
            int(food_down),
            food_type_regular,
            food_type_special,
            food_type_rare,
            trap_exists,
            trap_left,
            trap_right,
            trap_up,
            trap_down,
            trap_warning_flag
        ]
        return np.array(state, dtype=int)

    def _is_collision(self, point):
        # Check if a point collides with the wall or with the snake's own body
        if (point[0] >= self.width or point[0] < 0 or 
            point[1] >= self.height or point[1] < 0):
            return True
        if point in self.snake[1:]:
            return True
        return False

    def step(self, action):
        """
        Takes an action (a one-hot list of length 3: [turn right, straight, turn left])
        and updates the environment.
        Returns: reward, game_over, score.
        """
        self.frame_iteration += 1
        reward = 0
        game_over = False
        current_time = time.time()

        # ---- Trap Logic ----
        if self.trap is None and current_time - self.last_trap_time >= 10:
            # Begin warning phase for 2 seconds before spawning trap
            if not self.trap_warning:
                self.trap_warning = True
                self.warning_start = current_time
            elif current_time - self.warning_start >= 2:
                self.trap = self._spawn_trap()
                self.trap_warning = False
                self.last_trap_time = current_time

        # ---- Food Expiration ----
        if self.food and current_time - self.food['spawn_time'] > 10:
            self.food = self._spawn_food()

        # ---- Compute Distance to Food ----
        prev_food_dist = math.dist(self.snake[0], self.food['position'])

        # ---- Update Direction Based on Action ----
        action_index = np.argmax(action)
        self.direction = get_new_direction(self.direction, action_index)

        new_head = self.snake[0].copy()
        if self.direction == "RIGHT":
            new_head[0] += self.block_size
        elif self.direction == "LEFT":
            new_head[0] -= self.block_size
        elif self.direction == "DOWN":
            new_head[1] += self.block_size
        elif self.direction == "UP":
            new_head[1] -= self.block_size

        # ---- Check for Collisions ----
        if self._is_collision(new_head):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # ---- Check for Trap Collision ----
        if self.trap and new_head == self.trap:
            # Hitting a trap penalizes the snake
            reward -= 20
            self.trap = None
            self.last_trap_time = current_time

        # ---- Proximity Reward ----
        new_food_dist = math.dist(new_head, self.food['position'])
        reward += (prev_food_dist - new_food_dist) * 0.2
        reward -= 0.1

        # ---- Move Snake ----
        self.snake.insert(0, new_head)

        # ---- Check if Food is Eaten ----
        if new_head == self.food['position']:
            self.score += self.food['points']
            reward += self.food['points']  # reward proportional to food type
            self.food = self._spawn_food()
        else:
            self.snake.pop()

        # ---- Timeout Penalty ----
        if self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10

        return reward, game_over, self.score

    def render(self):
        # Draw the environment
        self.screen.fill((0, 0, 0))

        # Draw snake segments
        for i, segment in enumerate(self.snake):
            color = (0, 128, 128) if i == 0 else (0, 150, 150)
            pygame.draw.rect(self.screen, color,
                             pygame.Rect(segment[0], segment[1], self.block_size, self.block_size))

        # Draw food with color based on type
        food_color = (255, 0, 0)  # default regular
        if self.food['type'] == 'special':
            food_color = (255, 215, 0)
        elif self.food['type'] == 'rare':
            food_color = (0, 0, 255)
        pygame.draw.rect(self.screen, food_color,
                         pygame.Rect(self.food['position'][0], self.food['position'][1],
                                     self.block_size, self.block_size))

        # Draw trap if it exists
        if self.trap:
            pygame.draw.rect(self.screen, (148, 0, 211),
                             pygame.Rect(self.trap[0], self.trap[1], self.block_size, self.block_size))

        # Optionally, display trap warning
        if self.trap_warning:
            font = pygame.font.SysFont("Arial", 25)
            text_surface = font.render("TRAP WARNING", True, (255, 165, 0))
            self.screen.blit(text_surface, (self.width // 2 - 50, self.height - 30))

        pygame.display.flip()

# ---------------------------
# Neural Network for Snake AI
# ---------------------------
class SnakeAI(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SnakeAI, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = self.linear3(x)
        return x

# ---------------------------
# Q-Learning Trainer with Target Network and Gradient Clipping
# ---------------------------
class QTrainer:
    def __init__(self, model, lr, gamma):
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        # Create target network using the same architecture as model.
        self.target_model = SnakeAI(model.input_size, model.hidden_size, model.output_size)
        self.target_model.load_state_dict(model.state_dict())
        self.target_update_counter = 0

    def train_step(self, state, action, reward, next_state, game_over):
        # Convert data to tensors (if not already in batched form)
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
        
        # Use target network for next state Q-values
        with torch.no_grad():
            q_next = self.target_model(next_state)
            max_q_next, _ = torch.max(q_next, dim=1)
        
        # Compute target Q values (for terminal states, future reward is 0)
        q_target = reward + (1 - done) * self.gamma * max_q_next
        
        # Gather predicted Q-values for the actions taken
        batch_indices = torch.arange(action.shape[0])
        action_indices = torch.argmax(action, dim=1)
        
        target = pred.clone()
        target[batch_indices, action_indices] = q_target
        
        self.optimizer.zero_grad()
        loss = self.criterion(pred, target)
        loss.backward()
        # Gradient clipping to prevent exploding gradients
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Update target network every 100 training steps.
        self.target_update_counter += 1
        if self.target_update_counter % 100 == 0:
            self.target_model.load_state_dict(self.model.state_dict())

# ---------------------------
# Training Loop with Experience Replay, Optimized Rendering, and TensorBoard Logging
# ---------------------------
def train():
    # Training hyperparameters
    MAX_MEMORY = 100_000
    BATCH_SIZE = 1000
    LR = 0.001
    GAMMA = 0.9
    
    # Initialize game, model, trainer, and replay memory
    # NOTE: The model's input size is now 20 to match the enhanced state representation.
    game = SnakeGameAI()
    model = SnakeAI(20, 256, 3)  # 20 input features, 256 hidden neurons, 3 actions
    trainer = QTrainer(model, lr=LR, gamma=GAMMA)
    memory = deque(maxlen=MAX_MEMORY)
    
    episodes = 1000
    best_score = 0
    writer = SummaryWriter()  # TensorBoard writer
    
    for episode in range(episodes):
        state = game.reset()
        total_reward = 0
        step_count = 0
        
        # Improved epsilon: minimum exploration rate is 0.01
        epsilon = max(0.01, 1 - episode / (episodes * 0.8))
        
        while True:
            step_count += 1
            
            # Process pygame events to allow a clean exit
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
            
            state_old = state
            
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action_index = random.randint(0, 2)
            else:
                with torch.no_grad():
                    state_tensor = torch.tensor(state, dtype=torch.float)
                    action_index = torch.argmax(model(state_tensor)).item()
            final_move = [0, 0, 0]
            final_move[action_index] = 1
            
            # Execute move and get feedback from the environment
            reward, game_over, score = game.step(final_move)
            state_new = game.get_state()
            
            # Short-term training on current move
            trainer.train_step(state_old, final_move, reward, state_new, game_over)
            
            # Save experience into replay memory
            memory.append((state_old, final_move, reward, state_new, game_over))
            
            # Train on a random sample from replay memory if enough samples are available
            if len(memory) > BATCH_SIZE:
                mini_sample = random.sample(memory, BATCH_SIZE)
                states, actions, rewards, next_states, dones = zip(*mini_sample)
                trainer.train_step(states, actions, rewards, next_states, dones)
            
            state = state_new
            total_reward += reward
            
            # Optimized rendering: render every 10 steps for episodes divisible by 50
            if episode % 50 == 0 and step_count % 10 == 0:
                game.render()
                game.clock.tick(30)
            else:
                game.clock.tick(300)
            
            if game_over:
                # Save the best model based on score
                if score > best_score:
                    best_score = score
                    torch.save(model.state_dict(), 'best_model.pth')
                
                print(f'Episode {episode}, Score: {score}, Total Reward: {total_reward}')
                writer.add_scalar('Score', score, episode)
                writer.add_scalar('Total Reward', total_reward, episode)
                break
        
        # Optional: Curriculum learning by adjusting difficulty after some episodes
        if episode > 300:
            game.block_size = 15
        
        # Periodically save model checkpoints
        if episode % 100 == 0:
            torch.save(model.state_dict(), f'snake_model_{episode}.pth')
            
    writer.close()

if __name__ == '__main__':
    train()
