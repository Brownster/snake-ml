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
    def __init__(self, width=600, height=400, block_size=20, render=True):
        self.width = width
        self.height = height
        self.block_size = block_size
        self.RENDER = render  # Toggle rendering

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
        
        # Initialize trap-related variables BEFORE spawning food
        self.trap = None
        self.trap_warning = False
        self.last_trap_time = time.time()
        self.warning_start = 0
    
        # Spawn food (ensuring it doesn't overlap snake or trap)
        self.food = self._spawn_food()
        
        self.score = 0
        self.frame_iteration = 0
        self.last_food_time = time.time()  # For additional reward shaping if needed
        return self.get_state()

    def _spawn_food(self):
        # Define food types with properties
        FOOD_TYPES = {
            'regular': {'color': (255, 0, 0), 'points': 10, 'chance': 70},
            'special': {'color': (255, 215, 0), 'points': 20, 'chance': 20},
            'rare':    {'color': (0, 0, 255), 'points': 50, 'chance': 10}
        }
        rand = random.randint(1, 100)
        cumulative = 0
        food_type = 'regular'
        for t, props in FOOD_TYPES.items():
            cumulative += props['chance']
            if rand <= cumulative:
                food_type = t
                break

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
        # Spawn a trap at a random location not occupied by the snake or current food
        while True:
            x = random.randrange(0, self.width, self.block_size)
            y = random.randrange(0, self.height, self.block_size)
            if [x, y] not in self.snake and (self.food is None or [x, y] != self.food['position']):
                return [x, y]

    def get_vision(self, window_size=5):
        """
        Returns a flattened vision grid centered on the snake's head.
        Encoding:
          0: empty, 1: snake body, 2: food, 3: wall.
        """
        half = window_size // 2
        vision = []
        head_x, head_y = self.snake[0]
        for dy in range(-half, half + 1):
            for dx in range(-half, half + 1):
                # Use block_size multiples for grid cells
                x = head_x + dx * self.block_size
                y = head_y + dy * self.block_size
                if x < 0 or x >= self.width or y < 0 or y >= self.height:
                    vision.append(3)  # wall
                elif [x, y] in self.snake:
                    vision.append(1)  # snake body
                elif self.food and [x, y] == self.food['position']:
                    vision.append(2)  # food
                else:
                    vision.append(0)
        return np.array(vision, dtype=int)

    def get_wall_distances(self):
        """
        Returns normalized distances from the snake's head to the walls in four directions.
        """
        head_x, head_y = self.snake[0]
        dist_left = head_x / self.block_size
        dist_right = (self.width - head_x) / self.block_size
        dist_up = head_y / self.block_size
        dist_down = (self.height - head_y) / self.block_size
        # Normalize by maximum number of blocks (approximate)
        max_blocks = max(self.width, self.height) / self.block_size
        return np.array([dist_left, dist_right, dist_up, dist_down]) / max_blocks

    def get_state(self):
        head = self.snake[0]
        
        # Adjacent points for collision/danger checks
        point_l = [head[0] - self.block_size, head[1]]
        point_r = [head[0] + self.block_size, head[1]]
        point_u = [head[0], head[1] - self.block_size]
        point_d = [head[0], head[1] + self.block_size]

        # Direction flags
        dir_l = self.direction == "LEFT"
        dir_r = self.direction == "RIGHT"
        dir_u = self.direction == "UP"
        dir_d = self.direction == "DOWN"

        # Basic danger flags (for immediate adjacent cells)
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
        trap_warning_flag = 1 if self.trap_warning else 0

        # Additional features: snake length and wall distances
        snake_length = len(self.snake)
        wall_distances = self.get_wall_distances()  # 4 features

        # Base state (20 features)
        base_state = np.array([
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
            trap_warning_flag,
            snake_length  # new feature: snake length
        ], dtype=int)
        # Now base_state has 21 features.

        # Get vision window (5x5 grid => 25 features)
        vision = self.get_vision(window_size=5)
        # Concatenate wall distances (4 features)
        full_state = np.concatenate((base_state, vision, wall_distances))
        # Total features: 21 + 25 + 4 = 50.
        return full_state

    def _is_collision(self, point):
        # Check if a point collides with the wall or snake's own body.
        if point[0] >= self.width or point[0] < 0 or point[1] >= self.height or point[1] < 0:
            return True
        if point in self.snake[1:]:
            return True
        return False

    def step(self, action):
        """
        Executes an action (one-hot list [turn right, straight, turn left])
        and updates the environment.
        Returns: reward, game_over, score.
        """
        self.frame_iteration += 1
        reward = 0
        game_over = False
        current_time = time.time()

        # ---- Trap Logic ----
        if self.trap is None and current_time - self.last_trap_time >= 10:
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

        # ---- Compute Distance to Food (for reward shaping)
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
            reward -= 20
            # End the game on trap collision (as suggested)
            game_over = True
            self.trap = None
            self.last_trap_time = current_time
            return reward, game_over, self.score

        # ---- Proximity Reward ----
        new_food_dist = math.dist(new_head, self.food['position'])
        reward += (prev_food_dist - new_food_dist) * 0.2
        reward -= 0.1

        # ---- Move Snake ----
        self.snake.insert(0, new_head)

        # ---- Check if Food is Eaten ----
        if new_head == self.food['position']:
            self.score += self.food['points']
            reward += self.food['points']
            self.food = self._spawn_food()
            self.last_food_time = current_time
        else:
            self.snake.pop()

        # ---- Timeout Penalty ----
        if self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10

        return reward, game_over, self.score

    def render(self):
        if not self.RENDER:
            return
        self.screen.fill((0, 0, 0))

        # Draw snake segments
        for i, segment in enumerate(self.snake):
            color = (0, 128, 128) if i == 0 else (0, 150, 150)
            pygame.draw.rect(self.screen, color,
                             pygame.Rect(segment[0], segment[1], self.block_size, self.block_size))

        # Draw food with color based on type
        food_color = (255, 0, 0)  # regular
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

        # Display trap warning if active
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
        # Convert data to tensors (if not already batched)
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
        
        # Get predictions for current state
        pred = self.model(state)
        
        # Use target network for next state Q-values
        with torch.no_grad():
            q_next = self.target_model(next_state)
            max_q_next, _ = torch.max(q_next, dim=1)
        
        # Compute target Q values (future reward is zero for terminal states)
        q_target = reward + (1 - done) * self.gamma * max_q_next
        
        # Gather predicted Q-values for the taken actions
        batch_indices = torch.arange(action.shape[0])
        action_indices = torch.argmax(action, dim=1)
        
        target = pred.clone()
        target[batch_indices, action_indices] = q_target
        
        self.optimizer.zero_grad()
        loss = self.criterion(pred, target)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Update target network every 100 training steps.
        self.target_update_counter += 1
        if self.target_update_counter % 100 == 0:
            self.target_model.load_state_dict(self.model.state_dict())
        
        return loss.item()

# ---------------------------
# Training Loop with Experience Replay, Optimized Rendering, and TensorBoard Logging
# ---------------------------
def train():
    # Hyperparameters
    MAX_MEMORY = 100_000
    BATCH_SIZE = 1000
    LR = 0.001
    GAMMA = 0.9
    episodes = 2000  # Set to 2000 episodes

    # Initialize game, model, trainer, and replay memory
    game = SnakeGameAI()
    # Input size is now 50 (21 base + 25 vision + 4 wall distances)
    model = SnakeAI(50, 256, 3)
    trainer = QTrainer(model, lr=LR, gamma=GAMMA)
    memory = deque(maxlen=MAX_MEMORY)
    
    best_score = 0
    writer = SummaryWriter()  # TensorBoard writer
    
    # Use an exponential decay for epsilon
    initial_epsilon = 1.0
    min_epsilon = 0.01
    decay_rate = 0.001  # Adjust this rate as needed
    
    for episode in range(episodes):
        state = game.reset()
        total_reward = 0
        step_count = 0
        
        # Compute current epsilon
        epsilon = max(min_epsilon, initial_epsilon * math.exp(-decay_rate * episode))
        
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
            
            # Execute move and get feedback from environment
            reward, game_over, score = game.step(final_move)
            state_new = game.get_state()
            
            # Short-term training on current move
            loss = trainer.train_step(state_old, final_move, reward, state_new, game_over)
            
            # Save experience into replay memory
            memory.append((state_old, final_move, reward, state_new, game_over))
            
            # Train on a random sample from replay memory if enough samples exist
            if len(memory) > BATCH_SIZE:
                mini_sample = random.sample(memory, BATCH_SIZE)
                states, actions, rewards, next_states, dones = zip(*mini_sample)
                trainer.train_step(states, actions, rewards, next_states, dones)
            
            state = state_new
            total_reward += reward
            
            # Rendering optimization: toggle rendering based on a flag or episode frequency
            if episode % 50 == 0 and step_count % 10 == 0:
                game.render()
                game.clock.tick(30)
            else:
                game.clock.tick(300)
            
            if game_over:
                if score > best_score:
                    best_score = score
                    torch.save(model.state_dict(), 'best_model.pth')
                
                print(f'Episode {episode}, Score: {score}, Total Reward: {total_reward}, Epsilon: {epsilon:.3f}, Loss: {loss:.4f}')
                writer.add_scalar('Score', score, episode)
                writer.add_scalar('Total Reward', total_reward, episode)
                writer.add_scalar('Epsilon', epsilon, episode)
                writer.add_scalar('Loss', loss, episode)
                break
        
        # Optional: Curriculum learning – for example, reducing block_size after 300 episodes
        if episode > 300:
            game.block_size = 15
        
        # Periodically save model checkpoints
        if episode % 100 == 0:
            torch.save(model.state_dict(), f'snake_model_{episode}.pth')
            
    writer.close()

if __name__ == '__main__':
    train()
