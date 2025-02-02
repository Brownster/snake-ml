import torch
import numpy as np
import pygame
import random
import math
import time

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
class SnakeAI(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SnakeAI, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.linear1 = torch.nn.Linear(input_size, hidden_size)
        self.linear2 = torch.nn.Linear(hidden_size, hidden_size)
        self.linear3 = torch.nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = self.linear3(x)
        return x

# ---------------------------
# Snake Game Environment for AI Play
# ---------------------------
class SnakeGameAI:
    def __init__(self, width=600, height=400, block_size=20):
        self.width = width
        self.height = height
        self.block_size = block_size

        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Snake AI Play")
        self.clock = pygame.time.Clock()

        self.reset()

    def reset(self):
        # Initialize snake in the middle of the screen
        self.snake = [[self.width // 2, self.height // 2]]
        self.direction = "RIGHT"
        
        # Initialize trap-related variables
        self.trap = None
        self.trap_warning = False
        self.last_trap_time = time.time()
        self.warning_start = 0
        
        # Spawn food using enhanced food spawning
        self.food = self._spawn_food()
        self.score = 0
        self.frame_iteration = 0
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
                x = head_x + dx * self.block_size
                y = head_y + dy * self.block_size
                if x < 0 or x >= self.width or y < 0 or y >= self.height:
                    vision.append(3)
                elif [x, y] in self.snake:
                    vision.append(1)
                elif self.food and [x, y] == self.food['position']:
                    vision.append(2)
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
        max_blocks = max(self.width, self.height) / self.block_size
        return np.array([dist_left, dist_right, dist_up, dist_down]) / max_blocks
    
    def get_state(self):
        head = self.snake[0]
        
        # Adjacent points for collision/danger checks.
        point_l = [head[0] - self.block_size, head[1]]
        point_r = [head[0] + self.block_size, head[1]]
        point_u = [head[0], head[1] - self.block_size]
        point_d = [head[0], head[1] + self.block_size]
        
        # Direction flags.
        dir_l = self.direction == "LEFT"
        dir_r = self.direction == "RIGHT"
        dir_u = self.direction == "UP"
        dir_d = self.direction == "DOWN"
        
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
        
        food_left = self.food['position'][0] < head[0]
        food_right = self.food['position'][0] > head[0]
        food_up = self.food['position'][1] < head[1]
        food_down = self.food['position'][1] > head[1]
        
        food_type_regular = 1 if self.food['type'] == 'regular' else 0
        food_type_special = 1 if self.food['type'] == 'special' else 0
        food_type_rare = 1 if self.food['type'] == 'rare' else 0
        
        trap_exists = 1 if self.trap is not None else 0
        if self.trap is not None:
            trap_left = 1 if self.trap[0] < head[0] else 0
            trap_right = 1 if self.trap[0] > head[0] else 0
            trap_up = 1 if self.trap[1] < head[1] else 0
            trap_down = 1 if self.trap[1] > head[1] else 0
        else:
            trap_left = trap_right = trap_up = trap_down = 0
        trap_warning_flag = 1 if self.trap_warning else 0
        
        # Additional features: snake length and wall distances.
        snake_length = len(self.snake)
        wall_distances = self.get_wall_distances()  # 4 features
        
        # Base state (21 features).
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
            snake_length
        ], dtype=int)
        
        # Vision window (5x5 grid => 25 features).
        vision = self.get_vision(window_size=5)
        full_state = np.concatenate((base_state, vision, wall_distances))
        # Total features = 21 + 25 + 4 = 50.
        return full_state
    
    def _is_collision(self, point):
        if point[0] >= self.width or point[0] < 0 or point[1] >= self.height or point[1] < 0:
            return True
        if point in self.snake[1:]:
            return True
        return False
    
    def step(self, action):
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
        
        self.snake.insert(0, new_head)
        
        # ---- Check for Trap Collision ----
        if self.trap and new_head == self.trap:
            reward -= 20
            game_over = True
            self.trap = None
            self.last_trap_time = current_time
            return reward, game_over, self.score
        
        # ---- Proximity Reward ----
        new_food_dist = math.dist(new_head, self.food['position'])
        reward += (prev_food_dist - new_food_dist) * 0.2
        reward -= 0.1
        
        # ---- Move Snake ----
        if new_head == self.food['position']:
            self.score += self.food['points']
            reward += self.food['points']
            self.food = self._spawn_food()
        else:
            self.snake.pop()
        
        # ---- Timeout Penalty ----
        if self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10
        
        return reward, game_over, self.score
    
    def render(self):
        self.screen.fill((0, 0, 0))
        for i, segment in enumerate(self.snake):
            color = (0, 128, 128) if i == 0 else (0, 150, 150)
            pygame.draw.rect(self.screen, color,
                             pygame.Rect(segment[0], segment[1], self.block_size, self.block_size))
        food_color = (255, 0, 0)
        if self.food['type'] == 'special':
            food_color = (255, 215, 0)
        elif self.food['type'] == 'rare':
            food_color = (0, 0, 255)
        pygame.draw.rect(self.screen, food_color,
                         pygame.Rect(self.food['position'][0], self.food['position'][1],
                                     self.block_size, self.block_size))
        if self.trap:
            pygame.draw.rect(self.screen, (148, 0, 211),
                             pygame.Rect(self.trap[0], self.trap[1], self.block_size, self.block_size))
        if self.trap_warning:
            font = pygame.font.SysFont("Arial", 25)
            text_surface = font.render("TRAP WARNING", True, (255, 165, 0))
            self.screen.blit(text_surface, (self.width // 2 - 50, self.height - 30))
        pygame.display.flip()

# ---------------------------
# Main Loop for Playing with Trained AI
# ---------------------------
def play():
    # Instantiate model with input size of 50 to match the enhanced state representation
    model = SnakeAI(50, 256, 3)
    model.load_state_dict(torch.load('best_model.pth'))  # Ensure the correct checkpoint exists
    model.eval()  # Set model to evaluation mode

    game = SnakeGameAI()
    state = game.reset()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        state = game.get_state()
        state_tensor = torch.tensor(state, dtype=torch.float)
        with torch.no_grad():
            action_values = model(state_tensor)
        action_index = torch.argmax(action_values).item()
        final_move = [0, 0, 0]
        final_move[action_index] = 1

        reward, game_over, score = game.step(final_move)
        game.render()
        game.clock.tick(30)

        if game_over:
            print(f"Game Over! Final Score: {score}")
            state = game.reset()

    pygame.quit()

if __name__ == '__main__':
    play()
