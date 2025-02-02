import torch
import numpy as np
import pygame
import random
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
        # Place the snake in the middle of the screen
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
        # Define points around the head for collision detection.
        point_l = [head[0] - self.block_size, head[1]]
        point_r = [head[0] + self.block_size, head[1]]
        point_u = [head[0], head[1] - self.block_size]
        point_d = [head[0], head[1] + self.block_size]
        
        # Direction flags
        dir_l = self.direction == "LEFT"
        dir_r = self.direction == "RIGHT"
        dir_u = self.direction == "UP"
        dir_d = self.direction == "DOWN"
        
        # Danger assessments
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
        
        # Food relative to head
        food_left = self.food[0] < head[0]
        food_right = self.food[0] > head[0]
        food_up = self.food[1] < head[1]
        food_down = self.food[1] > head[1]
        
        # Since in play you might not use food type and trap info,
        # you can assume defaults (if your training always used these):
        food_type_regular = 1  # e.g., assume food is always regular
        food_type_special = 0
        food_type_rare = 0
        trap_exists = 0
        trap_left = 0
        trap_right = 0
        trap_up = 0
        trap_down = 0
        trap_warning_flag = 0
        
        # Construct state vector with 20 features
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
        # Check for wall collisions
        if point[0] >= self.width or point[0] < 0 or point[1] >= self.height or point[1] < 0:
            return True
        # Check for self-collision
        if point in self.snake[1:]:
            return True
        return False
    
    def step(self, action):
        self.frame_iteration += 1
        
        # Convert one-hot action into index (0 = turn right, 1 = straight, 2 = turn left)
        action_index = np.argmax(action)
        # Update the snake's direction using the helper function
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
        
        # Check for collisions
        if self._is_collision(new_head):
            return -10, True, self.score
        
        self.snake.insert(0, new_head)
        
        # Check if food is eaten
        if new_head == self.food:
            self.score += 1
            reward = 10
            self.food = self._spawn_food()
        else:
            self.snake.pop()
            reward = 0
        
        # Timeout penalty: if the snake takes too long
        if self.frame_iteration > 100 * len(self.snake):
            return -10, True, self.score
        
        return reward, False, self.score
    
    def render(self):
        self.screen.fill((0, 0, 0))
        # Draw snake
        for i, segment in enumerate(self.snake):
            color = (0, 128, 128) if i == 0 else (0, 150, 150)
            pygame.draw.rect(self.screen, color, 
                             pygame.Rect(segment[0], segment[1], self.block_size, self.block_size))
        # Draw food
        pygame.draw.rect(self.screen, (255, 0, 0), 
                         pygame.Rect(self.food[0], self.food[1], self.block_size, self.block_size))
        pygame.display.flip()

# ---------------------------
# Main Loop for Playing with Trained AI
# ---------------------------
def play():
    # Load the trained model weights
    model = SnakeAI(20, 256, 3)
    model.load_state_dict(torch.load('best_model.pth'))  # Make sure this file exists
    model.eval()  # Set model to evaluation mode
    
    # Initialize the game environment
    game = SnakeGameAI()
    
    state = game.reset()
    running = True
    
    while running:
        # Process events to allow for quitting
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Get the current state and predict the action using the trained model
        state = game.get_state()
        state_tensor = torch.tensor(state, dtype=torch.float)
        with torch.no_grad():
            action_values = model(state_tensor)
        action_index = torch.argmax(action_values).item()
        
        # Convert action index to one-hot vector
        final_move = [0, 0, 0]
        final_move[action_index] = 1
        
        # Apply the action to the game environment
        reward, game_over, score = game.step(final_move)
        
        # Render the game
        game.render()
        game.clock.tick(30)
        
        if game_over:
            print(f"Game Over! Final Score: {score}")
            state = game.reset()
    
    pygame.quit()

if __name__ == '__main__':
    play()
