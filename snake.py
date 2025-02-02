import pygame
import sys
import random
import time
import math

# Initialize Pygame and Mixer
pygame.init()
pygame.mixer.init()

# ---------------------------
# Game Settings and Constants
# ---------------------------
WINDOW_WIDTH = 600
WINDOW_HEIGHT = 400
BLOCK_SIZE = 20
INITIAL_FPS = 10

# Colors (R, G, B)
BLACK      = (0, 0, 0)
WHITE      = (255, 255, 255)
TEAL       = (0, 128, 128)        # Snake head
LIGHT_TEAL = (0, 150, 150)        # Snake body
RED        = (255, 0, 0)          # Regular food
GOLD       = (255, 215, 0)        # Special food
BLUE       = (0, 0, 255)          # Rare food
PURPLE     = (148, 0, 211)        # Trap
ORANGE     = (255, 165, 0)        # Warning / trap hit flash
YELLOW     = (255, 255, 0)

# Set up the game window
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Enhanced Snake Game")

# Set up the clock and font
clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 25)

# Load sound effects (ensure these files exist in your directory)
try:
    FOOD_SOUND = pygame.mixer.Sound('bite.wav')
    TRAP_SOUND = pygame.mixer.Sound('trap.wav')
except Exception as e:
    print("Sound files not found or error loading sounds:", e)
    FOOD_SOUND = None
    TRAP_SOUND = None

# Food Types and their properties
FOOD_TYPES = {
    'regular': {'color': RED, 'points': 10, 'chance': 70},
    'special': {'color': GOLD, 'points': 20, 'chance': 20},
    'rare':    {'color': BLUE, 'points': 50, 'chance': 10}
}

# ---------------------------
# Helper Classes
# ---------------------------
class FoodItem:
    def __init__(self, position, food_type):
        self.position = position
        self.type = food_type
        self.properties = FOOD_TYPES[food_type]
        self.spawn_time = time.time()

# ---------------------------
# Main Game Class
# ---------------------------
class Game:
    def __init__(self):
        self.reset_game()
    
    def reset_game(self):
        self.snake = [[WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2]]
        self.direction = "RIGHT"
        self.change_to = self.direction
        self.trap = None  # Initialize trap here before calling spawn_food()
        self.food = self.spawn_food()  # Now spawn_food can safely check self.trap
        self.score = 0
        self.high_score = self.load_high_score()
        self.fps = INITIAL_FPS
        self.last_trap_time = time.time()
        self.trap_warning = False
        self.warning_start = 0
        # For trap hit visual feedback
        self.trap_hit_effect = False
        self.effect_start = 0
        # For snake growth animation
        self.growth_effect = False
        self.growth_effect_start = 0
        self.growth_effect_pos = None


    def load_high_score(self):
        try:
            with open('high_score.txt', 'r') as f:
                return int(f.read())
        except Exception:
            return 0

    def save_high_score(self):
        try:
            new_high = max(self.score, self.high_score)
            if new_high > self.high_score:
                self.high_score = new_high
            with open('high_score.txt', 'w') as f:
                f.write(str(self.high_score))
        except Exception as e:
            print(f"Error saving high score: {e}")

    def spawn_food(self):
        max_attempts = 100
        for _ in range(max_attempts):
            # Determine food type based on chances
            rand = random.randint(1, 100)
            cumulative = 0
            selected_type = 'regular'
            for food_type, props in FOOD_TYPES.items():
                cumulative += props['chance']
                if rand <= cumulative:
                    selected_type = food_type
                    break

            x = random.randrange(0, WINDOW_WIDTH, BLOCK_SIZE)
            y = random.randrange(0, WINDOW_HEIGHT, BLOCK_SIZE)
            # Ensure the new food does not overlap the snake or the trap.
            if [x, y] not in self.snake and ([x, y] != self.trap if self.trap else True):
                return FoodItem([x, y], selected_type)
        # If no valid position is found after many attempts, end the game.
        self.game_over()
        return None

    def spawn_trap(self):
        while True:
            x = random.randrange(0, WINDOW_WIDTH, BLOCK_SIZE)
            y = random.randrange(0, WINDOW_HEIGHT, BLOCK_SIZE)
            if [x, y] not in self.snake and (self.food is None or [x, y] != self.food.position):
                return [x, y]

    def handle_input(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP and self.direction != "DOWN":
                self.change_to = "UP"
            elif event.key == pygame.K_DOWN and self.direction != "UP":
                self.change_to = "DOWN"
            elif event.key == pygame.K_LEFT and self.direction != "RIGHT":
                self.change_to = "LEFT"
            elif event.key == pygame.K_RIGHT and self.direction != "LEFT":
                self.change_to = "RIGHT"
            elif event.key == pygame.K_p:
                self.pause_game()

    def update(self):
        current_time = time.time()

        # --- Trap Timing & Warning ---
        if self.trap is None and current_time - self.last_trap_time >= 10:
            # Begin warning phase for 2 seconds before trap spawns.
            if not self.trap_warning:
                self.trap_warning = True
                self.warning_start = current_time
            elif current_time - self.warning_start >= 2:
                self.trap = self.spawn_trap()
                self.trap_warning = False
                self.last_trap_time = current_time

        # --- Food Expiration with Position Validation ---
        if self.food and current_time - self.food.spawn_time > 10:
            new_food = self.spawn_food()
            if new_food:  # Only replace if valid food spawned
                self.food = new_food
            else:
                # End game if spawn fails
                self.game_over()
                return False

        # Update snake movement
        self.direction = self.change_to
        head_x, head_y = self.snake[0]
        if self.direction == "UP":
            head_y -= BLOCK_SIZE
        elif self.direction == "DOWN":
            head_y += BLOCK_SIZE
        elif self.direction == "LEFT":
            head_x -= BLOCK_SIZE
        elif self.direction == "RIGHT":
            head_x += BLOCK_SIZE
        
        new_head = [head_x, head_y]
        
        # Check for collisions with boundaries or self
        if (head_x < 0 or head_x >= WINDOW_WIDTH or
            head_y < 0 or head_y >= WINDOW_HEIGHT or
            new_head in self.snake):
            self.game_over()
            return False

        self.snake.insert(0, new_head)

        # --- Check for Trap Collision with Visual & Sound Feedback ---
        if self.trap and new_head == self.trap:
            self.trap_hit_effect = True
            self.effect_start = current_time
            if TRAP_SOUND:
                TRAP_SOUND.play()
            remove_count = max(1, len(self.snake) // 5)
            # Remove a portion from the tail
            self.snake = self.snake[:-remove_count]
            self.trap = None
            self.last_trap_time = current_time
            self.score = max(0, self.score - 20)

        # --- Check for Food Collision and Trigger Growth Animation ---
        elif self.food and new_head == self.food.position:
            if FOOD_SOUND:
                FOOD_SOUND.play()
            self.score += self.food.properties['points']
            self.growth_effect = True
            self.growth_effect_start = current_time
            self.growth_effect_pos = self.food.position.copy()  # store position for the animation
            self.food = self.spawn_food()
            # Gradual speed increase: increase fps as score increases (capped at 20)
            self.fps = min(INITIAL_FPS + (self.score // 10), 20)
        else:
            self.snake.pop()
        
        return True

    def draw(self):
        # If a trap was hit recently, flash the screen with ORANGE for 1 second.
        if self.trap_hit_effect:
            if time.time() - self.effect_start < 1:
                screen.fill(ORANGE)
            else:
                self.trap_hit_effect = False
                screen.fill(BLACK)
        else:
            screen.fill(BLACK)
        
        # --- Draw Trap Warning ---
        if self.trap_warning:
            self.draw_text("TRAP INCOMING!", WINDOW_WIDTH // 2 - 100, WINDOW_HEIGHT - 30, ORANGE)
            # Also display the countdown (2-second warning)
            countdown = max(0, 2 - int(time.time() - self.warning_start))
            self.draw_text(f"Trap in: {countdown}s", WINDOW_WIDTH - 150, WINDOW_HEIGHT - 30, ORANGE)
        else:
            # If no trap is active, show countdown until next trap.
            time_left = max(0, 10 - int(time.time() - self.last_trap_time))
            self.draw_text(f"Trap in: {time_left}s", WINDOW_WIDTH - 150, WINDOW_HEIGHT - 30, WHITE)
        
        # --- Draw Trap with Pulsating Effect ---
        if self.trap:
            # Calculate a pulsating scale factor between 0.9 and 1.1.
            scale = 1 + 0.1 * math.sin(time.time() * 5)
            size = BLOCK_SIZE * scale
            offset = (size - BLOCK_SIZE) / 2
            trap_rect = pygame.Rect(
                self.trap[0] - offset,
                self.trap[1] - offset,
                size,
                size
            )
            pygame.draw.rect(screen, PURPLE, trap_rect)
        
        # --- Draw Snake ---
        # Draw the head
        pygame.draw.rect(screen, TEAL, pygame.Rect(
            self.snake[0][0], self.snake[0][1], BLOCK_SIZE, BLOCK_SIZE))
        # Draw the body
        for segment in self.snake[1:]:
            pygame.draw.rect(screen, LIGHT_TEAL, pygame.Rect(
                segment[0], segment[1], BLOCK_SIZE, BLOCK_SIZE))
        
        # --- Draw Food ---
        if self.food:
            pygame.draw.rect(screen, self.food.properties['color'], pygame.Rect(
                self.food.position[0], self.food.position[1], BLOCK_SIZE, BLOCK_SIZE))
            # If the snake is nearby, display the food's point value.
            head = self.snake[0]
            food_pos = self.food.position
            if abs(head[0] - food_pos[0]) <= BLOCK_SIZE * 3 and abs(head[1] - food_pos[1]) <= BLOCK_SIZE * 3:
                self.draw_text(f"+{self.food.properties['points']}", 
                               food_pos[0] - 10, food_pos[1] - 20, self.food.properties['color'])
        
        # --- Draw Snake Growth Animation ---
        if self.growth_effect and self.growth_effect_pos:
            effect_duration = 0.5  # seconds
            elapsed = time.time() - self.growth_effect_start
            if elapsed < effect_duration:
                # Expand circle radius from BLOCK_SIZE to 2*BLOCK_SIZE over the duration
                effect_radius = int(BLOCK_SIZE * (1 + elapsed / effect_duration))
                pygame.draw.circle(screen, YELLOW, self.growth_effect_pos, effect_radius)
            else:
                self.growth_effect = False

        # --- Draw Score and Info ---
        self.draw_text(f"Score: {self.score}", 10, 10)
        self.draw_text(f"High Score: {self.high_score}", 10, 40)
        self.draw_text(f"Length: {len(self.snake)}", WINDOW_WIDTH - 120, 10)
        
        pygame.display.update()

    def draw_text(self, text, x, y, color=WHITE):
        text_surface = font.render(text, True, color)
        screen.blit(text_surface, (x, y))

    def pause_game(self):
        paused = True
        pause_start = time.time()  # Record when pause starts
        self.draw_text("PAUSED", WINDOW_WIDTH // 2 - 50, WINDOW_HEIGHT // 2, YELLOW)
        pygame.display.update()
        
        while paused:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                    paused = False
                    pause_duration = time.time() - pause_start
                    # Adjust trap timers and effect timers by the pause duration
                    self.last_trap_time += pause_duration
                    if self.trap_warning:
                        self.warning_start += pause_duration
                    if self.trap_hit_effect:
                        self.effect_start += pause_duration
                    if self.growth_effect:
                        self.growth_effect_start += pause_duration

    def game_over(self):
        self.save_high_score()
        screen.fill(BLACK)
        self.draw_text("GAME OVER", WINDOW_WIDTH // 2 - 80, WINDOW_HEIGHT // 2 - 60, RED)
        self.draw_text(f"Score: {self.score}", WINDOW_WIDTH // 2 - 50, WINDOW_HEIGHT // 2 - 20, YELLOW)
        self.draw_text(f"Final Length: {len(self.snake)}", WINDOW_WIDTH // 2 - 70, WINDOW_HEIGHT // 2 + 10, TEAL)
        self.draw_text("Press R to Restart or Q to Quit", WINDOW_WIDTH // 2 - 160, WINDOW_HEIGHT // 2 + 40, WHITE)
        pygame.display.update()
        
        pygame.time.wait(500)  # Brief delay to prevent accidental restart
        
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        self.reset_game()
                        waiting = False
                    elif event.key == pygame.K_q:
                        pygame.quit()
                        sys.exit()

# ---------------------------
# Main Loop
# ---------------------------
def main():
    game = Game()
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            game.handle_input(event)
        
        if game.update():
            game.draw()
            clock.tick(game.fps)

if __name__ == '__main__':
    main()
