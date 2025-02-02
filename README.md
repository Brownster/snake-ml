Snake Game AI

A comprehensive implementation of the classic Snake game with multiple modes and a deep reinforcement learning (DRL) framework. This repository contains:

    Human Playable Version: snake.py
    A fully playable Snake game built with Pygame for human control.

    AI Training Module: snake_ai.py
    A training script that uses deep Q-learning with PyTorch to train an agent to play Snake. This module includes an enhanced state representation (vision window, wall distances, snake length, trap logic, etc.), dynamic epsilon decay, curriculum learning, and reward shaping.

    AI Play Mode: play_snake.py
    A script that uses a previously trained model to control the Snake game, allowing you to observe the trained AI in action.

Features

    Enhanced State Representation:
    Combines traditional danger flags with additional features such as:
        A 5×5 vision grid around the snake’s head.
        Normalized distances to the walls.
        The snake’s current length.

    Dynamic Exploration:
    Uses an exponential epsilon decay schedule and Boltzmann exploration to balance exploration and exploitation.

    Reward Shaping:
    Incorporates multiple reward components:
        Proximity rewards based on reducing the distance to the food.
        Time penalties to discourage stalling.
        Trap avoidance rewards.
        Significant penalties for collisions (with walls, self, or traps).

    Trap and Food Variation:
    Implements varied food types (regular, special, rare) with different point values and a trap mechanic that adds challenge by ending the game upon collision.

    Curriculum Learning:
    Gradually increases difficulty (e.g., reducing block size after a set number of episodes) as the agent improves.

    TensorBoard Logging:
    Tracks training metrics such as score, total reward, epsilon value, loss, and average Q-value.

Requirements

    Python 3.6+
    PyTorch
    Pygame
    TensorBoard (for logging)

Install dependencies via pip (or use a requirements.txt file):

pip install torch torchvision torchaudio pygame tensorboard

Usage
Human Playable Version

Run the human-controlled Snake game:

python snake.py

Training the AI

Train the deep Q-learning agent on the Snake game:

python snake_ai.py

    The training module will run for 2000 episodes.
    Training progress (score, reward, loss, epsilon, etc.) is logged to TensorBoard.
    Model checkpoints are saved periodically and the best model is stored as best_model.pth.

Playing with the Trained AI

Watch the trained model play the Snake game:

python play_snake.py

    Ensure that best_model.pth (or another checkpoint) is available.
    This mode disables exploration to let the agent exploit what it has learned.

Project Structure

.
├── snake.py         # Human playable Snake game
├── snake_ai.py      # Deep Q-learning training script
├── play_snake.py    # Script to run the game using the trained AI model
├── README.md        # This file
└── (optional files, e.g., requirements.txt, checkpoints/)

Training Details & Further Improvements

    Hyperparameter Tuning:
    Experiment with different learning rates, discount factors, and batch sizes to optimize training.

    Exploration Strategies:
    Consider alternative exploration techniques (e.g., Noisy Networks or Boltzmann exploration) for improved performance.

    Model Architecture:
    If needed, expand the network architecture (e.g., add more layers or experiment with convolutional layers if the vision grid is increased).

    Reward Shaping Enhancements:
    Additional penalties (e.g., for repetitive movement) or rewards (e.g., for trap avoidance) can further guide learning.

    Evaluation Phase:
    Integrate a dedicated evaluation phase where the agent’s performance is measured without exploration (epsilon = 0).

    Saving & Resuming Training:
    Add functionality to load a saved model checkpoint to resume training without starting over.


Hyperparameter and Configuration Tuning

Our Snake AI project exposes several key levers that can be tuned to improve performance and training stability. Below is an overview of the major components and suggestions for adjusting them:
1. Exploration vs. Exploitation

    Epsilon (Exploration Rate):
    Controls the balance between taking random actions (exploration) and following the learned policy (exploitation).
        How to Adjust: Modify initial_epsilon, min_epsilon, and decay_rate in the training loop.
        Example:

    initial_epsilon = 1.0  # Start with high exploration
    min_epsilon = 0.01     # Minimum exploration rate
    decay_rate = 0.001     # Controls how quickly epsilon decays

    Effect: A slower decay (lower decay_rate) maintains exploration longer, potentially helping the agent discover better strategies, while a faster decay leads to quicker convergence but may risk premature exploitation.

Exploration Strategy:
We currently use epsilon‑greedy exploration. An alternative is Boltzmann (softmax) exploration, which selects actions based on a probability distribution over Q-values.

    Example:

        def boltzmann_exploration(q_values, temperature=1.0):
            probabilities = torch.softmax(q_values / temperature, dim=-1)
            return torch.multinomial(probabilities, 1).item()

        Usage: Replace or combine with epsilon‑greedy to yield more nuanced exploration.

2. Reward Shaping

    Reward Values:
    Rewards are assigned based on the food’s point value and penalties for collisions or timeouts.
        Example:

    reward += self.food['points']  # Reward for eating food
    reward -= 10                   # Penalty for hitting a wall or trap
    reward -= 0.1                  # Small penalty per step

    Effect: Increasing collision penalties encourages caution; higher food rewards incentivize aggressive food-seeking.

Proximity Reward:
The agent receives extra reward based on how much closer it gets to the food.

    Example:

    reward += (prev_food_dist - new_food_dist) * 0.2

    Effect: A higher multiplier focuses the agent on approaching food more directly.

Time Penalty:
A penalty can be applied for taking too long to eat food.

    Example:

        time_since_last_food = current_time - self.last_food_time
        if time_since_last_food > 5:
            reward -= 0.05 * time_since_last_food

        Effect: This encourages the agent to move quickly toward food rather than wandering aimlessly.

3. Neural Network Architecture

    Hidden Layer Size:
    The number of neurons in each hidden layer affects the model’s capacity.
        Example:

    model = SnakeAI(50, 256, 3)  # 256 neurons per hidden layer

    Effect: More neurons can improve learning of complex strategies but may slow training or lead to overfitting.

Number of Layers:
Our baseline model has two hidden layers. You can experiment with adding more layers.

    Example:

    self.linear1 = nn.Linear(input_size, hidden_size)
    self.linear2 = nn.Linear(hidden_size, hidden_size)
    self.linear3 = nn.Linear(hidden_size, hidden_size)  # Additional layer
    self.linear4 = nn.Linear(hidden_size, output_size)

Activation Functions:
We use ReLU activations by default; consider alternatives like LeakyReLU or ELU.

    Example:

        self.activation = nn.LeakyReLU(negative_slope=0.01)

4. Training Hyperparameters

    Learning Rate (LR):
    Determines how quickly weights are updated.
        Example:

    LR = 0.001

    Effect: Higher rates speed up learning but risk instability; lower rates are more stable but slower.

Discount Factor (GAMMA):
Values how much the agent values future rewards.

    Example:

    GAMMA = 0.9

    Effect: Higher gamma (e.g., 0.99) encourages long-term planning, while lower gamma focuses on immediate rewards.

Batch Size (BATCH_SIZE):
The number of experiences sampled from memory for each training step.

    Example:

    BATCH_SIZE = 1000

    Effect: Larger batches stabilize training but require more memory and computation.

Memory Size (MAX_MEMORY):
The size of the replay buffer.

    Example:

        MAX_MEMORY = 100_000

        Effect: A larger memory stores more diverse experiences but increases memory usage.

5. Environment Settings

    Block Size:
    The size of each cell in the game grid.
        Example:

    block_size = 20

    Effect: Smaller blocks increase game complexity by increasing the number of decisions.

Trap Frequency:
Controls how often traps appear.

    Example:

    if self.trap is None and current_time - self.last_trap_time >= 10:

    Effect: Lowering the threshold (e.g., 5 seconds) increases game difficulty.

Food Expiration:
How long food remains on the grid before it is replaced.

    Example:

        if self.food and current_time - self.food['spawn_time'] > 10:

        Effect: A shorter expiration time (e.g., 5 seconds) forces faster food collection, increasing difficulty.

6. Curriculum Learning

    Block Size Reduction:
    To gradually increase difficulty, the block size is reduced after a set number of episodes.
        Example:

        if episode > 300:
            game.block_size = 15

        Effect: A smaller block size forces the agent to navigate more precisely.

7. TensorBoard Logging and Visualization

    Metrics Logged:
    Currently, we log score, total reward, epsilon, and loss. You can extend this by logging:
        Average Q-values.
        Exploration rate.
        Number of trap collisions.
        Example:

    writer.add_scalar('Average Q-value', q_values.mean().item(), episode)

Rendering Frequency:
Adjust rendering frequency during training to speed up learning without sacrificing too much visual feedback.

    Example:

        if episode % 50 == 0 and step_count % 10 == 0:
            game.render()

Summary of Key Levers
Lever	Parameter/Component	Effect
Exploration	epsilon, decay_rate	Balances random exploration vs. following the learned policy.
Reward Shaping	Reward values, proximity multiplier, time penalty	Encourages desired behaviors (e.g., food-seeking, trap avoidance) and discourages indecision.
Neural Network	Hidden layers, neurons, activation functions	Controls the model’s capacity to learn complex strategies.
Training Hyperparameters	LR, GAMMA, BATCH_SIZE, MAX_MEMORY	Affects convergence speed, stability, and memory usage.
Environment Settings	block_size, trap frequency, food expiration	Adjusts the overall difficulty and complexity of the game.
Curriculum Learning	Gradual changes (e.g., block size reduction)	Increases difficulty as the agent improves to promote robust learning.
Logging & Visualization	TensorBoard metrics, rendering frequency	Provides insights into the training process and helps with debugging.

By adjusting these parameters, you can fine‑tune the training process to achieve better performance or faster convergence. Experimentation is key—try different settings to see which combination works best for your specific implementation.




Contributing

Contributions, suggestions, and bug reports are welcome! Feel free to open an issue or submit a pull request.
License

This project is licensed under the MIT License. See the LICENSE file for details.

Happy coding and good luck training your Snake AI!

This README outlines the project, its features, installation instructions, usage, and areas for further improvement. Adjust details (like hyperparameter values or curriculum strategies) as your project evolves.
