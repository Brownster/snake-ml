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

Contributing

Contributions, suggestions, and bug reports are welcome! Feel free to open an issue or submit a pull request.
License

This project is licensed under the MIT License. See the LICENSE file for details.

Happy coding and good luck training your Snake AI!

This README outlines the project, its features, installation instructions, usage, and areas for further improvement. Adjust details (like hyperparameter values or curriculum strategies) as your project evolves.
