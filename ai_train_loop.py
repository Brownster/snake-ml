import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import deque
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import logging
import requests
import random

class PlateauDetector:
    """Detects performance plateaus for early stopping and parameter adjustment"""
    def __init__(self, window_size=50, threshold=0.01):
        self.window_size = window_size
        self.threshold = threshold
        self.scores = []
        self.improvements = []
    
    def update(self, score):
        """Update scores and calculate improvement"""
        if self.scores:
            improvement = score - self.scores[-1]
            self.improvements.append(improvement)
        self.scores.append(score)
        
        if len(self.scores) > self.window_size:
            self.scores.pop(0)
        if len(self.improvements) > self.window_size:
            self.improvements.pop(0)
    
    def is_plateau(self):
        """Check if performance has plateaued"""
        if len(self.scores) < self.window_size:
            return False
        
        recent_scores = self.scores[-self.window_size:]
        recent_improvements = self.improvements[-self.window_size:]
        
        # Check for both low variance and minimal improvement
        score_std = np.std(recent_scores)
        avg_improvement = np.mean(recent_improvements) if recent_improvements else 0
        
        return score_std < self.threshold and abs(avg_improvement) < self.threshold

class ModelCheckpointer:
    """Handles model saving and loading with comprehensive tracking"""
    def __init__(self, log_dir):
        self.log_dir = Path(log_dir) / 'checkpoints'
        self.log_dir.mkdir(exist_ok=True)
        self.best_score = float('-inf')
        self.saved_models = []
        self.model_metrics = {}
    
    def save_checkpoint(self, model, metrics, config, cycle):
        """Save model with detailed metrics and config"""
        score = metrics['avg_score']
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        model_info = {
            'score': score,
            'metrics': metrics,
            'config': config,
            'cycle': cycle,
            'timestamp': timestamp
        }
        
        # Save if it's a new best model
        if score > self.best_score:
            self.best_score = score
            model_path = self.log_dir / f"best_model_cycle{cycle}_score{score:.2f}.pth"
            torch.save({
                'model_state_dict': model.state_dict(),
                'info': model_info
            }, model_path)
            
            self.saved_models.append(model_path)
            self.model_metrics[model_path] = model_info
            logging.info(f"New best model saved: {model_path}")
        
        # Save periodic checkpoint
        if cycle % 5 == 0:
            checkpoint_path = self.log_dir / f"checkpoint_cycle{cycle}.pth"
            torch.save({
                'model_state_dict': model.state_dict(),
                'info': model_info
            }, checkpoint_path)
    
    def load_best_model(self):
        """Load the best performing model"""
        if not self.saved_models:
            return None, None
        
        best_model_path = max(self.saved_models, 
                            key=lambda x: self.model_metrics[x]['score'])
        checkpoint = torch.load(best_model_path)
        return checkpoint['model_state_dict'], checkpoint['info']

class MetricsTracker:
    """Comprehensive metrics tracking and analysis"""
    def __init__(self):
        self.episode_metrics = []
        self.cycle_metrics = []
        self.global_metrics = {
            'best_score': float('-inf'),
            'best_config': None,
            'learning_curves': [],
            'parameter_impact': {}
        }
    
    def update_episode(self, metrics):
        """Update episode-level metrics"""
        self.episode_metrics.append(metrics)
        
        # Update learning curves
        if len(self.episode_metrics) % 100 == 0:
            self.global_metrics['learning_curves'].append({
                'episodes': len(self.episode_metrics),
                'avg_score': np.mean([m['score'] for m in self.episode_metrics[-100:]]),
                'avg_loss': np.mean([m['avg_loss'] for m in self.episode_metrics[-100:]])
            })
    
    def update_cycle(self, cycle_results, config):
        """Update cycle-level metrics"""
        avg_score = np.mean([r['metrics']['avg_score'] for r in cycle_results])
        self.cycle_metrics.append({
            'config': config,
            'avg_score': avg_score,
            'results': cycle_results
        })
        
        # Update parameter impact analysis
        for param, value in config.items():
            if param not in self.global_metrics['parameter_impact']:
                self.global_metrics['parameter_impact'][param] = []
            self.global_metrics['parameter_impact'][param].append({
                'value': value,
                'score': avg_score
            })
        
        # Update best performance
        if avg_score > self.global_metrics['best_score']:
            self.global_metrics['best_score'] = avg_score
            self.global_metrics['best_config'] = config.copy()
    
    def get_parameter_recommendations(self):
        """Analyze parameter impact and make recommendations"""
        recommendations = {}
        for param, results in self.global_metrics['parameter_impact'].items():
            if not results:
                continue
            
            # Sort by performance
            sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
            best_value = sorted_results[0]['value']
            
            # Calculate performance-weighted average
            scores = np.array([r['score'] for r in results])
            values = np.array([r['value'] for r in results])
            weights = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
            weighted_avg = np.average(values, weights=weights)
            
            recommendations[param] = {
                'best_value': best_value,
                'weighted_avg': weighted_avg,
                'performance_range': [scores.min(), scores.max()]
            }
        
        return recommendations
    def __init__(self, grid_width, grid_height, block_size):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.block_size = block_size
        self.grid_cols = grid_width // block_size
        self.grid_rows = grid_height // block_size
        self.reset()
    
    def reset(self):
        self.position_history = []
        self.food_positions = []
        self.collision_positions = []
        self.trap_positions = []
        self.action_history = []
        self.scores = []
        self.moves_per_food = []
        self.current_moves = 0
    
    def _normalize_position(self, pos):
        return (pos[0] // self.block_size, pos[1] // self.block_size)
    
    def update(self, game_state, action, reward, done):
        """Update tracker with normalized positions and additional metrics"""
        head_pos = game_state.snake[0]
        norm_pos = self._normalize_position(head_pos)
        self.position_history.append(norm_pos)
        self.action_history.append(action)
        self.current_moves += 1
        
        if done:
            if game_state._is_collision(head_pos):
                self.collision_positions.append(norm_pos)
            self.scores.append(game_state.score)
        
        if game_state.food and head_pos == game_state.food['position']:
            self.food_positions.append(norm_pos)
            self.moves_per_food.append(self.current_moves)
            self.current_moves = 0
        
        if game_state.trap and head_pos == game_state.trap:
            self.trap_positions.append(norm_pos)

    def get_metrics(self):
        """Calculate comprehensive gameplay metrics"""
        return {
            'avg_score': np.mean(self.scores) if self.scores else 0,
            'max_score': max(self.scores) if self.scores else 0,
            'avg_moves_per_food': np.mean(self.moves_per_food) if self.moves_per_food else 0,
            'collision_rate': len(self.collision_positions) / len(self.position_history) if self.position_history else 0,
            'food_efficiency': len(self.food_positions) / self.current_moves if self.current_moves > 0 else 0
        }

class Visualizer:
    """Enhanced visualizer with dynamic grid sizing and advanced metrics plotting"""
    def __init__(self, log_dir, grid_cols, grid_rows):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.writer = SummaryWriter(f"{log_dir}/runs/{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.grid_cols = grid_cols
        self.grid_rows = grid_rows
    
    def create_heatmap(self, data, title, normalize=True):
        """Generate heatmap from normalized position data with optional normalization"""
        grid = np.zeros((self.grid_rows, self.grid_cols))
        for x, y in data:
            if 0 <= x < self.grid_cols and 0 <= y < self.grid_rows:
                grid[y, x] += 1
        
        if normalize and grid.max() > 0:
            grid = grid / grid.max()
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(grid, cmap='YlOrRd', annot=False)
        plt.title(title)
        
        filename = f"{title.lower().replace(' ', '_')}_{int(time.time())}.png"
        plt.savefig(self.log_dir / filename)
        self.writer.add_figure(title, plt.gcf())
        plt.close()
        return grid
    
    def plot_training_progress(self, metrics_history, config):
        """Plot comprehensive training metrics"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot scores
        scores = [m['avg_score'] for m in metrics_history]
        ax1.plot(scores)
        ax1.set_title('Average Score per Episode')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Score')
        
        # Plot efficiency
        efficiency = [m['food_efficiency'] for m in metrics_history]
        ax2.plot(efficiency)
        ax2.set_title('Food Collection Efficiency')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Food/Move Ratio')
        
        # Plot collision rate
        collisions = [m['collision_rate'] for m in metrics_history]
        ax3.plot(collisions)
        ax3.set_title('Collision Rate')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Collision Rate')
        
        # Plot moves per food
        moves = [m['avg_moves_per_food'] for m in metrics_history]
        ax4.plot(moves)
        ax4.set_title('Average Moves per Food')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Moves')
        
        plt.tight_layout()
        filename = f"training_progress_{int(time.time())}.png"
        plt.savefig(self.log_dir / filename)
        self.writer.add_figure('Training Progress', fig)
        plt.close()

class HyperparameterTuner:
    """Improved hyperparameter tuner with range validation and adaptive updates"""
    def __init__(self, base_config):
        self.base_config = base_config
        self.param_ranges = {
            'LR': [0.0001, 0.001, 0.01],
            'GAMMA': [0.85, 0.9, 0.95],
            'HIDDEN_SIZE': [128, 256, 512],
            'BATCH_SIZE': [500, 1000, 2000]
        }
        self.param_history = []
    
    def generate_configs(self):
        """Generate valid parameter configurations with history tracking"""
        configs = []
        for param, values in self.param_ranges.items():
            for value in values:
                if not self._validate_param(param, value):
                    continue
                config = self.base_config.copy()
                config[param] = value
                config['MODEL_NAME'] = f"model_{param}_{value}_{int(time.time())}.pth"
                configs.append(config)
                self.param_history.append((param, value))
        return configs
    
    def _validate_param(self, param, value):
        """Enhanced parameter validation with logging"""
        try:
            if param == 'LR' and (value <= 0 or value > 0.1):
                logging.warning(f"Invalid learning rate: {value}")
                return False
            if param == 'GAMMA' and (value <= 0 or value >= 1):
                logging.warning(f"Invalid gamma value: {value}")
                return False
            if param == 'HIDDEN_SIZE' and (value <= 0 or not isinstance(value, int)):
                logging.warning(f"Invalid hidden size: {value}")
                return False
            if param == 'BATCH_SIZE' and (value <= 0 or not isinstance(value, int)):
                logging.warning(f"Invalid batch size: {value}")
                return False
            return True
        except Exception as e:
            logging.error(f"Parameter validation error: {str(e)}")
            return False

    def update_ranges(self, results):
        """Adaptive parameter range updates based on performance"""
        if not results:
            return
        
        # Calculate performance for each parameter value
        param_performance = {}
        for param in self.param_ranges.keys():
            param_performance[param] = {}
            for result in results:
                value = result['config'][param]
                score = result['metrics']['avg_score']
                if value not in param_performance[param]:
                    param_performance[param][value] = []
                param_performance[param][value].append(score)
        
        # Update ranges based on performance
        for param, performances in param_performance.items():
            if not performances:
                continue
            
            # Calculate mean performance for each value
            mean_performances = {
                value: np.mean(scores) 
                for value, scores in performances.items()
            }
            
            # Find best performing value
            best_value = max(mean_performances.items(), key=lambda x: x[1])[0]
            
            # Generate new range around best value
            if param == 'LR':
                new_range = [best_value * 0.5, best_value, best_value * 2.0]
            elif param == 'GAMMA':
                step = 0.05
                new_range = [
                    max(0.8, best_value - step),
                    best_value,
                    min(0.99, best_value + step)
                ]
            elif param in ['HIDDEN_SIZE', 'BATCH_SIZE']:
                new_range = [
                    max(64, best_value // 2),
                    best_value,
                    best_value * 2
                ]
            
            self.param_ranges[param] = sorted(list(set(new_range)))

class QTrainer(nn.Module):
    """Enhanced QTrainer with advanced features"""
    def __init__(self, model, lr, gamma):
        super().__init__()
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        self.target_model = type(model)(model.input_size, model.hidden_size, model.output_size)
        self.target_model.load_state_dict(model.state_dict())
        self.target_update_counter = 0
        self.loss_history = []
        
        # Initialize momentum buffer
        self.momentum = None
        self.beta = 0.9  # momentum coefficient
    
    def train_step(self, states, actions, rewards, next_states, dones):
        """Training step with momentum and gradient clipping"""
        # Convert inputs to tensors
        states = torch.tensor(np.array(states), dtype=torch.float)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float)
        actions = torch.tensor(np.array(actions), dtype=torch.long)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float)
        dones = torch.tensor(np.array(dones), dtype=torch.float)
        
        if len(states.shape) == 1:
            states = torch.unsqueeze(states, 0)
            next_states = torch.unsqueeze(next_states, 0)
            actions = torch.unsqueeze(actions, 0)
            rewards = torch.unsqueeze(rewards, 0)
            dones = torch.unsqueeze(dones, 0)
        
        # Compute current Q values
        current_q = self.model(states)
        
        # Compute target Q values using target network
        with torch.no_grad():
            next_q = self.target_model(next_states)
            max_next_q = next_q.max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * max_next_q
        
        # Compute loss
        q_values = current_q.gather(1, actions.unsqueeze(1)).squeeze()
        loss = self.criterion(q_values, target_q)
        
        # Optimize with momentum
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Apply momentum
        if self.momentum is None:
            self.momentum = []
            for param in self.model.parameters():
                self.momentum.append(torch.zeros_like(param.data))
        
        for i, param in enumerate(self.model.parameters()):
            if param.grad is not None:
                self.momentum[i] = self.beta * self.momentum[i] + (1 - self.beta) * param.grad
                param.grad = self.momentum[i]
        
        self.optimizer.step()
        
        # Update target network
        self.target_update_counter += 1
        if self.target_update_counter % 100 == 0:
            self.target_model.load_state_dict(self.model.state_dict())
        
        # Store loss
        self.loss_history.append(loss.item())
        return loss.item()

    def get_metrics(self):
        """Calculate training metrics"""
        return {
            'avg_loss': np.mean(self.loss_history) if self.loss_history else 0,
            'loss_std': np.std(self.loss_history) if self.loss_history else 0,
            'min_loss': min(self.loss_history) if self.loss_history else 0,
            'max_loss': max(self.loss_history) if self.loss_history else 0
        }

class MLFeedbackLoop:
    """Integrated feedback loop system with comprehensive monitoring"""
    def __init__(self, game_env, model_class, log_dir='logs'):
        self.game_env = game_env
        self.model_class = model_class
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Initialize components
        grid_width = game_env.width
        grid_height = game_env.height
        block_size = game_env.block_size
        
        self.state_tracker = GameStateTracker(grid_width, grid_height, block_size)
        self.visualizer = Visualizer(log_dir, 
                                   self.state_tracker.grid_cols, 
                                   self.state_tracker.grid_rows)
        self.tuner = HyperparameterTuner({
            'MAX_MEMORY': 100_000,
            'BATCH_SIZE': 1000,
            'LR': 0.001,
            'GAMMA': 0.9,
            'HIDDEN_SIZE': 256,
            'EPISODES': 1000
        })
        
        self.results = []
        self.best_model = None
        self.metrics_history = []
        
        # Add new components while preserving original functionality
        self.checkpointer = ModelCheckpointer(log_dir)
        self.plateau_detector = PlateauDetector()
        self.metrics_tracker = MetricsTracker()
        
        # Configure logging with multiple handlers
        self._setup_logging()
    
    def _setup_logging(self):
        """Configure comprehensive logging system"""
        logger = logging.getLogger('MLFeedbackLoop')
        logger.setLevel(logging.INFO)
        
        # Clear any existing handlers
        logger.handlers = []
        
        # File handler for detailed logging
        fh = logging.FileHandler(
            self.log_dir / f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        )
        fh.setLevel(logging.INFO)
        
        # Console handler for basic progress
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Create formatters and add to handlers
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        simple_formatter = logging.Formatter(
            '%(levelname)s - %(message)s'
        )
        
        fh.setFormatter(detailed_formatter)
        ch.setFormatter(simple_formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        self.logger = logger
    
    def train_model(self, config):
        """Complete training cycle with metrics tracking"""
        model = self.model_class(20, config['HIDDEN_SIZE'], 3)
        memory = deque(maxlen=config['MAX_MEMORY'])
        trainer = QTrainer(model, config['LR'], config['GAMMA'])
        
        for episode in range(config['EPISODES']):
            state = self.game_env.reset()
            self.state_tracker.reset()
            episode_losses = []
            
            while True:
                state_tensor = torch.tensor(state, dtype=torch.float)
                epsilon = max(0.01, 1 - episode/(config['EPISODES']*0.8))
                
                if random.random() < epsilon:
                    action = random.randint(0, 2)
                else:
                    action = torch.argmax(model(state_tensor)).item()
                
                # Environment step
                final_move = [0, 0, 0]
                final_move[action] = 1
                reward, done, _ = self.game_env.step(final_move)
                next_state = self.game_env.get_state()
                
                # Store experience and update tracker
                memory.append((state, action, reward, next_state, done))
                self.state_tracker.update(self.game_env, action, reward, done)
                
                # Train batch
                if len(memory) > config['BATCH_SIZE']:
                    batch = random.sample(memory, config['BATCH_SIZE'])
                    states, actions, rewards, next_states, dones = zip(*batch)
                    loss = trainer.train_step(states, actions, rewards, next_states, dones)
                    episode_losses.append(loss)
                
                if done:
                    # Update metrics
                    metrics = self.state_tracker.get_metrics()
                    metrics.update({
                        'episode': episode,
                        'avg_loss': np.mean(episode_losses) if episode_losses else 0,
                        'epsilon': epsilon,
                        'memory_size': len(memory)
                    })
                    self.metrics_history.append(metrics)
                    
                    # Periodic logging and visualization
                    if episode % 100 == 0:
                        self.logger.info(
                            f"Episode {episode} | "
                            f"Score: {metrics['avg_score']:.2f} | "
                            f"Loss: {metrics['avg_loss']:.4f} | "
                            f"Epsilon: {epsilon:.2f}"
                        )
                        self.visualizer.plot_training_progress(self.metrics_history, config)
                    break
                
                state = next_state
        
        return model, trainer.get_metrics()
    
    def evaluate_model(self, model, num_games=100):
        """Comprehensive model evaluation"""
        model.eval()  # Set evaluation mode
        scores = []
        self.state_tracker.reset()
        
        for _ in range(num_games):
            state = self.game_env.reset()
            done = False
            
            while not done:
                with torch.no_grad():
                    state_tensor = torch.tensor(state, dtype=torch.float)
                    action = torch.argmax(model(state_tensor)).item()
                
                final_move = [0, 0, 0]
                final_move[action] = 1
                reward, done, score = self.game_env.step(final_move)
                
                self.state_tracker.update(self.game_env, action, reward, done)
                if done:
                    scores.append(score)
                
                state = self.game_env.get_state()
        
        model.train()  # Reset to training mode
        metrics = self.state_tracker.get_metrics()
        metrics.update({
            'avg_score': np.mean(scores),
            'std_score': np.std(scores),
            'max_score': max(scores),
            'min_score': min(scores),
            'games_played': num_games
        })
        return metrics
    
    def run_cycle(self):
        """Complete optimization cycle"""
        configs = self.tuner.generate_configs()
        if not configs:
            self.logger.warning("No valid configurations generated!")
            return None
        
        cycle_results = []
        for config in configs:
            try:
                model, training_metrics = self.train_model(config)
                eval_metrics = self.evaluate_model(model)
                
                result = {
                    'config': config,
                    'training_metrics': training_metrics,
                    'evaluation_metrics': eval_metrics,
                    'timestamp': datetime.now().isoformat()
                }
                cycle_results.append(result)
                
                # Update best model
                if eval_metrics['avg_score'] > (self.best_model['score'] if self.best_model else -np.inf):
                    self.best_model = {
                        'model': model.state_dict(),
                        'score': eval_metrics['avg_score'],
                        'config': config
                    }
                    torch.save(model.state_dict(), self.log_dir / 'best_model.pth')
                
                # Generate visualizations
                self.visualizer.create_heatmap(
                    self.state_tracker.position_history,
                    f"Position Heatmap - {config['LR']}_{config['GAMMA']}"
                )
                
            except Exception as e:
                self.logger.error(f"Training failed for config {config}: {str(e)}")
                continue
        
        # Update parameter ranges based on results
        if cycle_results:
            self.tuner.update_ranges(cycle_results)
        
        self.results.extend(cycle_results)
        return cycle_results
    def __init__(self, game_env, model_class, log_dir='logs'):
        self.game_env = game_env
        self.model_class = model_class
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Initialize components with game dimensions
        grid_width = game_env.width
        grid_height = game_env.height
        block_size = game_env.block_size
        
        self.state_tracker = GameStateTracker(grid_width, grid_height, block_size)
        self.visualizer = Visualizer(log_dir, 
                                   self.state_tracker.grid_cols, 
                                   self.state_tracker.grid_rows)
        self.tuner = HyperparameterTuner({
            'MAX_MEMORY': 100_000,
            'BATCH_SIZE': 1000,
            'LR': 0.001,
            'GAMMA': 0.9,
            'HIDDEN_SIZE': 256,
            'EPISODES': 1000
        })
        
        self.results = []
        self.best_model = None
        self.metrics_history = []
        
        # Configure logging
        logging.basicConfig(
            filename=self.log_dir / f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
