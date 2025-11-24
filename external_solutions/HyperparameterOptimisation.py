import numpy as np
import random
import time
from typing import List, Dict, Tuple, Any
import os

# External imports (assumed to be available in your file system):
from enviroment.calico_env import CalicoEnv
from enviroment.calico_potential_scoring import evaluate_move
from enviroment.calico_scoring import get_total_score_on_board

# --- 1. ENVIRONMENT/UTILITY CONSTANTS ---
BOARD_SIZE = 7
PLAYER_HAND_SIZE = 3
NR_OF_TILES_IN_SHOP = 3
NO_TILE_VALUE = -1
CHECKPOINT_FILENAME = "best_calico_weights2.npz"

# Placeholder initial weights (these will be overwritten by the NN output)
EVALUATION_CONFIG = {
    "WEIGHT_FINAL_SCORE": 1.0,
    "WEIGHT_CAT_POTENTIAL": 1.0,
    "WEIGHT_COLOR_POTENTIAL": 1.0,
    "WEIGHT_OBJECTIVE_VIABILITY": 1.0,
}


# --- 2. THE NEURAL NETWORK (Weight Generator) ---

class CalicoWeightGenerator:
    """
    A simple feed-forward network to generate 4 evaluation weights
    based on the current game state.
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int, weights=None):
        # Neural Network Structure
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        if weights is None:
            # Randomly initialize weights
            self.W1 = np.random.randn(input_size, hidden_size) * 0.1
            self.b1 = np.zeros(hidden_size)
            self.W2 = np.random.randn(hidden_size, output_size) * 0.1
            self.b2 = np.zeros(output_size)
        else:
            self.W1, self.b1, self.W2, self.b2 = weights

        # Store weights for genetic algorithm operations (No @property needed here)
        self._params = [self.W1, self.b1, self.W2, self.b2]

    @property
    def params(self):
        """Used by the trainer for saving, loading, crossover, and mutation."""
        return self._params

    def predict_weights(self, flat_state: np.ndarray) -> Dict[str, float]:
        """
        Runs the forward pass and returns the scaled evaluation config dictionary.
        """
        # Reshape input to (1, input_size) if it's 1D
        state = flat_state.reshape(1, -1)

        # Hidden layer (ReLU activation)
        h = np.dot(state, self.W1) + self.b1
        h_relu = np.maximum(0, h)

        # Output layer (Tanh activation)
        output = np.dot(h_relu, self.W2) + self.b2
        final_output = np.tanh(output).flatten()

        # Scale Tanh output [-1, 1] to positive weights [0.1, 5.0]
        weights = 4.95 * (final_output + 1.0) + 0.1

        # Map the 4 outputs to the evaluation config keys
        config_keys = ["WEIGHT_FINAL_SCORE", "WEIGHT_CAT_POTENTIAL", "WEIGHT_COLOR_POTENTIAL",
                       "WEIGHT_OBJECTIVE_VIABILITY"]

        return dict(zip(config_keys, weights))


# --- 3. THE AGENT FUNCTION (Modified Lookahead) ---

def one_step_lookahead_move(env: CalicoEnv, config: Dict[str, float]):
    """
    The original lookahead, modified to use the learned 'config' weights.
    We assume remaining_recursions=0 for pure 1-step lookahead as requested.
    """
    max_score = -np.inf
    best_action = None

    for action in env.get_legal_actions():
        env.perform_action(action)

        # Use the learned config in the evaluation function
        score = evaluate_move(env.board_matrix, env.cat_tiles, config)

        env.undo_action()  # Efficient undo (critical for performance)

        if score > max_score:
            max_score = score
            best_action = action

    if best_action:
        return best_action, max_score

    return None, max_score


# --- 4. NEUROEVOLUTIONARY TRAINER ---

class NeuroevolutionaryTrainer:
    def __init__(self, population_size=50, num_games_per_nn=5, generations=10):
        # The environment must be instantiated once
        self.env = CalicoEnv()

        # Define NN dimensions based on the environment state
        self.input_size = 61  # Fixed size based on previous traceback
        print(f"Trainer Initialized: NN Input Size set to {self.input_size} (Required size for CalicoEnv state).")

        self.hidden_size = 16
        self.output_size = 4

        self.population_size = population_size
        self.num_games_per_nn = num_games_per_nn
        self.generations = generations

        # --- Load Existing Weights for Continuous Training ---
        initial_weights = self._load_agent_weights()

        # Initialize population
        self.population = []
        for i in range(population_size):
            if initial_weights is not None:
                # Initialize all agents with the best loaded weights to resume training
                self.population.append(
                    CalicoWeightGenerator(self.input_size, self.hidden_size, self.output_size, weights=initial_weights)
                )
            else:
                # Start from random weights
                self.population.append(
                    CalicoWeightGenerator(self.input_size, self.hidden_size, self.output_size)
                )

        # Optional: Verify the hardcoded size is correct by starting the env once.
        self.env.start_game()
        actual_size = len(self.env.get_flat_state())
        if actual_size != self.input_size:
            print(
                f"CRITICAL WARNING: Calculated state size ({actual_size}) does not match hardcoded size ({self.input_size}). Check CalicoEnv.get_flat_state().")

    def _save_best_agent(self, generator: CalicoWeightGenerator, filename=CHECKPOINT_FILENAME):
        """Saves the parameters of the best agent for checkpointing."""
        # Use a dictionary to save the parameters by index
        param_dict = {f'param_{i}': p for i, p in enumerate(generator.params)}
        np.savez_compressed(filename, **param_dict)
        print(f"\nSaved best agent weights to {filename} for checkpointing.")

    def _load_agent_weights(self, filename=CHECKPOINT_FILENAME):
        """Loads parameters from a checkpoint file."""
        try:
            if os.path.exists(filename) and os.path.getsize(filename) > 0:
                data = np.load(filename)
                # Ensure we load all 4 parameters (W1, b1, W2, b2)
                params = [data[f'param_{i}'] for i in range(4)]
                print(f"Loaded existing weights from {filename}. Resuming training.")
                return params
            else:
                raise FileNotFoundError
        except FileNotFoundError:
            print(f"No existing weights found at {filename}. Starting training from scratch.")
            return None

    def run_game(self, generator: CalicoWeightGenerator, agent_index: int, game_index: int, generation: int) -> float:
        """Runs a single game using the given NN to determine move weights."""
        self.env.start_game()

        # Add print to show game start
        is_debug_agent = (agent_index == 0 and game_index == 0)
        if is_debug_agent:
            print(f"\n--- DEBUG START | Gen {generation + 1} | Agent {agent_index + 1}, Game {game_index + 1} ---")

        # Assuming 25 moves (5x5 board) for a full game loop,
        # adjusted to 50 for max safety if buy/place count as 1 move each.
        max_moves = 50
        moves_played = 0

        while not self.env.is_game_over() and moves_played < max_moves:
            # 1. Get current state and generate weights
            flat_state = self.env.get_flat_state()
            current_config = generator.predict_weights(flat_state)

            # --- Conditional Print for First Move ---
            if is_debug_agent and moves_played % 10 == 0:
                print(f"Move {str(moves_played)}/50: NN Generated Weights (Lookahead Heuristic):\n{current_config}")
                print("-" * 20)

            # 2. Find and perform the best move
            best_action, _ = one_step_lookahead_move(self.env, current_config)

            if best_action:
                self.env.perform_action(best_action)
                moves_played += 1
            else:
                # No legal moves, break early
                break

        # Calculate final, actual score (using the standard scoring, not the potential)
        final_score = get_total_score_on_board(self.env.board_matrix, self.env.cat_tiles)

        if is_debug_agent:
            print(f"--- DEBUG END | Game Over after {moves_played} moves. Final Score: {final_score:.2f} ---\n")

        return final_score

    def train(self):
        print(f"Starting Neuroevolution: {self.generations} generations, {self.population_size} population.")
        print("-" * 50)

        for generation in range(self.generations):
            fitnesses = {}
            start_time = time.time()

            # 1. Evaluate Population Fitness
            for i, nn in enumerate(self.population):
                total_score = 0
                for j in range(self.num_games_per_nn):
                    # Pass agent index (i), game index (j), and current generation
                    total_score += self.run_game(nn, i, j, generation)

                avg_score = total_score / self.num_games_per_nn
                fitnesses[i] = avg_score

            # Sort by fitness (descending)
            sorted_fitness = sorted(fitnesses.items(), key=lambda item: item[1], reverse=True)
            best_fitness = sorted_fitness[0][1]
            best_nn_index = sorted_fitness[0][0]  # Get the index of the best agent
            avg_fitness = np.mean(list(fitnesses.values()))

            # --- Save the best agent found in this generation ---
            self._save_best_agent(self.population[best_nn_index])

            print(
                f"Gen {generation + 1}/{self.generations} | Best Score: {best_fitness:.2f} | Avg Score: {avg_fitness:.2f} | Time: {time.time() - start_time:.2f}s")

            # 2. Selection (Elitism)
            num_elite = int(0.1 * self.population_size)  # Top 10% survive
            elite_indices = [idx for idx, score in sorted_fitness[:num_elite]]
            new_population: List[CalicoWeightGenerator] = []

            for idx in elite_indices:
                new_population.append(self.population[idx])  # Preserve elite members

            # 3. Reproduction (Crossover and Mutation)
            while len(new_population) < self.population_size:
                # Select two parents from the top half of the population
                parent1_idx = random.choice(sorted_fitness[:self.population_size // 2])[0]
                parent2_idx = random.choice(sorted_fitness[:self.population_size // 2])[0]

                parent1 = self.population[parent1_idx]
                parent2 = self.population[parent2_idx]

                child_params: List[np.ndarray] = []
                # Crossover and Mutation on each parameter matrix/vector
                for p1_mat, p2_mat in zip(parent1.params, parent2.params):
                    # Simple Crossover (50/50 chance for each element)
                    mask = np.random.rand(*p1_mat.shape) < 0.5
                    child_mat = np.where(mask, p1_mat, p2_mat)

                    # Mutation (Add small random noise)
                    mutation_rate = 0.05
                    mutation_mask = np.random.rand(*child_mat.shape) < mutation_rate
                    child_mat += mutation_mask * np.random.randn(*child_mat.shape) * 0.1

                    child_params.append(child_mat)

                # Create the new child NN
                child = CalicoWeightGenerator(self.input_size, self.hidden_size, self.output_size, weights=child_params)
                new_population.append(child)

            self.population = new_population

        print("\nTraining Complete.")

        # Return the best network found
        return self.population[best_nn_index]


# Main execution block
if __name__ == "__main__":
    trainer = NeuroevolutionaryTrainer(
        population_size=25,  # Number of NN agents per generation
        num_games_per_nn=5,  # Games to average score for fitness
        generations=20  # Number of evolutionary steps
    )

    best_agent = trainer.train()
    print("\n--- Final Best Agent Weights (Sample Output) ---")

    # Run a quick test game with the final best agent
    final_state = trainer.env.get_flat_state()
    final_weights = best_agent.predict_weights(final_state)

    print(f"Best Agent Weight Configuration:\n{final_weights}")
    print(
        "\nNote: The scores and weights shown are based on the training run. Replace the PLACEHOLDER functions to train effectively.")