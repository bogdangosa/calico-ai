import numpy as np
import random
import time
from typing import List, Dict, Tuple, Any
import os

# NOTE: Since the CalicoEnv and get_total_score_on_board are
# imported externally, we rely on them being available and correctly defined.

# External imports (assumed to be available in your file system):
from enviroment.calico_env import CalicoEnv
from enviroment.calico_scoring import get_total_score_on_board

# --- 1. ENVIRONMENT/UTILITY CONSTANTS ---
BOARD_SIZE = 7
PLAYER_HAND_SIZE = 3
NR_OF_TILES_IN_SHOP = 3
NO_TILE_VALUE = -1
CHECKPOINT_FILENAME = "../agents/best_calico_agent.npz"


# --- 2. THE NEURAL NETWORK (Value Estimator) ---

class CalicoValueEstimator:
    """
    A simple feed-forward network to directly estimate the final score (value)
    of the current game state. This completely replaces the heuristic function.
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int = 1, weights=None):
        # The output size is fixed at 1 (the score estimate)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size  # Always 1

        if weights is None:
            # Randomly initialize weights
            self.W1 = np.random.randn(input_size, hidden_size) * 0.1
            self.b1 = np.zeros(hidden_size)
            self.W2 = np.random.randn(hidden_size, output_size) * 0.1
            self.b2 = np.zeros(output_size)
        else:
            self.W1, self.b1, self.W2, self.b2 = weights

        # Store weights for genetic algorithm operations
        self.params = [self.W1, self.b1, self.W2, self.b2]

    def predict_score(self, flat_state: np.ndarray) -> float:
        """
        Runs the forward pass and returns the estimated final score.
        """
        # Reshape input to (1, input_size)
        state = flat_state.reshape(1, -1)

        # Hidden layer (ReLU activation)
        h = np.dot(state, self.W1) + self.b1
        h_relu = np.maximum(0, h)

        # Output layer (Tanh activation, outputting a single value)
        output = np.dot(h_relu, self.W2) + self.b2
        final_output = np.tanh(output).flatten()[0]  # Get the single score value

        # Scale Tanh output [-1, 1] to a plausible Calico score range [0, 120]
        # Max score is typically around 100, so 120 gives some headroom.
        estimated_score = 60 * (final_output + 1.0)

        # Return the float score
        return float(estimated_score)


# --- 3. THE AGENT FUNCTION (Value-Based Lookahead WITHOUT Pruning) ---

def one_step_estimator_move(env: CalicoEnv, estimator: CalicoValueEstimator):
    """
    Lookahead based purely on the Neural Network's value estimation.
    This uses the full search space (all legal actions) without pruning.
    """
    max_score = -np.inf
    best_action = None

    # *** CORE CHANGE: No Pruning ***
    # Iterate over the full set of legal actions
    for action in env.get_legal_actions():
        env.perform_action(action)

        # Get score directly from the Neural Network
        flat_state = env.get_flat_state()
        score = estimator.predict_score(flat_state)

        env.undo_action()

        if score > max_score:
            max_score = score
            best_action = action

    if best_action:
        return best_action, max_score

    return None, max_score


# --- 4. NEUROEVOLUTIONARY TRAINER ---

class NeuroevolutionaryTrainer:
    def __init__(self, population_size=50, num_games_per_nn=5, generations=50):
        # The environment must be instantiated once
        self.env = CalicoEnv()

        # Define NN dimensions based on the environment state
        self.input_size = 61  # Fixed size based on previous traceback
        print(f"Trainer Initialized: NN Input Size set to {self.input_size} (Required size for CalicoEnv state).")

        self.hidden_size = 16
        self.output_size = 1  # Fixed for value estimation

        self.population_size = population_size
        self.num_games_per_nn = num_games_per_nn
        self.generations = generations

        # Mutation rate for exploration
        self.mutation_rate = 0.1

        # --- Load Existing Weights for Continuous Training ---
        initial_weights = self._load_agent_weights()

        # Initialize population using the new Value Estimator class
        self.population = []
        for i in range(population_size):
            if initial_weights is not None:
                self.population.append(
                    CalicoValueEstimator(self.input_size, self.hidden_size, self.output_size, weights=initial_weights)
                )
            else:
                self.population.append(
                    CalicoValueEstimator(self.input_size, self.hidden_size, self.output_size)
                )

        self.env.start_game()
        actual_size = len(self.env.get_flat_state())
        if actual_size != self.input_size:
            print(
                f"CRITICAL WARNING: Calculated state size ({actual_size}) does not match hardcoded size ({self.input_size}). Check CalicoEnv.get_flat_state().")

    def _save_best_agent(self, estimator: CalicoValueEstimator, filename=CHECKPOINT_FILENAME):
        """Saves the parameters of the best agent for checkpointing."""
        param_dict = {f'param_{i}': p for i, p in enumerate(estimator.params)}
        np.savez_compressed(filename, **param_dict)
        print(f"\nSaved best agent weights to {filename} for checkpointing.")

    def _load_agent_weights(self, filename=CHECKPOINT_FILENAME):
        """Loads parameters from a checkpoint file."""
        try:
            # Check if file exists and is not empty
            if os.path.exists(filename) and os.path.getsize(filename) > 0:
                data = np.load(filename)
                params = [data[f'param_{i}'] for i in range(4)]
                print(f"Loaded existing weights from {filename}. Resuming training.")
                return params
            else:
                raise FileNotFoundError
        except FileNotFoundError:
            print(f"No existing weights found at {filename}. Starting training from scratch.")
            return None

    def run_game(self, estimator: CalicoValueEstimator, agent_index: int, game_index: int, generation: int) -> float:
        """Runs a single game using the given NN to determine move scores."""
        self.env.start_game()

        is_debug_agent = (agent_index == 0 and game_index == 0)
        if is_debug_agent:
            print(f"\n--- DEBUG START | Gen {generation + 1} | Agent {agent_index + 1}, Game {game_index + 1} ---")

        max_moves = 50
        moves_played = 0

        while not self.env.is_game_over() and moves_played < max_moves:
            # 1. Find and perform the best move based on NN's value estimate
            best_action, estimated_score = one_step_estimator_move(self.env, estimator)

            # --- Conditional Print for First Move ---
            if is_debug_agent and moves_played % 10 == 0:
                # Need to use the estimator directly here since the move hasn't been performed yet
                initial_flat_state = self.env.get_flat_state()
                initial_estimated_score = estimator.predict_score(initial_flat_state)
                print(f"Move {str(moves_played)}/50: NN Estimated Initial Position Score: {initial_estimated_score:.2f}")
                print(f"Move {str(moves_played)}/50: Best Action Estimated Score: {estimated_score:.2f}")
                print("-" * 20)

            if best_action:
                self.env.perform_action(best_action)
                moves_played += 1
            else:
                break

        # Calculate final, actual score (true fitness)
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
            for i, estimator in enumerate(self.population):
                total_score = 0
                for j in range(self.num_games_per_nn):
                    total_score += self.run_game(estimator, i, j, generation)

                avg_score = total_score / self.num_games_per_nn
                fitnesses[i] = avg_score

            # Sort by fitness (descending)
            sorted_fitness = sorted(fitnesses.items(), key=lambda item: item[1], reverse=True)
            best_fitness = sorted_fitness[0][1]
            avg_fitness = np.mean(list(fitnesses.values()))

            print(
                f"Gen {generation + 1}/{self.generations} | Best Score: {best_fitness:.2f} | Avg Score: {avg_fitness:.2f} | Time: {time.time() - start_time:.2f}s")

            # 2. Selection (Elitism)
            num_elite = int(0.1 * self.population_size)
            elite_indices = [idx for idx, score in sorted_fitness[:num_elite]]
            best_nn_index = sorted_fitness[0][0]

            # 3. Save the single best agent
            self._save_best_agent(self.population[best_nn_index])

            new_population: List[CalicoValueEstimator] = []

            for idx in elite_indices:
                new_population.append(self.population[idx])

                # 4. Reproduction (Crossover and Mutation)
            while len(new_population) < self.population_size:
                parent1_idx = random.choice(sorted_fitness[:self.population_size // 2])[0]
                parent2_idx = random.choice(sorted_fitness[:self.population_size // 2])[0]

                parent1 = self.population[parent1_idx]
                parent2 = self.population[parent2_idx]

                child_params: List[np.ndarray] = []

                for p1_mat, p2_mat in zip(parent1.params, parent2.params):
                    # Simple Crossover
                    mask = np.random.rand(*p1_mat.shape) < 0.5
                    child_mat = np.where(mask, p1_mat, p2_mat)

                    # Mutation
                    mutation_rate = self.mutation_rate
                    mutation_mask = np.random.rand(*child_mat.shape) < mutation_rate
                    child_mat += mutation_mask * np.random.randn(*child_mat.shape) * 0.1

                    child_params.append(child_mat)

                # Create the new child NN
                child = CalicoValueEstimator(self.input_size, self.hidden_size, self.output_size, weights=child_params)

                new_population.append(child)

            self.population = new_population

        print("\nTraining Complete.")

        return self.population[best_nn_index]


# Main execution block
if __name__ == "__main__":
    trainer = NeuroevolutionaryTrainer(
        population_size=25,  # Number of NN agents per generation
        num_games_per_nn=5,  # Games to average score for fitness
        generations=50  # Number of evolutionary steps
    )

    best_agent = trainer.train()
    print("\n--- Final Best Agent (Value Estimator) ---")

    # Run a quick test on a new game state
    trainer.env.start_game()
    final_state = trainer.env.get_flat_state()
    final_estimated_score = best_agent.predict_score(final_state)

    print(f"Final Agent's Estimated Score for a fresh board: {final_estimated_score:.2f}")

    # Run a full test game and print the true final score
    true_score = trainer.run_game(best_agent, 99, 99, 99)  # Use dummy indices for final test
    print(f"Final Agent achieved True Final Score: {true_score:.2f}")