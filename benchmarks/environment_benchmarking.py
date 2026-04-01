import time
import copy
import numpy as np

# Adjust imports to match your project structure
try:
    from enviroment.calico_env import CalicoEnv
    from enviroment.calico_scoring import get_total_score_on_board
except ImportError:
    print("Ensure you are running this from the root directory or adjust imports.")

ITERATIONS = 2000


def run_benchmark(name, func, setup_func=None):
    """Helper to run and print results for a specific test."""
    print(f"Running {name} benchmark...")

    # Run setup if provided (to initialize env)
    env = setup_func() if setup_func else CalicoEnv()

    start = time.perf_counter()
    for _ in range(ITERATIONS):
        func(env)
    end = time.perf_counter()

    total_time = end - start
    avg_time_ms = (total_time / ITERATIONS) * 1000
    print(f"  -> Total: {total_time:.4f}s | Avg: {avg_time_ms:.4f} ms")
    print("-" * 40)


# --- SETUP HELPERS ---
def setup_standard():
    env = CalicoEnv()
    env.start_game()
    return env


def setup_buying_mode():
    env = CalicoEnv()
    env.start_game()
    env.mode = "buying"
    return env


# --- TEST CASES ---

def test_get_flat_state(env):
    """Benchmarks generating the NN input."""
    _ = env.get_flat_state()


def test_get_legal_actions(env):
    """Benchmarks calculating valid moves."""
    _ = env.get_legal_actions()


def test_perform_place_action(env):
    """Benchmarks placing a tile (logic + history recording)."""
    # We place at 1,1 repeatedly.
    # Since we don't care about game logic validty for speed testing,
    # we just overwrite the memory.
    action = ('place', 0, 1, 1)  # index 0, row 1, col 1
    env.perform_action(action)
    # We must undo or reset, otherwise history grows infinitely and skews result
    # However, to measure pure 'perform' speed, we usually just accept the growth
    # or manually pop history to keep it light without triggering full undo logic.
    env.move_history.pop()


def test_perform_buy_action(env):
    """Benchmarks buying a tile."""
    # Action: buy, shop_index 0
    action = ('buy', 0, None, None)
    env.perform_action(action)
    env.move_history.pop()  # Keep history small


def test_undo_action(env):
    """Benchmarks the undo logic."""
    # Setup: Do one action first
    action = ('place', 0, 1, 1)
    env.perform_action(action)
    # Time the undo
    env.undo_action()


def test_env_deepcopy(env):
    """Benchmarks cloning the whole environment (Critical for MCTS)."""
    _ = copy.deepcopy(env)


# --- MAIN RUNNER ---

if __name__ == "__main__":
    print(f"=== BENCHMARK SUITE ({ITERATIONS} iterations) ===\n")


    # 1. Scoring (From your original code)
    def run_scoring(env):
        get_total_score_on_board(env.board_matrix, env.cat_tiles)


    def setup_full_board():
        env = CalicoEnv()
        env.start_game()
        env.fill_board_randomly()
        return env


    run_benchmark("Scoring", run_scoring, setup_full_board)

    # 2. Starting Game
    run_benchmark("Start Game", lambda e: e.start_game())

    # 3. Observation Speed (Critical for NN Training)
    run_benchmark("Get Flat State (NN Input)", test_get_flat_state, setup_standard)

    # 4. Action Masking
    run_benchmark("Get Legal Actions", test_get_legal_actions, setup_standard)

    # 5. Transition Speed (Place)
    run_benchmark("Perform Action (Place)", test_perform_place_action, setup_standard)

    # 6. Transition Speed (Buy)
    # Note: This is expected to be slower due to copy.deepcopy in your buy_tile logic
    run_benchmark("Perform Action (Buy)", test_perform_buy_action, setup_buying_mode)

    # 7. Undo Speed
    # We pass a lambda to handle the setup/teardown within the loop logic specific to undo
    # But for simplicity in the helper, let's just run the test function which handles the 'do'
    # Actually, the test_undo_action does a do-then-undo.
    run_benchmark("Place & Undo Cycle", test_undo_action, setup_standard)

    # 8. MCTS Cloning
    run_benchmark("Full Env Deepcopy", test_env_deepcopy, setup_standard)