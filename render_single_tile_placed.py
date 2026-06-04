import numpy as np
import os
from src.utils.config import load_config
from src.agents.temporal_difference.tabular_td_agent import TabularTDAgent
from src.engine.environments.calico_env import CalicoEnv
from src.engine.scoring.scoring import ScoringCalculator
from src.ui.table_renderer import TableRenderer


def render_single_tile_states():
    config_path = "config/micro_calico_settings_v2.json"
    v_table_path = "agent_models/micro_calico_v2/tabular_v_table_v1.3.pkl"

    if not os.path.exists(v_table_path):
        fallback = "agent_models/micro_calico_v2/tabular_q_table_v1.2.pkl"
        if os.path.exists(fallback):
            v_table_path = fallback

    if not os.path.exists(v_table_path):
        print(f"Error: V-table not found.")
        return

    print(f"Rendering states from: {v_table_path}")
    config = load_config(config_path)
    empty_value = config.board.no_tile_value

    env = CalicoEnv(config)
    env.start_game()
    base_board = env.board_matrix.copy()
    playable_mask = (base_board == empty_value)

    agent = TabularTDAgent(config)
    agent.load(v_table_path)

    renderer = TableRenderer(config)
    scorer = ScoringCalculator(config)

    match_count = 0

    for state_key, state_value in agent.v_table.items():
        hand_display_info = ""

        if len(state_key) == 4:
            tiles_placed = sum(1 for tile in state_key if tile != empty_value)
            if tiles_placed != 1:
                continue
            reconstructed_board = base_board.copy()
            reconstructed_board[playable_mask] = state_key

        elif len(state_key) == 6:
            board_slots = state_key[:4]
            hand_slots = state_key[4:]
            tiles_placed = sum(1 for tile in board_slots if tile != empty_value)
            if tiles_placed != 1:
                continue
            reconstructed_board = base_board.copy()
            reconstructed_board[playable_mask] = board_slots
            env.player_hand = list(hand_slots)
            hand_display_info = f" | Hand Tiles: {hand_slots}"

        elif len(state_key) == 16:
            flat_key = np.array(state_key)
            flat_mask = playable_mask.flatten()
            playable_elements = flat_key[flat_mask]
            tiles_placed = sum(1 for tile in playable_elements if tile != empty_value)
            if tiles_placed != 1:
                continue
            reconstructed_board = np.array(state_key).reshape((4, 4))

        else:
            continue

        match_count += 1
        print("\n" + "=" * 60)
        print(f" SINGLE TILE STATE #{match_count} | Table Value: {state_value:.4f}{hand_display_info} ")
        print("=" * 60)

        env.board_matrix = reconstructed_board
        renderer.render(env, scorer)


if __name__ == "__main__":
    render_single_tile_states()
