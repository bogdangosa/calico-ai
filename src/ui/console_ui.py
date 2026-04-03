import numpy as np
from src.engine.environments.calico_env import CalicoEnv
from src.engine.scoring.scoring import ScoringCalculator
from src.ui.table_renderer import TableRenderer


class CalicoConsoleUI:
    def __init__(self, env : CalicoEnv, renderer: TableRenderer,scorer : ScoringCalculator):
        self.env = env
        self.scorer = scorer
        self.renderer = renderer
        self.config = env.config
        self.empty_val = self.config.board.no_tile_value
        self.game_over = False

    def _get_input(self, prompt, valid_range=None):
        """Helper to get and validate integer input."""
        try:
            val = int(input(prompt))
            if valid_range and val not in valid_range:
                print(f"Invalid choice. Please pick from {list(valid_range)}.")
                return None
            return val
        except ValueError:
            print("Input error! Please enter an integer.")
            return None

    def handle_placing_phase(self, legal_actions):
        """Logic for the 'place' mode."""
        print(f"\n--- PLACING PHASE ---")

        hand_idx = self._get_input(f"Pick tile index from hand (0-{len(self.env.player_tiles) - 1}): ",
                                   range(len(self.env.player_tiles)))
        if hand_idx is None: return None

        row = self._get_input(f"Enter row (0-{self.config.board.size - 1}): ")
        col = self._get_input(f"Enter col (0-{self.config.board.size - 1}): ")

        for a in legal_actions:
            if a.action_type == "place" and a.tile_index == hand_idx and a.row == row and a.col == col:
                return a

        print("Invalid placement! Spot is either taken, an objective, or out of bounds.")
        return None

    def handle_buying_phase(self, legal_actions):
        """Logic for the 'buy' mode."""
        print(f"\n--- BUYING PHASE ---")
        print(f"Shop Tiles: {self.env.shop_tiles}")

        shop_idx = self._get_input(f"Pick tile index to buy from shop (0-{len(self.env.shop_tiles) - 1}): ",
                                   range(len(self.env.shop_tiles)))
        if shop_idx is None: return None

        for a in legal_actions:
            if a.action_type == "buy" and a.tile_index == shop_idx:
                return a

        print("Invalid shop index!")
        return None

    def get_user_action(self):
        """Determines the current mode and routes to the correct input handler."""
        legal_actions = self.env.get_legal_actions()
        if not legal_actions:
            self.game_over = True
            return None

        if len(self.env.player_tiles) == self.config.player_hand_size:
            return self.handle_placing_phase(legal_actions)
        else:
            return self.handle_buying_phase(legal_actions)

    def run(self):
        """The main game loop."""
        self.env.start_game()
        print("=== CALICO CONSOLE ===")

        while not self.game_over:
            self.renderer.render(self.env, self.scorer)

            action = self.get_user_action()

            if action:
                self.env.perform_action(action)
                print(f"Action Successful: {action.action_type}")

            if self.env.is_game_over():
                self.renderer.render(self.env, self.scorer)
                print("\nBoard is full! Game Over.")
                self.game_over = True

        total, *details = self.scorer.get_total_detailed_score(self.env.board_matrix, self.env.cat_tiles)
        print(f"Final Score: {total}")