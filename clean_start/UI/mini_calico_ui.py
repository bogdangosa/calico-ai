import numpy as np

from clean_start.environments.mini_calico_env import MicroCalicoEnv
from clean_start.environments.scheme import CalicoAction
from enviroment.mini_calico_env import Colors

PLAYER_HAND_SIZE=2
TILE_PATTERNS = 3
TILE_COLORS=3
NR_OF_IDENTICAL_TILES=2
BOARD_SIZE = 2
NO_TILE_VALUE = 37


class CalicoConsoleUI:
    def __init__(self, env):
        self.env = env
        self.game_over = False

    def get_user_input(self):
        """Prompts user and returns a CalicoAction or None if input is junk."""
        try:
            legal_actions = self.env.get_legal_actions()
            if not legal_actions:
                print("No legal moves left!")
                self.game_over = True
                return None

            print(f"\nAvailable Hand Indices: 0 to {len(self.env.player_tiles) - 1}")
            tile_idx = int(input("Pick tile index from hand: "))
            row = int(input("Enter row: "))
            col = int(input("Enter col: "))

            # Create the action object
            action = CalicoAction(action_type="place", tile_index=tile_idx, row=row, col=col)

            # Simple validation: Check if this action is in the legal list
            if any(a.tile_index == action.tile_index and a.row == action.row and a.col == action.col
                   for a in legal_actions):
                return action
            else:
                print(f"{Colors.RED}Invalid Move! Spot taken or out of bounds.{Colors.RESET}")
                return None

        except (ValueError, IndexError):
            print(f"{Colors.RED}Input error! Please enter integers only.{Colors.RESET}")
            return None

    def run(self):
        """The main game loop."""
        self.env.start_game()

        print(f"{Colors.BOLD}--- Welcome to Micro Calico ---{Colors.RESET}")

        while not self.game_over:
            self.env.render()

            # 1. Get Move
            action = self.get_user_input()

            # 2. Apply Move
            if action:
                self.env.perform_action(action)

            # 3. Check for Game Over (Board full)
            if not np.any(self.env.board_matrix == NO_TILE_VALUE):
                self.env.render()
                print(f"{Colors.BOLD}Board is full! Game Over.{Colors.RESET}")
                self.game_over = True


# --- Testing Script ---
if __name__ == "__main__":
    env = MicroCalicoEnv()
    ui = CalicoConsoleUI(env)
    ui.run()

