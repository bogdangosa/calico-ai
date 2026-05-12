import numpy as np
from typing import List, Optional, Dict

from src.engine.environments.calico_player_env import CalicoPlayerEnv
from src.models.game_config import GameSettings
from src.models.game_models import CalicoAction, ActionType

class CalicoMultiplayerEnv:
    """Orchestrates multiple players, shared tile pool, and shop."""
    def __init__(self, config: GameSettings, nr_of_players: int):
        self.config = config
        self.nr_of_players = nr_of_players
        self.players: List[CalicoPlayerEnv] = [
            CalicoPlayerEnv(i, config) for i in range(nr_of_players)
        ]
        
        self.tile_pool: np.ndarray = np.array([])
        self.shop_tiles: List[int] = []
        self.cat_tiles: np.ndarray = np.array([])
        
        self.current_player_idx: int = 0
        self.mode: ActionType = ActionType.PLACE

    def start_game(self, seed: Optional[int] = 42):
        if seed is not None:
            np.random.seed(seed)
            
        self.tile_pool = self._initiate_tile_pool()
        self.cat_tiles = self._initialize_cat_tiles()
        self.shop_tiles = [self._generate_random_tile() for _ in range(self.config.nr_of_tiles_in_shop)]
        
        # Assign different border colors if available
        available_colors = list(self.config.board.borders.keys())
        for i, player in enumerate(self.players):
            initial_hand = [self._generate_random_tile() for _ in range(self.config.player_hand_size)]
            color = available_colors[i % len(available_colors)]
            player.initialize_player(initial_hand, color)
            
        self.current_player_idx = 0
        self.mode = ActionType.PLACE

    def _initiate_tile_pool(self) -> np.ndarray:
        return np.full(self.config.tiles.colors * self.config.tiles.patterns, self.config.tiles.identical_tiles)

    def _initialize_cat_tiles(self) -> np.ndarray:
        cat_tiles = np.arange(1, self.config.tiles.cat_types + 1)
        np.random.shuffle(cat_tiles)
        return cat_tiles

    def _generate_random_tile(self) -> int:
        valid_indices = np.where(self.tile_pool > 0)[0]
        if len(valid_indices) == 0:
            return self.config.tiles.none_left
        tile_id = int(np.random.choice(valid_indices))
        self.tile_pool[tile_id] -= 1
        return tile_id

    def get_current_player(self) -> CalicoPlayerEnv:
        return self.players[self.current_player_idx]

    def get_legal_actions(self) -> List[CalicoAction]:
        current_player = self.get_current_player()
        legal_actions = []
        
        if self.mode == ActionType.PLACE:
            empty_slots = [
                (r, c) for r in range(self.config.board.size) for c in range(self.config.board.size)
                if current_player.board_matrix[r][c] == self.config.board.no_tile_value
            ]
            for idx, tile_id in enumerate(current_player.player_tiles):
                if tile_id == self.config.board.no_tile_value:
                    continue
                for r, c in empty_slots:
                    legal_actions.append(CalicoAction(
                        action_type=ActionType.PLACE,
                        tile_index=idx,
                        row=r,
                        col=c
                    ))
        elif self.mode == ActionType.BUY:
            for shop_idx in range(len(self.shop_tiles)):
                legal_actions.append(CalicoAction(
                    action_type=ActionType.BUY,
                    tile_index=shop_idx
                ))
        return legal_actions

    def perform_action(self, action: CalicoAction):
        current_player = self.get_current_player()
        
        if action.action_type == ActionType.PLACE:
            current_player.place_tile(action.row, action.col, action.tile_index)
            self.mode = ActionType.BUY
            
        elif action.action_type == ActionType.BUY:
            # Transfer tile from shop to player hand
            bought_tile_id = self.shop_tiles[action.tile_index]
            current_player.player_tiles[current_player.selected_player_tile_index] = bought_tile_id
            
            # Refill shop
            self.shop_tiles[action.tile_index] = self._generate_random_tile()
            
            # Advance turn
            self._advance_turn()

    def _advance_turn(self):
        self.current_player_idx = (self.current_player_idx + 1) % self.nr_of_players
        self.mode = ActionType.PLACE

    def is_game_over(self) -> bool:
        # Game is over when all players have full boards
        return all(p.is_board_full() for p in self.players)

    def get_state(self) -> Dict:
        return {
            "mode": self.mode,
            "current_player_idx": self.current_player_idx,
            "shop_tiles": self.shop_tiles,
            "players": [
                {
                    "player_id": p.player_id,
                    "player_tiles": p.player_tiles,
                    "board": p.board_matrix.tolist()
                } for p in self.players
            ],
            "is_game_over": self.is_game_over()
        }
