from typing import Optional
from src.models.game_models import CalicoAction, ActionType


class ActionMapper:
    def __init__(self, board_size: int, hand_size: int, shop_size: int):
        self.board_size = board_size
        self.inner_size = board_size - 2
        self.hand_size = hand_size
        self.shop_size = shop_size
        self.place_actions = self.hand_size * self.inner_size * self.inner_size
        self.total_actions = self.place_actions + self.shop_size

    def action_to_index(self, action: CalicoAction) -> Optional[int]:
        if action.action_type == ActionType.PLACE:
            row = action.row if action.row is not None else 0
            col = action.col if action.col is not None else 0

            inner_row = row - 1
            inner_col = col - 1

            if not (0 <= inner_row < self.inner_size and 0 <= inner_col < self.inner_size):
                return None
                
            slot_idx = inner_row * self.inner_size + inner_col
            return action.tile_index * (self.inner_size * self.inner_size) + slot_idx
            
        elif action.action_type == ActionType.BUY:
            return self.place_actions + action.tile_index
        return None

    def index_to_action(self, index: int) -> Optional[CalicoAction]:
        if index < 0 or index >= self.total_actions:
            return None
            
        if index < self.place_actions:
            inner_area = self.inner_size * self.inner_size
            tile_index = index // inner_area
            rem = index % inner_area
            inner_row = rem // self.inner_size
            inner_col = rem % self.inner_size
            
            row = inner_row + 1
            col = inner_col + 1
            return CalicoAction(action_type=ActionType.PLACE, tile_index=tile_index, row=row, col=col)
        else:
            shop_index = index - self.place_actions
            return CalicoAction(action_type=ActionType.BUY, tile_index=shop_index)
