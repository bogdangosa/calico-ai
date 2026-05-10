from src.models.game_models import CalicoAction, ActionType


class ActionMapper:
    def __init__(self, board_size: int, hand_size: int, shop_size: int):
        self.board_size = board_size
        self.hand_size = hand_size
        self.shop_size = shop_size
        self.total_actions = hand_size * board_size * board_size + shop_size

    def action_to_index(self, action: CalicoAction) -> int:
        if action.action_type == ActionType.PLACE:
            row = action.row if action.row is not None else 0
            col = action.col if action.col is not None else 0
            return action.tile_index * (self.board_size * self.board_size) + row * self.board_size + col
        elif action.action_type == ActionType.BUY:
            return self.hand_size * self.board_size * self.board_size + action.tile_index
        return 0

    def index_to_action(self, index: int) -> CalicoAction:
        place_actions = self.hand_size * self.board_size * self.board_size
        if index < place_actions:
            tile_index = index // (self.board_size * self.board_size)
            rem = index % (self.board_size * self.board_size)
            row = rem // self.board_size
            col = rem % self.board_size
            return CalicoAction(action_type=ActionType.PLACE, tile_index=tile_index, row=row, col=col)
        else:
            shop_index = index - place_actions
            return CalicoAction(action_type=ActionType.BUY, tile_index=shop_index)
