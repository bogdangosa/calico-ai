import copy
from src.models.game_models import ActionType, CalicoAction


class HistoryManager:
    def __init__(self, env):
        self.env = env
        self.history = []

    def record_and_perform(self, action: CalicoAction):
        record = {
            "action_type": action.action_type,
            "prev_mode": self.env.mode,
            "prev_selected_index": self.env.selected_player_tile_index
        }

        if action.action_type == ActionType.PLACE:
            record.update({
                "row": action.row,
                "col": action.col,
                "hand_index": action.tile_index,
                "hand_value": self.env.player_tiles[action.tile_index],
                "prev_tile": self.env.board_matrix[action.row, action.col],
            })
            self.env.place_tile(action.row, action.col, action.tile_index)

        elif action.action_type == ActionType.BUY:
            record.update({
                "shop_index": action.tile_index,
                "hand_value": self.env.player_tiles[self.env.selected_player_tile_index],
                "shop_snapshot": copy.deepcopy(self.env.shop_tiles)
            })
            self.env.buy_tile(action.tile_index)
            record["new_shop_snapshot"] = copy.deepcopy(self.env.shop_tiles)

        self.history.append(record)

    def undo(self):
        if not self.history:
            return

        record = self.history.pop()
        action_type = record["action_type"]

        if action_type == ActionType.PLACE:
            self.env.board_matrix[record["row"], record["col"]] = record["prev_tile"]
            self.env.player_tiles[record["hand_index"]] = record["hand_value"]
            self.env.mode = record["prev_mode"]
            self.env.selected_player_tile_index = record["prev_selected_index"]

        elif action_type == ActionType.BUY:
            self.env.shop_tiles = record["shop_snapshot"]
            self.env.player_tiles[self.env.selected_player_tile_index] = record["hand_value"]

            new_shop = record["new_shop_snapshot"]
            for tile_id in new_shop:
                self.env.tile_pool[tile_id] += 1

            self.env.mode = record["prev_mode"]
            self.env.selected_player_tile_index = record["prev_selected_index"]