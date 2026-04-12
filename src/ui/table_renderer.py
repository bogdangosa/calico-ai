class TableRenderer:
    PATTERNS = ["●", "✚", "▲", "■", "♦", "*"]

    def __init__(self, config):
        self.config = config
        self.color_map = {
            0: "\033[95m",  # Purple
            1: "\033[94m",  # Blue
            2: "\033[92m",  # Green
            3: "\033[93m",  # Yellow
            4: "\033[91m",  # Red
            5: "\033[96m",  # Cyan
            "reset": "\033[0m",
            "bold": "\033[1m",
            "gray": "\033[90m"
        }

    def _get_tile_visual(self, tile_id):
        """Converts a tile ID into a colored symbol string."""
        if tile_id == self.config.board.no_tile_value:
            return f"{self.color_map['gray']} . {self.color_map['reset']}"

        if tile_id < 0:
            return f"{self.color_map['bold']}OB{abs(tile_id)}{self.color_map['reset']}"

        color_idx = tile_id // self.config.tiles.patterns
        pattern_idx = tile_id % self.config.tiles.patterns

        c_code = self.color_map.get(color_idx, self.color_map['reset'])
        sym = self.PATTERNS[pattern_idx % len(self.PATTERNS)]

        return f"{c_code} {sym} {self.color_map['reset']}"

    def _render_board(self, env) -> list:
        """Generates the hex-grid visual representation of the board."""
        board_rows = []
        for r in range(env.size):
            indent = "  " if r % 2 == 1 else ""
            line_parts = [f"{r:2} {indent}"]

            for c in range(env.size):
                tile_id = env.board_matrix[r, c]
                tile_visual = self._get_tile_visual(tile_id)
                line_parts.append(f"[{tile_visual}]")

            board_rows.append(" ".join(line_parts))
        return board_rows

    def _render_cat_objectives(self, cat_tiles) -> str:
        """Formats the active cat pattern requirements."""
        symbols = [self.PATTERNS[(p - 1) % len(self.PATTERNS)] for p in cat_tiles]
        label = f"{self.color_map['bold']}Cats want patterns:{self.color_map['reset']}"
        return f"{label} {' '.join(symbols)}"

    def _render_tile_list(self, label, tiles) -> str:
        """Generic helper to render a list of tiles (Hand or Shop)."""
        visuals = [self._get_tile_visual(t) for t in tiles]
        bold_label = f"{self.color_map['bold']}{label}:{self.color_map['reset']}"
        padding = " " * (10 - len(label))
        return f"{bold_label}{padding}{' '.join(visuals)}"

    def _render_score(self, env, scorer) -> str:
        """Calculates and formats the current score display."""
        total, *details = scorer.get_total_detailed_score(env.board_matrix, env.cat_tiles)
        label = f"{self.color_map['bold']}Current Score:{self.color_map['reset']}"
        return f"{label} {total}"

    def render(self, env, scorer=None):
        """Renders the current state of the Calico game to the console."""
        res = [f"\n{self.color_map['bold']}=== CALICO BOARD ==={self.color_map['reset']}"]

        res.extend(self._render_board(env))
        res.append("-" * 30)

        res.append(self._render_cat_objectives(env.cat_tiles))
        res.append(self._render_tile_list("Your Hand", env.player_tiles))
        res.append(self._render_tile_list("Shop", env.shop_tiles))

        if scorer:
            res.append(self._render_score(env, scorer))

        res.append("=" * 30 + "\n")
        print("\n".join(res))