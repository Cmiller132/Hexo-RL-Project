"""PNG match snapshots for dashboard and autonomous run inspection.

The renderer intentionally has no third-party image dependency.  It writes a
small RGB PNG directly so the tuning orchestrator can request board images even
from a minimal training environment.
"""

from __future__ import annotations

import math
import re
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from hexorl.dashboard.replay import Move, decode_move_history


HEX_SIZE = 24.0
NEIGHBORS = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1))


@dataclass(frozen=True)
class MatchSnapshotOptions:
    width: int = 1280
    height: int = 960
    turn_index: int | None = None
    context_rings: int = 2
    show_numbers: bool = True
    show_legal: bool = False
    fit: str = "played"
    title: str = ""


def render_match_snapshot_png(
    history: bytes,
    *,
    options: MatchSnapshotOptions | None = None,
    legal_moves: Iterable[dict[str, Any]] | None = None,
    metadata: dict[str, Any] | None = None,
) -> bytes:
    """Render one game history as a dashboard-style PNG."""
    opts = options or MatchSnapshotOptions()
    width = _clamp(int(opts.width), 320, 4096)
    height = _clamp(int(opts.height), 260, 4096)
    moves = _truncate_moves(decode_move_history(history), opts.turn_index)
    stones = [
        {"player": int(player), "q": int(q), "r": int(r), "move": idx + 1}
        for idx, (player, q, r) in enumerate(moves)
    ]
    legal = _legal_pairs(legal_moves) if opts.show_legal else set()
    cells = _snapshot_cells(stones, legal, max(0, min(int(opts.context_rings), 8)))
    canvas = _Canvas(width, height, _rgb("#0a0e14"))
    canvas.fill_rect(0, 0, width, height, _rgb("#0d1117"))
    canvas.fill_rect(0, 0, width, 54, _rgb("#060a0f"))
    canvas.draw_line(0, 54, width - 1, 54, _rgb("#30363d"), thickness=1)

    transform = _fit_transform(cells, stones, width, height, opts.fit)
    stone_by_cell = {(s["q"], s["r"]): s for s in stones}
    last = (stones[-1]["q"], stones[-1]["r"]) if stones else None

    for q, r in sorted(cells, key=lambda item: (item[1], item[0])):
        x, y, size = _screen_hex(q, r, transform)
        if x < -size or x > width + size or y < 54 - size or y > height + size:
            continue
        stone = stone_by_cell.get((q, r))
        is_legal = (q, r) in legal
        if stone:
            fill, stroke = _player_colors(int(stone["player"]))
            canvas.fill_polygon(_hex_points(x, y, size * 0.98), fill)
            canvas.stroke_polygon(_hex_points(x, y, size * 0.98), stroke, thickness=max(1, int(size / 16)))
        else:
            fill = _rgb("#101923") if is_legal else _rgb("#111820")
            stroke = _rgb("#285284") if is_legal else _rgb("#1b3c62")
            canvas.fill_polygon(_hex_points(x, y, size * 0.98), fill, alpha=0.72 if is_legal else 0.52)
            canvas.stroke_polygon(_hex_points(x, y, size * 0.98), stroke, thickness=1)
        if last == (q, r):
            canvas.stroke_polygon(_hex_points(x, y, size * 1.03), _rgb("#ffffff"), thickness=max(2, int(size / 10)))
        if stone and opts.show_numbers:
            number = str(stone["move"])
            scale = max(1, min(4, int(size / max(8.5, len(number) * 3.2))))
            canvas.draw_text_centered(number, int(x), int(y - 4 * scale), _rgb("#ffffff"), scale=scale, stroke=True)

    title = opts.title or _title_from_metadata(metadata or {})
    if title:
        canvas.draw_text(_clean_label(title), 18, 16, _rgb("#f0f6fc"), scale=2, stroke=True)
    current_player = _current_player_after(len(moves))
    remaining = _placements_remaining_after(len(moves))
    subtitle = f"MOVE {len(moves)}  P{current_player} TO MOVE  PLACES {remaining}"
    if metadata:
        run_id = str(metadata.get("run_id") or "")
        game_id = str(metadata.get("game_id") or "")
        if run_id or game_id:
            subtitle = f"{subtitle}  GAME {game_id}  RUN {run_id}"
    canvas.draw_text(_clean_label(subtitle), 18, 38, _rgb("#8b949e"), scale=1, stroke=False)
    return canvas.to_png()


def write_match_snapshot_png(
    path: Path | str,
    history: bytes,
    *,
    options: MatchSnapshotOptions | None = None,
    legal_moves: Iterable[dict[str, Any]] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(
        render_match_snapshot_png(
            history,
            options=options,
            legal_moves=legal_moves,
            metadata=metadata,
        )
    )
    return target


def snapshot_filename(row: dict[str, Any], *, turn_index: int | None = None) -> str:
    run = _slug(str(row.get("run_id") or "run"))
    game = _slug(str(row.get("game_id") or row.get("external_game_id") or "game"))
    move = "final" if turn_index is None or int(turn_index) < 0 else f"move_{int(turn_index):04d}"
    return f"{run}_game_{game}_{move}.png"


def _truncate_moves(moves: list[Move], turn_index: int | None) -> list[Move]:
    if turn_index is None or int(turn_index) < 0:
        return moves
    return moves[: max(0, min(int(turn_index), len(moves)))]


def _legal_pairs(legal_moves: Iterable[dict[str, Any]] | None) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for move in legal_moves or []:
        if "q" in move and "r" in move:
            pairs.add((int(move["q"]), int(move["r"])))
    return pairs


def _snapshot_cells(
    stones: list[dict[str, int]],
    legal: set[tuple[int, int]],
    context_rings: int,
) -> set[tuple[int, int]]:
    coords = {(int(s["q"]), int(s["r"])) for s in stones}
    if legal:
        coords |= legal
    if not coords:
        coords = {(0, 0)}
    frontier = set(coords)
    for _ in range(context_rings):
        next_frontier: set[tuple[int, int]] = set()
        for q, r in frontier:
            for dq, dr in NEIGHBORS:
                nxt = (q + dq, r + dr)
                if nxt not in coords:
                    next_frontier.add(nxt)
        coords |= next_frontier
        frontier = next_frontier
    return coords


def _fit_transform(
    cells: set[tuple[int, int]],
    stones: list[dict[str, int]],
    width: int,
    height: int,
    fit: str,
) -> dict[str, float]:
    focus_coords = {(int(s["q"]), int(s["r"])) for s in stones}
    if str(fit).lower() in {"all", "legal", "full"} or not focus_coords:
        focus_coords = cells
    raw = [_raw_hex(q, r) for q, r in focus_coords]
    min_x = min(x - HEX_SIZE for x, _ in raw)
    max_x = max(x + HEX_SIZE for x, _ in raw)
    min_y = min(y - HEX_SIZE for _, y in raw)
    max_y = max(y + HEX_SIZE for _, y in raw)
    content_w = max(HEX_SIZE * 6.0, max_x - min_x + HEX_SIZE * 4.0)
    content_h = max(HEX_SIZE * 6.0, max_y - min_y + HEX_SIZE * 4.0)
    available_w = max(120.0, width - 64.0)
    available_h = max(120.0, height - 94.0)
    scale = min(2.8, max(0.18, min(available_w / content_w, available_h / content_h)))
    cx = (min_x + max_x) / 2.0
    cy = (min_y + max_y) / 2.0
    return {
        "scale": scale,
        "offset_x": width / 2.0 - cx * scale,
        "offset_y": 54.0 + available_h / 2.0 - cy * scale + 18.0,
    }


def _screen_hex(q: int, r: int, transform: dict[str, float]) -> tuple[float, float, float]:
    x, y = _raw_hex(q, r)
    scale = transform["scale"]
    return x * scale + transform["offset_x"], y * scale + transform["offset_y"], HEX_SIZE * scale


def _raw_hex(q: int, r: int) -> tuple[float, float]:
    return HEX_SIZE * (1.5 * q), HEX_SIZE * ((math.sqrt(3.0) / 2.0) * q + math.sqrt(3.0) * r)


def _hex_points(cx: float, cy: float, size: float) -> list[tuple[float, float]]:
    return [
        (cx + size * math.cos(math.pi / 3.0 * i), cy + size * math.sin(math.pi / 3.0 * i))
        for i in range(6)
    ]


def _player_colors(player: int) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    if player == 1:
        return _rgb("#dd3333"), _rgb("#ff9d93")
    return _rgb("#3377ee"), _rgb("#95b8ff")


def _current_player_after(move_count: int) -> int:
    if move_count <= 0:
        return 0
    return 1 if ((move_count - 1) // 2) % 2 == 0 else 0


def _placements_remaining_after(move_count: int) -> int:
    if move_count <= 0:
        return 1
    return 2 if (move_count - 1) % 2 == 0 else 1


def _title_from_metadata(metadata: dict[str, Any]) -> str:
    source = str(metadata.get("source") or "MATCH")
    outcome = metadata.get("outcome")
    if outcome is None:
        return source
    return f"{source}  OUTCOME {outcome}"


def _clean_label(value: str) -> str:
    value = value.upper()
    return "".join(ch if ch in _FONT or ch == " " else " " for ch in value)[:120]


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return slug.strip("._") or "item"


def _rgb(value: str) -> tuple[int, int, int]:
    value = value.strip().lstrip("#")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def _clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


class _Canvas:
    def __init__(self, width: int, height: int, bg: tuple[int, int, int]) -> None:
        self.width = width
        self.height = height
        self.pixels = bytearray(bytes(bg) * (width * height))

    def fill_rect(self, x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int], alpha: float = 1.0) -> None:
        x0 = _clamp(x0, 0, self.width)
        x1 = _clamp(x1, 0, self.width)
        y0 = _clamp(y0, 0, self.height)
        y1 = _clamp(y1, 0, self.height)
        for y in range(y0, y1):
            for x in range(x0, x1):
                self._set_pixel(x, y, color, alpha)

    def fill_polygon(self, points: list[tuple[float, float]], color: tuple[int, int, int], alpha: float = 1.0) -> None:
        if len(points) < 3:
            return
        min_y = max(0, int(math.floor(min(y for _, y in points))))
        max_y = min(self.height - 1, int(math.ceil(max(y for _, y in points))))
        n = len(points)
        for y in range(min_y, max_y + 1):
            scan_y = y + 0.5
            xs: list[float] = []
            for i in range(n):
                x1, y1 = points[i]
                x2, y2 = points[(i + 1) % n]
                if (y1 <= scan_y < y2) or (y2 <= scan_y < y1):
                    t = (scan_y - y1) / (y2 - y1)
                    xs.append(x1 + t * (x2 - x1))
            xs.sort()
            for i in range(0, len(xs) - 1, 2):
                x0 = max(0, int(math.ceil(xs[i])))
                x1 = min(self.width - 1, int(math.floor(xs[i + 1])))
                for x in range(x0, x1 + 1):
                    self._set_pixel(x, y, color, alpha)

    def stroke_polygon(self, points: list[tuple[float, float]], color: tuple[int, int, int], thickness: int = 1) -> None:
        for idx, (x1, y1) in enumerate(points):
            x2, y2 = points[(idx + 1) % len(points)]
            self.draw_line(int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)), color, thickness=thickness)

    def draw_line(
        self,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        color: tuple[int, int, int],
        *,
        thickness: int = 1,
    ) -> None:
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        x, y = x0, y0
        radius = max(0, thickness // 2)
        while True:
            self._dot(x, y, radius, color)
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x += sx
            if e2 <= dx:
                err += dx
                y += sy

    def draw_text(
        self,
        text: str,
        x: int,
        y: int,
        color: tuple[int, int, int],
        *,
        scale: int = 1,
        stroke: bool = False,
    ) -> None:
        if stroke:
            for ox, oy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                self._draw_text_raw(text, x + ox * scale, y + oy * scale, _rgb("#000000"), scale)
        self._draw_text_raw(text, x, y, color, scale)

    def draw_text_centered(
        self,
        text: str,
        cx: int,
        cy: int,
        color: tuple[int, int, int],
        *,
        scale: int = 1,
        stroke: bool = False,
    ) -> None:
        width = _text_width(text, scale)
        self.draw_text(text, cx - width // 2, cy, color, scale=scale, stroke=stroke)

    def _draw_text_raw(self, text: str, x: int, y: int, color: tuple[int, int, int], scale: int) -> None:
        cursor = x
        for char in text:
            glyph = _FONT.get(char, _FONT[" "])
            for row, pattern in enumerate(glyph):
                for col, bit in enumerate(pattern):
                    if bit == "1":
                        self.fill_rect(cursor + col * scale, y + row * scale, cursor + (col + 1) * scale, y + (row + 1) * scale, color)
            cursor += (len(glyph[0]) + 1) * scale

    def _dot(self, cx: int, cy: int, radius: int, color: tuple[int, int, int]) -> None:
        for y in range(cy - radius, cy + radius + 1):
            for x in range(cx - radius, cx + radius + 1):
                if radius <= 1 or (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2:
                    self._set_pixel(x, y, color)

    def _set_pixel(self, x: int, y: int, color: tuple[int, int, int], alpha: float = 1.0) -> None:
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return
        idx = (y * self.width + x) * 3
        if alpha >= 0.999:
            self.pixels[idx : idx + 3] = bytes(color)
            return
        inv = 1.0 - alpha
        self.pixels[idx] = int(self.pixels[idx] * inv + color[0] * alpha)
        self.pixels[idx + 1] = int(self.pixels[idx + 1] * inv + color[1] * alpha)
        self.pixels[idx + 2] = int(self.pixels[idx + 2] * inv + color[2] * alpha)

    def to_png(self) -> bytes:
        raw = bytearray()
        stride = self.width * 3
        for y in range(self.height):
            raw.append(0)
            start = y * stride
            raw.extend(self.pixels[start : start + stride])
        return _png_bytes(self.width, self.height, bytes(raw))


def _png_bytes(width: int, height: int, raw: bytes) -> bytes:
    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, level=6))
        + chunk(b"IEND", b"")
    )


def _text_width(text: str, scale: int) -> int:
    width = 0
    for char in text:
        glyph = _FONT.get(char, _FONT[" "])
        width += (len(glyph[0]) + 1) * scale
    return max(0, width - scale)


_FONT: dict[str, tuple[str, ...]] = {
    " ": ("000", "000", "000", "000", "000", "000", "000"),
    "0": ("111", "101", "101", "101", "101", "101", "111"),
    "1": ("010", "110", "010", "010", "010", "010", "111"),
    "2": ("111", "001", "001", "111", "100", "100", "111"),
    "3": ("111", "001", "001", "111", "001", "001", "111"),
    "4": ("101", "101", "101", "111", "001", "001", "001"),
    "5": ("111", "100", "100", "111", "001", "001", "111"),
    "6": ("111", "100", "100", "111", "101", "101", "111"),
    "7": ("111", "001", "010", "010", "010", "010", "010"),
    "8": ("111", "101", "101", "111", "101", "101", "111"),
    "9": ("111", "101", "101", "111", "001", "001", "111"),
    "A": ("010", "101", "101", "111", "101", "101", "101"),
    "B": ("110", "101", "101", "110", "101", "101", "110"),
    "C": ("111", "100", "100", "100", "100", "100", "111"),
    "D": ("110", "101", "101", "101", "101", "101", "110"),
    "E": ("111", "100", "100", "110", "100", "100", "111"),
    "F": ("111", "100", "100", "110", "100", "100", "100"),
    "G": ("111", "100", "100", "101", "101", "101", "111"),
    "H": ("101", "101", "101", "111", "101", "101", "101"),
    "I": ("111", "010", "010", "010", "010", "010", "111"),
    "J": ("001", "001", "001", "001", "001", "101", "111"),
    "K": ("101", "101", "110", "100", "110", "101", "101"),
    "L": ("100", "100", "100", "100", "100", "100", "111"),
    "M": ("101", "111", "111", "101", "101", "101", "101"),
    "N": ("101", "111", "111", "111", "111", "111", "101"),
    "O": ("111", "101", "101", "101", "101", "101", "111"),
    "P": ("111", "101", "101", "111", "100", "100", "100"),
    "Q": ("111", "101", "101", "101", "111", "001", "001"),
    "R": ("110", "101", "101", "110", "110", "101", "101"),
    "S": ("111", "100", "100", "111", "001", "001", "111"),
    "T": ("111", "010", "010", "010", "010", "010", "010"),
    "U": ("101", "101", "101", "101", "101", "101", "111"),
    "V": ("101", "101", "101", "101", "101", "101", "010"),
    "W": ("101", "101", "101", "101", "111", "111", "101"),
    "X": ("101", "101", "101", "010", "101", "101", "101"),
    "Y": ("101", "101", "101", "010", "010", "010", "010"),
    "Z": ("111", "001", "001", "010", "100", "100", "111"),
    "-": ("000", "000", "000", "111", "000", "000", "000"),
    "_": ("000", "000", "000", "000", "000", "000", "111"),
    ".": ("000", "000", "000", "000", "000", "000", "010"),
    ":": ("000", "010", "000", "000", "000", "010", "000"),
    "/": ("001", "001", "010", "010", "010", "100", "100"),
    "#": ("101", "111", "101", "101", "111", "101", "101"),
    "+": ("000", "010", "010", "111", "010", "010", "000"),
}
