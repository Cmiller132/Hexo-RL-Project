import { useMemo, useState } from "react";
import type { AnyRow } from "../api/client";

const HEX_SIZE = 24;
const NEIGHBORS = [[1, 0], [-1, 0], [0, 1], [0, -1], [1, -1], [-1, 1]];

export function Board({ position, interactive = false, onCellClick, overlayMoves = [], viewKey }: {
  position?: AnyRow | null;
  interactive?: boolean;
  onCellClick?: (q: number, r: number) => void;
  overlayMoves?: AnyRow[];
  viewKey?: string | number | null;
}) {
  const [hover, setHover] = useState<AnyRow | null>(null);
  const geometry = useMemo(() => buildBoardGeometry(position, overlayMoves), [position, overlayMoves, viewKey]);
  const stones = Array.isArray(position?.stones) ? position.stones as AnyRow[] : [];
  const legal = Array.isArray(position?.legal_moves) ? position.legal_moves as AnyRow[] : [];
  const moves = Array.isArray(position?.moves) ? position.moves as AnyRow[] : [];
  const legalSet = new Set(legal.map((m) => `${m.q},${m.r}`));
  const overlayMap = new Map(overlayMoves.map((m) => [`${m.q},${m.r}`, m]));
  const moveNum = new Map(moves.map((m, i) => [`${m.q},${m.r}`, i + 1]));
  const stoneMap = new Map(stones.map((s) => [`${s.q},${s.r}`, s]));
  const currentPlayer = Number(position?.current_player ?? 0);
  const clickCell = (q: number, r: number) => {
    if (interactive && legalSet.has(`${q},${r}`)) onCellClick?.(q, r);
  };
  return (
    <div className="viewerBoardArea">
      <svg className={`board ${interactive ? "interactive" : ""}`} viewBox={`0 0 ${geometry.width} ${geometry.height}`} onMouseLeave={() => setHover(null)}>
        {geometry.cells.map((cell) => {
          const key = `${cell.q},${cell.r}`;
          const stone = stoneMap.get(key);
          const overlay = overlayMap.get(key);
          const isLegal = legalSet.has(key);
          const score = Number(overlay?.score ?? 0);
          return (
            <g key={key}>
              <path
                d={hexPath(cell.x, cell.y, 23)}
                className={["hexCell", stone ? `stone p${stone.player}` : "empty", isLegal ? "legal" : "", overlay ? "overlay" : "", interactive && isLegal ? "clickable" : ""].filter(Boolean).join(" ")}
                style={overlay ? { "--overlay-alpha": Math.min(0.82, 0.16 + Math.abs(score) * 0.36) } as React.CSSProperties : undefined}
                onClick={() => clickCell(cell.q, cell.r)}
                onMouseEnter={() => setHover({ q: cell.q, r: cell.r, legal: isLegal, score })}
              />
              {stone && <text className="moveNumber" x={cell.x} y={cell.y + 4}>{moveNum.get(key) || ""}</text>}
              {overlay && !stone && <text className="overlayValue" x={cell.x} y={cell.y + 3}>{score.toFixed(2)}</text>}
            </g>
          );
        })}
        <g className="boardBadge">
          <rect x="8" y="8" width="128" height="42" rx="5" />
          <circle cx="22" cy="24" r="6" className={`badgeDot p${currentPlayer}`} />
          <text x="34" y="28">P{currentPlayer} to move</text>
          <text x="22" y="43">Move {String(position?.turn_index ?? 0)}</text>
        </g>
      </svg>
      <div className="coordTip">{hover ? `(${hover.q}, ${hover.r}) ${hover.legal ? "legal" : "not legal"}` : "Hover a cell"}</div>
    </div>
  );
}

function buildBoardGeometry(position: AnyRow | null | undefined, overlayMoves: AnyRow[]) {
  const coords = new Set<string>();
  const add = (q: number, r: number, neighbors = true) => {
    coords.add(`${q},${r}`);
    if (neighbors) NEIGHBORS.forEach(([dq, dr]) => coords.add(`${q + dq},${r + dr}`));
  };
  for (const list of [position?.stones, position?.legal_moves, position?.moves, overlayMoves]) {
    if (Array.isArray(list)) list.forEach((m: AnyRow) => add(Number(m.q), Number(m.r)));
  }
  if (!coords.size) for (let q = -3; q <= 3; q++) for (let r = -3; r <= 3; r++) add(q, r, false);
  const parsed = [...coords].map((key) => {
    const [q, r] = key.split(",").map(Number);
    const c = hexCenter(q, r);
    return { q, r, rawX: c.x, rawY: c.y };
  });
  const minX = Math.min(...parsed.map((c) => c.rawX - HEX_SIZE));
  const maxX = Math.max(...parsed.map((c) => c.rawX + HEX_SIZE));
  const minY = Math.min(...parsed.map((c) => c.rawY - HEX_SIZE));
  const maxY = Math.max(...parsed.map((c) => c.rawY + HEX_SIZE));
  return {
    width: Math.max(360, maxX - minX + 44),
    height: Math.max(360, maxY - minY + 44),
    cells: parsed.map((c) => ({ q: c.q, r: c.r, x: c.rawX - minX + 22, y: c.rawY - minY + 22 })).sort((a, b) => a.r - b.r || a.q - b.q)
  };
}

function hexCenter(q: number, r: number) {
  return { x: HEX_SIZE * (1.5 * q), y: HEX_SIZE * ((Math.sqrt(3) / 2) * q + Math.sqrt(3) * r) };
}

function hexPath(cx: number, cy: number, size: number) {
  const pts = [];
  for (let i = 0; i < 6; i++) {
    const a = (Math.PI / 3) * i;
    pts.push(`${(cx + size * Math.cos(a)).toFixed(2)},${(cy + size * Math.sin(a)).toFixed(2)}`);
  }
  return `M${pts.join("L")}Z`;
}
