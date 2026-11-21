// Sokoban solver entry for make.fun's sokoban-solver-01-run-20261121 (program/v1).
// Runs as `node /bot/main.ts` on Node 22: erasable TypeScript only, node: built-ins, no network.
//
// Strategy: generated levels decompose into rooms separated by the hallway. Each room
// holds as many boxes as goals, so every room can be solved to its own push optimum
// independently with A* over pushes (admissible box-to-goal matching heuristic +
// deadlock pruning); the player then walks through the hallway between rooms.
// Deterministic: no clock, fixed iteration order, bounded searches.
import { mkdirSync, readdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const DIRS: [number, number, string][] = [
  [0, -1, 'u'],
  [0, 1, 'd'],
  [-1, 0, 'l'],
  [1, 0, 'r'],
];

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

interface Level {
  w: number;
  h: number;
  wall: Uint8Array;
  goal: Uint8Array;
  box: Uint8Array;
  player: number;
}

function parse(text: string): Level {
  const rows = text.split('\n');
  let w = 0;
  for (const row of rows) {
    if (row.length > w) {
      w = row.length;
    }
  }
  const h = rows.length;
  const n = w * h;
  const wall = new Uint8Array(n).fill(1);
  const goal = new Uint8Array(n);
  const box = new Uint8Array(n);
  let player = -1;
  for (let y = 0; y < h; y++) {
    const row = rows[y];
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      const ch = x < row.length ? row[x] : '#';
      if (ch === '#') {
        continue;
      }
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') {
        goal[i] = 1;
      }
      if (ch === '$' || ch === '*') {
        box[i] = 1;
      }
      if (ch === '@' || ch === '+') {
        player = i;
      }
    }
  }
  return { w, h, wall, goal, box, player };
}

function inBounds(w: number, h: number, x: number, y: number): boolean {
  return x >= 0 && y >= 0 && x < w && y < h;
}

function idx(w: number, x: number, y: number): number {
  return y * w + x;
}

function floorNeighbors(w: number, h: number, wall: Uint8Array, i: number): number[] {
  const out: number[] = [];
  const x = i % w;
  const y = (i / w) | 0;
  for (let d = 0; d < 4; d++) {
    const qx = x + DIRS[d][0];
    const qy = y + DIRS[d][1];
    if (!inBounds(w, h, qx, qy)) {
      continue;
    }
    const j = qy * w + qx;
    if (!wall[j]) {
      out.push(j);
    }
  }
  return out;
}

// Squares on which a box could never be pushed onto any goal (static reverse search).
function deadSquares(w: number, h: number, wall: Uint8Array, goal: Uint8Array, allowed: Uint8Array): Uint8Array {
  const n = w * h;
  const can = new Uint8Array(n);
  const queue: number[] = [];
  for (let i = 0; i < n; i++) {
    if (goal[i] && allowed[i]) {
      can[i] = 1;
      queue.push(i);
    }
  }
  for (let qi = 0; qi < queue.length; qi++) {
    const g = queue[qi]!;
    const gx = g % w;
    const gy = (g / w) | 0;
    for (let d = 0; d < 4; d++) {
      const p = g - DIRS[d][0] - DIRS[d][1] * w;
      const stand = p - DIRS[d][0] - DIRS[d][1] * w;
      if (!inBounds(w, h, p % w, (p / w) | 0)) {
        continue;
      }
      if (!inBounds(w, h, stand % w, (stand / w) | 0)) {
        continue;
      }
      if (p < 0 || p >= n || stand < 0 || stand >= n) {
        continue;
      }
      if (Math.abs((p % w) - gx) + Math.abs(((p / w) | 0) - gy) !== 1) {
        continue;
      }
      if (wall[p] || !allowed[p] || can[p]) {
        continue;
      }
      if (wall[stand] || !allowed[stand]) {
        continue;
      }
      can[p] = 1;
      queue.push(p);
    }
  }
  const dead = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    if (!can[i]) {
      dead[i] = 1;
    }
  }
  return dead;
}

// Exact min-cost assignment of boxes to goals (bitmask DP).
function matchingCost(boxes: number[], goals: number[], w: number): number {
  const m = boxes.length;
  if (m === 0) {
    return 0;
  }
  const cost: number[][] = [];
  for (let i = 0; i < m; i++) {
    cost[i] = [];
    const bx = boxes[i] % w;
    const by = (boxes[i] / w) | 0;
    for (let j = 0; j < m; j++) {
      const gx = goals[j] % w;
      const gy = (goals[j] / w) | 0;
      cost[i][j] = Math.abs(bx - gx) + Math.abs(by - gy);
    }
  }
  const size = 1 << m;
  const dp: number[] = new Array(size).fill(Infinity);
  dp[0] = 0;
  for (let mask = 0; mask < size; mask++) {
    const k = bitCount(mask);
    if (k >= m) {
      continue;
    }
    const base = dp[mask];
    if (base === Infinity) {
      continue;
    }
    for (let j = 0; j < m; j++) {
      if ((mask >> j) & 1) {
        continue;
      }
      const nm = mask | (1 << j);
      const v = base + cost[k][j];
      if (v < dp[nm]) {
        dp[nm] = v;
      }
    }
  }
  return dp[size - 1];
}

function bitCount(x: number): number {
  let c = 0;
  while (x > 0) {
    c += x & 1;
    x >>= 1;
  }
  return c;
}

function allOnGoals(boxes: number[], goal: Uint8Array): boolean {
  for (const b of boxes) {
    if (!goal[b]) {
      return false;
    }
  }
  return true;
}

// Shortest walk path (letters) from start to target avoiding box squares; null if none.
function walkPath(w: number, h: number, wall: Uint8Array, boxed: Uint8Array, start: number, target: number): string | null {
  const n = w * h;
  if (boxed[start] || boxed[target]) {
    return null;
  }
  const prevDir = new Int8Array(n).fill(-1);
  const seen = new Uint8Array(n);
  const queue: number[] = [start];
  seen[start] = 1;
  let head = 0;
  while (head < queue.length) {
    const i = queue[head++]!;
    if (i === target) {
      break;
    }
    const x = i % w;
    const y = (i / w) | 0;
    for (let d = 0; d < 4; d++) {
      const qx = x + DIRS[d][0];
      const qy = y + DIRS[d][1];
      if (!inBounds(w, h, qx, qy)) {
        continue;
      }
      const j = qy * w + qx;
      if (wall[j] || boxed[j] || seen[j]) {
        continue;
      }
      seen[j] = 1;
      prevDir[j] = d;
      queue.push(j);
    }
  }
  if (!seen[target]) {
    return null;
  }
  const letters: string[] = [];
  let cur = target;
  while (cur !== start) {
    letters.push(DIRS[prevDir[cur]!][2]);
    cur = cur - DIRS[prevDir[cur]!][0] - DIRS[prevDir[cur]!][1] * w;
  }
  letters.reverse();
  return letters.join('');
}

// Can the player walk from start to any of `exits` avoiding the boxes?
function exitsReachable(w: number, h: number, wall: Uint8Array, boxed: Uint8Array, start: number, exits: number[]): boolean {
  const n = w * h;
  if (boxed[start]) {
    return false;
  }
  const isExit = new Uint8Array(n);
  for (const e of exits) {
    isExit[e] = 1;
  }
  const seen = new Uint8Array(n);
  const queue: number[] = [start];
  seen[start] = 1;
  let head = 0;
  while (head < queue.length) {
    const i = queue[head++]!;
    if (isExit[i]) {
      return true;
    }
    const x = i % w;
    const y = (i / w) | 0;
    for (let d = 0; d < 4; d++) {
      const qx = x + DIRS[d][0];
      const qy = y + DIRS[d][1];
      if (!inBounds(w, h, qx, qy)) {
        continue;
      }
      const j = qy * w + qx;
      if (wall[j] || boxed[j] || seen[j]) {
        continue;
      }
      seen[j] = 1;
      queue.push(j);
    }
  }
  return false;
}

function cornerDeadlock(w: number, h: number, wall: Uint8Array, i: number): boolean {
  const x = i % w;
  const y = (i / w) | 0;
  const get = (dx: number, dy: number): boolean => {
    const qx = x + dx;
    const qy = y + dy;
    return !inBounds(w, h, qx, qy) || wall[qy * w + qx] === 1;
  };
  return (
    (get(0, -1) && get(-1, 0)) ||
    (get(0, -1) && get(1, 0)) ||
    (get(0, 1) && get(-1, 0)) ||
    (get(0, 1) && get(1, 0))
  );
}

function boxedOf(boxes: number[], n: number): Uint8Array {
  const boxed = new Uint8Array(n);
  for (const b of boxes) {
    boxed[b] = 1;
  }
  return boxed;
}

// A* over pushes inside one room. Boxes confined to `allowed` squares.
// Returns a solution move string or null if unsolvable within `cap` expansions.
function solveRoom(w: number, h: number, wall: Uint8Array, goal: Uint8Array,
  allowed: Uint8Array, dead: Uint8Array,
  compGoals: number[], boxes0: number[], player0: number, exits: number[],
  requireExit: boolean, cap: number,
): { moves: string; boxes: number[]; player: number; x: number } | null {
  const m = boxes0.length;
  boxes0 = boxes0.slice().sort((a, b) => a - b);
  const key = (boxes: number[], p: number): string =>
    String.fromCharCode(p, ...boxes);

  let expansions = 0;
  let bestFailed: { g: number; sid: number } | null = null;

  const sp: number[] = [player0];
  const sb: number[][] = [boxes0];
  const sg: number[] = [0];
  const sm: string[] = [''];
  const parent: number[] = [-1];
  const bestG = new Map<string, number>([[key(boxes0, player0), 0]]);
  const h0 = matchingCost(boxes0, compGoals, w);
  const heap: [number, number, number, number][] = [[h0, 0, 0, 0]];
  const less = (a: [number, number, number, number], b: [number, number, number, number]): boolean => {
    if (a[0] !== b[0]) {
      return a[0] < b[0];
    }
    if (a[1] !== b[1]) {
      return a[1] < b[1];
    }
    return a[2] < b[2];
  };
  const pushHeap = (e: [number, number, number, number]): void => {
    let i = heap.length;
    heap.push(e);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!less(heap[i]!, heap[p]!)) {
        break;
      }
      [heap[i], heap[p]] = [heap[p]!, heap[i]!];
      i = p;
    }
  };
  const popHeap = (): [number, number, number, number] => {
    const top = heap[0]!;
    const last = heap.pop()!;
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0;; i++) {
        let c = 2 * i + 1;
        if (c >= heap.length) {
          break;
        }
        if (c + 1 < heap.length && less(heap[c + 1]!, heap[c]!)) {
          c++;
        }
        if (!less(heap[c]!, heap[i]!)) {
          break;
        }
        [heap[i], heap[c]] = [heap[c]!, heap[i]!];
        i = c;
      }
    }
    return top;
  };

  let counter = 1;
  const pathOf = (sid: number): string => {
    let path = '';
    for (let t = sid; parent[t] >= 0; t = parent[t]!) {
      path = sm[t]! + path;
    }
    return path;
  };

  while (heap.length > 0) {
    const [f, g, , sid] = popHeap()!;
    const boxes = sb[sid]!;
    const p = sp[sid]!;
    const curG = bestG.get(key(boxes, p));
    if (curG === undefined || curG < g) {
      continue;
    }
    expansions++;
    if (expansions > cap) {
      break;
    }
    if (allOnGoals(boxes, goal)) {
      if (!requireExit || exitsReachable(w, h, wall, boxedOf(boxes, w * h), p, exits)) {
        return { moves: pathOf(sid), boxes, p, x: expansions };
      }
      if (bestFailed === null || g < bestFailed.g) {
        bestFailed = { g, sid };
      }
      continue;
    }
    const boxed = boxedOf(boxes, w * h);
    const reachable = new Uint8Array(w * h);
    const walkTo: string[] = [];
    walkTo[p] = '';
    reachable[p] = 1;
    const wqueue: number[] = [p];
    let whead = 0;
    while (whead < wqueue.length) {
      const i = wqueue[whead++]!;
      const walk = walkTo[i]!;
      const x = i % w;
      const y = (i / w) | 0;
      for (let d = 0; d < 4; d++) {
        const qx = x + DIRS[d][0];
        const qy = y + DIRS[d][1];
        if (!inBounds(w, h, qx, qy)) {
          continue;
        }
        const j = qy * w + qx;
        if (wall[j] || boxed[j] || reachable[j]) {
          continue;
        }
        reachable[j] = 1;
        walkTo[j] = walk + DIRS[d][2];
        wqueue.push(j);
      }
    }
    for (const i of wqueue) {
      const x = i % w;
      const y = (i / w) | 0;
      const walk = walkTo[i]!;
      for (let d = 0; d < 4; d++) {
        const qx = x + DIRS[d][0];
        const qy = y + DIRS[d][1];
        if (!inBounds(w, h, qx, qy)) {
          continue;
        }
        const q = qy * w + qx;
        if (!boxed[q]) {
          continue;
        }
        const r = q + DIRS[d][0] + DIRS[d][1] * w;
        if (!inBounds(w, h, r % w, (r / w) | 0)) {
          continue;
        }
        if (wall[r] || boxed[r] || !allowed[r]) {
          continue;
        }
        const at = boxes.indexOf(q);
        const nboxes = boxes.slice();
        nboxes[at] = r;
        nboxes.sort((a, b) => a - b);
        let deadEnd = false;
        if (!goal[r] && dead[r]) {
          deadEnd = true;
        }
        if (!deadEnd && !goal[r] && cornerDeadlock(w, h, wall, r)) {
          deadEnd = true;
        }
        if (!deadEnd) {
          for (const b of nboxes) {
            if (!goal[b] && dead[b]) {
              deadEnd = true;
              break;
            }
          }
        }
        if (deadEnd) {
          continue;
        }
        const ng = g + 1;
        const k = key(nboxes, q);
        const seen = bestG.get(k);
        if (seen !== undefined && seen <= ng) {
          continue;
        }
        bestG.set(k, ng);
        const id = sp.length;
        sp.push(q);
        sb.push(nboxes);
        sg.push(ng);
        parent.push(sid);
        sm.push(walk + DIRS[d][2]);
        const heur = matchingCost(nboxes, compGoals, w);
        pushHeap([ng + heur, ng, counter++, id]);
      }
    }
  }
  const finalRes = bestFailed !== null ? { moves: pathOf(bestFailed.sid), boxes: sb[bestFailed.sid]!, p: sp[bestFailed.sid]!, x: expansions } : null;
  if (VERBOSE && finalRes === null) {
    console.error('solveRoom NULL m=' + m + ' expansions=' + expansions + ' cap=' + cap);
  }
  return finalRes;
}

const VERBOSE = process.env.DEBUG === '1';

// One-square-wide hallway: squares inside straight runs of length at least 2.
// Doors and short (length-1) corridors stay part of rooms, which keeps rooms intact.
function classifyHallway(w: number, h: number, wall: Uint8Array): Uint8Array {
  const n = w * h;
  const hall = new Uint8Array(n);
  const straight2 = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    if (!wall[i]) {
      straight2[i] = isStraight2(w, h, wall, i) ? 1 : 0;
    }
  }
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      if (wall[i] || !straight2[i]) {
        continue;
      }
      const nb = floorNeighbors(w, h, wall, i);
      if (straight2[nb[0]!] && straight2[nb[1]!]) {
        hall[i] = 1;
      }
    }
  }
  return hall;
}

function isStraight2(w: number, h: number, wall: Uint8Array, i: number): boolean {
  const x = i % w;
  const y = (i / w) | 0;
  const nb = floorNeighbors(w, h, wall, i);
  if (nb.length !== 2) {
    return false;
  }
  const a = nb[0];
  const b = nb[1];
  return (a % w === x && b % w === x) || ((a / w) | 0 === y && (b / w) | 0 === y);
}

// Connected components of floor squares that are not hallway: the rooms.
function components(w: number, h: number, wall: Uint8Array, hall: Uint8Array): number[][] {
  const n = w * h;
  const seen = new Uint8Array(n);
  const comps: number[][] = [];
  for (let s = 0; s < n; s++) {
    if (wall[s] || hall[s] || seen[s]) {
      continue;
    }
    const comp: number[] = [];
    const stack: number[] = [s];
    seen[s] = 1;
    while (stack.length > 0) {
      const i = stack.pop()!;
      comp.push(i);
      for (const j of floorNeighbors(w, h, wall, i)) {
        if (hall[j] || seen[j]) {
          continue;
        }
        seen[j] = 1;
        stack.push(j);
      }
    }
    comps.push(comp.filter((a: number, b: number) => comp.indexOf(a) === b).sort((a, c) => a - c));
  }
  return comps;
}

function solve(text: string): string {
  const lv = parse(text);
  const { w, h, wall, goal: goalLv, box: boxLv, player } = lv;
  const n = w * h;
  const boxList: number[] = [];
  const goalList: number[] = [];
  for (let i = 0; i < n; i++) {
    if (boxLv[i]) {
      boxList.push(i);
    }
    if (goalLv[i]) {
      goalList.push(i);
    }
  }
  boxList.sort((a, b) => a - b);
  goalList.sort((a, b) => a - b);
  if (boxList.length !== goalList.length) {
    return '';
  }

  const hall = classifyHallway(w, h, wall);
  const comps = components(w, h, wall, hall);
  const compOf = new Int32Array(n).fill(-1);
  for (let c = 0; c < comps.length; c++) {
    for (const s of comps[c]) {
      compOf[s] = c;
    }
  }

  const compBoxes: number[][] = comps.map(() => []);
  const compGoals: number[][] = comps.map(() => []);
  for (let i = 0; i < n; i++) {
    const c = compOf[i];
    if (c < 0) {
      continue;
    }
    if (boxLv[i]) {
      compBoxes[c].push(i);
    }
    if (goalLv[i]) {
      compGoals[c].push(i);
    }
  }

  const compExits: number[][] = comps.map((comp) => {
    const exits: number[] = [];
    for (const s of comp) {
      const x = s % w;
      const y = (s / w) | 0;
      for (let d = 0; d < 4; d++) {
        const qx = x + DIRS[d][0];
        const qy = y + DIRS[d][1];
        if (!inBounds(w, h, qx, qy)) {
          continue;
        }
        const j = qy * w + qx;
        if (!wall[j] && compOf[j] !== comps.indexOf(comp)) {
          exits.push(s);
          break;
        }
      }
    }
    exits.sort((a, b) => a - b);
    return exits;
  });

  let balanced = true;
  for (let c = 0; c < comps.length; c++) {
    if (compBoxes[c].length !== compGoals[c].length) {
      balanced = false;
      break;
    }
  }
  if (VERBOSE) {
    console.error('comps=' + comps.length + ' balanced=' + balanced + ' counts=' + comps.map((c, i) => compBoxes[i].length + '/' + compGoals[i].length).join(','));
  }

  const order: number[] = [];
  for (let c = 0; c < comps.length; c++) {
    if (compBoxes[c].length > 0) {
      order.push(c);
    }
  }
  order.sort((a, b) => comps[a][0] - comps[b][0]);

  const curBoxes = boxList.slice();
  let curPlayer = player;
  let moves = '';
  if (balanced) {
    for (let ci = 0; ci < order.length; ci++) {
      const c = order[ci]!;
      const boxes = compBoxes[c].slice();
      const goalsSet = new Uint8Array(n);
      for (const go of compGoals[c]) {
        goalsSet[go] = 1;
      }
      const dead = deadSquares(w, h, wall, goalsSet, compMask(comps[c], n));
      const boxedNow = boxedOf(curBoxes, n);
      let entry = -1;
      const candExits: number[] = [];
      for (const e of compExits[c]) { candExits.push(e); }
      for (const s of comps[c]) { if (compExits[c].indexOf(s) < 0) { candExits.push(s); } }
      for (const e of candExits) {
        if (boxedNow[e]) { continue; }
        const pth = walkPath(w, h, wall, boxedNow, curPlayer, e);
        if (pth !== null) { entry = e; moves += pth; curPlayer = e; break; }
      }
      if (entry < 0) { return ''; }
      if (VERBOSE) {
        console.error('comp c=' + c + ' entry=' + entry + ' boxes=' + boxes.length + ' goals=' + compGoals[c].length);
      }
      const capRoom = boxes.length <= 4 ? 400_000 : 1_000_000;
      const res = solveRoom(w, h, wall, goalsSet, compMask(comps[c], n), dead, compGoals[c], boxes, entry, [], false, capRoom);
      if (res === null) {
        return '';
      }
      moves += res.moves;
      for (let k = 0; k < boxes.length; k++) {
        const at = curBoxes.indexOf(boxes[k]);
        curBoxes[at] = res.boxes[k];
      }
      curPlayer = res.p;
    }
  }
  if (!balanced) {
    return '';
  }
  return moves;
}

function compMask(comp: number[], n: number): Uint8Array {
  const mask = new Uint8Array(n);
  for (const s of comp) {
    mask[s] = 1;
  }
  return mask;
}

function solveFallback(w: number, h: number, wall: Uint8Array, goal: Uint8Array,
  _box: Uint8Array, player: number, boxes: number[], goals: number[], budget: number,
): string {
  const n = w * h;
  const allFloor = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    if (!wall[i]) {
      allFloor[i] = 1;
    }
  }
  const dead = deadSquares(w, h, wall, goal, allFloor);
  const res = solveRoom(w, h, wall, goal, allFloor, dead, goals, boxes.slice().sort((a, b) => a - b), player, [], false, budget);
  return res === null ? '' : res.moves;
}

for (const id of readdirSync(INPUT_DIR).sort()) {
  const here = join(INPUT_DIR, id);
  let isDir = false;
  try {
    isDir = statSync(here).isDirectory();
  } catch {
    continue;
  }
  if (!isDir) {
    continue;
  }
  const text = readFileSync(join(here, 'level.txt'), 'utf8');
  const moves = solve(text);
  mkdirSync(join(OUTPUT_DIR, id), { recursive: true });
  writeFileSync(join(OUTPUT_DIR, id, 'solution.txt'), moves + '\n');
}