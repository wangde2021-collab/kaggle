# ConnectX agent: Pascal Pons style bitboard solver (numba) + Pons opening book.
# The build script (build_submission.py) injects the opening book blob into BOOK_BLOB.
# Credits: algorithm follows Pascal Pons' Connect 4 solver (AGPL-3.0) - http://connect4.gamesolver.org
import time
import base64
import zlib

import numpy as np

BOOK_BLOB = None  # <<BOOK_BLOB>>
BOOK_PATH = None  # local testing only

try:
    import numba
    from numba import njit

    NUMBA = True
except Exception:  # pragma: no cover - pure python fallback
    numba = None
    NUMBA = False

    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def deco(f):
            return f

        return deco


W = 7
H = 6
H1 = 7
SIZE = 42
MIN_SCORE = -18
MAX_SCORE = 18
BOTTOM = 4432676798593  # sum(1 << (c*7) for c in range(7))
BOARD_MASK = 279258638311359  # BOTTOM * 63
ORDER = (3, 2, 4, 1, 5, 0, 6)
TT_SIZE = 8388617
UNKNOWN = 999

if NUMBA:

    @njit(cache=False)
    def now_ns():
        with numba.objmode(t="int64"):
            t = time.perf_counter_ns()
        return t

else:

    def now_ns():
        return time.perf_counter_ns()


@njit(cache=False)
def cdiv(a, b):
    if a >= 0:
        return a // b
    return -((-a) // b)


@njit(cache=False)
def popcount(x):
    c = 0
    while x:
        x &= x - 1
        c += 1
    return c


@njit(cache=False)
def winning_pos(position, mask):
    r = (position << 1) & (position << 2) & (position << 3)
    p = (position << 7) & (position << 14)
    r |= p & (position << 21)
    r |= p & (position >> 7)
    p = (position >> 7) & (position >> 14)
    r |= p & (position << 7)
    r |= p & (position >> 21)
    p = (position << 6) & (position << 12)
    r |= p & (position << 18)
    r |= p & (position >> 6)
    p = (position >> 6) & (position >> 12)
    r |= p & (position << 6)
    r |= p & (position >> 18)
    p = (position << 8) & (position << 16)
    r |= p & (position << 24)
    r |= p & (position >> 8)
    p = (position >> 8) & (position >> 16)
    r |= p & (position << 8)
    r |= p & (position >> 24)
    return r & (BOARD_MASK ^ mask)


@njit(cache=False)
def possible(mask):
    return (mask + BOTTOM) & BOARD_MASK


@njit(cache=False)
def nonlosing_moves(position, mask):
    poss = possible(mask)
    opp_win = winning_pos(position ^ mask, mask)
    forced = poss & opp_win
    if forced:
        if forced & (forced - 1):
            return 0
        poss = forced
    return poss & ~(opp_win >> 1)


@njit(cache=False)
def column_mask(col):
    return 63 << (col * 7)


@njit(cache=False)
def partial_key3(key, col, position, mask):
    pos = 1 << (col * 7)
    while pos & mask:
        key *= 3
        if pos & position:
            key += 1
        else:
            key += 2
        pos <<= 1
    key *= 3
    return key


@njit(cache=False)
def key3(position, mask):
    kf = 0
    for i in range(7):
        kf = partial_key3(kf, i, position, mask)
    kr = 0
    for i in range(6, -1, -1):
        kr = partial_key3(kr, i, position, mask)
    if kf < kr:
        return kf // 3
    return kr // 3


@njit(cache=False)
def book_get(position, mask, nb, bk, bv, bsize, bdepth):
    if nb > bdepth or bsize == 0:
        return 0
    k = key3(position, mask)
    i = k % bsize
    if bk[i] == (k & 255):
        return bv[i]
    return 0


@njit(cache=False)
def negamax(position, mask, nb, alpha, beta, tk, tv, bk, bv, bsize, bdepth, st, wm, ws):
    st[0] += 1
    if (st[0] & 1023) == 0:
        if now_ns() > st[2]:
            st[1] = 1
    if st[1] != 0:
        return 0
    nxt = nonlosing_moves(position, mask)
    if nxt == 0:
        return cdiv(-(SIZE - nb), 2)
    if nb >= SIZE - 2:
        return 0
    mn = cdiv(-(SIZE - 2 - nb), 2)
    if alpha < mn:
        alpha = mn
        if alpha >= beta:
            return alpha
    mx = cdiv(SIZE - 1 - nb, 2)
    if beta > mx:
        beta = mx
        if alpha >= beta:
            return beta
    key = position + mask
    ti = key % TT_SIZE
    if tk[ti] == key:
        val = tv[ti]
        if val != 0:
            if val > MAX_SCORE - MIN_SCORE + 1:
                mn = val + 2 * MIN_SCORE - MAX_SCORE - 2
                if alpha < mn:
                    alpha = mn
                    if alpha >= beta:
                        return alpha
            else:
                mx = val + MIN_SCORE - 1
                if beta > mx:
                    beta = mx
                    if alpha >= beta:
                        return beta
    bval = book_get(position, mask, nb, bk, bv, bsize, bdepth)
    if bval != 0:
        return bval + MIN_SCORE - 1
    sz = 0
    for i in range(6, -1, -1):
        mv = nxt & column_mask(ORDER[i])
        if mv:
            sc = popcount(winning_pos(position | mv, mask))
            p = sz
            sz += 1
            while p > 0 and ws[nb, p - 1] > sc:
                wm[nb, p] = wm[nb, p - 1]
                ws[nb, p] = ws[nb, p - 1]
                p -= 1
            wm[nb, p] = mv
            ws[nb, p] = sc
    while sz > 0:
        sz -= 1
        mv = wm[nb, sz]
        score = -negamax(position ^ mask, mask | mv, nb + 1, -beta, -alpha,
                         tk, tv, bk, bv, bsize, bdepth, st, wm, ws)
        if st[1] != 0:
            return 0
        if score >= beta:
            tk[ti] = key
            tv[ti] = score + MAX_SCORE - 2 * MIN_SCORE + 2
            return score
        if score > alpha:
            alpha = score
    tk[ti] = key
    tv[ti] = alpha - MIN_SCORE + 1
    return alpha


@njit(cache=False)
def solve(position, mask, nb, weak, tk, tv, bk, bv, bsize, bdepth, st, wm, ws):
    """Exact (weak=0) or win/draw/loss (weak=1) score for side to move. UNKNOWN on timeout."""
    if winning_pos(position, mask) & possible(mask):
        return cdiv(SIZE + 1 - nb, 2)
    mn = cdiv(-(SIZE - nb), 2)
    mx = cdiv(SIZE + 1 - nb, 2)
    if weak:
        mn = -1
        mx = 1
    while mn < mx:
        med = mn + cdiv(mx - mn, 2)
        if med <= 0 and cdiv(mn, 2) < med:
            med = cdiv(mn, 2)
        elif med >= 0 and cdiv(mx, 2) > med:
            med = cdiv(mx, 2)
        r = negamax(position, mask, nb, med, med + 1, tk, tv, bk, bv, bsize, bdepth, st, wm, ws)
        if st[1] != 0:
            return UNKNOWN
        if r <= med:
            mx = r
        else:
            mn = r
    return mn


# ----------------------------------------------------------------------------- engine state
class Engine:
    def __init__(self):
        self.tk = np.zeros(TT_SIZE, dtype=np.int64)
        self.tv = np.zeros(TT_SIZE, dtype=np.int8)
        self.st = np.zeros(3, dtype=np.int64)
        self.wm = np.zeros((44, 8), dtype=np.int64)
        self.ws = np.zeros((44, 8), dtype=np.int64)
        self.bk = np.zeros(1, dtype=np.uint8)
        self.bv = np.zeros(1, dtype=np.uint8)
        self.bsize = 0
        self.bdepth = -1
        self._load_book()
        self.warm = False

    def _load_book(self):
        raw = None
        try:
            if BOOK_BLOB is not None:
                raw = zlib.decompress(base64.b85decode(BOOK_BLOB))
            elif BOOK_PATH is not None:
                with open(BOOK_PATH, "rb") as f:
                    raw = f.read()
        except Exception:
            raw = None
        if not raw or len(raw) < 6:
            return
        width, height, depth, key_size, value_size, log_size = raw[0], raw[1], raw[2], raw[3], raw[4], raw[5]
        if width != 7 or height != 6 or key_size != 1 or value_size != 1:
            return
        n = (len(raw) - 6) // 2
        arr = np.frombuffer(raw, dtype=np.uint8, offset=6)
        self.bk = np.ascontiguousarray(arr[:n])
        self.bv = np.ascontiguousarray(arr[n:2 * n])
        self.bsize = n
        self.bdepth = depth

    def solve(self, position, mask, nb, weak, deadline_ns):
        self.st[0] = 0
        self.st[1] = 0
        self.st[2] = deadline_ns
        return solve(np.int64(position), np.int64(mask), np.int64(nb), np.int64(weak), self.tk, self.tv,
                     self.bk, self.bv, np.int64(self.bsize), np.int64(self.bdepth), self.st, self.wm, self.ws)

    def warmup(self):
        if self.warm:
            return
        # trigger JIT compilation on a trivial position
        self.solve(0, 0, 0, 1, now_ns() + 50_000_000)
        p, m = 0, 0
        for c in (3, 3, 2, 4, 2, 4, 1, 5):
            mv = (m + (1 << (c * 7))) & (63 << (c * 7))
            p, m = p ^ m, m | mv
        self.solve(p, m, 8, 0, now_ns() + 50_000_000)
        self.warm = True


ENGINE = None


def _bit_col(b):
    return (b & -b).bit_length() // 7 - (1 if ((b & -b).bit_length() % 7) == 0 else 0)


def bit_to_col(b):
    idx = (b & -b).bit_length() - 1
    return idx // 7


def parse(board, mark):
    position = 0
    mask = 0
    nb = 0
    for r in range(6):
        for c in range(7):
            v = board[r * 7 + c]
            if v:
                bit = 1 << (c * 7 + 5 - r)
                mask |= bit
                nb += 1
                if v == mark:
                    position |= bit
    return position, mask, nb


def move_bit(mask, col):
    return (mask + (1 << (col * 7))) & (63 << (col * 7))


def heuristic_order(position, mask, cols):
    def key(c):
        mv = move_bit(mask, c)
        sc = popcount(int(winning_pos(position | mv, mask)))
        return (-sc, abs(3 - c))
    return sorted(cols, key=key)


def choose(board, mark, base_budget=1.0, extra_budget=0.0, verbose=False):
    eng = ENGINE
    t0 = now_ns()
    position, mask, nb = parse(board, mark)
    poss = int(possible(mask))
    legal = [c for c in range(7) if poss & (63 << (c * 7))]
    wins = int(winning_pos(position, mask)) & poss
    if wins:
        return bit_to_col(wins), "win-now"
    nl = int(nonlosing_moves(position, mask))
    if nl == 0:
        opp = int(winning_pos(position ^ mask, mask)) & poss
        if opp:
            return bit_to_col(opp), "lost-block"
        return heuristic_order(position, mask, legal)[0], "lost"
    cands = [c for c in range(7) if nl & (63 << (c * 7))]
    cands = heuristic_order(position, mask, cands)
    if len(cands) == 1:
        return cands[0], "forced"

    deadline = t0 + int(base_budget * 1e9)
    hard_deadline = deadline + int(extra_budget * 1e9)
    res = {c: UNKNOWN for c in cands}
    slice_ns = int(0.02e9)
    while True:
        unresolved = [c for c in cands if res[c] == UNKNOWN]
        if not unresolved:
            break
        now = now_ns()
        limit = deadline
        # use extra (overage) time only if no proven win/draw yet
        if now >= deadline:
            if any(res[c] != UNKNOWN and res[c] >= 0 for c in cands) or now >= hard_deadline:
                break
            limit = hard_deadline
        for c in unresolved:
            now = now_ns()
            if now >= limit:
                break
            mv = move_bit(mask, c)
            r = eng.solve(position ^ mask, mask | mv, nb + 1, 1, min(limit, now + slice_ns))
            if r != UNKNOWN:
                res[c] = -int(r)
                if res[c] > 0:
                    return c, "win"
        slice_ns *= 3
        if now_ns() >= hard_deadline:
            break

    draws = [c for c in cands if res[c] == 0]
    unknown = [c for c in cands if res[c] == UNKNOWN]
    if draws:
        return draws[0], "draw"
    if unknown:
        return unknown[0], "unknown"
    # all proven losing: prefer slowest loss with remaining time (strong solve)
    best, best_s = cands[0], -UNKNOWN
    end = max(deadline, now_ns() + int(0.3e9))
    for c in cands:
        now = now_ns()
        if now >= end:
            break
        mv = move_bit(mask, c)
        r = eng.solve(position ^ mask, mask | mv, nb + 1, 0, end)
        if r == UNKNOWN:
            continue
        s = -int(r)
        if s > best_s:
            best, best_s = c, s
    return best, "loss"


def generic_agent(obs, config):
    cols = config.columns
    board = obs.board
    valid = [c for c in range(cols) if board[c] == 0]
    center = cols // 2
    return sorted(valid, key=lambda c: abs(center - c))[0]


def agent(obs, config):
    global ENGINE
    try:
        if config.columns != 7 or config.rows != 6 or config.inarow != 4:
            return generic_agent(obs, config)
        if ENGINE is None:
            ENGINE = Engine()
            ENGINE.warmup()
        overage = float(obs.get("remainingOverageTime", 0) or 0) if hasattr(obs, "get") else 0.0
        extra = max(0.0, min(4.0, overage - 20.0))
        col, _ = choose(obs.board, obs.mark, base_budget=1.0, extra_budget=extra)
        if obs.board[col] != 0:
            raise ValueError("illegal")
        return int(col)
    except Exception:
        return generic_agent(obs, config)
