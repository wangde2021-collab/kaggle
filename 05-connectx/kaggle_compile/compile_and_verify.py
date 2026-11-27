# Kaggle CPU kernel: compile solver.cpp statically for Linux x86-64, download 7x6.book,
# build self-contained submission.py, and run rigorous kaggle_environments validation.
import base64
import json
import os
import py_compile
import shutil
import subprocess
import time
import urllib.request
import zlib
from collections import Counter
from pathlib import Path

WORK = Path("/kaggle/working")
WORK.mkdir(parents=True, exist_ok=True)

SOLVER_CPP = r"""// Fast ConnectX (7x6, 4-in-a-row) solver based on Pascal Pons' Connect4 bitboard solver (AGPL-3.0).
// Supports exact 14-ply OpeningBook lookup, persistent TranspositionTable, time-bounded
// weak/strong exact solving, practical counterplay on lost openings, and iterative-deepening
// heuristic alpha-beta fallback.

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr int W = 7;
constexpr int H = 6;
constexpr int SIZE = W * H;          // 42
constexpr int MIN_SCORE = -SIZE / 2 + 3;  // -18
constexpr int MAX_SCORE = (SIZE + 1) / 2 - 3;  // 18
constexpr int UNKNOWN_SCORE = 999;

constexpr uint64_t make_bottom_mask() {
    uint64_t m = 0;
    for (int c = 0; c < W; ++c) m |= (uint64_t(1) << (c * (H + 1)));
    return m;
}
constexpr uint64_t BOTTOM_MASK = make_bottom_mask();
constexpr uint64_t BOARD_MASK = BOTTOM_MASK * ((uint64_t(1) << H) - 1);

constexpr uint64_t col_mask(int c) {
    return ((uint64_t(1) << H) - 1) << (c * (H + 1));
}
constexpr uint64_t top_mask_col(int c) {
    return uint64_t(1) << ((H - 1) + c * (H + 1));
}
constexpr uint64_t bottom_mask_col(int c) {
    return uint64_t(1) << (c * (H + 1));
}

inline int popcount64(uint64_t x) {
    return __builtin_popcountll(x);
}

inline int ctz64(uint64_t x) {
    return __builtin_ctzll(x);
}

struct Position {
    uint64_t cur = 0;
    uint64_t mask = 0;
    int moves = 0;

    static uint64_t compute_winning_position(uint64_t pos, uint64_t mask) {
        // vertical
        uint64_t r = (pos << 1) & (pos << 2) & (pos << 3);

        // horizontal
        uint64_t p = (pos << (H + 1)) & (pos << 2 * (H + 1));
        r |= p & (pos << 3 * (H + 1));
        r |= p & (pos >> (H + 1));
        p = (pos >> (H + 1)) & (pos >> 2 * (H + 1));
        r |= p & (pos << (H + 1));
        r |= p & (pos >> 3 * (H + 1));

        // diagonal 1
        p = (pos << H) & (pos << 2 * H);
        r |= p & (pos << 3 * H);
        r |= p & (pos >> H);
        p = (pos >> H) & (pos >> 2 * H);
        r |= p & (pos << H);
        r |= p & (pos >> 3 * H);

        // diagonal 2
        p = (pos << (H + 2)) & (pos << 2 * (H + 2));
        r |= p & (pos << 3 * (H + 2));
        r |= p & (pos >> (H + 2));
        p = (pos >> (H + 2)) & (pos >> 2 * (H + 2));
        r |= p & (pos << (H + 2));
        r |= p & (pos >> 3 * (H + 2));

        return r & (BOARD_MASK ^ mask);
    }

    bool can_play(int col) const {
        return (mask & top_mask_col(col)) == 0;
    }

    uint64_t possible() const {
        return (mask + BOTTOM_MASK) & BOARD_MASK;
    }

    uint64_t winning_position() const {
        return compute_winning_position(cur, mask);
    }

    uint64_t opponent_winning_position() const {
        return compute_winning_position(cur ^ mask, mask);
    }

    bool can_win_next() const {
        return (winning_position() & possible()) != 0;
    }

    bool is_winning_move(int col) const {
        return (winning_position() & possible() & col_mask(col)) != 0;
    }

    uint64_t possible_non_losing_moves() const {
        uint64_t poss = possible();
        uint64_t opp_win = opponent_winning_position();
        uint64_t forced = poss & opp_win;
        if (forced) {
            if (forced & (forced - 1)) return 0;  // double threat -> hopeless
            poss = forced;
        }
        return poss & ~(opp_win >> 1);
    }

    void play(uint64_t move_bit) {
        cur ^= mask;
        mask |= move_bit;
        ++moves;
    }

    void play_col(int col) {
        play((mask + bottom_mask_col(col)) & col_mask(col));
    }

    uint64_t key() const {
        return cur + mask;
    }

    void partial_key3(uint64_t& k, int col) const {
        for (uint64_t p = uint64_t(1) << (col * (H + 1)); p & mask; p <<= 1) {
            k *= 3;
            if (p & cur) k += 1;
            else k += 2;
        }
        k *= 3;
    }

    uint64_t key3() const {
        uint64_t kf = 0;
        for (int i = 0; i < W; ++i) partial_key3(kf, i);
        uint64_t kr = 0;
        for (int i = W - 1; i >= 0; --i) partial_key3(kr, i);
        return (kf < kr ? kf : kr) / 3;
    }

    int move_score(uint64_t move_bit) const {
        return popcount64(compute_winning_position(cur | move_bit, mask));
    }

    int col_height(int c) const {
        return popcount64(mask & col_mask(c));
    }
};

constexpr uint64_t next_prime_const(uint64_t n) {
    while (true) {
        bool ok = true;
        for (uint64_t d = 2; d * d <= n; ++d) {
            if (n % d == 0) { ok = false; break; }
        }
        if (ok) return n;
        ++n;
    }
}

class OpeningBook {
public:
    int depth = -1;
    uint64_t size = 0;
    std::vector<uint8_t> keys;
    std::vector<uint8_t> vals;

    bool load(const std::string& path) {
        depth = -1;
        size = 0;
        keys.clear();
        vals.clear();
        std::ifstream ifs(path, std::ios::binary);
        if (!ifs) return false;
        uint8_t hdr[6]{};
        ifs.read(reinterpret_cast<char*>(hdr), 6);
        if (!ifs) return false;
        if (hdr[0] != W || hdr[1] != H || hdr[3] != 1 || hdr[4] != 1 || hdr[5] > 30) {
            return false;
        }
        uint64_t sz = next_prime_const(uint64_t(1) << hdr[5]);
        keys.resize(sz);
        vals.resize(sz);
        ifs.read(reinterpret_cast<char*>(keys.data()), sz);
        ifs.read(reinterpret_cast<char*>(vals.data()), sz);
        if (!ifs) {
            keys.clear();
            vals.clear();
            return false;
        }
        depth = hdr[2];
        size = sz;
        return true;
    }

    int get(const Position& p) const {
        if (p.moves > depth || size == 0) return 0;
        uint64_t k = p.key3();
        uint64_t idx = k % size;
        if (keys[idx] == static_cast<uint8_t>(k)) return vals[idx];
        return 0;
    }
};

class TranspositionTable {
public:
    static constexpr uint64_t TT_SIZE = 8388617;  // prime (~8.4M entries, ~76MB)
    std::vector<uint64_t> keys;
    std::vector<uint8_t> vals;

    TranspositionTable() : keys(TT_SIZE, 0), vals(TT_SIZE, 0) {}

    void put(uint64_t k, uint8_t v) {
        uint64_t idx = k % TT_SIZE;
        keys[idx] = k;
        vals[idx] = v;
    }

    uint8_t get(uint64_t k) const {
        uint64_t idx = k % TT_SIZE;
        return keys[idx] == k ? vals[idx] : 0;
    }
};

struct MoveSorter {
    struct Entry {
        uint64_t move = 0;
        int score = 0;
    };
    std::array<Entry, W> entries{};
    int size = 0;

    void add(uint64_t move, int score) {
        int pos = size++;
        for (; pos > 0 && entries[pos - 1].score > score; --pos) {
            entries[pos] = entries[pos - 1];
        }
        entries[pos] = {move, score};
    }

    uint64_t get_next() {
        return size > 0 ? entries[--size].move : 0;
    }
};

constexpr std::array<int, W> COL_ORDER = {3, 4, 2, 5, 1, 6, 0};

class Solver {
public:
    OpeningBook book;
    TranspositionTable tt;
    uint64_t node_count = 0;
    bool aborted = false;
    std::chrono::steady_clock::time_point deadline;

    bool check_timeout() {
        if ((node_count & 4095ULL) == 0) {
            if (std::chrono::steady_clock::now() >= deadline) {
                aborted = true;
            }
        }
        return aborted;
    }

    int negamax(const Position& p, int alpha, int beta) {
        ++node_count;
        if (check_timeout()) return 0;

        uint64_t poss = p.possible_non_losing_moves();
        if (poss == 0) return -(SIZE - p.moves) / 2;
        if (p.moves >= SIZE - 2) return 0;

        int min_val = -(SIZE - 2 - p.moves) / 2;
        if (alpha < min_val) {
            alpha = min_val;
            if (alpha >= beta) return alpha;
        }
        int max_val = (SIZE - 1 - p.moves) / 2;
        if (beta > max_val) {
            beta = max_val;
            if (alpha >= beta) return beta;
        }

        uint64_t k = p.key();
        if (uint8_t val = tt.get(k)) {
            if (val > MAX_SCORE - MIN_SCORE + 1) {
                int lb = val + 2 * MIN_SCORE - MAX_SCORE - 2;
                if (alpha < lb) {
                    alpha = lb;
                    if (alpha >= beta) return alpha;
                }
            } else {
                int ub = val + MIN_SCORE - 1;
                if (beta > ub) {
                    beta = ub;
                    if (alpha >= beta) return beta;
                }
            }
        }

        if (int bval = book.get(p)) {
            return bval + MIN_SCORE - 1;
        }

        MoveSorter moves;
        for (int i = W - 1; i >= 0; --i) {
            uint64_t mv = poss & col_mask(COL_ORDER[i]);
            if (mv) moves.add(mv, p.move_score(mv));
        }

        while (uint64_t next = moves.get_next()) {
            Position p2(p);
            p2.play(next);
            int score = -negamax(p2, -beta, -alpha);
            if (aborted) return 0;
            if (score >= beta) {
                tt.put(k, static_cast<uint8_t>(score + MAX_SCORE - 2 * MIN_SCORE + 2));
                return score;
            }
            if (score > alpha) alpha = score;
        }
        tt.put(k, static_cast<uint8_t>(alpha - MIN_SCORE + 1));
        return alpha;
    }

    int solve(const Position& p, bool weak, std::chrono::steady_clock::time_point dl) {
        aborted = false;
        deadline = dl;
        if (p.can_win_next()) return (SIZE + 1 - p.moves) / 2;

        int mn = -(SIZE - p.moves) / 2;
        int mx = (SIZE + 1 - p.moves) / 2;
        if (weak) {
            mn = -1;
            mx = 1;
        }
        while (mn < mx) {
            int med = mn + (mx - mn) / 2;
            if (med <= 0 && mn / 2 < med) med = mn / 2;
            else if (med >= 0 && mx / 2 > med) med = mx / 2;
            int r = negamax(p, med, med + 1);
            if (aborted) return UNKNOWN_SCORE;
            if (r <= med) mx = r;
            else mn = r;
        }
        return mn;
    }

    // Fast heuristic evaluation (used only if exact weak search runs out of time)
    static int heuristic_eval(const Position& p) {
        uint64_t my_win = p.winning_position();
        uint64_t opp_win = p.opponent_winning_position();
        int my_threats = popcount64(my_win);
        int opp_threats = popcount64(opp_win);
        // Even/odd row threat parity bonus (odd rows from bottom = bits 0,2,4 for P1)
        constexpr uint64_t ODD_ROWS = BOTTOM_MASK * 21ULL;   // bits 0,2,4 -> (1+4+16)=21
        constexpr uint64_t EVEN_ROWS = BOTTOM_MASK * 42ULL;  // bits 1,3,5 -> (2+8+32)=42
        bool p1_turn = (p.moves % 2 == 0);
        uint64_t my_pref = p1_turn ? ODD_ROWS : EVEN_ROWS;
        uint64_t opp_pref = p1_turn ? EVEN_ROWS : ODD_ROWS;
        int parity_bonus = 6 * popcount64(my_win & my_pref) - 6 * popcount64(opp_win & opp_pref);

        int center_score = 0;
        for (int c = 0; c < W; ++c) {
            int w = 4 - std::abs(c - 3);
            uint64_t cm = col_mask(c);
            center_score += w * (popcount64(p.cur & cm) - popcount64((p.cur ^ p.mask) & cm));
        }
        return 14 * (my_threats - opp_threats) + parity_bonus + 2 * center_score;
    }

    int heur_negamax(const Position& p, int depth, int alpha, int beta) {
        ++node_count;
        if (check_timeout()) return 0;
        if (p.can_win_next()) return 100000 + (SIZE - p.moves);
        uint64_t poss = p.possible_non_losing_moves();
        if (poss == 0) return -100000 - (SIZE - p.moves);
        if (p.moves >= SIZE - 2) return 0;
        if (depth <= 0) return heuristic_eval(p);

        MoveSorter moves;
        for (int i = W - 1; i >= 0; --i) {
            uint64_t mv = poss & col_mask(COL_ORDER[i]);
            if (mv) moves.add(mv, p.move_score(mv));
        }
        int best = -200000;
        while (uint64_t next = moves.get_next()) {
            Position p2(p);
            p2.play(next);
            int sc = -heur_negamax(p2, depth - 1, -beta, -alpha);
            if (aborted) return 0;
            if (sc > best) best = sc;
            if (best > alpha) alpha = best;
            if (alpha >= beta) break;
        }
        return best;
    }

    int heuristic_pick(const Position& p, const std::vector<int>& cands, int budget_ms) {
        int best_col = cands[0];
        auto dl = std::chrono::steady_clock::now() + std::chrono::milliseconds(budget_ms);
        for (int depth = 2; depth <= 18; ++depth) {
            aborted = false;
            deadline = dl;
            int alpha = -200000;
            const int beta = 200000;
            int iter_best_col = best_col;
            bool completed = true;
            for (int c : cands) {
                Position p2(p);
                p2.play_col(c);
                int sc = -heur_negamax(p2, depth - 1, -beta, -alpha);
                if (aborted) {
                    completed = false;
                    break;
                }
                if (sc > alpha) {
                    alpha = sc;
                    iter_best_col = c;
                }
            }
            if (!completed) break;
            best_col = iter_best_col;
            if (alpha > 90000 || std::chrono::steady_clock::now() >= dl) break;
        }
        return best_col;
    }
};

Position parse_board(const std::string& board_str, int mark) {
    Position p;
    for (int r = 0; r < H; ++r) {
        for (int c = 0; c < W; ++c) {
            char ch = board_str[r * W + c];
            if (ch == '1' || ch == '2') {
                uint64_t bit = uint64_t(1) << (c * (H + 1) + (H - 1 - r));
                p.mask |= bit;
                ++p.moves;
                if ((ch - '0') == mark) p.cur |= bit;
            }
        }
    }
    return p;
}

int counterplay_penalty(const Position& p, int col) {
    Position p2(p);
    p2.play_col(col);
    uint64_t opp_safe = p2.possible_non_losing_moves();
    int safe_replies = popcount64(opp_safe);
    int forced_pen = (safe_replies == 1) ? 1000 : 0;
    int opp_threats = popcount64(p2.winning_position());
    int height_sq = 0;
    for (int c = 0; c < W; ++c) {
        int h = p2.col_height(c);
        height_sq += h * h;
    }
    return 4 * opp_threats + 3 * height_sq + forced_pen;
}

struct Decision {
    int col = 3;
    std::string tag = "fallback";
    int score = 0;
    uint64_t nodes = 0;
};

Decision choose_move(Solver& solver, const Position& p, int budget_ms) {
    using clock = std::chrono::steady_clock;
    auto t0 = clock::now();
    auto hard_dl = t0 + std::chrono::milliseconds(budget_ms);
    solver.node_count = 0;

    // 1. Immediate win
    for (int c : COL_ORDER) {
        if (p.can_play(c) && p.is_winning_move(c)) {
            return {c, "win-now", (SIZE + 1 - p.moves) / 2, 1};
        }
    }

    // 2. Non-losing moves
    uint64_t nl = p.possible_non_losing_moves();
    if (nl == 0) {
        uint64_t poss = p.possible();
        uint64_t opp = p.opponent_winning_position() & poss;
        if (opp) {
            for (int c : COL_ORDER) {
                if (opp & col_mask(c)) {
                    return {c, "lost-block", -(SIZE - p.moves) / 2, 1};
                }
            }
        }
        for (int c : COL_ORDER) {
            if (p.can_play(c)) return {c, "lost-any", -(SIZE - p.moves) / 2, 1};
        }
    }

    std::vector<int> cands;
    for (int c : COL_ORDER) {
        if (nl & col_mask(c)) cands.push_back(c);
    }
    std::stable_sort(cands.begin(), cands.end(), [&](int a, int b) {
        uint64_t ma = (p.mask + bottom_mask_col(a)) & col_mask(a);
        uint64_t mb = (p.mask + bottom_mask_col(b)) & col_mask(b);
        int sa = p.move_score(ma);
        int sb = p.move_score(mb);
        if (sa != sb) return sa > sb;
        return std::abs(a - 3) < std::abs(b - 3);
    });

    if (cands.size() == 1) {
        return {cands[0], "forced", 0, 1};
    }

    // 3. Opening book horizon: ply <= 13 -> children are <= 14 -> exact solve is instant
    if (p.moves <= 13) {
        std::array<int, W> scores{};
        scores.fill(UNKNOWN_SCORE);
        bool all_done = true;
        for (int c : cands) {
            Position p2(p);
            p2.play_col(c);
            int r = solver.solve(p2, false, hard_dl);
            if (r == UNKNOWN_SCORE) {
                all_done = false;
                break;
            }
            scores[c] = -r;
        }
        if (all_done) {
            int best_s = -1000;
            int best_c = cands[0];
            for (int c : cands) {
                if (scores[c] > best_s) {
                    best_s = scores[c];
                    best_c = c;
                }
            }
            if (best_s < 0 && p.moves <= 12) {
                int best_pen = 1000000000;
                for (int c : cands) {
                    if (scores[c] >= best_s - 1) {
                        int pen = counterplay_penalty(p, c);
                        if (pen < best_pen) {
                            best_pen = pen;
                            best_c = c;
                        }
                    }
                }
                return {best_c, "book-counterplay", best_s, solver.node_count};
            }
            return {best_c, best_s > 0 ? "book-win" : (best_s == 0 ? "book-draw" : "book-loss"),
                    best_s, solver.node_count};
        }
    }

    // 4. Time-sliced weak solve across non-losing candidates
    std::array<int, W> wres{};
    wres.fill(UNKNOWN_SCORE);
    int slice_ms = 15;
    while (clock::now() < hard_dl) {
        bool any_unknown = false;
        for (int c : cands) {
            if (wres[c] != UNKNOWN_SCORE) continue;
            any_unknown = true;
            auto now = clock::now();
            if (now >= hard_dl) break;
            auto dl = std::min(hard_dl, now + std::chrono::milliseconds(slice_ms));
            Position p2(p);
            p2.play_col(c);
            int r = solver.solve(p2, true, dl);
            if (r != UNKNOWN_SCORE) {
                wres[c] = -r;
                if (wres[c] > 0) {
                    return {c, "exact-win", wres[c], solver.node_count};
                }
            }
        }
        if (!any_unknown) break;
        slice_ms = std::min(slice_ms * 3, budget_ms);
    }

    std::vector<int> draws, unknowns;
    for (int c : cands) {
        if (wres[c] == 0) draws.push_back(c);
        else if (wres[c] == UNKNOWN_SCORE) unknowns.push_back(c);
    }

    // 5. All candidates proven losing -> spend remaining/small extra budget for slowest loss
    if (draws.empty() && unknowns.empty()) {
        int best_c = cands[0];
        int best_s = -1000;
        auto dl = std::max(hard_dl, clock::now() + std::chrono::milliseconds(150));
        for (int c : cands) {
            if (clock::now() >= dl) break;
            Position p2(p);
            p2.play_col(c);
            int r = solver.solve(p2, false, dl);
            if (r != UNKNOWN_SCORE && -r > best_s) {
                best_s = -r;
                best_c = c;
            }
        }
        return {best_c, "exact-loss", best_s, solver.node_count};
    }

    // 6. If only draws (no unknowns), pick best draw (or heuristic among draws)
    if (unknowns.empty()) {
        if (draws.size() == 1) return {draws[0], "exact-draw", 0, solver.node_count};
        int col = solver.heuristic_pick(p, draws, 80);
        return {col, "exact-draw-heur", 0, solver.node_count};
    }

    // 7. Budget expired with unresolved moves: search only {unknowns + draws}, never proven losses
    std::vector<int> pool = unknowns;
    pool.insert(pool.end(), draws.begin(), draws.end());
    int col = solver.heuristic_pick(p, pool, 120);
    return {col, wres[col] == 0 ? "heur-draw" : "heur", 0, solver.node_count};
}

}  // namespace

int main(int argc, char** argv) {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    Solver solver;
    if (argc >= 2) {
        if (!solver.book.load(argv[1])) {
            std::cerr << "WARN: failed to load book: " << argv[1] << "\n";
        }
    }
    std::cout << "READY\n" << std::flush;

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        std::istringstream iss(line);
        std::string cmd;
        iss >> cmd;
        if (cmd == "QUIT") break;
        if (cmd == "PING") {
            std::cout << "PONG\n" << std::flush;
            continue;
        }
        if (cmd == "SOLVE") {
            std::string board_str;
            int mark = 1;
            int budget_ms = 800;
            iss >> board_str >> mark >> budget_ms;
            if (board_str.size() != SIZE) {
                std::cout << "3 bad-board 0 0\n" << std::flush;
                continue;
            }
            Position p = parse_board(board_str, mark);
            Decision d = choose_move(solver, p, std::max(50, std::min(budget_ms, 1800)));
            std::cout << d.col << ' ' << d.tag << ' ' << d.score << ' ' << d.nodes << '\n' << std::flush;
        }
    }
    return 0;
}
"""

SUBMISSION_TEMPLATE = r'''"""ConnectX Agent v2: Statically-linked C++17 Pascal Pons Bitboard Solver + 14-ply Book + Counterplay."""
import base64
import os
import select
import shutil
import subprocess
import sys
import time
import zlib

BOOK_B85 = __BOOK_B85__
BIN_B85 = __BIN_B85__
CPP_B85 = __CPP_B85__

_BOOK_PATH = "/tmp/cx_7x6_v2.bin"
_BIN_PATH = "/tmp/cx_solver_v2"
_PROC = None
_LOGGED_STATUS = False
_ORDER = (3, 4, 2, 5, 1, 6, 0)
_BOTTOM = 4432676798593
_BOARD_MASK = 279258638311359


def _log_once(msg):
    global _LOGGED_STATUS
    if not _LOGGED_STATUS:
        _LOGGED_STATUS = True
        print(msg, file=sys.stderr, flush=True)


def _start_binary(bin_path):
    proc = subprocess.Popen(
        [bin_path, _BOOK_PATH],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    rlist, _, _ = select.select([proc.stdout], [], [], 10.0)
    if not rlist:
        rc = proc.poll()
        err = proc.stderr.read() if rc is not None else "timeout"
        proc.kill()
        raise RuntimeError(f"rc={rc} err={err[:200]}")
    ready = proc.stdout.readline().strip()
    if ready != "READY":
        rc = proc.poll()
        err = proc.stderr.read() if rc is not None else ""
        proc.kill()
        raise RuntimeError(f"banner={ready!r} rc={rc} err={err[:200]}")
    return proc


def _ensure_worker():
    global _PROC
    if _PROC is not None and _PROC.poll() is None:
        return _PROC
    if not os.path.exists(_BOOK_PATH) or os.path.getsize(_BOOK_PATH) < 33_000_000:
        tmp_b = _BOOK_PATH + ".tmp"
        with open(tmp_b, "wb") as f:
            f.write(zlib.decompress(base64.b85decode(BOOK_B85)))
        os.replace(tmp_b, _BOOK_PATH)
    if not os.path.exists(_BIN_PATH) or os.path.getsize(_BIN_PATH) < 10_000:
        tmp_x = _BIN_PATH + ".tmp"
        with open(tmp_x, "wb") as f:
            f.write(zlib.decompress(base64.b85decode(BIN_B85)))
        os.chmod(tmp_x, 0o755)
        os.replace(tmp_x, _BIN_PATH)

    try:
        _PROC = _start_binary(_BIN_PATH)
        _log_once("cx-v2 worker READY (static)")
        return _PROC
    except Exception as e1:
        gpp = shutil.which("g++")
        if gpp:
            try:
                cpp_p = "/tmp/cx_solver_v2.cpp"
                loc_bin = "/tmp/cx_solver_v2_local"
                with open(cpp_p, "wb") as f:
                    f.write(zlib.decompress(base64.b85decode(CPP_B85)))
                subprocess.check_call([gpp, "-O3", "-std=c++17", cpp_p, "-o", loc_bin])
                _PROC = _start_binary(loc_bin)
                _log_once(f"cx-v2 worker READY (local g++ after: {e1})")
                return _PROC
            except Exception as e2:
                _log_once(f"cx-v2 worker FAIL: static={e1} | local={e2}")
                raise
        _log_once(f"cx-v2 worker FAIL: {e1}")
        raise


def _bb_win_pos(pos, mask):
    r = (pos << 1) & (pos << 2) & (pos << 3)
    for sh in (7, 6, 8):
        p = (pos << sh) & (pos << (2 * sh))
        r |= p & (pos << (3 * sh))
        r |= p & (pos >> sh)
        p = (pos >> sh) & (pos >> (2 * sh))
        r |= p & (pos << sh)
        r |= p & (pos >> (3 * sh))
    return r & (_BOARD_MASK ^ mask)


def _py_bitboard_fallback(board, mark):
    pos = mask = nb = 0
    for r in range(6):
        for c in range(7):
            v = board[r * 7 + c]
            if v:
                b = 1 << (c * 7 + 5 - r)
                mask |= b
                nb += 1
                if v == mark:
                    pos |= b

    poss = (mask + _BOTTOM) & _BOARD_MASK
    wins = _bb_win_pos(pos, mask) & poss
    for c in _ORDER:
        if wins & (63 << (c * 7)):
            return c

    opp_win = _bb_win_pos(pos ^ mask, mask)
    forced = poss & opp_win
    if forced:
        for c in _ORDER:
            if forced & (63 << (c * 7)):
                return c

    def eval_pos(p, m):
        my_w = _bb_win_pos(p, m).bit_count()
        op_w = _bb_win_pos(p ^ m, m).bit_count()
        ctr = 0
        for c in range(7):
            w = 4 - abs(c - 3)
            cm = 63 << (c * 7)
            ctr += w * ((p & cm).bit_count() - ((p ^ m) & cm).bit_count())
        return 14 * (my_w - op_w) + 2 * ctr

    def ab(p, m, ply, depth, alpha, beta):
        ps = (m + _BOTTOM) & _BOARD_MASK
        if _bb_win_pos(p, m) & ps:
            return 10000 - ply
        ow = _bb_win_pos(p ^ m, m)
        fc = ps & ow
        if fc:
            if fc & (fc - 1):
                return -10000 + ply
            ps = fc
        else:
            safe = ps & ~(ow >> 1)
            if safe:
                ps = safe
        if ply >= 42:
            return 0
        if depth == 0:
            return eval_pos(p, m)
        best = -20000
        for c in _ORDER:
            mv = ps & (63 << (c * 7))
            if mv:
                sc = -ab(p ^ m, m | mv, ply + 1, depth - 1, -beta, -alpha)
                if sc > best:
                    best = sc
                if best > alpha:
                    alpha = best
                if alpha >= beta:
                    break
        return best

    nl = poss & ~(opp_win >> 1)
    cands = [c for c in _ORDER if (nl if nl else poss) & (63 << (c * 7))]
    best_c, best_s = cands[0], -30000
    for c in cands:
        mv = (mask + (1 << (c * 7))) & (63 << (c * 7))
        sc = -ab(pos ^ mask, mask | mv, nb + 1, 4, -30000, 30000)
        if sc > best_s:
            best_s, best_c = sc, c
    return best_c


def _generic_fallback(board, mark, rows, cols, inarow):
    valid = [c for c in range(cols) if board[c] == 0]
    if not valid:
        return 0
    valid.sort(key=lambda c: abs(cols // 2 - c))
    return valid[0]


def agent(observation, configuration):
    board = observation.board
    mark = int(observation.mark)
    rows = int(configuration.rows)
    cols = int(configuration.columns)
    inarow = int(configuration.inarow)

    if rows != 6 or cols != 7 or inarow != 4:
        return _generic_fallback(board, mark, rows, cols, inarow)

    valid = [c for c in _ORDER if board[c] == 0]
    if not valid:
        return 0
    if len(valid) == 1:
        return valid[0]

    nb = sum(1 for x in board if x != 0)
    overage = float(getattr(observation, "remainingOverageTime", 0.0) or 0.0)
    if nb <= 13:
        budget_ms = 550
    elif nb <= 24:
        extra = int(max(0.0, min(400.0, (overage - 25.0) * 15.0)))
        budget_ms = 850 + extra
    else:
        budget_ms = 650

    try:
        proc = _ensure_worker()
        bstr = "".join(str(int(x)) for x in board)
        proc.stdin.write(f"SOLVE {bstr} {mark} {budget_ms}\n")
        proc.stdin.flush()
        wait_s = min(1.55, budget_ms / 1000.0 + 0.35)
        rlist, _, _ = select.select([proc.stdout], [], [], wait_s)
        if not rlist:
            raise TimeoutError("solver response timeout")
        resp = proc.stdout.readline().strip().split()
        col = int(resp[0])
        if 0 <= col < 7 and board[col] == 0:
            return col
    except Exception:
        global _PROC
        if _PROC is not None:
            try:
                _PROC.kill()
            except Exception:
                pass
            _PROC = None

    return _py_bitboard_fallback(board, mark)
'''


def main():
    cpp_path = WORK / "solver.cpp"
    bin_path = WORK / "cx_solver"
    book_path = WORK / "7x6.book"
    sub_path = WORK / "submission.py"

    cpp_path.write_text(SOLVER_CPP, encoding="utf-8")
    cmd = [
        "g++", "-O3", "-std=c++17", "-march=x86-64",
        "-static", "-pthread",
        str(cpp_path), "-o", str(bin_path),
    ]
    print("Compiling statically:", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)
    ldd_proc = subprocess.run(["ldd", str(bin_path)], capture_output=True, text=True)
    print("ldd rc:", ldd_proc.returncode, "out:", ldd_proc.stdout.strip(), "err:", ldd_proc.stderr.strip(), flush=True)
    file_out = subprocess.check_output(["file", str(bin_path)], text=True).strip()
    print("file:", file_out, flush=True)
    assert "statically linked" in file_out, f"Not statically linked: {file_out}"

    url = "https://github.com/PascalPons/connect4/releases/download/book/7x6.book"
    print("Downloading 7x6.book ...", flush=True)
    urllib.request.urlretrieve(url, str(book_path))
    raw_book = book_path.read_bytes()
    assert len(raw_book) == 33554524

    p = subprocess.Popen(
        [str(bin_path), str(book_path)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1,
    )
    assert p.stdout.readline().strip() == "READY"
    p.stdin.write("SOLVE " + ("0" * 42) + " 1 700\n")
    p.stdin.flush()
    out_empty = p.stdout.readline().strip()
    print("Empty board solve:", out_empty, flush=True)
    assert out_empty.startswith("3 ")
    p.stdin.write("QUIT\n")
    p.stdin.flush()
    p.wait(timeout=5)

    comp_book = zlib.compress(raw_book, level=6)
    comp_bin = zlib.compress(bin_path.read_bytes(), level=9)
    comp_cpp = zlib.compress(SOLVER_CPP.encode("utf-8"), level=9)

    def wrap_b85(b):
        s = base64.b85encode(b).decode("ascii")
        lines = [repr(s[i:i + 120]) for i in range(0, len(s), 120)]
        return "(\n    " + "\n    ".join(lines) + "\n)"

    sub_code = (
        SUBMISSION_TEMPLATE
        .replace("__BOOK_B85__", wrap_b85(comp_book))
        .replace("__BIN_B85__", wrap_b85(comp_bin))
        .replace("__CPP_B85__", wrap_b85(comp_cpp))
    )
    sub_path.write_text(sub_code, encoding="utf-8")
    py_compile.compile(str(sub_path), doraise=True)
    sz = sub_path.stat().st_size
    print(f"Built submission.py: {sz / 1e6:.2f} MB ({sz} bytes)", flush=True)
    assert sz < 45_000_000

    book_path.unlink(missing_ok=True)
    cpp_path.unlink(missing_ok=True)
    bin_path.unlink(missing_ok=True)

    subprocess.check_call(["pip", "install", "-q", "kaggle-environments"])
    from kaggle_environments import make

    env = make("connectx", debug=True)
    t_start = time.perf_counter()
    steps = env.run([str(sub_path), str(sub_path)])
    t_self = time.perf_counter() - t_start
    last = steps[-1]
    print(f"Self-play finished in {t_self:.2f}s, rewards={[last[0]['reward'], last[1]['reward']]}, statuses={[last[0]['status'], last[1]['status']]}", flush=True)
    assert last[0]["status"] == "DONE" and last[1]["status"] == "DONE"

    ns = {}
    exec(compile(sub_code, str(sub_path), "exec"), ns)
    fn = ns["agent"]

    move_times = []
    def timed_agent(obs, cfg):
        t0 = time.perf_counter()
        mv = fn(obs, cfg)
        move_times.append(time.perf_counter() - t0)
        return mv

    results = {1: Counter(), 2: Counter()}
    for mark in (1, 2):
        for _g in range(5):
            e = make("connectx", debug=True)
            agents = [timed_agent, "negamax"] if mark == 1 else ["negamax", timed_agent]
            st = e.run(agents)[-1]
            assert st[0]["status"] == "DONE" and st[1]["status"] == "DONE"
            results[mark][st[mark - 1]["reward"]] += 1
        print(f"vs negamax as P{mark}: {dict(results[mark])}", flush=True)

    shutil.rmtree(WORK / "__pycache__", ignore_errors=True)

    move_times.sort()
    p50 = move_times[len(move_times) // 2]
    p99 = move_times[int(0.99 * (len(move_times) - 1))]
    mx = move_times[-1]
    summary = {
        "submission_bytes": sz,
        "self_play_rewards": [last[0]["reward"], last[1]["reward"]],
        "vs_negamax_p1": dict(results[1]),
        "vs_negamax_p2": dict(results[2]),
        "first_move_s": round(move_times[0], 4),
        "p50_s": round(p50, 4),
        "p99_s": round(p99, 4),
        "max_s": round(mx, 4),
    }
    (WORK / "verify_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("ALL CONNECTX V2 CHECKS PASSED:", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
