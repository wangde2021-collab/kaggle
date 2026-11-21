# NFL Big Data Bowl 2027 — Phase 3b: regular-season game kinematics.
#
# Aggregates per-player regular-season 10 Hz game kinematics from
# data/parquet/game_tracking_*.parquet (REG only, ETL-filtered).
# Scheme §3c: trim each (game_id, play_id, nfl_id) trajectory to the
# ball_snap -> end-event window (tackle / pass_outcome_* / out_of_bounds /
# touchdown; fall back to ball_snap + 6.0 s), drop sensor jumps
# (dis > 1.3 yd or s > 12.5 yd/s), recompute signed a_t / a_n / omega from
# smoothed (x, y, s, dir) -- the raw `a` column is NEVER used (2025 sign
# break). Per-player outputs: game_s_p99, game_accel_p95, game_decel_p95,
# game_lat_accel_p95, snap_burst_1s, plus KTR ratios vs combine v_max /
# decel_p95 (joined later in model_phase3.py).
#
# Round-1 fixes (Gemini review 2026-11-21):
# - END_EVENTS no longer contains "handoff" / "run" / "fumble": a handoff
#   or run is the START of the ball-carrier's movement, not its end, and a
#   fumble does not end the play. Including them truncated most rushing
#   plays to a few frames.
# - Trajectories with no ball_snap event are DROPPED (they used to be kept
#   whole, which mixed pre-snap alignment shuffling into the kinematics).
# - Plays nullified by penalty are filtered via player_play
#   (play_nullified_by_penalty == 'N' only), matching the Y1/Y2 filter.
#
# Usage:
#   python3 src/game_kinematics.py [--parquet data/parquet]
#       [--out data/derived] [--seasons 2023,2024,2025]
# Writes <out>/game_kinematics.parquet (one row per player).
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kinematics import compute_kinematics  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Round-1 fix: handoff / run / fumble removed (see header).
END_EVENTS = {"tackle", "pass_outcome_caught", "pass_outcome_incomplete",
              "out_of_bounds", "touchdown", "pass_outcome_touchdown",
              "pass_outcome_interception", "qb_sack", "dropped_pass",
              "autoevent_passinterrupted"}


def _trim_window(traj):
    """Keep ball_snap -> first end event (or ball_snap + 6 s).

    Returns None when the trajectory has no ball_snap event (Round-1 fix:
    such trajectories are dropped instead of being kept whole).
    """
    ev = traj["event"].fillna("")
    snap = traj.index[ev == "ball_snap"]
    if len(snap) == 0:
        return None
    i0 = snap[0]
    after = traj.index[(traj.index > i0) & (ev.isin(END_EVENTS))]
    i1 = after[0] if len(after) else None
    sub = traj.loc[i0:i1] if i1 is not None else traj.loc[i0:]
    t0 = sub["time"].iloc[0]
    sub = sub[sub["time"] <= t0 + pd.Timedelta(seconds=6.0)]
    return sub


def process_season(path, reg_game_ids, nullified_plays):
    """Return per-player raw arrays (s, a_t, a_n) and snap-burst values."""
    pf = pd.read_parquet(path)
    pf["time"] = pd.to_datetime(pf["time"])
    pf = pf[pf["game_id"].isin(reg_game_ids)]
    if nullified_plays:
        keys = list(zip(pf["game_id"], pf["play_id"]))
        keep = np.array([k not in nullified_plays for k in keys])
        pf = pf[keep]
    pf = pf.sort_values(["game_id", "play_id", "nfl_id", "time"])
    # physical cleaning first (same bounds as combine)
    bad = (pf["dis"] > 1.30) | (pf["s"] > 12.5)
    pf = pf[~bad]
    per_player = {}
    dropped = {"no_snap": 0, "too_short": 0}
    for _, traj in pf.groupby(["game_id", "play_id", "nfl_id"], sort=False):
        if len(traj) < 10:
            dropped["too_short"] += 1
            continue
        sub = _trim_window(traj)
        if sub is None:
            dropped["no_snap"] += 1
            continue
        if len(sub) < 10:
            dropped["too_short"] += 1
            continue
        kin = compute_kinematics(sub)
        nid = int(traj["nfl_id"].iloc[0])
        d = per_player.setdefault(nid, {"s": [], "a_t": [], "a_n": [],
                                        "burst": []})
        d["s"].append(kin["s_smooth"].to_numpy())
        d["a_t"].append(kin["a_t"].to_numpy())
        d["a_n"].append(kin["a_n"].to_numpy())
        # snap burst: max speed gain in first 1.0 s after ball_snap
        t = kin["t_sec"].to_numpy()
        s = kin["s_smooth"].to_numpy()
        m = t <= 1.0
        if m.sum() >= 2:
            d["burst"].append(float(np.max(s[m]) - s[m][0]))
    return per_player, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default=os.path.join(BASE, "data", "parquet"))
    ap.add_argument("--out", default=os.path.join(BASE, "data", "derived"))
    ap.add_argument("--seasons", default="2023,2024,2025")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    games = pd.read_parquet(os.path.join(args.parquet, "games.parquet"))
    reg_game_ids = set(games.loc[games["season_type"] == "REG", "game_id"])
    pp = pd.read_parquet(os.path.join(args.parquet, "player_play.parquet"),
                         columns=["game_id", "play_id", "play_nullified_by_penalty"])
    nullified = set(map(tuple, pp.loc[
        pp["play_nullified_by_penalty"] != "N",
        ["game_id", "play_id"]].drop_duplicates().to_numpy()))
    print(f"REG games: {len(reg_game_ids)}; nullified plays: {len(nullified)}")

    all_players = {}
    total_dropped = {"no_snap": 0, "too_short": 0}
    for yr in args.seasons.split(","):
        path = os.path.join(args.parquet, f"game_tracking_{yr.strip()}.parquet")
        if not os.path.exists(path):
            print(f"skip missing {path}")
            continue
        print(f"processing {path} ...", flush=True)
        season, dropped = process_season(path, reg_game_ids, nullified)
        for k in total_dropped:
            total_dropped[k] += dropped[k]
        for nid, d in season.items():
            agg = all_players.setdefault(nid, {"s": [], "a_t": [], "a_n": [],
                                               "burst": []})
            for k in agg:
                agg[k].extend(d[k])
        print(f"  players so far: {len(all_players)}; dropped this season: {dropped}",
              flush=True)

    rows = []
    for nid, d in all_players.items():
        s = np.concatenate(d["s"]) if d["s"] else np.array([])
        a_t = np.concatenate(d["a_t"]) if d["a_t"] else np.array([])
        a_n = np.concatenate(d["a_n"]) if d["a_n"] else np.array([])
        rows.append({
            "nfl_id": nid,
            "game_frames": int(len(s)),
            "game_s_p99": float(np.percentile(s, 99)) if len(s) else np.nan,
            "game_s_p50": float(np.percentile(s, 50)) if len(s) else np.nan,
            "game_accel_p95": float(np.percentile(a_t[a_t > 0], 95))
            if (a_t > 0).sum() >= 20 else np.nan,
            "game_decel_p95": float(np.percentile(np.abs(a_t[a_t < 0]), 95))
            if (a_t < 0).sum() >= 20 else np.nan,
            "game_lat_accel_p95": float(np.percentile(a_n, 95)) if len(a_n) else np.nan,
            "game_snap_burst_1s": float(np.median(d["burst"])) if d["burst"] else np.nan,
        })
    out = pd.DataFrame(rows)
    dest = os.path.join(args.out, "game_kinematics.parquet")
    out.to_parquet(dest, engine="pyarrow", compression="zstd", index=False)
    print(f"wrote {dest}: {len(out)} players, "
          f"frames total {int(out['game_frames'].sum()):,}, "
          f"dropped trajectories {total_dropped}")
    print(out.describe().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
