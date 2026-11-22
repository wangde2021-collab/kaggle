"""Shared 10 Hz tracking kinematics (NFL Big Data Bowl 2027, Phase 2).

Implements the scheme's Step 0: Savitzky-Golay smoothing and recomputation of
signed kinematic primitives from (x, y, s, dir) -- the raw `a` column is never
used (its definition differs across seasons: unsigned in 2023/2024, signed
values in 2025).

Round-1 fixes (Gemini review 2026-11-21):
- `dir` is now unwrapped AND Savitzky-Golay smoothed before differentiation
  (previously raw diffs turned ~9 deg of optical jitter at 10 Hz into 90 deg/s
  phantom angular-velocity spikes, polluting lat_accel and cut detection).
- Duplicate timestamps are dropped; non-uniform sampling (gaps left by bad
  frame removal) is repaired by linear interpolation onto a uniform 0.1 s grid
  before filtering (Savitzky-Golay assumes equal spacing).
- `cut_events` uses a cut-arc definition: a maximal run of |omega| above
  threshold whose apex speed may drop as low as 0.5 yd/s (true 90/180 deg cuts
  bottom out at 0.5-2.0 yd/s), with speed retention measured against the
  pre-braking entry speed and clipped to [0, 1.2].

Conventions (all speeds in yd/s, time in s, angles in degrees in / radians out):
- a_t(t)  = ds/dt            (signed tangential acceleration; <0 = braking)
- omega  = wrap(dDir)/dt    (rad/s; 0/360 wrap handled)
- a_n    = s * |omega|       (normal / centripetal acceleration)
- jerk   = da_t/dt
"""
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

SAVGOL_WINDOW = 5
SAVGOL_POLY = 2
DT_TARGET = 0.1


def wrap_deg(d):
    """Wrap degree differences to [-180, 180)."""
    return (d + 180.0) % 360.0 - 180.0


def smooth_series(v, window=SAVGOL_WINDOW, poly=SAVGOL_POLY):
    v = np.asarray(v, dtype=float)
    if len(v) < window:
        return v.copy()
    w = window if len(v) >= window else (len(v) // 2 * 2 + 1)
    if w < 3:
        return v.copy()
    return savgol_filter(v, window_length=w, polyorder=min(poly, w - 1))


def _to_uniform_grid(df):
    """Resample a trajectory onto a uniform 0.1 s grid (linear interp).

    Only called when timestamps are duplicated or gaps exceed 0.15 s.
    Returns a DataFrame with the same columns (time rebuilt from the grid).
    """
    t0 = df["time"].iloc[0]
    t = (df["time"] - t0).dt.total_seconds().to_numpy(dtype=float)
    t_new = np.arange(0.0, t[-1] + 1e-9, DT_TARGET)
    if len(t_new) < 3:
        return df
    out = pd.DataFrame({"time": t0 + pd.to_timedelta(t_new, unit="s")})
    for c in df.columns:
        if c == "time":
            continue
        v = df[c].to_numpy()
        if np.issubdtype(v.dtype, np.number):
            out[c] = np.interp(t_new, t, v)
        else:
            # categorical columns (event, drill names): nearest value
            idx = np.searchsorted(t, t_new, side="left")
            idx = np.clip(idx, 0, len(v) - 1)
            out[c] = v[idx]
    return out


def compute_kinematics(traj):
    """Add signed kinematic primitives to one trajectory DataFrame.

    Input columns: time (datetime64), x, y, s, dir. Rows must already be
    sorted by time. Returns a copy with dt, t_sec, s_smooth, dir_smooth,
    a_t, omega, a_n, jerk.
    """
    df = traj.sort_values("time").copy()
    # drop duplicate timestamps (dt == 0 would divide by zero downstream)
    df = df.drop_duplicates(subset=["time"], keep="first")
    if len(df) < 5:
        df = df.copy()
        df["t_sec"] = (df["time"] - df["time"].iloc[0]).dt.total_seconds()
        for c in ("s_smooth", "dir_smooth", "a_t", "omega", "a_n", "jerk"):
            df[c] = np.nan
        df["dt"] = np.nan
        return df
    t = (df["time"] - df["time"].iloc[0]).dt.total_seconds().to_numpy(dtype=float)
    dt = np.diff(t)
    if (dt <= 0).any() or (dt > 0.15).any() or (dt < 0.05).any():
        df = _to_uniform_grid(df)
        t = (df["time"] - df["time"].iloc[0]).dt.total_seconds().to_numpy(dtype=float)
    s = smooth_series(df["s"].to_numpy(dtype=float))
    # unwrap direction to radians, smooth, then differentiate (Round-1 fix)
    dir_unwrapped = np.unwrap(np.deg2rad(df["dir"].to_numpy(dtype=float)))
    dir_smooth = smooth_series(dir_unwrapped)
    df["t_sec"] = t
    df["s_smooth"] = s
    df["dir_smooth"] = np.rad2deg(dir_smooth) % 360.0
    df["dt"] = np.gradient(t)
    df["a_t"] = np.gradient(s, t)
    df["omega"] = np.gradient(dir_smooth, t)
    df["a_n"] = s * np.abs(df["omega"].to_numpy())
    df["jerk"] = np.gradient(df["a_t"].to_numpy(dtype=float), t)
    return df


def cut_events(df, omega_thresh_deg=60.0, s_apex_min=0.5):
    """Detect cut (change-of-direction) events.

    A cut is a maximal run of frames with |omega| > threshold (default 60
    deg/s on the *smoothed* direction signal). The apex speed may drop as low
    as ``s_apex_min`` (true 90/180 deg cuts bottom out at 0.5-2.0 yd/s, so the
    old s > 2.5 yd/s gate filtered out exactly the cuts that matter).
    Returns a list of dicts with apex index, retention and re-acceleration
    slope. Retention is apex speed vs the pre-braking entry speed (max speed
    in the 0.5 s before the window) and is clipped to [0, 1.2].
    """
    om = np.abs(np.rad2deg(df["omega"].to_numpy()))
    s = df["s_smooth"].to_numpy()
    a_t = df["a_t"].to_numpy()
    t = df["t_sec"].to_numpy()
    mask = om > omega_thresh_deg
    events = []
    i, n = 0, len(df)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and mask[j + 1]:
            j += 1
        apex = i + int(np.argmax(om[i:j + 1]))
        # entry speed: max speed in the 0.5 s before the window starts
        before = (t >= t[i] - 0.5) & (t < t[i])
        s_entry = float(np.max(s[before])) if before.any() else float(s[i])
        retention = s[apex] / s_entry if s_entry > 0 else np.nan
        retention = float(np.clip(retention, 0.0, 1.2)) if np.isfinite(retention) else np.nan
        # re-acceleration: mean positive a_t in the 0.5 s after the apex
        after = (t > t[apex]) & (t <= t[apex] + 0.5)
        reacc = float(np.mean(a_t[after][a_t[after] > 0])) if after.any() and (a_t[after] > 0).any() else np.nan
        # cumulative turn angle across the window (deg), using actual dt
        dt_arr = df["dt"].to_numpy(dtype=float)
        turn_angle = float(np.nansum(np.abs(np.rad2deg(
            df["omega"].to_numpy()[i:j + 1])) * dt_arr[i:j + 1]))
        events.append({"apex_idx": apex, "omega_deg": float(om[apex]),
                       "s_apex": float(s[apex]), "retention": retention,
                       "reaccel_slope": reacc, "turn_angle_deg": turn_angle})
        i = j + 1
    return events


def fit_sprint_curve(t, v, s_peak=None):
    """Fit Furusawa-Hill / Samozino sprint model v(t) = v_max * (1 - exp(-t/tau)).

    Round-2 fix (Gemini review 2026-11-21): the old no-offset fit forced the
    curve through (0, 0), so the ~0.8-0.9 s low-speed start-up tail (crouch
    wobble / sensor lag before the 3-frame s > 0.5 yd/s trigger) inflated
    tau to ~1.4-1.5 s and pushed the asymptotic v_max to 12.20-12.27 yd/s
    for WRs -- faster than any Combine athlete ever ran (~25 mph). The fit
    now includes a start-time offset t0:
        v(t) = v_max * (1 - exp(-max(t - t0, 0) / tau)), t0 in [-0.3, 1.5]
    and a physical cap v_max <= s_peak * 1.12 (s_peak = measured peak
    speed of the same attempt; no stopwatch split is used as a constraint,
    so TRACK_PURE stays stopwatch-free). Returns dict(v_max, tau, a0, r2,
    t0) or NaN values on failure.
    """
    from scipy.optimize import curve_fit

    t = np.asarray(t, dtype=float)
    v = np.asarray(v, dtype=float)
    ok = np.isfinite(t) & np.isfinite(v)
    t, v = t[ok], v[ok]
    if len(t) < 10 or t.max() < 1.0:
        return {"v_max": np.nan, "tau": np.nan, "a0": np.nan, "r2": np.nan,
                "t0": np.nan}
    v0 = float(np.max(v))
    cap = float(s_peak) * 1.12 if (s_peak is not None
                                   and np.isfinite(s_peak)
                                   and s_peak > 0) else v0 * 1.5 + 1.0
    cap = max(cap, v0 * 1.02)  # never cap below the observed peak

    def model(tt, vmax, tau, t0):
        return vmax * (1.0 - np.exp(-np.maximum(tt - t0, 0.0) / tau))

    try:
        popt, _ = curve_fit(model, t, v, p0=[min(v0, cap), 0.5, 0.0],
                            bounds=([max(v0 * 0.8, 1.0), 0.05, -0.3],
                                    [cap, 3.0, 1.5]),
                            maxfev=5000)
        vmax, tau, t0 = float(popt[0]), float(popt[1]), float(popt[2])
        pred = model(t, vmax, tau, t0)
        ss_res = float(np.sum((v - pred) ** 2))
        ss_tot = float(np.sum((v - np.mean(v)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        return {"v_max": vmax, "tau": tau, "a0": vmax / tau, "r2": r2,
                "t0": t0}
    except Exception:
        return {"v_max": np.nan, "tau": np.nan, "a0": np.nan, "r2": np.nan,
                "t0": np.nan}
