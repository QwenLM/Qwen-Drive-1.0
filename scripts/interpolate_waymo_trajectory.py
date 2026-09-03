#!/usr/bin/env python3
"""Resample Waymo WOD-E2E 4 Hz ego trajectories to the 10 Hz scene format.
The Waymo Open Dataset end-to-end driving metadata stores the past (16 points,
-1.5 s .. 0 s) and future (20 points, 0.25 s .. 5 s) ego trajectory at 4 Hz.
Qwen-Drive consumes trajectories at 10 Hz (see docs/data.md), so both halves
are resampled:
* History: a cubic spline over the 16 raw positions evaluated on the 10 Hz
  grid, headings taken from the velocity direction and linearly interpolated,
  velocity/acceleration linearly resampled.
* Future: a natural cubic spline over the current pose plus the 20 raw future
  positions evaluated on the 10 Hz grid, headings obtained by integrating the
  spline curvature directly on the 10 Hz grid, velocity/acceleration from the
  spline derivatives.
Everything is expressed in the ego frame of the current timestamp: x forward,
y left, heading positive for a left turn, current pose (0, 0, 0).
:func:`interpolate_history` / :func:`interpolate_future` are the resampling
kernels; each takes the corresponding raw metadata dict and returns the
(trajectory, velocity, acceleration) arrays of the 10 Hz scene format.
"""
from __future__ import annotations
import math
import numpy as np
from scipy.interpolate import CubicSpline
RAW_HIST_DT_S = 0.25
RAW_HIST_TIMES = np.round(np.arange(-15, 1, dtype=np.float64) / 4.0, 10)
HIST_10HZ_TIMES = np.round(np.arange(-15, 1, dtype=np.float64) / 10.0, 10)
RAW_FUTURE_TIMES = np.round(np.arange(1, 21, dtype=np.float64) / 4.0, 10)
FUTURE_10HZ_TIMES = np.round(np.arange(1, 51, dtype=np.float64) / 10.0, 10)
FUTURE_HEADING_MIN_SPEED = 0.3
MIN_HEADING_SPEED = 0.2
def wrap_angle(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))
def require_vector_pair(container, x_key, y_key, expected):
    x = np.asarray(container.get(x_key), dtype=np.float64)
    y = np.asarray(container.get(y_key), dtype=np.float64)
    valid = (
        x.shape == (expected,)
        and y.shape == (expected,)
        and np.isfinite(x).all()
        and np.isfinite(y).all()
    )
    if not valid:
        raise ValueError(
            f"invalid {x_key}/{y_key}; expected {expected} finite values"
        )
    return np.column_stack([x, y])
def linear_resample(source_times, values, target_times):
    values = np.asarray(values, dtype=np.float64)
    return np.column_stack(
        [
            np.interp(target_times, source_times, values[:, channel])
            for channel in range(values.shape[1])
        ]
    )
def rotate_vectors_to_body(vectors_cf, headings):
    """Rotate (N, 2) course-frame vectors into the body frame of each heading."""
    cosine = np.cos(headings)
    sine = np.sin(headings)
    return np.column_stack(
        [
            vectors_cf[:, 0] * cosine + vectors_cf[:, 1] * sine,
            -vectors_cf[:, 0] * sine + vectors_cf[:, 1] * cosine,
        ]
    )
def headings_from_velocity(velocity_cf, anchor_heading=0.0):
    """Headings from velocity direction, anchored at the current frame.
    Low-speed points keep the previous heading; speeds near the anchor are
    flipped by pi so the curve stays continuous around the current pose.
    """
    velocity_cf = np.asarray(velocity_cf, dtype=np.float64)
    result = np.zeros(len(velocity_cf), dtype=np.float64)
    running = float(anchor_heading)
    for i in range(len(velocity_cf) - 1, -1, -1):
        vx, vy = velocity_cf[i]
        if math.hypot(vx, vy) >= MIN_HEADING_SPEED:
            raw = math.atan2(vy, vx)
            delta = math.atan2(math.sin(raw - running), math.cos(raw - running))
            if abs(delta) > math.pi / 2:
                raw = math.atan2(math.sin(raw + math.pi), math.cos(raw + math.pi))
            running = raw
        result[i] = running
    if len(result):
        result = np.unwrap(result)
        result -= result[-1]
    return wrap_angle(result)
def integrate_curvature(times_s, velocity_cf, acceleration_cf):
    """Heading from curvature integration: yaw rate = (v x a) / |v|^2.
    The yaw rate is trapezoidally integrated over ``times_s`` and anchored to
    zero at the first sample. Samples slower than FUTURE_HEADING_MIN_SPEED
    contribute zero yaw rate. Returns the unwrapped heading at every sample.
    """
    times_s = np.asarray(times_s, dtype=np.float64)
    velocity_cf = np.asarray(velocity_cf, dtype=np.float64)
    acceleration_cf = np.asarray(acceleration_cf, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        speed_squared = np.sum(velocity_cf * velocity_cf, axis=1)
        observable = speed_squared >= FUTURE_HEADING_MIN_SPEED**2
        cross = (
            velocity_cf[:, 0] * acceleration_cf[:, 1]
            - velocity_cf[:, 1] * acceleration_cf[:, 0]
        )
        yawrate = np.zeros(len(times_s), dtype=np.float64)
        yawrate[observable] = cross[observable] / speed_squared[observable]
        yawrate[0] = 0.0
        heading = np.zeros(len(times_s), dtype=np.float64)
        heading[1:] = np.cumsum(
            0.5 * (yawrate[1:] + yawrate[:-1]) * np.diff(times_s)
        )
    if not (np.isfinite(yawrate).all() and np.isfinite(heading).all()):
        raise ValueError("nonfinite heading from curvature integration")
    return heading
def interpolate_history(past):
    """Resample the raw 4 Hz past trajectory to the 10 Hz history fields.
    ``past`` is the metadata record's ``past_trajectory`` dict with ``pos_x``,
    ``pos_y``, ``vel_x``, ``vel_y`` (16 values each, -1.5 s .. 0 s at 4 Hz) and
    ``accel_x``, ``accel_y`` in velocity units per raw step. Returns
    ``hist_traj_10hz`` [16, 3] (x, y, heading), ``hist_vel_10hz`` and
    ``hist_acc_10hz`` [16, 2], all anchored at the current pose (0, 0, 0).
    """
    history_xy = require_vector_pair(past, "pos_x", "pos_y", 16)
    velocity_cf = require_vector_pair(past, "vel_x", "vel_y", 16)
    acceleration_cf = (
        require_vector_pair(past, "accel_x", "accel_y", 16) / RAW_HIST_DT_S
    )
    xy_10hz = CubicSpline(RAW_HIST_TIMES, history_xy, axis=0)(HIST_10HZ_TIMES)
    xy_10hz[-1] = [0.0, 0.0]
    headings_10hz = np.interp(
        HIST_10HZ_TIMES,
        RAW_HIST_TIMES,
        np.unwrap(headings_from_velocity(velocity_cf)),
    )
    headings_10hz = wrap_angle(headings_10hz)
    headings_10hz[-1] = 0.0
    velocity_10hz = rotate_vectors_to_body(
        linear_resample(RAW_HIST_TIMES, velocity_cf, HIST_10HZ_TIMES),
        headings_10hz,
    )
    acceleration_10hz = rotate_vectors_to_body(
        linear_resample(RAW_HIST_TIMES, acceleration_cf, HIST_10HZ_TIMES),
        headings_10hz,
    )
    return (
        np.column_stack([xy_10hz, headings_10hz]),
        velocity_10hz,
        acceleration_10hz,
    )
def interpolate_future(future):
    """Resample the raw 4 Hz future trajectory to the 10 Hz future fields.
    ``future`` is the metadata record's ``future_trajectory`` dict with
    ``pos_x`` and ``pos_y`` (20 values each, 0.25 s .. 5 s at 4 Hz). Positions
    come from a natural cubic spline anchored at the current pose; headings
    integrate the spline curvature directly on the 10 Hz grid rather than
    interpolating a coarser 4 Hz heading curve. Returns ``future_traj_10hz``
    [50, 3] (x, y, heading), ``future_vel_10hz`` and ``future_acc_10hz``
    [50, 2].
    """
    future_xy = require_vector_pair(future, "pos_x", "pos_y", 20)
    spline = CubicSpline(
        np.concatenate([[0.0], RAW_FUTURE_TIMES]),
        np.vstack([[0.0, 0.0], future_xy]),
        axis=0,
        bc_type="natural",
    )
    xy_10hz = spline(FUTURE_10HZ_TIMES)
    velocity_cf = spline(FUTURE_10HZ_TIMES, 1)
    acceleration_cf = spline(FUTURE_10HZ_TIMES, 2)
    heading = integrate_curvature(
        np.concatenate([[0.0], FUTURE_10HZ_TIMES]),
        spline(np.concatenate([[0.0], FUTURE_10HZ_TIMES]), 1),
        spline(np.concatenate([[0.0], FUTURE_10HZ_TIMES]), 2),
    )[1:]
    headings_10hz = wrap_angle(heading)
    return (
        np.column_stack([xy_10hz, headings_10hz]),
        rotate_vectors_to_body(velocity_cf, headings_10hz),
        rotate_vectors_to_body(acceleration_cf, headings_10hz),
    )
