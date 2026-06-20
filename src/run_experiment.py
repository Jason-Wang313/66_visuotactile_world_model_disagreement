import csv
import math
import os
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np


BASE_SEED = 231575465
DEFAULT_SEED_COUNT = 8
DEFAULT_EPISODES_PER_SEED = 24
DEFAULT_ABLATION_EPISODES_PER_SEED = 24
DEFAULT_STRESS_EPISODES_PER_SEED = 12
MAX_WORKERS = max(1, min(4, int(os.environ.get("PAPER66_WORKERS", "4"))))

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

OBJECT_HALF = 0.04
FINGER_RADIUS = 0.015
CONTACT_GAP = OBJECT_HALF + FINGER_RADIUS + 0.008
SUCCESS_RADIUS = 0.075
CONTACT_LIMIT = 520.0

METHODS = [
    "random_push",
    "vision_only_mpc",
    "tactile_only_mpc",
    "mean_fusion_mpc",
    "ensemble_uncertainty_mpc",
    "conformal_risk_filter",
    "diagnostic_probe_then_mpc",
    "robust_minimax_mpc",
    "particle_belief_mpc",
    "old_vt_disagreement_branch_mpc",
    "cvtb_mpc_v5",
    "cvtb_no_probe",
    "oracle_mode_mpc",
]

ABLATIONS = [
    "cvtb_mpc_v5",
    "cvtb_no_probe",
    "no_sensor_health",
    "no_branch_preservation",
    "no_value_of_information",
    "no_cvar_tail",
    "no_contact_safety",
    "no_reliability_fallback",
    "mean_only_fusion",
    "tactile_trust_high",
    "small_branch_set",
    "old_vt_disagreement_branch_mpc",
    "ensemble_uncertainty_mpc",
    "robust_minimax_mpc",
    "oracle_mode_mpc",
]

MAIN_SPLITS = [
    "nominal",
    "high_friction",
    "low_friction",
    "vision_bias",
    "tactile_noise",
    "sticky_contact",
    "combined_shift",
    "sensor_conflict",
    "contact_dropout",
    "delayed_touch_sticky",
]


@dataclass(frozen=True)
class RolloutResult:
    final_pos: np.ndarray
    object_path: float
    pusher_path: float
    contact_impulse: float
    max_contact_force: float
    contact_steps: int
    first_contact_pusher: np.ndarray | None


MODEL_CACHE: dict[tuple[float, float], mujoco.MjModel] = {}


def stable_int(text: str) -> int:
    return sum((idx + 1) * ord(ch) for idx, ch in enumerate(text))


def unit(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-8:
        return np.array([1.0, 0.0], dtype=float)
    return vec / norm


def rotate(vec: np.ndarray, angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array([c * vec[0] - s * vec[1], s * vec[0] + c * vec[1]], dtype=float)


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def ci95(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    arr = np.asarray(values, dtype=float)
    return float(1.96 * arr.std(ddof=1) / math.sqrt(len(arr)))


def normal_p_from_t(t_stat: float) -> float:
    return float(math.erfc(abs(t_stat) / math.sqrt(2.0)))


def mode_config(split: str, rng: np.random.Generator) -> dict:
    bias_dir = unit(rng.normal(0.0, 1.0, size=2))
    tactile_bias_dir = unit(rng.normal(0.0, 1.0, size=2))
    configs = {
        "nominal": dict(friction=0.55, drag=0.995, vision_bias=0.0, vision_noise=0.006, tactile_noise=0.006, actuator_scale=1.00),
        "high_friction": dict(friction=1.25, drag=0.985, vision_bias=0.0, vision_noise=0.006, tactile_noise=0.008, actuator_scale=0.96),
        "low_friction": dict(friction=0.22, drag=0.999, vision_bias=0.0, vision_noise=0.006, tactile_noise=0.008, actuator_scale=1.04),
        "vision_bias": dict(friction=0.60, drag=0.995, vision_bias=0.052, vision_noise=0.009, tactile_noise=0.007, actuator_scale=1.00),
        "tactile_noise": dict(friction=0.62, drag=0.995, vision_bias=0.0, vision_noise=0.006, tactile_noise=0.040, actuator_scale=1.00),
        "sticky_contact": dict(friction=1.55, drag=0.950, vision_bias=0.010, vision_noise=0.007, tactile_noise=0.010, actuator_scale=0.92),
        "combined_shift": dict(friction=1.35, drag=0.960, vision_bias=0.045, vision_noise=0.010, tactile_noise=0.030, actuator_scale=0.94),
        "sensor_conflict": dict(friction=0.74, drag=0.985, vision_bias=0.060, vision_noise=0.011, tactile_noise=0.044, tactile_bias=0.055, actuator_scale=0.96, contact_dropout=0.10),
        "contact_dropout": dict(friction=0.70, drag=0.990, vision_bias=0.025, vision_noise=0.009, tactile_noise=0.020, tactile_bias=0.012, actuator_scale=0.98, contact_dropout=0.45),
        "delayed_touch_sticky": dict(friction=1.65, drag=0.940, vision_bias=0.040, vision_noise=0.010, tactile_noise=0.028, tactile_bias=0.035, actuator_scale=0.90, tactile_delay=0.70),
    }
    cfg = configs[split].copy()
    cfg["vision_bias_vec"] = bias_dir * cfg.pop("vision_bias")
    cfg["tactile_bias_vec"] = tactile_bias_dir * cfg.pop("tactile_bias", 0.0)
    cfg["contact_dropout"] = cfg.get("contact_dropout", 0.0)
    cfg["tactile_delay"] = cfg.get("tactile_delay", 0.0)
    cfg["split"] = split
    return cfg


def stress_config(level: float, rng: np.random.Generator) -> dict:
    bias_dir = unit(rng.normal(0.0, 1.0, size=2))
    tactile_bias_dir = unit(rng.normal(0.0, 1.0, size=2))
    return {
        "split": f"stress_{level:.2f}",
        "friction": 0.55 + 0.85 * level,
        "drag": 0.995 - 0.045 * level,
        "vision_noise": 0.006 + 0.004 * level,
        "tactile_noise": 0.006 + 0.030 * level,
        "actuator_scale": 1.0 - 0.07 * level,
        "vision_bias_vec": bias_dir * (0.055 * level),
        "tactile_bias_vec": tactile_bias_dir * (0.040 * level),
        "contact_dropout": 0.18 * level,
        "tactile_delay": 0.35 * level,
    }


def model_xml(friction: float, drag: float) -> str:
    # Drag is handled in the rollout loop; the key keeps separate caches for sticky modes.
    del drag
    table_friction = max(0.15, friction)
    obj_friction = max(0.12, friction)
    return f"""
<mujoco model="visuotactile_pusher">
  <compiler angle="radian" coordinate="local"/>
  <option timestep="0.01" gravity="0 0 -9.81" integrator="RK4" cone="elliptic"/>
  <default>
    <geom condim="4" solref="0.008 1" solimp="0.9 0.95 0.001"/>
  </default>
  <worldbody>
    <geom name="table" type="plane" size="1.0 1.0 0.05" friction="{table_friction:.4f} 0.004 0.0001" rgba="0.80 0.82 0.84 1"/>
    <body name="object" pos="0 0 {OBJECT_HALF}">
      <freejoint name="object_free"/>
      <geom name="object_geom" type="box" size="{OBJECT_HALF} {OBJECT_HALF} {OBJECT_HALF}" mass="0.16"
            friction="{obj_friction:.4f} 0.004 0.0001" rgba="0.10 0.42 0.82 1"/>
    </body>
    <body name="pusher" pos="0 0 {OBJECT_HALF}">
      <joint name="px" type="slide" axis="1 0 0" range="-0.75 0.75" damping="4"/>
      <joint name="py" type="slide" axis="0 1 0" range="-0.55 0.55" damping="4"/>
      <geom name="finger_geom" type="capsule" fromto="0 -0.035 0 0 0.035 0" size="{FINGER_RADIUS}"
            mass="0.08" friction="1.6 0.005 0.0001" rgba="0.86 0.24 0.14 1"/>
    </body>
  </worldbody>
  <actuator>
    <position name="ax" joint="px" kp="650" ctrlrange="-0.75 0.75"/>
    <position name="ay" joint="py" kp="650" ctrlrange="-0.55 0.55"/>
  </actuator>
</mujoco>
"""


def get_model(friction: float, drag: float) -> mujoco.MjModel:
    key = (round(float(friction), 3), round(float(drag), 3))
    if key not in MODEL_CACHE:
        MODEL_CACHE[key] = mujoco.MjModel.from_xml_string(model_xml(*key))
    return MODEL_CACHE[key]


def contact_force(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    obj_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "object_geom")
    finger_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "finger_geom")
    force = np.zeros(6, dtype=float)
    total = 0.0
    for cidx in range(data.ncon):
        contact = data.contact[cidx]
        if {contact.geom1, contact.geom2} == {obj_gid, finger_gid}:
            mujoco.mj_contactForce(model, data, cidx, force)
            total += float(np.linalg.norm(force[:3]))
    return total


def rollout_push(cfg: dict, object_pos: np.ndarray, pusher_start: np.ndarray, pusher_end: np.ndarray, steps: int) -> RolloutResult:
    model = get_model(cfg["friction"], cfg["drag"])
    data = mujoco.MjData(model)
    obj_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    px_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "px")
    py_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "py")
    px_adr = model.jnt_qposadr[px_jid]
    py_adr = model.jnt_qposadr[py_jid]

    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.qpos[0] = object_pos[0]
    data.qpos[1] = object_pos[1]
    data.qpos[2] = OBJECT_HALF
    data.qpos[3] = 1.0
    data.qpos[px_adr] = pusher_start[0]
    data.qpos[py_adr] = pusher_start[1]
    data.ctrl[0] = pusher_start[0]
    data.ctrl[1] = pusher_start[1]
    mujoco.mj_forward(model, data)

    path_obj = 0.0
    path_pusher = 0.0
    contact_impulse = 0.0
    max_force = 0.0
    contact_steps = 0
    first_contact = None
    prev_obj = data.xpos[obj_bid][:2].copy()
    prev_pusher = pusher_start.copy()

    for _ in range(10):
        data.ctrl[:] = pusher_start
        mujoco.mj_step(model, data)

    for step in range(steps):
        alpha = (step + 1) / steps
        desired = pusher_start * (1.0 - alpha) + pusher_end * alpha
        desired = pusher_start + (desired - pusher_start) * cfg["actuator_scale"]
        data.ctrl[0] = clamp(float(desired[0]), -0.72, 0.72)
        data.ctrl[1] = clamp(float(desired[1]), -0.52, 0.52)
        mujoco.mj_step(model, data)

        if cfg["drag"] < 0.995:
            data.qvel[0] *= cfg["drag"]
            data.qvel[1] *= cfg["drag"]

        obj_pos = data.xpos[obj_bid][:2].copy()
        pusher_pos = np.array([data.qpos[px_adr], data.qpos[py_adr]], dtype=float)
        path_obj += float(np.linalg.norm(obj_pos - prev_obj))
        path_pusher += float(np.linalg.norm(pusher_pos - prev_pusher))
        prev_obj = obj_pos
        prev_pusher = pusher_pos

        f = contact_force(model, data)
        if f > 1e-6:
            contact_steps += 1
            contact_impulse += f
            max_force = max(max_force, f)
            if first_contact is None:
                first_contact = pusher_pos.copy()

    return RolloutResult(
        final_pos=data.xpos[obj_bid][:2].copy(),
        object_path=path_obj,
        pusher_path=path_pusher,
        contact_impulse=contact_impulse,
        max_contact_force=max_force,
        contact_steps=contact_steps,
        first_contact_pusher=first_contact,
    )


def make_episode(seed: int, episode: int, split: str, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    object_pos = rng.uniform([-0.035, -0.075], [0.035, 0.075])
    target_angle = rng.uniform(-0.45, 0.45)
    target_dist = rng.uniform(0.19, 0.30)
    if split in {"sensor_conflict", "contact_dropout", "delayed_touch_sticky"}:
        target_angle = rng.uniform(-0.70, 0.70)
        target_dist = rng.uniform(0.23, 0.34)
    target = object_pos + rotate(np.array([1.0, 0.0]), target_angle) * target_dist
    if split in {"low_friction", "vision_bias"}:
        target[1] += rng.uniform(-0.035, 0.035)
    if split in {"sensor_conflict", "delayed_touch_sticky"}:
        target[1] += rng.choice([-1.0, 1.0]) * rng.uniform(0.020, 0.055)
    target[0] = clamp(float(target[0]), 0.12, 0.38)
    target[1] = clamp(float(target[1]), -0.18, 0.18)
    return object_pos.astype(float), target.astype(float)


def common_probe(cfg: dict, object_pos: np.ndarray, vision_obs: np.ndarray, target: np.ndarray, rng: np.random.Generator) -> dict:
    direction = unit(target - vision_obs)
    start = vision_obs - direction * CONTACT_GAP
    end = vision_obs + direction * 0.018
    result = rollout_push(cfg, object_pos, start, end, 42)

    if result.first_contact_pusher is None:
        tactile_est = vision_obs + rng.normal(0.0, cfg["tactile_noise"] * 1.4, size=2)
        confidence = 0.12
        contact_seen = 0
    else:
        tactile_est = result.first_contact_pusher + direction * CONTACT_GAP
        tactile_est = tactile_est + rng.normal(0.0, cfg["tactile_noise"], size=2)
        confidence = clamp(0.92 - 10.0 * cfg["tactile_noise"], 0.28, 0.92)
        contact_seen = 1

    if contact_seen and rng.random() < cfg.get("contact_dropout", 0.0):
        tactile_est = vision_obs + rng.normal(0.0, cfg["tactile_noise"] * 1.6, size=2)
        confidence *= 0.25
        contact_seen = 0
    if contact_seen:
        tactile_est = tactile_est + cfg.get("tactile_bias_vec", np.zeros(2))
        delay = cfg.get("tactile_delay", 0.0)
        if delay > 0.0:
            tactile_est = tactile_est * (1.0 - delay) + vision_obs * delay
            confidence *= 1.0 - 0.55 * delay

    force_per_step = result.contact_impulse / max(1, result.contact_steps)
    displacement = float(np.linalg.norm(result.final_pos - object_pos))
    friction_signal = clamp((force_per_step / 260.0) - 2.2 * displacement, 0.2, 2.4)
    disagreement = float(np.linalg.norm(vision_obs - tactile_est))
    return {
        "actual_after_probe": result.final_pos,
        "tactile_est": tactile_est,
        "tactile_confidence": confidence,
        "contact_seen": contact_seen,
        "probe_force": force_per_step,
        "probe_displacement": displacement,
        "friction_signal": friction_signal,
        "disagreement": disagreement,
        "probe_energy": result.pusher_path + 0.0008 * result.contact_impulse,
        "probe_max_contact": result.max_contact_force,
    }


def diagnostic_probe(cfg: dict, object_pos: np.ndarray, anchor: np.ndarray, target: np.ndarray, rng: np.random.Generator) -> dict:
    direction = unit(target - anchor)
    start = anchor - direction * CONTACT_GAP
    end = anchor + direction * 0.035
    result = rollout_push(cfg, object_pos, start, end, 46)
    if result.first_contact_pusher is None:
        tactile_est = anchor + rng.normal(0.0, cfg["tactile_noise"] * 1.5, size=2)
        confidence = 0.18
        contact_seen = 0
    else:
        tactile_est = result.first_contact_pusher + direction * CONTACT_GAP
        tactile_est = tactile_est + rng.normal(0.0, cfg["tactile_noise"] * 0.8, size=2)
        confidence = clamp(0.96 - 8.0 * cfg["tactile_noise"], 0.35, 0.96)
        contact_seen = 1
    if contact_seen and rng.random() < 0.65 * cfg.get("contact_dropout", 0.0):
        tactile_est = anchor + rng.normal(0.0, cfg["tactile_noise"] * 1.2, size=2)
        confidence *= 0.30
        contact_seen = 0
    if contact_seen:
        tactile_est = tactile_est + 0.65 * cfg.get("tactile_bias_vec", np.zeros(2))
        delay = cfg.get("tactile_delay", 0.0)
        if delay > 0.0:
            tactile_est = tactile_est * (1.0 - 0.55 * delay) + anchor * (0.55 * delay)
            confidence *= 1.0 - 0.40 * delay
    force_per_step = result.contact_impulse / max(1, result.contact_steps)
    return {
        "actual_after_probe": result.final_pos,
        "tactile_est": tactile_est,
        "tactile_confidence": confidence,
        "contact_seen": contact_seen,
        "probe_force": force_per_step,
        "probe_displacement": float(np.linalg.norm(result.final_pos - object_pos)),
        "probe_energy": result.pusher_path + 0.0008 * result.contact_impulse,
        "probe_max_contact": result.max_contact_force,
    }


def estimate_friction(obs: dict) -> float:
    return clamp(0.55 + 0.55 * obs["friction_signal"], 0.22, 1.75)


def is_cvtb_method(method: str) -> bool:
    if method in {"cvtb_mpc_v5", "cvtb_no_probe"}:
        return True
    if method.startswith("ablation:"):
        return method.split(":", 1)[1] in {
            "cvtb_mpc_v5",
            "cvtb_no_probe",
            "no_sensor_health",
            "no_value_of_information",
            "no_cvar_tail",
            "no_contact_safety",
            "no_reliability_fallback",
            "tactile_trust_high",
            "small_branch_set",
        }
    return False


def ablation_name(method: str) -> str | None:
    return method.split(":", 1)[1] if method.startswith("ablation:") else None


def sensor_reliability(obs: dict, method: str) -> tuple[float, float, float]:
    ablation = ablation_name(method)
    disagreement = float(obs["disagreement"])
    tactile_noise = float(obs.get("tactile_noise", 0.0))
    contact_seen = float(obs["contact_seen"])
    tactile_conf = float(obs["tactile_confidence"]) if obs["contact_seen"] else 0.08
    force = float(obs["probe_force"])
    displacement = float(obs["probe_displacement"])
    sticky_signal = clamp(force / 520.0 - 3.0 * displacement, 0.0, 1.0)
    vision_rel = clamp(0.86 - 5.0 * max(0.0, disagreement - 0.018) - 1.5 * float(obs.get("vision_bias_norm", 0.0)), 0.12, 0.92)
    tactile_rel = clamp(0.12 + 0.72 * contact_seen + 0.40 * tactile_conf - 8.0 * tactile_noise - 0.28 * float(obs.get("contact_dropout_rate", 0.0)), 0.08, 0.94)
    if ablation == "no_sensor_health":
        vision_rel = clamp(0.55 + 0.35 * (1.0 - contact_seen), 0.15, 0.90)
        tactile_rel = clamp(0.55 + 4.0 * disagreement, 0.15, 0.90)
    if ablation == "tactile_trust_high":
        tactile_rel = 0.92
        vision_rel = min(vision_rel, 0.45)
    return vision_rel, tactile_rel, sticky_signal


def make_branches(method: str, obs: dict) -> list[dict]:
    vision = obs["vision_obs"]
    tactile = obs["tactile_est"]
    target = obs["target"]
    disagreement = obs["disagreement"]
    tactile_conf = obs["tactile_confidence"] if obs["contact_seen"] else 0.10
    friction_est = estimate_friction(obs)
    visual_weight = clamp(0.78 - 7.0 * disagreement, 0.15, 0.88)
    tactile_weight = clamp(tactile_conf + 8.0 * disagreement, 0.12, 0.90)
    total = visual_weight + tactile_weight

    if method == "vision_only_mpc":
        return [dict(state=vision, friction=0.55, weight=1.0, tag="vision")]
    if method == "tactile_only_mpc":
        state = tactile if obs["contact_seen"] else vision
        return [dict(state=state, friction=friction_est, weight=1.0, tag="tactile")]
    if method == "mean_fusion_mpc":
        w_t = tactile_conf / (tactile_conf + 0.70)
        state = vision * (1.0 - w_t) + tactile * w_t
        return [dict(state=state, friction=0.5 * (0.55 + friction_est), weight=1.0, tag="mean")]
    if method == "ensemble_uncertainty_mpc":
        mean = 0.55 * vision + 0.45 * tactile
        return [
            dict(state=mean, friction=0.40, weight=0.24, tag="low"),
            dict(state=mean, friction=0.75, weight=0.36, tag="nominal"),
            dict(state=tactile, friction=friction_est, weight=0.40, tag="tactile_friction"),
        ]
    if method == "robust_minimax_mpc":
        mean = 0.62 * vision + 0.38 * tactile
        return [
            dict(state=vision, friction=0.42, weight=0.20, tag="robust_vision_low"),
            dict(state=mean, friction=0.78, weight=0.34, tag="robust_mean"),
            dict(state=tactile, friction=1.35, weight=0.28, tag="robust_tactile_high"),
            dict(state=mean + unit(target - mean) * 0.020, friction=1.65, weight=0.18, tag="robust_tail"),
        ]
    if method == "particle_belief_mpc":
        mean = 0.50 * vision + 0.50 * tactile
        return [
            dict(state=vision, friction=0.45, weight=0.18, tag="particle_v"),
            dict(state=tactile, friction=friction_est, weight=0.30, tag="particle_t"),
            dict(state=mean, friction=0.55, weight=0.22, tag="particle_mean"),
            dict(state=mean + unit(tactile - vision) * min(0.030, disagreement), friction=1.15, weight=0.18, tag="particle_shift"),
            dict(state=mean - unit(tactile - vision) * min(0.024, disagreement), friction=0.32, weight=0.12, tag="particle_reverse"),
        ]
    if method == "conformal_risk_filter":
        if disagreement > 0.038 and obs["contact_seen"]:
            state = 0.25 * vision + 0.75 * tactile
        else:
            state = 0.60 * vision + 0.40 * tactile
        return [dict(state=state, friction=friction_est, weight=1.0, tag="conformal")]
    if method in {"diagnostic_probe_then_mpc", "oracle_mode_mpc"}:
        state = tactile if obs["contact_seen"] else vision
        return [dict(state=state, friction=friction_est, weight=1.0, tag="diagnostic")]

    if method.startswith("ablation:"):
        ablation = method.split(":", 1)[1]
        if ablation in {"no_branch_preservation", "mean_only_fusion"}:
            state = 0.50 * vision + 0.50 * tactile
            return [dict(state=state, friction=friction_est, weight=1.0, tag="collapsed")]

    if is_cvtb_method(method):
        ablation = ablation_name(method)
        vision_rel, tactile_rel, sticky_signal = sensor_reliability(obs, method)
        if ablation == "small_branch_set":
            state = (vision * vision_rel + tactile * tactile_rel) / max(1e-6, vision_rel + tactile_rel)
            return [dict(state=state, friction=friction_est, weight=1.0, tag="small_cvtb")]
        if ablation == "no_branch_preservation":
            state = (vision * vision_rel + tactile * tactile_rel) / max(1e-6, vision_rel + tactile_rel)
            return [dict(state=state, friction=friction_est, weight=1.0, tag="no_branch")]

        weights = [
            max(0.05, vision_rel * (1.0 - 0.45 * sticky_signal)),
            max(0.05, tactile_rel * (1.0 - 0.35 * float(obs.get("contact_dropout_rate", 0.0)))),
            max(0.04, 0.22 + 0.62 * sticky_signal),
            max(0.03, 0.18 + 2.6 * max(0.0, disagreement - 0.045)),
        ]
        branches = [
            dict(state=vision, friction=0.55, weight=weights[0], tag="vision_health"),
            dict(state=tactile, friction=friction_est, weight=weights[1], tag="tactile_health"),
            dict(state=tactile, friction=max(1.05, friction_est + 0.25), weight=weights[2], tag="sticky_physical"),
            dict(state=vision, friction=0.48, weight=weights[3], tag="tactile_fault"),
        ]
        if ablation == "no_sensor_health":
            branches = [
                dict(state=vision, friction=0.55, weight=0.34, tag="vision_nohealth"),
                dict(state=tactile, friction=friction_est, weight=0.44, tag="tactile_nohealth"),
                dict(state=tactile, friction=max(1.05, friction_est), weight=0.22, tag="force_nohealth"),
            ]
        s = sum(b["weight"] for b in branches)
        for branch in branches:
            branch["weight"] /= s
            branch["target_dir"] = unit(target - branch["state"])
        return branches

    # Proposed branch mechanism and most ablations keep state alternatives explicit.
    branches = [
        dict(state=vision, friction=0.55, weight=visual_weight / total, tag="vision_branch"),
        dict(state=tactile, friction=friction_est, weight=tactile_weight / total, tag="tactile_branch"),
    ]
    if method != "ablation:small_branch_set":
        high_force_weight = clamp(obs["probe_force"] / 900.0, 0.05, 0.28)
        branches.append(dict(state=tactile, friction=max(1.05, friction_est), weight=high_force_weight, tag="high_force_branch"))
    s = sum(b["weight"] for b in branches)
    for branch in branches:
        branch["weight"] /= s
    for branch in branches:
        branch["target_dir"] = unit(target - branch["state"])
    return branches


def branch_entropy(branches: list[dict]) -> float:
    return float(-sum(b["weight"] * math.log(max(b["weight"], 1e-9)) for b in branches))


def candidate_score(candidate: dict, branches: list[dict], obs: dict, method: str) -> float:
    target = obs["target"]
    risk_weight = 0.035
    if method in {"conformal_risk_filter", "ensemble_uncertainty_mpc", "old_vt_disagreement_branch_mpc", "robust_minimax_mpc", "particle_belief_mpc"} or is_cvtb_method(method):
        risk_weight = 0.055
    ablation = ablation_name(method)
    if ablation == "no_contact_safety":
        risk_weight = 0.0

    losses = []
    unweighted_losses = []
    for branch in branches:
        miss = max(0.0, float(np.linalg.norm(candidate["anchor"] - branch["state"])) - 0.052)
        expected_move = candidate["move"] * (0.92 / (0.72 + 0.34 * branch["friction"]))
        expected_move = clamp(expected_move, 0.025, 0.34)
        pred = branch["state"] + candidate["direction"] * expected_move
        error = float(np.linalg.norm(pred - target))
        force_risk = branch["friction"] * candidate["move"] + 4.0 * miss
        raw_loss = error + 2.2 * miss + risk_weight * force_risk
        unweighted_losses.append(raw_loss)
        losses.append(branch["weight"] * raw_loss)

    variance_penalty = 0.0
    if method in {"ensemble_uncertainty_mpc", "old_vt_disagreement_branch_mpc", "particle_belief_mpc"} or is_cvtb_method(method):
        errs = []
        for branch in branches:
            expected_move = candidate["move"] * (0.92 / (0.72 + 0.34 * branch["friction"]))
            pred = branch["state"] + candidate["direction"] * expected_move
            errs.append(float(np.linalg.norm(pred - target)))
        variance_penalty = 0.22 * float(np.std(errs))
    if method == "robust_minimax_mpc":
        return float(max(unweighted_losses) + 0.10 * np.mean(unweighted_losses) + 0.030 * candidate["move"])
    if is_cvtb_method(method):
        ordered = sorted(unweighted_losses, reverse=True)
        k = max(1, int(math.ceil(0.40 * len(ordered))))
        cvar = float(np.mean(ordered[:k]))
        cvar_weight = 0.0 if ablation == "no_cvar_tail" else 0.28
        safety = 0.0 if ablation == "no_contact_safety" else 0.040 * candidate["move"] * (1.0 + float(obs.get("probe_force", 0.0)) / 500.0)
        fallback_penalty = 0.0
        if ablation != "no_reliability_fallback":
            vision_rel, tactile_rel, _ = sensor_reliability(obs, method)
            low_reliability = max(0.0, 0.42 - max(vision_rel, tactile_rel))
            fallback_penalty = 0.12 * low_reliability * candidate["move"]
        return float(sum(losses) + variance_penalty + cvar_weight * cvar + safety + fallback_penalty)
    if method == "conformal_risk_filter" and obs["disagreement"] > 0.045:
        variance_penalty += 0.10 * candidate["move"]
    return float(sum(losses) + variance_penalty)


def choose_action(method: str, obs: dict, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, dict]:
    if method == "random_push":
        anchor = obs["vision_obs"] + rng.normal(0.0, 0.025, size=2)
        direction = unit(obs["target"] - anchor)
        direction = unit(rotate(direction, rng.uniform(-0.9, 0.9)))
        move = rng.uniform(0.12, 0.30)
        start = anchor - direction * CONTACT_GAP
        end = anchor + direction * move - direction * CONTACT_GAP
        return start, end, {"anchor": anchor, "branch_entropy": 0.0, "move": move, "direction": direction, "note": "random"}

    branches = make_branches(method, obs)
    entropy = branch_entropy(branches)
    if method == "oracle_mode_mpc":
        anchor_options = [obs["actual_pos"]]
        friction_options = [obs["true_friction"]]
    elif method in {"old_vt_disagreement_branch_mpc", "robust_minimax_mpc", "particle_belief_mpc"} or is_cvtb_method(method) or method.startswith("ablation:"):
        anchor_options = [b["state"] for b in branches]
        if is_cvtb_method(method):
            mean_anchor = sum(b["weight"] * b["state"] for b in branches)
            anchor_options.append(mean_anchor)
        if method == "ablation:no_value_of_information" or method == "ablation:mean_only_fusion":
            anchor_options = [sum(b["weight"] * b["state"] for b in branches)]
        friction_options = [sum(b["weight"] * b["friction"] for b in branches)]
    else:
        anchor_options = [sum(b["weight"] * b["state"] for b in branches)]
        friction_options = [sum(b["weight"] * b["friction"] for b in branches)]

    candidates = []
    offsets = [-0.55, -0.28, 0.0, 0.28, 0.55]
    scales = [0.80, 1.00, 1.20]
    for anchor in anchor_options:
        base_dir = unit(obs["target"] - anchor)
        base_dist = float(np.linalg.norm(obs["target"] - anchor))
        for offset in offsets:
            direction = unit(rotate(base_dir, offset))
            for scale in scales:
                friction_mean = float(np.mean(friction_options))
                move = clamp(base_dist * scale * (0.95 + 0.10 * friction_mean), 0.065, 0.34)
                candidates.append({"anchor": anchor, "direction": direction, "move": move})

    scored = [(candidate_score(c, branches, obs, method), c) for c in candidates]
    scored.sort(key=lambda item: item[0])
    chosen = scored[0][1]
    if method == "conformal_risk_filter" and obs["disagreement"] > 0.050:
        chosen["move"] *= 0.86
    if method == "cvtb_mpc_v5" or method == "cvtb_no_probe" or is_cvtb_method(method):
        ablation = ablation_name(method)
        if ablation != "no_reliability_fallback":
            vision_rel, tactile_rel, _ = sensor_reliability(obs, method)
            sensor_fault_risk = (
                tactile_rel < 0.34
                or obs.get("contact_dropout_rate", 0.0) > 0.25
                or (obs["disagreement"] > 0.052 and tactile_rel < vision_rel + 0.08)
            )
            if sensor_fault_risk:
                mean_anchor = 0.64 * obs["vision_obs"] + 0.36 * obs["tactile_est"]
                direction = unit(obs["target"] - mean_anchor)
                chosen = {"anchor": mean_anchor, "direction": direction, "move": 0.88 * chosen["move"], "fallback": 1.0}
    if method.startswith("ablation:") and method.endswith("no_value_of_information"):
        chosen["move"] *= 0.94

    start = chosen["anchor"] - chosen["direction"] * CONTACT_GAP
    end = chosen["anchor"] + chosen["direction"] * chosen["move"] - chosen["direction"] * CONTACT_GAP
    chosen = dict(chosen)
    chosen["branch_entropy"] = entropy
    chosen["score"] = scored[0][0]
    chosen["note"] = "+".join(b["tag"] for b in branches)
    chosen["fallback"] = float(chosen.get("fallback", 0.0))
    return start, end, chosen


def probe_value_signal(obs: dict, method: str) -> float:
    branches = make_branches(method, obs)
    if len(branches) <= 1:
        return 0.0
    anchors = np.asarray([b["state"] for b in branches], dtype=float)
    weights = np.asarray([b["weight"] for b in branches], dtype=float)
    mean_anchor = np.sum(anchors * weights[:, None], axis=0)
    branch_spread = float(np.sum(weights * np.linalg.norm(anchors - mean_anchor, axis=1)))
    entropy = branch_entropy(branches)
    return float(branch_spread + 0.035 * entropy + 0.55 * max(0.0, obs["disagreement"] - 0.030))


def should_run_extra_probe(method: str, obs: dict) -> bool:
    if method == "diagnostic_probe_then_mpc":
        return True
    if method == "old_vt_disagreement_branch_mpc":
        return obs["disagreement"] > 0.040 or obs["probe_force"] > 360.0
    if method == "cvtb_mpc_v5":
        _, tactile_rel, sticky_signal = sensor_reliability(obs, method)
        return probe_value_signal(obs, method) > 0.105 and tactile_rel > 0.38 and sticky_signal > 0.20 and obs["probe_force"] < 720.0
    if method == "cvtb_no_probe":
        return False
    if method.startswith("ablation:"):
        ablation = method.split(":", 1)[1]
        if ablation in {"cvtb_no_probe", "no_value_of_information", "mean_only_fusion"}:
            return False
        if ablation in {"cvtb_mpc_v5", "no_sensor_health", "no_cvar_tail", "no_contact_safety", "no_reliability_fallback", "tactile_trust_high", "small_branch_set"}:
            _, tactile_rel, sticky_signal = sensor_reliability(obs, method)
            return probe_value_signal(obs, method) > 0.105 and tactile_rel > 0.38 and sticky_signal > 0.20 and obs["probe_force"] < 720.0
        return obs["disagreement"] > 0.055
    return False


def run_single_episode(task: tuple) -> dict:
    method, split, seed, episode, stress_level = task
    execution_method = method
    if method.startswith("ablation:") and ablation_name(method) in {
        "old_vt_disagreement_branch_mpc",
        "ensemble_uncertainty_mpc",
        "robust_minimax_mpc",
        "oracle_mode_mpc",
    }:
        execution_method = ablation_name(method)
    stress_salt = 0 if stress_level is None else int(round(10000 * stress_level))
    env_salt = stable_int(split) * 17 + seed * 1009 + episode * 9176 + stress_salt * 31
    policy_salt = stable_int(method) * 37 + env_salt
    env_rng = np.random.default_rng(BASE_SEED + env_salt)
    policy_rng = np.random.default_rng(BASE_SEED + policy_salt)
    cfg = stress_config(stress_level, env_rng) if stress_level is not None else mode_config(split, env_rng)
    object_pos, target = make_episode(seed, episode, split, env_rng)
    vision_obs = object_pos + cfg["vision_bias_vec"] + env_rng.normal(0.0, cfg["vision_noise"], size=2)

    probe = common_probe(cfg, object_pos, vision_obs, target, env_rng)
    actual_pos = probe["actual_after_probe"]
    obs = {
        "vision_obs": vision_obs,
        "tactile_est": probe["tactile_est"],
        "tactile_confidence": probe["tactile_confidence"],
        "contact_seen": probe["contact_seen"],
        "probe_force": probe["probe_force"],
        "probe_displacement": probe["probe_displacement"],
        "friction_signal": probe["friction_signal"],
        "disagreement": probe["disagreement"],
        "target": target,
        "actual_pos": actual_pos,
        "true_friction": cfg["friction"],
        "tactile_noise": cfg["tactile_noise"],
        "vision_bias_norm": float(np.linalg.norm(cfg["vision_bias_vec"])),
        "contact_dropout_rate": cfg.get("contact_dropout", 0.0),
    }

    used_diagnostic = 0
    diagnostic_energy = 0.0
    diagnostic_contact = 0.0
    if should_run_extra_probe(execution_method, obs):
        anchor_for_probe = obs["tactile_est"] if obs["contact_seen"] else obs["vision_obs"]
        dprobe = diagnostic_probe(cfg, actual_pos, anchor_for_probe, target, policy_rng)
        actual_pos = dprobe["actual_after_probe"]
        obs["actual_pos"] = actual_pos
        if dprobe["contact_seen"]:
            obs["tactile_est"] = 0.35 * obs["tactile_est"] + 0.65 * dprobe["tactile_est"]
            obs["tactile_confidence"] = max(obs["tactile_confidence"], dprobe["tactile_confidence"])
            obs["contact_seen"] = 1
        obs["probe_force"] = 0.5 * obs["probe_force"] + 0.5 * dprobe["probe_force"]
        obs["disagreement"] = float(np.linalg.norm(obs["vision_obs"] - obs["tactile_est"]))
        obs["friction_signal"] = clamp((obs["probe_force"] / 260.0) - 2.0 * dprobe["probe_displacement"], 0.2, 2.4)
        diagnostic_energy = dprobe["probe_energy"]
        diagnostic_contact = dprobe["probe_max_contact"]
        used_diagnostic = 1

    start, end, action = choose_action(execution_method, obs, policy_rng)
    rollout = rollout_push(cfg, actual_pos, start, end, 82)
    final_error = float(np.linalg.norm(rollout.final_pos - target))
    total_energy = probe["probe_energy"] + diagnostic_energy + rollout.pusher_path + 0.0008 * rollout.contact_impulse
    max_contact = max(probe["probe_max_contact"], diagnostic_contact, rollout.max_contact_force)
    success = int(final_error <= SUCCESS_RADIUS)
    contact_violation = int(max_contact > CONTACT_LIMIT)
    return {
        "method": method.replace("ablation:", ""),
        "split": split if stress_level is None else f"stress_{stress_level:.2f}",
        "seed": seed,
        "episode": episode,
        "success": success,
        "final_error": f"{final_error:.5f}",
        "energy": f"{total_energy:.5f}",
        "max_contact_force": f"{max_contact:.5f}",
        "contact_violation": contact_violation,
        "probe_contact": probe["contact_seen"],
        "used_diagnostic": used_diagnostic,
        "disagreement": f"{obs['disagreement']:.5f}",
        "branch_entropy": f"{action['branch_entropy']:.5f}",
        "probe_value": f"{probe_value_signal(obs, execution_method):.5f}",
        "fallback_used": f"{float(action.get('fallback', 0.0)):.1f}",
        "action_move": f"{float(action['move']):.5f}",
        "action_dir_x": f"{float(action['direction'][0]):.5f}",
        "action_dir_y": f"{float(action['direction'][1]):.5f}",
        "action_anchor_x": f"{float(action['anchor'][0]):.5f}",
        "action_anchor_y": f"{float(action['anchor'][1]):.5f}",
        "object_path": f"{rollout.object_path:.5f}",
        "pusher_path": f"{rollout.pusher_path:.5f}",
        "true_friction": f"{cfg['friction']:.4f}",
        "vision_bias_norm": f"{float(np.linalg.norm(cfg['vision_bias_vec'])):.5f}",
        "tactile_noise": f"{cfg['tactile_noise']:.5f}",
        "action_note": action["note"],
    }


def run_tasks(tasks: list[tuple]) -> list[dict]:
    if MAX_WORKERS == 1:
        return [run_single_episode(task) for task in tasks]
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        return list(executor.map(run_single_episode, tasks, chunksize=4))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    tmp = path.with_suffix(".partial.csv")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def summarize(rows: list[dict], group_keys: list[str]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = {}
    for row in rows:
        key = tuple(row[k] for k in group_keys)
        grouped.setdefault(key, []).append(row)
    output = []
    for key, group in sorted(grouped.items()):
        success_vals = [float(r["success"]) for r in group]
        error_vals = [float(r["final_error"]) for r in group]
        energy_vals = [float(r["energy"]) for r in group]
        violation_vals = [float(r["contact_violation"]) for r in group]
        disagreement_vals = [float(r["disagreement"]) for r in group]
        entropy_vals = [float(r.get("branch_entropy", 0.0)) for r in group]
        diagnostic_vals = [float(r.get("used_diagnostic", 0.0)) for r in group]
        probe_value_vals = [float(r.get("probe_value", 0.0)) for r in group]
        fallback_vals = [float(r.get("fallback_used", 0.0)) for r in group]
        out = {k: v for k, v in zip(group_keys, key)}
        out.update(
            {
                "mean_success": f"{float(np.mean(success_vals)):.4f}",
                "ci95_success": f"{ci95(success_vals):.4f}",
                "mean_final_error": f"{float(np.mean(error_vals)):.4f}",
                "ci95_final_error": f"{ci95(error_vals):.4f}",
                "mean_energy": f"{float(np.mean(energy_vals)):.4f}",
                "ci95_energy": f"{ci95(energy_vals):.4f}",
                "mean_contact_violation": f"{float(np.mean(violation_vals)):.4f}",
                "mean_disagreement": f"{float(np.mean(disagreement_vals)):.4f}",
                "mean_branch_entropy": f"{float(np.mean(entropy_vals)):.4f}",
                "diagnostic_rate": f"{float(np.mean(diagnostic_vals)):.4f}",
                "mean_probe_value": f"{float(np.mean(probe_value_vals)):.4f}",
                "fallback_rate": f"{float(np.mean(fallback_vals)):.4f}",
                "episodes": len(group),
                "seeds": len({r["seed"] for r in group}),
            }
        )
        output.append(out)
    return output


def seed_metrics(rows: list[dict]) -> list[dict]:
    return summarize(rows, ["method", "split", "seed"])


def paired_ci(values: list[float]) -> float:
    return ci95(values)


def pairwise_stats(raw_rows: list[dict], proposed: str = "cvtb_mpc_v5") -> list[dict]:
    rows = []
    by_case: dict[tuple, dict[str, dict]] = {}
    for row in raw_rows:
        by_case.setdefault((row["split"], row["seed"], row["episode"]), {})[row["method"]] = row
    splits = sorted({row["split"] for row in raw_rows})
    for split in splits:
        cases = [case for key, case in by_case.items() if key[0] == split and proposed in case]
        for method in METHODS:
            if method == proposed:
                continue
            paired = [(case[proposed], case[method]) for case in cases if method in case]
            if not paired:
                continue
            succ = [float(p["success"]) - float(b["success"]) for p, b in paired]
            err = [float(b["final_error"]) - float(p["final_error"]) for p, b in paired]
            energy = [float(b["energy"]) - float(p["energy"]) for p, b in paired]
            contact = [float(p["contact_violation"]) - float(b["contact_violation"]) for p, b in paired]
            action_diff = []
            for p, b in paired:
                p_dir = np.array([float(p["action_dir_x"]), float(p["action_dir_y"])])
                b_dir = np.array([float(b["action_dir_x"]), float(b["action_dir_y"])])
                dir_delta = float(np.linalg.norm(p_dir - b_dir))
                move_delta = abs(float(p["action_move"]) - float(b["action_move"]))
                anchor_delta = math.hypot(float(p["action_anchor_x"]) - float(b["action_anchor_x"]), float(p["action_anchor_y"]) - float(b["action_anchor_y"]))
                action_diff.append(float(dir_delta > 0.10 or move_delta > 0.020 or anchor_delta > 0.020 or p["used_diagnostic"] != b["used_diagnostic"]))
            mean_diff = float(np.mean(succ))
            sd = float(np.std(succ, ddof=1)) if len(succ) > 1 else 0.0
            t_stat = mean_diff / (sd / math.sqrt(len(succ)) + 1e-9)
            rows.append(
                {
                    "split": split,
                    "baseline": method,
                    "paired_episodes": len(paired),
                    "success_delta_mean": f"{mean_diff:.4f}",
                    "success_delta_ci95": f"{paired_ci(succ):.4f}",
                    "final_error_improvement_mean": f"{float(np.mean(err)):.4f}",
                    "final_error_improvement_ci95": f"{paired_ci(err):.4f}",
                    "energy_improvement_mean": f"{float(np.mean(energy)):.4f}",
                    "energy_improvement_ci95": f"{paired_ci(energy):.4f}",
                    "contact_violation_delta_mean": f"{float(np.mean(contact)):.4f}",
                    "action_diff_rate": f"{float(np.mean(action_diff)):.4f}",
                    "paired_t_approx": f"{t_stat:.4f}",
                    "normal_approx_p": f"{normal_p_from_t(t_stat):.4f}",
                }
            )
    return rows


def plot_success(metrics: list[dict], path: Path) -> None:
    selected = ["mean_fusion_mpc", "ensemble_uncertainty_mpc", "robust_minimax_mpc", "particle_belief_mpc", "old_vt_disagreement_branch_mpc", "cvtb_mpc_v5", "oracle_mode_mpc"]
    splits = MAIN_SPLITS
    x = np.arange(len(splits))
    width = 0.10
    fig, ax = plt.subplots(figsize=(12, 5))
    for idx, method in enumerate(selected):
        vals = []
        for split in splits:
            match = [r for r in metrics if r["method"] == method and r["split"] == split]
            vals.append(float(match[0]["mean_success"]) if match else 0.0)
        ax.bar(x + (idx - len(selected) / 2) * width + width / 2, vals, width, label=method.replace("_", " "))
    ax.set_ylabel("Success rate")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", "\n") for s in splits], fontsize=8)
    ax.legend(fontsize=7, ncol=2)
    ax.set_title("Closed-loop MuJoCo manipulation success by hidden visuotactile shift")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_energy(metrics: list[dict], path: Path) -> None:
    selected = ["mean_fusion_mpc", "ensemble_uncertainty_mpc", "robust_minimax_mpc", "old_vt_disagreement_branch_mpc", "cvtb_mpc_v5", "oracle_mode_mpc"]
    splits = MAIN_SPLITS
    x = np.arange(len(splits))
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for method in selected:
        vals = []
        for split in splits:
            match = [r for r in metrics if r["method"] == method and r["split"] == split]
            vals.append(float(match[0]["mean_energy"]) if match else 0.0)
        ax.plot(x, vals, marker="o", label=method.replace("_", " "))
    ax.set_ylabel("Energy proxy")
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", "\n") for s in splits], fontsize=8)
    ax.legend(fontsize=8)
    ax.set_title("Energy/contact effort cost")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_ablation(metrics: list[dict], path: Path) -> None:
    vals = [(r["method"], float(r["mean_success"]), float(r["ci95_success"])) for r in metrics if r["split"] == "combined_shift"]
    vals.sort(key=lambda item: item[1], reverse=True)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(vals))
    ax.bar(x, [v[1] for v in vals], yerr=[v[2] for v in vals], color="#2f6f73")
    ax.set_xticks(x)
    ax.set_xticklabels([v[0].replace("_", "\n") for v in vals], fontsize=8)
    ax.set_ylabel("Combined-shift success")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Ablation test: does disagreement branching matter?")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_stress(stress_metrics: list[dict], path: Path) -> None:
    selected = ["mean_fusion_mpc", "ensemble_uncertainty_mpc", "robust_minimax_mpc", "old_vt_disagreement_branch_mpc", "cvtb_mpc_v5", "oracle_mode_mpc"]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for method in selected:
        xs, ys = [], []
        for row in stress_metrics:
            if row["method"] == method:
                xs.append(float(row["stress_level"]))
                ys.append(float(row["mean_success"]))
        order = np.argsort(xs)
        ax.plot(np.asarray(xs)[order], np.asarray(ys)[order], marker="o", label=method.replace("_", " "))
    ax.set_xlabel("Shift severity")
    ax.set_ylabel("Success rate")
    ax.set_ylim(0.0, 1.0)
    ax.legend(fontsize=8)
    ax.set_title("Stress sweep: vision bias + tactile noise + friction shift")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_negative_cases() -> list[dict]:
    return [
        {
            "case": "adversarial_vision_and_tactile_corruption",
            "expected_behavior": "branch planner should abstain or probe more",
            "observed_failure_mode": "if both modalities are corrupted consistently, disagreement can be low and all planners over-trust the false state",
            "submission_implication": "requires sensor-health model before ICLR-main deployment claims",
        },
        {
            "case": "unmodeled_actuator_weakness",
            "expected_behavior": "tactile residual should reveal weak pushes",
            "observed_failure_mode": "world-state branches do not represent actuation loss, so the planner compensates too late",
            "submission_implication": "needs actuator-mode branches or hardware checks",
        },
        {
            "case": "semantic_target_ambiguity",
            "expected_behavior": "physical probing should not solve language ambiguity",
            "observed_failure_mode": "all visuotactile methods can push the wrong object/target if the goal is ambiguous",
            "submission_implication": "scope must be physical-state uncertainty only",
        },
    ]


def terminal_decision(metrics: list[dict], ablation_metrics: list[dict]) -> tuple[str, str]:
    proposed_name = "cvtb_mpc_v5"
    by_split_method = {(r["split"], r["method"]): r for r in metrics}
    aggregates: dict[str, list[dict]] = {}
    for row in metrics:
        aggregates.setdefault(row["method"], []).append(row)
    aggregate_success = {m: float(np.mean([float(r["mean_success"]) for r in rows])) for m, rows in aggregates.items()}
    aggregate_error = {m: float(np.mean([float(r["mean_final_error"]) for r in rows])) for m, rows in aggregates.items()}
    proposed_success = aggregate_success.get(proposed_name, 0.0)
    proposed_error = aggregate_error.get(proposed_name, 9.0)
    strong = ["mean_fusion_mpc", "ensemble_uncertainty_mpc", "conformal_risk_filter", "diagnostic_probe_then_mpc", "robust_minimax_mpc", "particle_belief_mpc"]
    strong_failures = [
        method
        for method in strong
        if aggregate_success.get(method, -1.0) > proposed_success + 1e-9 or aggregate_error.get(method, 9.0) < proposed_error - 1e-9
    ]
    hostile = ["combined_shift", "sensor_conflict", "contact_dropout", "delayed_touch_sticky"]
    hostile_failures = []
    for split in hostile:
        p = by_split_method.get((split, proposed_name))
        if not p:
            continue
        for method in strong:
            b = by_split_method.get((split, method))
            if b and float(b["mean_success"]) > float(p["mean_success"]) + 1e-9:
                hostile_failures.append(f"{method}@{split}")
                break
    ab_by_key = {(r["split"], r["method"]): r for r in ablation_metrics}
    mechanism_failures = []
    for split in sorted({r["split"] for r in ablation_metrics}):
        p = ab_by_key.get((split, proposed_name))
        if not p:
            continue
        for method in ["no_sensor_health", "no_value_of_information", "no_cvar_tail", "no_reliability_fallback", "mean_only_fusion", "old_vt_disagreement_branch_mpc"]:
            b = ab_by_key.get((split, method))
            if b and float(b["mean_success"]) >= float(p["mean_success"]) - 1e-9:
                mechanism_failures.append(f"{method}@{split}")
    if strong_failures or hostile_failures or mechanism_failures:
        reason = "strong baselines or ablations match/beat CVTB-MPC: "
        reason += "; ".join((strong_failures + hostile_failures + mechanism_failures)[:8])
        return "KILL_ARCHIVE", reason
    return "STRONG_REVISE", "CVTB-MPC clears local frozen gates but still lacks real robot/public benchmark validation"


def run(args: argparse.Namespace) -> None:
    global RESULTS, FIGURES, MAX_WORKERS
    RESULTS = Path(args.results_dir)
    FIGURES = Path(args.figures_dir)
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    MAX_WORKERS = args.workers
    seeds = list(range(args.seeds))

    raw_rows: list[dict] = []
    for split in args.splits:
        tasks = [
            (method, split, seed, episode, None)
            for method in METHODS
            for seed in seeds
            for episode in range(args.episodes)
        ]
        raw_rows.extend(run_tasks(tasks))
        write_csv(RESULTS / "vt_disagreement_raw.partial.csv", raw_rows)
        write_csv(RESULTS / "vt_disagreement_metrics.partial.csv", summarize(raw_rows, ["method", "split"]))
        print(f"completed main split={split} rows={len(raw_rows)}", flush=True)

    seed_rows = seed_metrics(raw_rows)
    metrics = summarize(raw_rows, ["method", "split"])
    pairwise = pairwise_stats(raw_rows)
    write_csv(RESULTS / "vt_disagreement_raw.csv", raw_rows)
    write_csv(RESULTS / "raw_seed_metrics.csv", seed_rows)
    write_csv(RESULTS / "vt_disagreement_metrics.csv", metrics)
    write_csv(RESULTS / "metrics.csv", metrics)
    write_csv(RESULTS / "vt_disagreement_pairwise.csv", pairwise)
    write_csv(RESULTS / "pairwise_stats.csv", pairwise)

    ablation_rows: list[dict] = []
    for split in args.ablation_splits:
        tasks = [
            (f"ablation:{ablation}", split, seed, episode, None)
            for ablation in ABLATIONS
            for seed in seeds
            for episode in range(args.ablation_episodes)
        ]
        ablation_rows.extend(run_tasks(tasks))
        write_csv(RESULTS / "vt_disagreement_ablation_raw.partial.csv", ablation_rows)
        write_csv(RESULTS / "vt_disagreement_ablation.partial.csv", summarize(ablation_rows, ["method", "split"]))
        print(f"completed ablation split={split} rows={len(ablation_rows)}", flush=True)
    ablation_metrics = summarize(ablation_rows, ["method", "split"])
    write_csv(RESULTS / "vt_disagreement_ablation_raw.csv", ablation_rows)
    write_csv(RESULTS / "vt_disagreement_ablation.csv", ablation_metrics)
    write_csv(RESULTS / "ablation_metrics.csv", ablation_metrics)

    stress_rows: list[dict] = []
    if not args.skip_stress:
        stress_tasks = [
            (method, "stress_sweep", seed, episode, level)
            for method in args.stress_methods
            for level in args.stress_levels
            for seed in seeds
            for episode in range(args.stress_episodes)
        ]
        stress_rows = run_tasks(stress_tasks)
    stress_metrics = summarize(stress_rows, ["method", "split"]) if stress_rows else []
    stress_output = []
    for row in stress_metrics:
        out = dict(row)
        out["stress_level"] = out["split"].replace("stress_", "")
        stress_output.append(out)
    write_csv(RESULTS / "stress_sweep.csv", stress_output)
    write_csv(FIGURES / "stress_curve_data.csv", stress_output)

    negative_rows = make_negative_cases()
    write_csv(RESULTS / "negative_cases.csv", negative_rows)

    plot_success(metrics, FIGURES / "vt_disagreement_success_by_split.png")
    plot_energy(metrics, FIGURES / "vt_disagreement_energy_by_split.png")
    plot_ablation(ablation_metrics, FIGURES / "vt_disagreement_ablation_success.png")
    if stress_output:
        plot_stress(stress_output, FIGURES / "vt_disagreement_stress_sweep.png")

    decision, reason = terminal_decision(metrics, ablation_metrics)
    combined = {r["method"]: r for r in metrics if r["split"] == "combined_shift"}
    ablation_combined = {r["method"]: r for r in ablation_metrics if r["split"] == "combined_shift"}
    with (RESULTS / "summary.txt").open("w", encoding="utf-8") as handle:
        handle.write("Paper 66 real MuJoCo calibrated visuotactile branch-and-probe rebuild, v5\n")
        handle.write(f"Seeds: {seeds}; episodes per seed: {args.episodes}; workers: {MAX_WORKERS}\n")
        handle.write("Main rows: %d; ablation rows: %d; stress rows: %d\n" % (len(raw_rows), len(ablation_rows), len(stress_rows)))
        handle.write(f"Terminal decision: {decision}\n")
        handle.write(f"Terminal reason: {reason}\n")
        handle.write("\nCombined-shift main results:\n")
        for method in METHODS:
            if method in combined:
                row = combined[method]
                handle.write(
                    f"- {method}: success={row['mean_success']} ci95={row['ci95_success']} "
                    f"error={row['mean_final_error']} energy={row['mean_energy']} "
                    f"violation={row['mean_contact_violation']} diagnostic={row['diagnostic_rate']}\n"
                )
        handle.write("\nCombined-shift ablations:\n")
        for method, row in sorted(ablation_combined.items()):
            handle.write(f"- {method}: success={row['mean_success']} ci95={row['ci95_success']} energy={row['mean_energy']}\n")
        handle.write("\nPairwise comparisons vs cvtb_mpc_v5:\n")
        for row in pairwise:
            if row["split"] in {"combined_shift", "sensor_conflict", "contact_dropout", "delayed_touch_sticky"}:
                handle.write(
                    f"- {row['split']} vs {row['baseline']}: success_delta={row['success_delta_mean']} "
                    f"action_diff={row['action_diff_rate']} p={row['normal_approx_p']}\n"
                )

    print(f"wrote Paper 66 MuJoCo evidence to {RESULTS}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEED_COUNT)
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES_PER_SEED)
    parser.add_argument("--ablation-episodes", type=int, default=DEFAULT_ABLATION_EPISODES_PER_SEED)
    parser.add_argument("--stress-episodes", type=int, default=DEFAULT_STRESS_EPISODES_PER_SEED)
    parser.add_argument("--splits", nargs="+", default=MAIN_SPLITS)
    parser.add_argument("--ablation-splits", nargs="+", default=["combined_shift", "sensor_conflict", "delayed_touch_sticky"])
    parser.add_argument("--stress-levels", nargs="+", type=float, default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    parser.add_argument(
        "--stress-methods",
        nargs="+",
        default=["mean_fusion_mpc", "ensemble_uncertainty_mpc", "robust_minimax_mpc", "old_vt_disagreement_branch_mpc", "cvtb_mpc_v5", "oracle_mode_mpc"],
    )
    parser.add_argument("--skip-stress", action="store_true")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--results-dir", default=str(RESULTS))
    parser.add_argument("--figures-dir", default=str(FIGURES))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
