"""Constantes et utilitaires partages entre levee.py, pivot_real.py et
levee_pivot.py -- avant ce fichier, les indices de jambe/genou, les angles de
posture de marche et les fonctions _checkpoint/_quintic_ease/_bend_knees/
_straighten_knees existaient en copie IDENTIQUE dans levee.py ET dans
pivot_real.py (copier-coller). Ce module est l'unique source pour tout ce qui
est partage ; ce qui reste local a un fichier (les durees de sequence par
exemple) reste dans ce fichier."""
import time

import numpy as np

RATE_HZ = 100

MOTION_STATE_TIMEOUT = 3.0

# --- Bras -------------------------------------------------------------------
LEFT_JOINT_INDICES = [13, 14, 15, 16, 17]
RIGHT_JOINT_INDICES = [18, 19, 20, 21, 22]
WRIST_CHAIN_INDEX = 4

Q_LEFT_HOME = np.array([0.000879, 0.075284, -0.000233, -0.126397, -0.000033])
Q_RIGHT_HOME = np.array([0.000885, -0.075161, 0.000241, -0.126390, 0.000033])

WAYPOINT_Q_LEFT = np.radians([30.0, 5.0, 0.0, -110.0, 0.0])
WAYPOINT_Q_RIGHT = np.radians([30.0, -5.0, 0.0, -110.0, 0.0])

# --- Buste (pivot) ------------------------------------------------------------
WAIST_JOINT_INDEX = 12
WAIST_KP = 150.0
WAIST_KD = 3.0

# --- Jambes / posture de marche (flexion genoux avant prise) -----------------
LEFT_HIP_PITCH_INDEX = 0
RIGHT_HIP_PITCH_INDEX = 6
LEFT_KNEE_PITCH_INDEX = 3
RIGHT_KNEE_PITCH_INDEX = 9
LEFT_ANKLE_PITCH_INDEX = 4
RIGHT_ANKLE_PITCH_INDEX = 10
WALK_STANCE_HIP_PITCH_L = np.radians(-6.9)
WALK_STANCE_HIP_PITCH_R = np.radians(-5.2)
WALK_STANCE_KNEE_L = np.radians(11.7)
WALK_STANCE_KNEE_R = np.radians(10.5)
WALK_STANCE_ANKLE_PITCH_L = np.radians(-4.8)
WALK_STANCE_ANKLE_PITCH_R = np.radians(-5.2)
WALK_STANCE_STIFFNESS_SCALE = 1.8


def _checkpoint(message, confirm):
    print(f"[ETAPE] {message}", flush=True)
    if confirm:
        input("        Verifie le robot, puis Entree pour continuer (Ctrl+C pour arreter)... ")


def _quintic_ease(t):
    t = max(0.0, min(1.0, t))
    return t ** 3 * (10 - 15 * t + 6 * t ** 2)


def _bend_knees(lever, scale, stiffness_scale, duration, rate_hz=RATE_HZ, dry_run=False):
    if dry_run:
        print(f"    [dry-run] flexion genoux -- scale={scale} duration={duration}s")
        return
    for idx, kp, kd in [
        (LEFT_HIP_PITCH_INDEX, 200.0, 5.0), (RIGHT_HIP_PITCH_INDEX, 200.0, 5.0),
        (LEFT_KNEE_PITCH_INDEX, 450.0, 5.0), (RIGHT_KNEE_PITCH_INDEX, 450.0, 5.0),
        (LEFT_ANKLE_PITCH_INDEX, 400.0, 2.0), (RIGHT_ANKLE_PITCH_INDEX, 400.0, 2.0),
    ]:
        lever.set_gains(idx, kp * stiffness_scale, kd * stiffness_scale)
    n = max(1, int(duration * rate_hz))
    for i in range(n + 1):
        a = _quintic_ease(i / n)
        lever[LEFT_HIP_PITCH_INDEX] = float(a * scale * WALK_STANCE_HIP_PITCH_L)
        lever[RIGHT_HIP_PITCH_INDEX] = float(a * scale * WALK_STANCE_HIP_PITCH_R)
        lever[LEFT_KNEE_PITCH_INDEX] = float(a * scale * WALK_STANCE_KNEE_L)
        lever[RIGHT_KNEE_PITCH_INDEX] = float(a * scale * WALK_STANCE_KNEE_R)
        lever[LEFT_ANKLE_PITCH_INDEX] = float(a * scale * WALK_STANCE_ANKLE_PITCH_L)
        lever[RIGHT_ANKLE_PITCH_INDEX] = float(a * scale * WALK_STANCE_ANKLE_PITCH_R)
        time.sleep(1.0 / rate_hz)


def _straighten_knees(lever, scale, stiffness_scale, duration, rate_hz=RATE_HZ, dry_run=False):
    if dry_run:
        print(f"    [dry-run] redressement genoux -- scale={scale} duration={duration}s")
        return
    n = max(1, int(duration * rate_hz))
    for i in range(n + 1):
        a = 1.0 - _quintic_ease(i / n)
        lever[LEFT_HIP_PITCH_INDEX] = float(a * scale * WALK_STANCE_HIP_PITCH_L)
        lever[RIGHT_HIP_PITCH_INDEX] = float(a * scale * WALK_STANCE_HIP_PITCH_R)
        lever[LEFT_KNEE_PITCH_INDEX] = float(a * scale * WALK_STANCE_KNEE_L)
        lever[RIGHT_KNEE_PITCH_INDEX] = float(a * scale * WALK_STANCE_KNEE_R)
        lever[LEFT_ANKLE_PITCH_INDEX] = float(a * scale * WALK_STANCE_ANKLE_PITCH_L)
        lever[RIGHT_ANKLE_PITCH_INDEX] = float(a * scale * WALK_STANCE_ANKLE_PITCH_R)
        time.sleep(1.0 / rate_hz)
