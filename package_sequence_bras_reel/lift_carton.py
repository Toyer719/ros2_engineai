import numpy as np

LEFT_CHAIN = [
    ("J13_SHOULDER_PITCH_L", np.array([0, 1, 0]), np.array([-0.027105, 0.12916, 0.21549])),
    ("J14_SHOULDER_ROLL_L",  np.array([1, 0, 0]), np.array([-0.0371, 0.066941, -0.020838])),
    ("J15_SHOULDER_YAW_L",   np.array([0, 0, 1]), np.array([0.0371, 0.017645, -0.070132])),
    ("J16_ELBOW_PITCH_L",    np.array([0, 1, 0]), np.array([0, 0.0065994, -0.10487])),
    ("J17_ELBOW_YAW_L",      np.array([0, 0, 1]), np.array([0.013817, 0.0097723, -0.1547])),
]
HAND_OFFSET_LEFT = np.array([0.03, -0.02, -0.14])

RIGHT_CHAIN = [
    ("J18_SHOULDER_PITCH_R", np.array([0, 1, 0]), np.array([-0.027105, -0.12916, 0.21549])),
    ("J19_SHOULDER_ROLL_R",  np.array([1, 0, 0]), np.array([-0.0371, -0.066941, -0.020838])),
    ("J20_SHOULDER_YAW_R",   np.array([0, 0, 1]), np.array([0.0371, -0.017644, -0.070132])),
    ("J21_ELBOW_PITCH_R",    np.array([0, 1, 0]), np.array([0, -0.006598, -0.10487])),
    ("J22_ELBOW_YAW_R",      np.array([0, 0, 1]), np.array([0.013817, -0.0097704, -0.1547])),
]
HAND_OFFSET_RIGHT = np.array([0.03, 0.02, -0.14])


def rotation_matrix(axis, angle):
    axis = axis / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)


def forward_kinematics(chain, hand_offset, q):
    pos = np.zeros(3)
    rot = np.eye(3)
    for (_, axis, offset), angle in zip(chain, q):
        pos = pos + rot @ offset
        rot = rot @ rotation_matrix(axis, angle)
    return pos + rot @ hand_offset


def numerical_jacobian(chain, hand_offset, q, eps=1e-6):
    p0 = forward_kinematics(chain, hand_offset, q)
    J = np.zeros((3, len(q)))
    for i in range(len(q)):
        dq = q.copy()
        dq[i] += eps
        J[:, i] = (forward_kinematics(chain, hand_offset, dq) - p0) / eps
    return J


def solve_ik(chain, hand_offset, target, q_init, iters=150, damping=0.05):
    q = q_init.copy()
    for _ in range(iters):
        error = target - forward_kinematics(chain, hand_offset, q)
        if np.linalg.norm(error) < 1e-6:
            break
        J = numerical_jacobian(chain, hand_offset, q)
        JJt = J @ J.T + damping ** 2 * np.eye(3)
        q = q + J.T @ np.linalg.solve(JJt, error)
    return q


def solve_arm_ik(chain, hand_offset, target, q_init, lock_index=None, lock_angle=None,
                  iters=200, damping=0.05, null_space_gain=0.2, null_space_pref=None):
    if lock_index is None:
        return solve_ik(chain, hand_offset, target, q_init, iters=iters, damping=damping)

    free_idx = [i for i in range(len(q_init)) if i != lock_index]
    q_pref = q_init.copy() if null_space_pref is None else null_space_pref.copy()
    q = q_init.copy()
    q[lock_index] = lock_angle
    n_free = len(free_idx)
    for _ in range(iters):
        p0 = forward_kinematics(chain, hand_offset, q)
        error = target - p0
        if np.linalg.norm(error) < 1e-7:
            break
        J = np.zeros((3, n_free))
        for k, i in enumerate(free_idx):
            dq = q.copy()
            dq[i] += 1e-6
            J[:, k] = (forward_kinematics(chain, hand_offset, dq) - p0) / 1e-6
        JJt = J @ J.T + damping ** 2 * np.eye(3)
        J_pinv = J.T @ np.linalg.inv(JJt)
        step = J_pinv @ error
        if null_space_gain:
            q_free = np.array([q[i] for i in free_idx])
            q_pref_free = np.array([q_pref[i] for i in free_idx])
            null_proj = np.eye(n_free) - J_pinv @ J
            step = step + null_space_gain * (null_proj @ (q_pref_free - q_free))
        for k, i in enumerate(free_idx):
            q[i] += step[k]
        q[lock_index] = lock_angle
    return q


def mirror_left_to_right(q_left):
    q_right = q_left.copy()
    q_right[1] *= -1
    q_right[2] *= -1
    q_right[4] *= -1
    return q_right


def ease(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)
