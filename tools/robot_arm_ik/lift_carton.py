                      
"""
Fait pincer le carton par les deux mains du PM01 (friction reelle, pas d'attache
artificielle) puis le soulever, dans une variante DYNAMIQUE de la scene
pm01_edu_carton generee a la volee.

carton.xml (utilise par run_mujoco.sh / le dialogue Arm Control) reste un corps
statique -- on ne le touche jamais. Ce script patche une copie temporaire de
serial_pm01_edu.xml (7 valeurs de qpos ajoutees a la keyframe 'floating_base_homing'
pour le <freejoint/> de assets/resource/environment/carton_liftable.xml) et assemble
une scene temporaire avec ca -- rien n'est ecrit dans le depot.

Sequence (calibree empiriquement, voir session de dev) :
  1. Approche : bras (a partir de leur VRAIE pose de repos, pas zero -- un demarrage
     a zero fait un a-coup qui pousse le carton avant meme la prise) -> position de
     pince validee par l'utilisateur (X=0.38 Y=+-0.17 Z=-0.05, repere torse).
  2. Serrage : les cibles Y se rapprochent au-dela de la surface du carton
     (+-0.17 -> +-0.14) pour generer une vraie force normale/friction. Un serrage
     trop fort (teste jusqu'a +-0.08) ejecte le carton au lieu de le tenir.
  3. Levee : les cibles Z montent ensemble (IK resolu a chaque pas, warm-starte
     depuis la solution precedente pour un mouvement fluide).

Attendu : le carton monte avec un certain retard/glissement (friction, pas une
prise rigide) et derive d'environ 10cm vers l'avant pendant la prise -- c'est une
vraie simulation physique, pas un mouvement scripte parfait.

SIMULATION MUJOCO UNIQUEMENT -- aucune commande envoyee au robot reel.
"""

import argparse
import math
import os
import re
import sys
import tempfile
import threading
import time

import lcm
import numpy as np

try:
    import mujoco
    import mujoco.viewer
except ImportError:
                                                                                   
                                                                                       
    mujoco = None

# Deplace vers stagiaire_1 le 2026-08-12 -- ce script reste physiquement hors de
# engineai_robotics_native_sdk, mais tous les assets qu'il lit (robot original,
# environnement, carton) vivent toujours dans le SDK (ou y sont symlinkes depuis
# stagiaire_1/assets/) -- donc REPO_DIR pointe explicitement sur le SDK, pas sur
# le dossier de ce fichier.
REPO_DIR = "/home/equansrobotic/engineai_robotics_native_sdk"
ROBOT_XML = os.path.join(REPO_DIR, "assets/resource/robot/pm01_edu/xml/serial_pm01_edu.xml")
ROBOT_DIR = os.path.dirname(ROBOT_XML)
ENV_DIR = os.path.join(REPO_DIR, "assets/resource/environment")

CARTON_XY = (1.8, 0.0)

# 2026-08-11 : recalcule pour le nouveau carton, tailles completes 275x191x292mm
# (demi-hauteur reelle 0.146m, contre 0.15 avant -- quasi identique) repose
# sur le podium (surface a z=0.65) -- 0.65+0.146=0.796.
CARTON_Z = 0.796
# 2026-09-11 : CARTON_XY/CARTON_Z ci-dessus sont des constantes FIGEES, perimees
# (scene live actuelle : x=1.77, z=0.946 -- voir carton_face_centers() plus bas)
# -- gardees pour ne pas casser build_dynamic_scene() (outil de calibrage IK hors
# ligne, sa propre scene generee, sans rapport avec la scene live de run_mujoco.sh).
# NE PAS s'en servir pour un calcul touchant la scene VIVANTE, utiliser
# carton_face_centers() a la place (lit le XML live a chaque appel, jamais perime).

LIVE_SCENE_XML = "/home/equansrobotic/engineai_robotics_native_sdk/assets/resource/pm01_edu_carton.xml"

SQUEEZE_OFFSET_Y = 0.010


def carton_face_centers(scene_xml=LIVE_SCENE_XML):
    """Coordonnees X,Y,Z (metres, REPERE MONDE) du centre des faces GAUCHE et
    DROITE du carton, lues directement dans le XML de la scene VIVANTE (celle
    que charge run_mujoco.sh) -- jamais perime, contrairement a une constante
    recopiee a la main (cf. CARTON_XY/CARTON_Z ci-dessus, ou le piege deja
    rencontre avec shadow_scene.CARTON_POS, voir memoire projet).

    Repere MuJoCo de ce projet (verifiable directement dans le XML) :
      - <body pos="x y z"> = position MONDE (les bodies podium/carton sont des
        enfants DIRECTS de <worldbody>, aucune imbrication -- pas de transform
        supplementaire a appliquer).
      - Un geom type="box" a un size="dx dy dz" = DEMI-dimensions locales (pas
        la taille totale) le long de X,Y,Z -- confirme en comparant a la taille
        REELLE connue du carton (275x191x292mm) : X=0.1375*2=275mm (profondeur),
        Y=0.0955*2=191mm (largeur), Z=0.146*2=292mm (hauteur).
      - +X = vers l'avant (direction dans laquelle le robot marche, deja
        confirme par toute la telemetrie sim_state de ce projet).
      - +Z = vers le haut (deja confirme partout : ~0.82m debout, ~0.08-0.1m
        tombe au sol).
      - +Y = vers la GAUCHE du robot (meme convention que pinch_y partout
        ailleurs dans ce projet : bras gauche vise +pinch_y, bras droit
        vise -pinch_y).
    Le carton n'a aucune rotation dans la scene (pas de quat/euler dans son
    <body>) -- ses faces gauche/droite sont donc exactement a Y = cy +- sy,
    memes X/Z que le centre du carton.

    Retourne (face_gauche, face_droite), chacune un np.array([x, y, z]) en
    repere MONDE -- PAS encore le repere du bassin du robot (pinch_x/y/z) : il
    faudrait encore soustraire la position du bassin (sim_state.
    base_link_position) et faire tourner par son quaternion pour obtenir une
    cible IK utilisable telle quelle (le robot ne regarde pas forcement le
    carton exactement de face)."""
    src = open(scene_xml).read()
    m = re.search(r'<body name="carton" pos="([^"]+)"', src)
    if not m:
        raise RuntimeError(f"Body 'carton' introuvable dans {scene_xml}.")
    cx, cy, cz = (float(v) for v in m.group(1).split())
    m = re.search(r'<geom name="carton_box"[^>]*\bsize="([^"]+)"', src)
    if not m:
        raise RuntimeError(f"Geom 'carton_box' introuvable dans {scene_xml}.")
    sx, sy, sz = (float(v) for v in m.group(1).split())
    face_gauche = np.array([cx, cy + sy, cz])
    face_droite = np.array([cx, cy - sy, cz])
    return face_gauche, face_droite


SIM_LCM_URL = "udpm://239.255.76.67:7667?ttl=1"
SIM_STATE_CHANNEL = "sim_state"


def _yaw_from_quaternion(w, x, y, z):
    """Cap (rotation autour de Z) a partir du quaternion (w,x,y,z) de
    base_link_quaternion -- copie de walk_to_xy.py::_yaw_from_quaternion (meme
    convention MuJoCo w,x,y,z que qpos)."""
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


class SimStateListener:
    """S'abonne au canal LCM `sim_state` (verite terrain MuJoCo) et garde le
    dernier message recu -- copie autonome de walk_to_xy.py::SimStateListener
    (evite d'importer walk_to_xy.py en entier, qui tire aussi gamepad_api.py).
    N'existe et ne fonctionne qu'avec la SIMULATION en cours d'execution."""

    def __init__(self, lcm_url=SIM_LCM_URL):
        sys.path.insert(0, "/home/equansrobotic/engineai_robotics_native_sdk/tools/virtual_gamepad")
        from lcm_msgs.data import SimState
        self._SimState = SimState
        self._lc = lcm.LCM(lcm_url)
        self._lc.subscribe(SIM_STATE_CHANNEL, self._on_message)
        self._latest = None
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def _on_message(self, channel, data):
        state = self._SimState.decode(data)
        with self._lock:
            self._latest = state

    def _spin(self):
        while True:
            self._lc.handle()

    def wait_for_first_message(self, timeout=5.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            with self._lock:
                if self._latest is not None:
                    return True
            time.sleep(0.05)
        return False

    def pose(self):
        """(x, y, z, yaw) courants du bassin, en repere MONDE, ou None si aucun
        message recu."""
        with self._lock:
            state = self._latest
        if state is None:
            return None
        x, y, z = state.base_link_position
        w, qx, qy, qz = state.base_link_quaternion
        return x, y, z, _yaw_from_quaternion(w, qx, qy, qz)


def world_to_robot_local(world_point, base_pose):
    """Convertit un point 3D en repere MONDE vers le repere LOCAL du bassin du
    robot (celui utilise par pinch_x/pinch_y/pinch_z, HAND_OFFSET_LEFT/RIGHT et
    LEFT_CHAIN/RIGHT_CHAIN partout ailleurs dans ce fichier) -- c'est
    l'operation INVERSE de _rotate_xy (lift.py/depose.py/pivot.py), qui va du
    repere local vers une correction de cap dans le repere du robot.

    world_point : np.array([x, y, z]) en repere monde (ex: un des points
      retournes par carton_face_centers()).
    base_pose : (x, y, z, yaw) du bassin, EXACTEMENT le tuple retourne par
      SimStateListener.pose() -- lire une pose fraiche a chaque appel, ne
      jamais reutiliser une pose mesuree avant un walk_to() precedent (deja
      documente ailleurs dans ce projet : le robot peut deriver entre-temps).

    Seule la rotation autour de Z (yaw) est prise en compte -- le robot est
    suppose rester vertical une fois debout (roll/pitch negligeables), meme
    hypothese implicite que pinch_yaw_offset partout ailleurs. Retourne
    np.array([x_local, y_local, z_local]) -- x_local/y_local/z_local
    correspondent directement a pinch_x/pinch_y/pinch_z (avant tout
    _rotate_xy -- le robot vise deja dans la bonne direction par construction,
    pinch_yaw_offset peut rester a 0.0)."""
    base_x, base_y, base_z, base_yaw = base_pose
    dx = world_point[0] - base_x
    dy = world_point[1] - base_y
    dz = world_point[2] - base_z
    c, s = math.cos(base_yaw), math.sin(base_yaw)
    local_x = dx * c + dy * s
    local_y = -dx * s + dy * c
    return np.array([local_x, local_y, dz])


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

                                                                                         
LEFT_LEG_CHAIN = [
    ("J00_HIP_PITCH_L",   np.array([0, 0.965926, -0.258819]), np.array([0.01541, 0.076141, -0.061208])),
    ("J01_HIP_ROLL_L",    np.array([1, 0, 0]),                np.array([0.048, 0.049359, -0.013226])),
    ("J02_HIP_YAW_L",     np.array([0, 0, 1]),                np.array([-0.03139, -0.0015951, -0.086016])),
    ("J03_KNEE_PITCH_L",  np.array([0, 1, 0]),                np.array([-0.02602, -0.000028566, -0.23655])),
    ("J04_ANKLE_PITCH_L", np.array([0, 1, 0]),                np.array([-0.026756, 0.00041994, -0.36305])),
    ("J05_ANKLE_ROLL_L",  np.array([1, 0, 0]),                np.array([0, 0, -0.015])),
]
FOOT_OFFSET_LEFT = np.array([0.0, 0.0, 0.0])

RIGHT_LEG_CHAIN = [
    ("J06_HIP_PITCH_R",   np.array([0, 0.965926, 0.258819]),  np.array([0.01541, -0.076141, -0.061208])),
    ("J07_HIP_ROLL_R",    np.array([1, 0, 0]),                np.array([0.048, -0.04936, -0.013226])),
    ("J08_HIP_YAW_R",     np.array([0, 0, 1]),                np.array([-0.03139, 0.0015966, -0.086016])),
    ("J09_KNEE_PITCH_R",  np.array([0, 1, 0]),                np.array([-0.02602, 0.000028566, -0.23655])),
    ("J10_ANKLE_PITCH_R", np.array([0, 1, 0]),                np.array([-0.026756, -0.00041994, -0.36305])),
    ("J11_ANKLE_ROLL_R",  np.array([1, 0, 0]),                np.array([0, 0, -0.015])),
]
FOOT_OFFSET_RIGHT = np.array([0.0, 0.0, 0.0])

STEP_PERIOD = 0.6                                                                       
STEP_HEIGHT = 0.10                                                        
BOB_HEIGHT = 0.025                                                                         
                                                                                          
                                                                                            
SWAY_WIDTH = 0.02                                                                           
                                                         

GAINS = {
    "HIP_PITCH": (250.0, 8.0), "HIP_ROLL": (250.0, 8.0), "HIP_YAW": (150.0, 5.0),
    "KNEE_PITCH": (250.0, 8.0), "ANKLE_PITCH": (120.0, 3.0), "ANKLE_ROLL": (120.0, 3.0),
    "WAIST_YAW": (150.0, 3.0), "SHOULDER_PITCH": (80.0, 2.0), "SHOULDER_ROLL": (80.0, 2.0),
    "SHOULDER_YAW": (60.0, 1.5), "ELBOW_PITCH": (60.0, 1.5), "ELBOW_YAW": (40.0, 1.0),
    "HEAD_YAW": (30.0, 1.0),
}


def gains_for(joint_name):
    for key, gains in GAINS.items():
        if key in joint_name:
            return gains
    return (100.0, 2.0)


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
    """IK (cinematique inverse) du bras : calcule les 5 angles articulaires (radians)
    qui amenent la MAIN a la position 3D `target` (X, Y, Z en metres, repere du
    bassin -- meme convention que pinch_x/pinch_y/pinch_z partout dans ce projet).

    2026-09-11 : consolide en UNE fonction reutilisable ce qui existait duplique
    (solve_ik ci-dessus + _solve_ik_locked_elbow dans lift.py ET depose.py) --
    demande explicite de l'utilisateur ("une fonction pour l'IK qui servira a
    controler les bras").

    Principe (Jacobienne amortie, "damped least squares") : le bras a 5
    articulations mais la cible n'a que 3 coordonnees (X,Y,Z) -- il existe donc
    en general une INFINITE de solutions (redondance de 2 DDL). L'algorithme part
    de `q_init` (position de depart) et AJUSTE les angles par petits pas pour
    reduire l'ecart entre la main et la cible, jusqu'a convergence -- il choisit
    donc la solution la plus PROCHE de q_init, pas une solution "canonique". C'est
    pourquoi le meme `target` peut donner des postures de coude tres differentes
    selon la valeur de q_init (voir la longue histoire de ce fichier/lift.py sur
    ce sujet -- q_init n'est jamais anodin).

    Parametres :
      chain, hand_offset : geometrie du bras (LEFT_CHAIN/HAND_OFFSET_LEFT ou
        RIGHT_CHAIN/HAND_OFFSET_RIGHT, definis plus haut dans ce fichier).
      target : np.array([x, y, z]), position 3D visee pour la main.
      q_init : np.array de 5 angles (radians), point de depart/seed du solveur.
      lock_index, lock_angle : optionnels -- si fournis, FIGE l'articulation
        d'index `lock_index` (0=SHOULDER_PITCH, 1=SHOULDER_ROLL, 2=SHOULDER_YAW,
        3=ELBOW_PITCH, 4=ELBOW_YAW) a `lock_angle` (radians) et ne resout la
        position qu'avec les 4 AUTRES articulations -- utile pour forcer une
        forme de bras precise (ex: coude tendu) a une position de main donnee,
        plutot que de laisser le solveur choisir au hasard parmi les solutions
        redondantes. Sans ces 2 parametres : IK standard, 5 DDL libres.
      null_space_gain : avec lock_index/lock_angle, il reste encore 1 DDL
        redondant parmi les 4 articulations libres (3 equations de position,
        4 inconnues) -- SANS regularisation, le solveur peut deriver vers une
        posture aberrante qui atteint quand meme la cible (ex: SHOULDER_YAW a
        -86deg au lieu de ~0deg, bras qui semble viser le centre du corps au
        lieu de la cible reelle -- constate le 2026-09-11 : meme erreur de
        position, poignet identique, mais tout le bras visuellement tordu).
        Ce parametre projette un rappel vers `q_init` dans le noyau (null
        space) du Jacobien -- ne change PAS la position finale de la main
        (le rappel est orthogonal a l'erreur de position), seulement laquelle
        des solutions redondantes est choisie. 0.0 desactive (comportement
        d'origine).

    Retourne : np.array de 5 angles (radians), a publier directement sur
      LEFT_JOINT_INDICES/RIGHT_JOINT_INDICES via Lever (voir lever.py)."""
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
        J_pinv = J.T @ np.linalg.inv(JJt)  # (n_free, 3), pseudo-inverse amortie
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


def build_dynamic_scene(tmpdir, attach_cymbals=False, cymbal_pos=(0.0334, 0.0258, -0.1825),
                         cymbal_euler=(-90, 0, 0)):
    """Genere une copie patchee de serial_pm01_edu.xml (keyframe +7 valeurs pour le
    freejoint du carton) et une scene temporaire l'assemblant avec carton_liftable.xml.
    Ne touche a aucun fichier du depot.

    attach_cymbals=True : soude une cymbale (visuelle, sans collision propre -- la
    prise reelle reste geree par la sphere de main deja validee) a chaque
    LINK_ELBOW_END_{L,R} dans une copie patchee de serial_links.xml (repere : c'est
    le lien TERMINAL du bras malgre son nom, celui ou HAND_OFFSET_LEFT/RIGHT place
    le point de pince -- donc bien "la main", pas le coude), pour que le robot ait
    visuellement une cymbale dans chaque main pendant qu'il soulve le carton.
    cymbal_pos/cymbal_euler : offset (X,Y,Z)/(rotX,rotY,rotZ deg) de la cymbale par
    rapport a ce lien, cote GAUCHE (le Y est neglige de signe automatiquement cote
    droit pour la symetrie miroir) -- ajuster ici si mal placee visuellement."""
    src = open(ROBOT_XML).read()
    links_path = os.path.join(ROBOT_DIR, "serial_links.xml")
    for fname in ["assets.xml", "serial_actuators.xml", "serial_sensors.xml"]:
        src = src.replace(f'file="{fname}"', f'file="{os.path.join(ROBOT_DIR, fname)}"')

    if attach_cymbals:
        links_src = open(links_path).read()
        cx, cy, cz = cymbal_pos
        ex, ey, ez = cymbal_euler
        cymbal_geom = (f'<geom type="mesh" mesh="cymbale" euler="{ex} {ey} {ez}" '
                        'pos="' + f'{cx} {{y}} {cz}" '
                        'rgba="0.85 0.7 0.2 1" contype="0" conaffinity="0" mass="0.05"/>')
        # Miroir gauche/droite = signe oppose de la MEME valeur fournie par l'utilisateur --
        # PAS force via abs() (bug 2026-08-12 : ecrasait silencieusement le signe voulu pour
        # la gauche, cassant le calibrage calcule par IK).
        links_src = links_src.replace(
            '<geom class="collision_left_elbow_end"/>',
            '<geom class="collision_left_elbow_end"/>\n                                ' + cymbal_geom.format(y=cy),
        )
        links_src = links_src.replace(
            '<geom class="collision_right_elbow_end"/>',
            '<geom class="collision_right_elbow_end"/>\n                                ' + cymbal_geom.format(y=-cy),
        )
        patched_links_path = os.path.join(tmpdir, "serial_links_cymbals.xml")
        open(patched_links_path, "w").write(links_src)
        src = src.replace('file="serial_links.xml"', f'file="{patched_links_path}"')
    else:
        src = src.replace('file="serial_links.xml"', f'file="{links_path}"')

    m = re.search(r'<key name="floating_base_homing" qpos="([^"]+)"/>', src)
    if not m:
        raise RuntimeError("Keyframe 'floating_base_homing' introuvable dans serial_pm01_edu.xml")
    box_qpos = f"{CARTON_XY[0]} {CARTON_XY[1]} {CARTON_Z} 1 0 0 0"
    patched = src.replace(m.group(0), f'<key name="floating_base_homing" qpos="{m.group(1)} {box_qpos}"/>')

    patched_robot_path = os.path.join(tmpdir, "serial_pm01_edu_liftdemo.xml")
    open(patched_robot_path, "w").write(patched)

    cymbal_asset = ""
    cymbal_compiler = ""
    if attach_cymbals:
        cymbal_compiler = f'<compiler meshdir="{os.path.join(ENV_DIR, "meshes")}"/>'
        cymbal_asset = '<asset><mesh name="cymbale" file="cymbale.STL" scale="0.001 0.001 0.001"/></asset>'

    top_path = os.path.join(tmpdir, "lift_demo_top.xml")
    open(top_path, "w").write(f'''<mujoco model="lift_demo">
    {cymbal_compiler}
    <include file="{patched_robot_path}"/>
    <include file="{os.path.join(ENV_DIR, "ground.xml")}"/>
    <include file="{os.path.join(ENV_DIR, "carton_liftable.xml")}"/>
    {cymbal_asset}

    <!-- Podium fusionne directement ici (n'existe plus comme fichier separe --
         voir assets/resource/pm01_edu_carton.xml pour la meme fusion). -->
    <worldbody>
        <body name="podium" pos="{CARTON_XY[0]} {CARTON_XY[1]} 0.315">
            <geom name="podium_top" type="box" size="0.19 0.17 0.02" pos="0 0 0.315" mass="20"
                  rgba="0.55 0.55 0.58 1" friction="1.0 0.005 0.0001" group="0"/>
            <geom name="podium_leg" type="cylinder" size="0.06 0.315" pos="0 0 0" mass="20"
                  rgba="0.35 0.35 0.38 1" friction="1.0 0.005 0.0001" group="0"/>
        </body>
    </worldbody>
</mujoco>''')
    return top_path


class PM01LiftSim:
    """Session de simulation PERSISTANTE (une seule fenetre MuJoCo, gardee ouverte) --
    contrairement a `main()`/le CLI qui relance une scene neuve a chaque execution, ceci
    permet d'enchainer marche()/soulever()/etc. sur le MEME robot, dans la MEME fenetre,
    comme une vraie choreographie. Utilisee par tools/virtual_gamepad/choreography.py.

    Le torse (`base_pose`) est deplace directement en le reecrivant a chaque pas (pas de
    solveur d'equilibre) -- mais les jambes, elles, executent une vraie alternance de pas
    (IK par jambe, un pied porteur pose au sol pendant que l'autre se souleve, avance et se
    repose) pilotee par marcher(), au lieu de rester figees en pose home et de trainer au
    sol. Les cibles bras (pinch_x/y/z) sont TOUJOURS relatives au torse (repere local) :
    elles restent valides ou que soit le torse dans le monde.
    """

    def __init__(self, attach_cymbals=False, cymbal_pos=(0.0334, 0.0258, -0.1825), cymbal_euler=(-90, 0, 0)):
        if mujoco is None:
            raise RuntimeError(
                "mujoco n'est pas installe dans cet interpreteur -- PM01LiftSim en a besoin "
                "(lance avec tools/robot_arm_ik/.venv/bin/python3). Si tu voulais juste "
                "reutiliser l'IK (solve_ik, LEFT_CHAIN, ...) sans simulation, ce n'est pas "
                "necessaire d'instancier PM01LiftSim."
            )
        self._tmpdir = tempfile.mkdtemp(prefix="pm01_lift_sim_")
        model_path = build_dynamic_scene(self._tmpdir, attach_cymbals=attach_cymbals,
                                          cymbal_pos=cymbal_pos, cymbal_euler=cymbal_euler)
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "floating_base_homing")
        if key_id < 0:
            raise RuntimeError("Keyframe 'floating_base_homing' introuvable dans le modele assemble")
        mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        self.home_qpos = self.model.key_qpos[key_id].copy()
        self.base_pose = self.home_qpos[0:7].copy()                                                    

        self.joints = []
        for jid in range(self.model.njnt):
            if self.model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE:
                continue
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, jid)
            actid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"motor_{name}")
            if actid < 0:
                continue
            kp, kd = gains_for(name)
            self.joints.append({
                "name": name, "qposadr": self.model.jnt_qposadr[jid], "dofadr": self.model.jnt_dofadr[jid],
                "actid": actid, "target": self.home_qpos[self.model.jnt_qposadr[jid]], "kp": kp, "kd": kd,
            })

        self.left_names = [n for n, _, _ in LEFT_CHAIN]
        self.right_names = [n for n, _, _ in RIGHT_CHAIN]
        self.q_left = np.array([self._home_joint_qpos(n) for n in self.left_names])
        self.q_right = np.array([self._home_joint_qpos(n) for n in self.right_names])

        self.left_leg_names = [n for n, _, _ in LEFT_LEG_CHAIN]
        self.right_leg_names = [n for n, _, _ in RIGHT_LEG_CHAIN]
        self.q_leg_L = np.array([self._home_joint_qpos(n) for n in self.left_leg_names])
        self.q_leg_R = np.array([self._home_joint_qpos(n) for n in self.right_leg_names])
                                                                                           
        self.foot_home_L = forward_kinematics(LEFT_LEG_CHAIN, FOOT_OFFSET_LEFT, self.q_leg_L)
        self.foot_home_R = forward_kinematics(RIGHT_LEG_CHAIN, FOOT_OFFSET_RIGHT, self.q_leg_R)
                                                                                         
                                                                              
        self.foot_x_local = {"L": 0.0, "R": 0.0}
        self.foot_y_local = {"L": 0.0, "R": 0.0}

        self.carton_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "carton")
        mujoco.mj_forward(self.model, self.data)                                          
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

    def _home_joint_qpos(self, name):
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        return self.home_qpos[self.model.jnt_qposadr[jid]]

    def _set_arm_targets(self, qL, qR):
        for j in self.joints:
            if j["name"] in self.left_names:
                j["target"] = qL[self.left_names.index(j["name"])]
            elif j["name"] in self.right_names:
                j["target"] = qR[self.right_names.index(j["name"])]

    def _set_leg_targets(self, qL, qR):
        for j in self.joints:
            if j["name"] in self.left_leg_names:
                j["target"] = qL[self.left_leg_names.index(j["name"])]
            elif j["name"] in self.right_leg_names:
                j["target"] = qR[self.right_leg_names.index(j["name"])]

    def distance_to_carton(self, pinch_x=0.38):
        """Distance (m, le long de X) qu'il faut faire marcher le torse depuis sa position
        HOME pour amener le carton a portee de bras (pinch_x)."""
        return CARTON_XY[0] - pinch_x - float(self.home_qpos[0])

    def _step(self):
        for j in self.joints:
            pos = self.data.qpos[j["qposadr"]]
            vel = self.data.qvel[j["dofadr"]]
            self.data.ctrl[j["actid"]] = j["kp"] * (j["target"] - pos) - j["kd"] * vel
        self.data.qpos[0:7] = self.base_pose
        self.data.qvel[0:6] = 0.0
        mujoco.mj_step(self.model, self.data)

    @property
    def is_running(self):
        return self.viewer.is_running()

    def hold(self, duree):
        """Ne change rien (pose bras/torse courantes) -- utilise pour repos()/passif()/debout()."""
        t0 = time.time()
        while self.is_running and time.time() - t0 < duree:
            self._step()
            self.viewer.sync()

    def marcher(self, x=0.0, y=0.0, vitesse=0.4, duree=None):
        """Fait marcher le torse de `x` metres (avant/arriere) et `y` metres (lateral) depuis
        sa position ACTUELLE, a `vitesse` m/s (ou en `duree` secondes si fournie).

        Vraie alternance de pas, pas un glissement : le trajet est decoupe en demi-enjambees
        de duree STEP_PERIOD (_foulee), chacune soulevant UN SEUL pied (l'autre reste plante
        au sol, IK jambe a chaque pas de simu) pendant que le bassin avance d'autant. Termine
        par un petit rassemblement des pieds sous le bassin (_rassembler)."""
        distance = (x**2 + y**2) ** 0.5
        if distance == 0.0:
            return
        duree = duree if duree is not None else distance / vitesse
        n_steps = max(2, round(duree / STEP_PERIOD))
        step_duration = duree / n_steps
        dx_step = x / n_steps
        dy_step = y / n_steps

        swing = "L"
        for _ in range(n_steps):
            self._foulee(swing, dx_step, dy_step, step_duration)
            swing = "R" if swing == "L" else "L"

        self._rassembler(step_duration / 2.0)

    def _foulee(self, swing_side, dx_pelvis, dy_pelvis, duration):
        """Une demi-enjambee : `swing_side` decolle, avance et se repose pendant que l'autre
        pied (porteur) reste plante au sol et que le bassin avance de (dx_pelvis, dy_pelvis).
        Symetrique : le pied porteur et le pied oscillant s'ecartent chacun de dx_pelvis/2 de
        part et d'autre de leur position home (repere torse).

        Le bassin (base_pose) n'avance pas juste en ligne droite : il monte legerement a
        mi-enjambee (BOB_HEIGHT, jambe porteuse la plus tendue) et bascule lateralement vers
        le pied porteur (SWAY_WIDTH, transfert de poids), sinon meme avec des jambes qui
        marchent vraiment en dessous le torse suit une trajectoire plate et se lit comme un
        glissement."""
        stance_side = "R" if swing_side == "L" else "L"
        chains = {"L": (LEFT_LEG_CHAIN, FOOT_OFFSET_LEFT, self.foot_home_L),
                  "R": (RIGHT_LEG_CHAIN, FOOT_OFFSET_RIGHT, self.foot_home_R)}
        q_leg = {"L": self.q_leg_L, "R": self.q_leg_R}

        amp_x, amp_y = dx_pelvis / 2.0, dy_pelvis / 2.0
        x0 = dict(self.foot_x_local)
        y0 = dict(self.foot_y_local)
        x_target = {swing_side: amp_x, stance_side: -amp_x}
        y_target = {swing_side: amp_y, stance_side: -amp_y}
        sway_sign = 1.0 if stance_side == "L" else -1.0

        px0, py0, pz0 = float(self.base_pose[0]), float(self.base_pose[1]), float(self.base_pose[2])
        t0 = time.time()
        while self.is_running and time.time() - t0 < duration:
            u = min(1.0, (time.time() - t0) / duration)
            a = ease(u)
            hump = np.sin(np.pi * u)                                                
            for side in ("L", "R"):
                chain, offset, home = chains[side]
                lx = x0[side] + a * (x_target[side] - x0[side])
                ly = y0[side] + a * (y_target[side] - y0[side])
                lift = STEP_HEIGHT * hump if side == swing_side else 0.0
                target = home + np.array([lx, ly, lift])
                q_leg[side] = solve_ik(chain, offset, target, q_leg[side], iters=30)
            self.q_leg_L, self.q_leg_R = q_leg["L"], q_leg["R"]
            self._set_leg_targets(self.q_leg_L, self.q_leg_R)
            self.base_pose[0] = px0 + a * dx_pelvis
            self.base_pose[1] = py0 + a * dy_pelvis + sway_sign * SWAY_WIDTH * hump
            self.base_pose[2] = pz0 + BOB_HEIGHT * hump
            self._step()
            self.viewer.sync()

        self.foot_x_local[swing_side], self.foot_x_local[stance_side] = amp_x, -amp_x
        self.foot_y_local[swing_side], self.foot_y_local[stance_side] = amp_y, -amp_y
        self.base_pose[0] = px0 + dx_pelvis
        self.base_pose[1] = py0 + dy_pelvis
        self.base_pose[2] = pz0

    def _rassembler(self, duration):
        """Ramene les deux pieds sous le bassin (ecart local -> 0) sans faire avancer le
        bassin, pour finir pieds joints plutot qu'en plein pas avant d'enchainer soulever()."""
        if all(abs(self.foot_x_local[s]) < 1e-4 and abs(self.foot_y_local[s]) < 1e-4 for s in ("L", "R")):
            return
        chains = {"L": (LEFT_LEG_CHAIN, FOOT_OFFSET_LEFT, self.foot_home_L),
                  "R": (RIGHT_LEG_CHAIN, FOOT_OFFSET_RIGHT, self.foot_home_R)}
        q_leg = {"L": self.q_leg_L, "R": self.q_leg_R}
        x0, y0 = dict(self.foot_x_local), dict(self.foot_y_local)

        t0 = time.time()
        while self.is_running and time.time() - t0 < duration:
            u = min(1.0, (time.time() - t0) / duration)
            a = ease(u)
            for side in ("L", "R"):
                chain, offset, home = chains[side]
                lx = x0[side] * (1 - a)
                ly = y0[side] * (1 - a)
                                                                                            
                                                                                           
                lift = 0.4 * STEP_HEIGHT * np.sin(np.pi * u)
                target = home + np.array([lx, ly, lift])
                q_leg[side] = solve_ik(chain, offset, target, q_leg[side], iters=30)
            self.q_leg_L, self.q_leg_R = q_leg["L"], q_leg["R"]
            self._set_leg_targets(self.q_leg_L, self.q_leg_R)
            self._step()
            self.viewer.sync()

        self.foot_x_local = {"L": 0.0, "R": 0.0}
        self.foot_y_local = {"L": 0.0, "R": 0.0}

    def soulever(self, pinch_x=0.38, pinch_y=0.28, pinch_z=-0.054, squeeze_y=0.086, lift_z=0.146,
                 approach_duration=1.5, squeeze_duration=1.0, lift_duration=2.5, hold_seconds=3.0):
        """Sequence approche->serrage->levee, cibles bras EN REPERE TORSE -- donc valides
        ou que le torse ait ete amene par marcher() au prealable."""
        pinch_L = np.array([pinch_x, pinch_y, pinch_z])
        pinch_R = np.array([pinch_x, -pinch_y, pinch_z])
        q_pinch_L = solve_ik(LEFT_CHAIN, HAND_OFFSET_LEFT, pinch_L, self.q_left)
        q_pinch_R = solve_ik(RIGHT_CHAIN, HAND_OFFSET_RIGHT, pinch_R, self.q_right)

        squeeze_L = np.array([pinch_x, squeeze_y, pinch_z])
        squeeze_R = np.array([pinch_x, -squeeze_y, pinch_z])
        q_squeeze_L = solve_ik(LEFT_CHAIN, HAND_OFFSET_LEFT, squeeze_L, q_pinch_L)
        q_squeeze_R = solve_ik(RIGHT_CHAIN, HAND_OFFSET_RIGHT, squeeze_R, q_pinch_R)

        z_start = float(self.data.xpos[self.carton_body][2])
        print(f"[INFO] Carton hauteur initiale (monde) : {z_start:.4f} m", flush=True)

        phases = [
            ("approche", approach_duration, self.q_left, q_pinch_L, self.q_right, q_pinch_R),
            ("serrage", squeeze_duration, q_pinch_L, q_squeeze_L, q_pinch_R, q_squeeze_R),
        ]
        for label, duration, qL0, qL1, qR0, qR1 in phases:
            phase_t0 = time.time()
            while self.is_running and time.time() - phase_t0 < duration:
                a = ease((time.time() - phase_t0) / duration) if duration > 0 else 1.0
                self._set_arm_targets(qL0 + a * (qL1 - qL0), qR0 + a * (qR1 - qR0))
                self._step()
                self.viewer.sync()
            print(f"[INFO] Phase '{label}' terminee -- carton z={self.data.xpos[self.carton_body][2]:.4f} m",
                  flush=True)

        qL_prev, qR_prev = q_squeeze_L.copy(), q_squeeze_R.copy()
        z0, z1 = pinch_z, lift_z
        lift_t0 = time.time()
        while self.is_running and time.time() - lift_t0 < lift_duration:
            a = ease((time.time() - lift_t0) / lift_duration)
            z = z0 + a * (z1 - z0)
            qL_prev = solve_ik(LEFT_CHAIN, HAND_OFFSET_LEFT, np.array([pinch_x, squeeze_y, z]), qL_prev, iters=30)
            qR_prev = solve_ik(RIGHT_CHAIN, HAND_OFFSET_RIGHT, np.array([pinch_x, -squeeze_y, z]), qR_prev, iters=30)
            self._set_arm_targets(qL_prev, qR_prev)
            self._step()
            self.viewer.sync()

        self.q_left, self.q_right = qL_prev, qR_prev                                                        
        z_final = float(self.data.xpos[self.carton_body][2])
        print(f"[INFO] Levee terminee -- carton z={z_final:.4f} m (depart {z_start:.4f} m, "
              f"gain {z_final - z_start:+.4f} m).", flush=True)
        self.hold(hold_seconds)

    def fermer(self):
        self.viewer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pinch-x", type=float, default=0.38, help="X de la position de pince (m, repere torse)")
    parser.add_argument("--pinch-y", type=float, default=0.28, help="|Y| de la position de pince avant serrage")
    parser.add_argument("--pinch-z", type=float, default=-0.054, help="Z de la position de pince")
    parser.add_argument("--squeeze-y", type=float, default=0.086,
                         help="|Y| apres serrage (doit rester > ~demi-largeur-0.03 : un serrage trop fort ejecte le carton)")
    parser.add_argument("--lift-z", type=float, default=0.146, help="Z cible en fin de levee (repere torse)")
    parser.add_argument("--approach-duration", type=float, default=1.5)
    parser.add_argument("--squeeze-duration", type=float, default=1.0)
    parser.add_argument("--lift-duration", type=float, default=2.5)
    parser.add_argument("--hold-seconds", type=float, default=-1.0,
                         help="Duree de maintien de la pose finale avant fermeture automatique "
                              "(defaut -1 : reste ouvert jusqu'a fermeture manuelle de la fenetre, "
                              "comme avant -- utile en usage interactif/dialogue Arm Control). "
                              "Mettre une valeur >= 0 pour un usage scripte/Grafcet qui doit "
                              "pouvoir continuer sans intervention.")
    parser.add_argument("--attach-cymbals", action="store_true",
                         help="Soude une cymbale (assets/resource/environment/meshes/cymbale.STL) "
                              "a chaque main, visuellement -- la prise du carton reste geree par "
                              "la sphere de main deja validee, la cymbale ne fait qu'accompagner "
                              "le mouvement du bras.")
    parser.add_argument("--cymbal-pos", type=float, nargs=3, default=[0.0334, 0.0258, -0.1825],
                         metavar=("X", "Y", "Z"),
                         help="Position (m) de la cymbale par rapport a LINK_ELBOW_END (le lien "
                              "terminal du bras, cote GAUCHE -- Y prend automatiquement le signe "
                              "oppose cote droit pour la symetrie miroir). Ajuster si mal placee.")
    parser.add_argument("--cymbal-euler", type=float, nargs=3, default=[-90, 0, 0],
                         metavar=("RX", "RY", "RZ"),
                         help="Rotation (degres, convention MuJoCo euler XYZ) de la cymbale par "
                              "rapport a LINK_ELBOW_END.")
    parser.add_argument("--home-only", action="store_true",
                         help="Saute approche/serrage/levee -- garde juste la pose de repos "
                              "(bras le long du corps, Q_LEFT_HOME/Q_RIGHT_HOME) a l'ouverture, "
                              "fenetre ouverte indefiniment. Pratique pour caler --cymbal-pos/"
                              "--cymbal-euler sans attendre la sequence a chaque essai -- combiner "
                              "avec le menu 'Frame' > 'Geom' du viewer pour voir les axes X/Y/Z.")
    args = parser.parse_args()

    sim = PM01LiftSim(attach_cymbals=args.attach_cymbals,
                       cymbal_pos=tuple(args.cymbal_pos), cymbal_euler=tuple(args.cymbal_euler))

    if args.home_only:
        print("[INFO] Pose de repos (bras le long du corps) -- fenetre ouverte, ferme-la pour "
              "quitter. Menu 'Frame' > 'Geom' dans le viewer pour voir les axes.", flush=True)
        while sim.is_running:
            sim.hold(1.0)
        return

    sim.soulever(
        pinch_x=args.pinch_x, pinch_y=args.pinch_y, pinch_z=args.pinch_z, squeeze_y=args.squeeze_y,
        lift_z=args.lift_z, approach_duration=args.approach_duration, squeeze_duration=args.squeeze_duration,
        lift_duration=args.lift_duration, hold_seconds=max(args.hold_seconds, 0.0),
    )

    if args.hold_seconds < 0:
        print("[INFO] Fenetre ouverte -- ferme-la pour quitter.", flush=True)
        while sim.is_running:
            sim.hold(1.0)
    else:
        print("[INFO] Fin du maintien -- fermeture automatique.", flush=True)
                                                                                             
                                                                                      
        sys.stdout.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
