                      
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
import os
import re
import sys
import tempfile
import time

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

PODIUM_XY = (0.6849, 0.0)              # 2026-08-26 (v3) : mesure REELLE utilisateur, 20cm entre
                                        # le bord du podium et l'extremite du bassin, AU REPOS.
                                        # La prise necessite ensuite un pas en avant (~14cm, voir
                                        # main()/marcher()) plutot qu'une position au repos deja
                                        # collee ou une flexion de genoux exageree (les deux
                                        # essayes et rejetes -- cf plan de session).
PODIUM_SIZE = (0.455, 0.355, 0.2675)   # demi-tailles X(longueur)/Y(largeur)/Z(hauteur) -- long./larg.
                                        # inversees le 2026-08-26 (etait 0.355/0.455) : c'est LE PODIUM
                                        # ("la boite qui supporte le carton"), pas le carton, qui devait
                                        # etre inverse -- carton restaure a sa taille d'origine ci-dessous.
PODIUM_Z = PODIUM_SIZE[2]              # bloc pose au sol (bas a z=0)

CARTON_XY = (0.3674, 0.0)

# 2026-08-26 : nouveau podium (bloc unique 91x53.5x71cm), carton repositionne a
# l'extremite du podium face au robot (flush avec le bord, pas de recouvrement),
# centre du carton a 32cm (distance 3D reelle) de l'EXTREMITE (surface avant) du
# bassin -- PAS de l'origine de LINK_BASE. Le bassin est modelise par une sphere
# de collision "collision_base_lower_box" (rayon 0.085m, offset local 0.02/0/-0.02
# depuis LINK_BASE). LINK_BASE monde (keyframe floating_base_homing) :
# -0.075145/0.014289/0.820203. Taille du carton INCHANGEE.
# Historique : 36cm -> pinch_x devait depasser ~0.44m, hors de portee du bras
# (erreur IK 328mm). 27cm corrigeait la portee (pinch_x~0.35, erreur quasi nulle)
# mais rapprochait trop le PODIUM (91cm de profondeur) du corps du robot -- "le
# robot est dans la table" (2026-08-26). 32cm = compromis : podium a 15.9cm du
# bassin (vs 10.4cm a 27cm), pinch_x~0.40 (erreur IK moderee ~40mm, acceptable).
CARTON_Z = 2 * PODIUM_SIZE[2] + 0.146

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
                                                         

# Posture jambes flechies -- valeurs portees telles quelles depuis
# virtual_gamepad_ros/lift.py::_bend_knees (deja validees stables sur le robot reel ET
# en simu ROS a l'echelle 3.0, cf session du 2026-08-25/26 -- ne pas reinventer ces
# angles, cf feedback "verifier les scripts de reference avant de deviner").
WALK_STANCE_SCALE = 3.0  # 5.0 teste puis abandonne (2026-08-26) -- rend le "flottement" du
                          # bassin fige (pas de vrai solveur d'equilibre dans PM01LiftSim)
                          # bien pire visuellement. Revenu a 3.0 (seule valeur validee sur
                          # le robot reel) -- voir plan de session pour la piste suivante
                          # (rapprocher le carton plutot que flechir plus).
WALK_STANCE_HIP_PITCH_L = np.radians(-6.9) * WALK_STANCE_SCALE
WALK_STANCE_HIP_PITCH_R = np.radians(-5.2) * WALK_STANCE_SCALE
WALK_STANCE_KNEE_L = np.radians(11.7) * WALK_STANCE_SCALE
WALK_STANCE_KNEE_R = np.radians(10.5) * WALK_STANCE_SCALE
WALK_STANCE_ANKLE_PITCH_L = np.radians(-4.8) * WALK_STANCE_SCALE
WALK_STANCE_ANKLE_PITCH_R = np.radians(-5.2) * WALK_STANCE_SCALE

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

    <!-- Podium fusionne directement ici -- bloc unique 91x53.5x71cm (2026-08-26,
         voir tools/robot_arm_ik_podium/pm01_edu_carton.xml pour la meme fusion). -->
    <worldbody>
        <body name="podium" pos="{PODIUM_XY[0]} {PODIUM_XY[1]} {PODIUM_Z}">
            <geom name="podium_block" type="box" size="{PODIUM_SIZE[0]} {PODIUM_SIZE[1]} {PODIUM_SIZE[2]}" mass="20"
                  rgba="0.55 0.55 0.58 1" friction="1.0 0.005 0.0001" group="0"/>
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

    def flechir_genoux(self, duration=1.5):
        """Flechit hanche+genou+cheville (posture 'walk_stance', valeurs eprouvees --
        voir WALK_STANCE_* en tete de fichier) depuis la pose home actuelle. Le bassin
        (base_pose) EST abaisse en compensation (moyenne des deux jambes) pour garder
        les pieds au sol -- sans ca, base_pose reste fige a la hauteur "jambes droites"
        et les pieds (suivant la cinematique directe depuis un bassin fixe) decollent du
        sol de plusieurs cm (constate 2026-08-26, "le robot flotte dans l'air")."""
        qL0, qR0 = self.q_leg_L.copy(), self.q_leg_R.copy()
        qL1, qR1 = qL0.copy(), qR0.copy()
        qL1[0], qL1[3], qL1[4] = WALK_STANCE_HIP_PITCH_L, WALK_STANCE_KNEE_L, WALK_STANCE_ANKLE_PITCH_L
        qR1[0], qR1[3], qR1[4] = WALK_STANCE_HIP_PITCH_R, WALK_STANCE_KNEE_R, WALK_STANCE_ANKLE_PITCH_R

        dz_L = forward_kinematics(LEFT_LEG_CHAIN, FOOT_OFFSET_LEFT, qL1)[2] - self.foot_home_L[2]
        dz_R = forward_kinematics(RIGHT_LEG_CHAIN, FOOT_OFFSET_RIGHT, qR1)[2] - self.foot_home_R[2]
        pelvis_dz = (dz_L + dz_R) / 2.0

        z0 = float(self.base_pose[2])
        t0 = time.time()
        while self.is_running and time.time() - t0 < duration:
            a = ease((time.time() - t0) / duration)
            self.q_leg_L = qL0 + a * (qL1 - qL0)
            self.q_leg_R = qR0 + a * (qR1 - qR0)
            self.base_pose[2] = z0 - a * pelvis_dz
            self._set_leg_targets(self.q_leg_L, self.q_leg_R)
            self._step()
            self.viewer.sync()
        self.q_leg_L, self.q_leg_R = qL1, qR1
        self.base_pose[2] = z0 - pelvis_dz

    def viser(self, pinch_x, pinch_y, pinch_z, duration=1.5, hold_seconds=-1.0):
        """Amene juste les bras a la position de pince (bras ecartes, vise le centre de
        chaque face du carton) SANS serrer ni lever -- pour valider visuellement la
        visee avant de tenter une vraie prise. Cibles EN REPERE TORSE, comme soulever()."""
        pinch_L = np.array([pinch_x, pinch_y, pinch_z])
        pinch_R = np.array([pinch_x, -pinch_y, pinch_z])
        q_pinch_L = solve_ik(LEFT_CHAIN, HAND_OFFSET_LEFT, pinch_L, self.q_left)
        q_pinch_R = solve_ik(RIGHT_CHAIN, HAND_OFFSET_RIGHT, pinch_R, self.q_right)

        qL0, qR0 = self.q_left.copy(), self.q_right.copy()
        t0 = time.time()
        while self.is_running and time.time() - t0 < duration:
            a = ease((time.time() - t0) / duration)
            self._set_arm_targets(qL0 + a * (q_pinch_L - qL0), qR0 + a * (q_pinch_R - qR0))
            self._step()
            self.viewer.sync()
        self.q_left, self.q_right = q_pinch_L, q_pinch_R
        err_L = np.linalg.norm(pinch_L - forward_kinematics(LEFT_CHAIN, HAND_OFFSET_LEFT, q_pinch_L))
        err_R = np.linalg.norm(pinch_R - forward_kinematics(RIGHT_CHAIN, HAND_OFFSET_RIGHT, q_pinch_R))
        print(f"[INFO] Visee terminee -- erreur IK gauche={err_L*1000:.1f}mm droite={err_R*1000:.1f}mm "
              "(residu de convergence solve_ik, pas une erreur physique).", flush=True)
        if hold_seconds >= 0:
            self.hold(hold_seconds)

    def soulever(self, pinch_x=0.30, pinch_y=0.1655, pinch_z=-0.1392, squeeze_y=0.1405, lift_z=0.146,
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
    # 2026-08-26 (v3, definitif) : podium/carton places pour respecter la mesure REELLE
    # utilisateur (20cm entre bord podium et extremite bassin, AU REPOS -- voir
    # PODIUM_XY/CARTON_XY). Cette distance seule laisse pinch_x=0.44, hors de la zone a
    # erreur IK quasi nulle (fiable jusqu'a pinch_x~0.30, verifie par sweep de session).
    # Deux pistes essayees et rejetees pour combler l'ecart : flexion de genoux plus
    # profonde (WALK_STANCE_SCALE=5) -> "flotte dans l'air", visuellement casse (le
    # torse est fige dans PM01LiftSim, pas de vrai solveur d'equilibre) ; position de
    # repos plus proche -> ne respecte plus les 20cm reels et ne laisse que ~6cm de
    # marge corps-podium. Solution retenue : --pas-avant (marcher() avant la visee,
    # deja dans l'outil) ramene pinch_x a 0.30 SANS toucher a la position de repos.
    #
    # pinch_y/squeeze_y (ci-dessous) : la main est une SPHERE de collision de rayon
    # 0.05m (collision_left/right_elbow_end, centree sur HAND_OFFSET). demi-largeur
    # carton = 0.0955 => la sphere touche la surface a |Y|=0.0955+0.05=0.1455.
    parser.add_argument("--pinch-x", type=float, default=0.30, help="X de la position de pince (m, repere torse)")
    parser.add_argument("--pas-avant", type=float, default=0.1425,
                         help="Pas en avant (m, marcher()) juste avant de viser -- ramene le "
                              "torse assez pres du carton sans que la position de repos ne "
                              "viole la marge corps-podium reelle. 0 pour desactiver.")
    parser.add_argument("--pinch-y", type=float, default=0.1655,
                         help="|Y| de la position de pince avant serrage -- doit rester > 0.1455 "
                              "(demi-largeur carton 0.0955 + rayon sphere main 0.05) pour ne pas "
                              "toucher des l'approche")
    parser.add_argument("--pinch-z", type=float, default=-0.1392, help="Z de la position de pince")
    parser.add_argument("--flechir-genoux", action="store_true",
                         help="Flechit hanche/genou/cheville (posture walk_stance) avant de "
                              "bouger les bras -- voir PM01LiftSim.flechir_genoux().")
    parser.add_argument("--aim-only", action="store_true",
                         help="Amene les bras a la position de pince (ecartes, vise chaque face "
                              "du carton) SANS serrer ni lever -- voir PM01LiftSim.viser().")
    parser.add_argument("--squeeze-y", type=float, default=0.1405,
                         help="|Y| apres serrage -- juste sous 0.1455 (contact leger sphere/carton, "
                              "~5mm de pression) ; un serrage bien plus fort ejecte le carton")
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

    if args.pas_avant:
        print(f"[INFO] Avance directe ({args.pas_avant}m, sans animation de pas -- "
              "marcher() donne un effet de 'vol' artificiel, torse fige dans cet outil)...",
              flush=True)
        sim.base_pose[0] += args.pas_avant

    if args.flechir_genoux:
        print("[INFO] Flexion hanche/genou/cheville (walk_stance)...", flush=True)
        sim.flechir_genoux()

    if args.aim_only:
        print(f"[INFO] Visee -- bras vers x={args.pinch_x} y=+-{args.pinch_y} z={args.pinch_z} "
              "(pas de serrage/levee).", flush=True)
        sim.viser(args.pinch_x, args.pinch_y, args.pinch_z, hold_seconds=max(args.hold_seconds, 0.0))
        if args.hold_seconds < 0:
            print("[INFO] Fenetre ouverte -- ferme-la pour quitter.", flush=True)
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
