"""Construit une scene MuJoCo TEMPORAIRE = la vraie scene (pm01_edu_carton.xml,
meme carton a pos="2.2 0 0.796", meme podium) + une camera ajoutee sur LINK_BASE
(bassin) -- pour le node "camera fantome" pelvis_camera_sim.py, qui rend cette
scene hors-ligne au lieu de recevoir une vraie image camera (aucune camera
n'existe dans la sim reelle, voir memoire projet -- ce module sert a en
simuler une SANS toucher au binaire C++ de run_mujoco.sh).

Ne touche a AUCUN fichier du depot -- meme technique que build_dynamic_scene()
dans tools/robot_arm_ik/lift_carton.py (patch texte + fichiers temporaires).

Limite assumee : la position du carton n'est PAS lue depuis la sim reelle
(aucun canal LCM ne l'expose, voir memoire) -- ce module suppose le carton
a sa position INITIALE du XML (2.2, 0, 0.796), valide tant qu'il n'a pas ete
saisi/deplace par la prise (cas d'usage : detecter + se placer AVANT la prise).

Ne dessine PAS le marqueur ArUco en tant que texture MuJoCo -- teste
empiriquement (session du 18/08) : mapping texture "plane"/"box" incoherent
sur cette version de MuJoCo (motif source 1680x1680 ecrase en pave 2x2, memes
avec texuniform/lighting corriges, indetectable par cv2.aruco). Le marqueur
est a la place COMPOSITE en 2D directement sur l'image rendue (voir
pelvis_camera_sim.py::_composite_marker -- cv2.warpPerspective des 4 coins 3D
connus du marqueur, projetes avec les intrinseques/extrinseques de la camera)
-- pixel-parfait, aucune dependance au moteur de rendu MuJoCo.
"""
import os

NATIVE_SDK_DIR = "/home/equansrobotic/engineai_robotics_native_sdk"
SCENE_XML = os.path.join(NATIVE_SDK_DIR, "assets/resource/pm01_edu_carton.xml")
ROBOT_XML = os.path.join(
    NATIVE_SDK_DIR, "assets/resource/robot/pm01_edu/xml/serial_pm01_edu_carton_liftable.xml"
)

CAMERA_NAME = "pelvis_cam"
# Legerement devant et au-dessus du centre du bassin (LINK_BASE, origine locale).
# xyaxes deduit pour regarder vers l'avant du robot (+X local, meme convention
# que pinch_x dans lift_carton.py) : une camera MuJoCo regarde le long de son
# axe -Z local. local X = (0,-1,0), local Y = (0,0,1) => local Z = X x Y =
# (-1,0,0) => la camera regarde vers -local_Z = (+1,0,0) = +X. "up" = local Y
# = +Z monde, donc l'image n'est pas la tete en bas.
CAMERA_POS = "0.08 0 0.05"
CAMERA_XYAXES = "0 -1 0 0 0 1"
CAMERA_FOVY = 60.0

# Camera EXTERIEURE fixe (20/08, demande utilisateur "vue exterieure, on voit
# le robot bien") -- placee dans <worldbody> DIRECTEMENT, PAS dans LINK_BASE :
# ne bouge PAS avec le robot, contrairement a CAMERA_NAME ci-dessus. Position
# calculee pour voir tout le trajet de marche (spawn x~0 -> carton x=2.2, y=0)
# de cote : decalee en -Y, regarde vers +Y. xyaxes derive de la meme
# convention que CAMERA_XYAXES (camera MuJoCo regarde le long de -local_Z) :
# local_X=(1,0,0) (droite image = +X monde), local_Y=(0,0,1) (haut image =
# +Z monde) => local_Z = X x Y = (0,-1,0) => regarde vers -local_Z = (0,1,0)
# = +Y monde, cote robot/carton (y=0).
EXTERNAL_CAMERA_NAME = "spectator_cam"
EXTERNAL_CAMERA_POS = "1.1 -3.5 1.3"
EXTERNAL_CAMERA_XYAXES = "1 0 0 0 0 1"
EXTERNAL_CAMERA_FOVY = 55.0

# Marqueur ArUco sur la face avant du carton (perpendiculaire a X, cote robot
# qui approche par x<2.2 -- voir PLAN_TEST_ARUCO_CARTON.txt etape 2). Taille
# THEORIQUE du PNG genere (101.6mm), pas une mesure d'impression reelle
# (rien n'est imprime en sim) -- utiliser --marker-size 0.1016 avec
# aruco_carton_test.py contre le topic de ce node.
MARKER_PNG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "aruco_carton_marker_DICT_4X4_50_id0.png")
MARKER_SIZE = 0.1016
CARTON_HALF_X = 0.1375  # demi-profondeur du carton (voir carton_box size dans pm01_edu_carton.xml)
CARTON_POS = (2.2, 0.0, 0.796)  # voir <body name="carton" pos="..."> dans pm01_edu_carton.xml

# PLAN_TEST_ARUCO_CARTON.txt etape 1 : le PNG genere (1680x1680px) a une marge
# blanche autour du motif -- feuille totale 142.2mm pour un motif de 101.6mm
# (1680*101.6/142.2 = 1200px de motif, marge de 240px de chaque cote). A
# utiliser pour ROGNER le PNG avant de le composer/mesurer -- coller le PNG
# ENTIER (avec sa marge) sur un carre de cote MARKER_SIZE fait apparaitre le
# motif plus PETIT que MARKER_SIZE dans le rendu, ce qui fausse la distance
# calculee par solvePnP d'un facteur ~1.4 (verifie empiriquement 2026-08-19 --
# a ne pas refaire).
MARKER_SHEET_MM = 142.2
MARKER_PATTERN_MM = 101.6


def load_marker_pattern_bgr(path=MARKER_PNG):
    """Charge le PNG et le rogne a la zone motif SEULE (sans la marge
    blanche) -- le resultat correspond exactement a MARKER_SIZE metres de
    cote, pret a etre compose/mesure sans facteur d'echelle correctif."""
    import cv2
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise RuntimeError(f"Impossible de charger {path}")
    h, w = gray.shape
    frac = MARKER_PATTERN_MM / MARKER_SHEET_MM
    margin_x = round(w * (1.0 - frac) / 2.0)
    margin_y = round(h * (1.0 - frac) / 2.0)
    cropped = gray[margin_y:h - margin_y, margin_x:w - margin_x]
    return cv2.cvtColor(cropped, cv2.COLOR_GRAY2BGR)


def build_shadow_scene(tmpdir: str) -> str:
    """Ecrit la scene patchee dans tmpdir, retourne son chemin (a charger avec
    mujoco.MjModel.from_xml_path)."""
    robot_dir = os.path.dirname(ROBOT_XML)
    robot_src = open(ROBOT_XML).read()
    # serial_pm01_edu_carton_liftable.xml a des <include file="..."/> relatifs
    # (assets.xml, serial_links.xml, serial_actuators.xml, serial_sensors.xml),
    # resolus a l'origine par rapport a robot_dir -- une fois copie dans tmpdir,
    # il faut les rendre absolus (meme piege/fix que build_dynamic_scene() dans
    # tools/robot_arm_ik/lift_carton.py).
    for fname in ["assets.xml", "serial_links.xml", "serial_actuators.xml", "serial_sensors.xml"]:
        robot_src = robot_src.replace(f'file="{fname}"', f'file="{os.path.join(robot_dir, fname)}"')
    camera_tag = (
        f'<camera name="{CAMERA_NAME}" pos="{CAMERA_POS}" xyaxes="{CAMERA_XYAXES}" '
        f'fovy="{CAMERA_FOVY}"/>'
    )
    # Marqueur unique juste avant <freejoint/> dans <body name="LINK_BASE">
    # (voir serial_pm01_edu_carton_liftable.xml) -- pas besoin de regex, cette
    # ligne est stable (site IMU, deja utilise ailleurs dans le repo).
    marker = '<site name="imu" pos="0.02311 0 -0.09631"/>'
    if marker not in robot_src:
        raise RuntimeError(
            f"Marqueur d'insertion introuvable dans {ROBOT_XML} -- fichier modifie depuis "
            "l'ecriture de shadow_scene.py ? Mettre a jour `marker` ci-dessus."
        )
    patched_robot = robot_src.replace(marker, marker + "\n            " + camera_tag)
    patched_robot_path = os.path.join(tmpdir, "serial_pm01_edu_carton_liftable_cam.xml")
    open(patched_robot_path, "w").write(patched_robot)

    scene_src = open(SCENE_XML).read()
    patched_scene = scene_src.replace(f'file="{ROBOT_XML}"', f'file="{patched_robot_path}"')
    if patched_scene == scene_src:
        raise RuntimeError(
            f"Include de {ROBOT_XML} introuvable dans {SCENE_XML} -- fichier modifie depuis "
            "l'ecriture de shadow_scene.py ?"
        )

    # Camera exterieure ajoutee directement sous <worldbody> (pas dans un
    # <body> mobile) -- reste fixe dans le monde, ne suit PAS le robot.
    external_camera_tag = (
        f'<camera name="{EXTERNAL_CAMERA_NAME}" pos="{EXTERNAL_CAMERA_POS}" '
        f'xyaxes="{EXTERNAL_CAMERA_XYAXES}" fovy="{EXTERNAL_CAMERA_FOVY}"/>'
    )
    patched_scene = patched_scene.replace("<worldbody>", "<worldbody>\n        " + external_camera_tag, 1)

    patched_scene_path = os.path.join(tmpdir, "pm01_edu_carton_cam.xml")
    open(patched_scene_path, "w").write(patched_scene)
    return patched_scene_path
