"""Marche PRECALIBREE (boucle ouverte, comme marche.py) sur le robot REEL,
mais dont le NOMBRE DE REPETITIONS est calcule a partir d'une mesure camera ArUco --
PAS une correction en continu pendant la marche.

Pourquoi cette conception precise (2026-08-31) : on a teste aujourd'hui, en simulation,
une marche corrigee en boucle FERMEE (walk_to_xy/approach_carton dans chef_node.py,
vision reevaluee entre chaque petit pas) -- ca n'a PAS converge de facon fiable (x
reste bloque ~0.70-0.71m sur 3 tentatives, les pulses courts de correction repartent
parfois en arriere d'un pulse a l'autre). Plutot que de reproduire ce meme risque sur
le robot reel, ce script ne fait qu'UNE SEULE mesure camera AVANT de marcher (pour
calculer combien de repetitions faire), exactement comme on mesurait au ruban a la
main vendredi -- la marche elle-meme reste entierement aveugle/en boucle ouverte,
methode deja validee. Une 2e mesure camera est prise APRES la marche, uniquement pour
verifier/rapporter le resultat -- jamais pour corriger en cours de route.

AUCUNE CORRECTION LATERALE (turn) calculee automatiquement ici -- si le robot devie,
cf. --turn de marche.py, calibre separement et manuellement (cf.
COMMENT_LANCER_ROBOT_REEL.txt).

Calibrage CM_PER_REP (2026-08-28/31, robot reel, forward=0.45 duree=3.0) : 6
repetitions ont couvert 188cm au total pour atteindre le podium depuis la position de
depart -- soit ~31.3cm/repetition en moyenne. C'est une MOYENNE mesuree sur CE
parcours precis, pas une constante physique garantie -- attendre une variabilite
reelle (le rythme mesure ce jour-la variait deja entre 37cm et 42cm par repetition
individuelle a duration=2.0, avant l'anomalie qui a force le passage a duration=3.0).
A recalibrer si --forward/--duration changent.

AUCUNE SIMULATION -- envoie de vraies commandes de marche au robot, calculees a partir
d'une vraie mesure camera. NE JAMAIS lancer sans validation explicite prealable, meme
regles de securite que marche.py (harnais, espace degage, telecommande
"passive" prete) -- PLUS le fait que ce script n'a jamais ete teste sur le robot reel
(ecrit le 2026-08-31 pendant que le robot etait a court de batterie) : TOUJOURS
commencer par --dry-run, puis un lancement avec le nombre de repetitions calcule mais
en verifiant/interrompant a chaque checkpoint (jamais --no-confirm au premier essai).
"""
import argparse
import math
import os
import statistics
import sys
import time

import rclpy

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import marche as walk_mod  # noqa: E402

# aruco_carton_test.py : cherche d'abord a cote de CE script (deploiement Nezha -- a
# copier ici, meme dossier que arm_test/), sinon retombe sur son emplacement d'origine
# sur le PC de dev (tools/virtual_gamepad/sequence_carton_export/vision/) -- evite de
# dupliquer le fichier tout en marchant dans les deux environnements sans modification.
sys.path.insert(0, _HERE)
sys.path.insert(0, "/home/equansrobotic/stagiaire_1/tools/virtual_gamepad/sequence_carton_export/vision")
from aruco_carton_test import ArucoCartonLocator, DEFAULT_DICT, DEFAULT_IMAGE_TOPIC, DEFAULT_MARKER_ID  # noqa: E402

# Calibre le 2026-08-28/31 sur CE robot, forward=0.45 duration=3.0 -- voir docstring
# module pour le detail et la mise en garde sur la variabilite reelle.
CM_PER_REP_DEFAULT = 0.313
MEASURE_WINDOW_S = 5.0        # duree de collecte des mesures cameras (avant et apres marche)
MIN_READINGS = 3              # nombre minimum de mesures valides pour faire confiance a la moyenne


def _checkpoint(message, confirm):
    print(f"[ETAPE] {message}", flush=True)
    if confirm:
        input("        Verifie le robot, puis Entree pour continuer (Ctrl+C pour arreter)... ")


def _measure_distance(args, window_s=MEASURE_WINDOW_S):
    """Cree un ArucoCartonLocator temporaire, collecte les mesures pendant `window_s`
    secondes, le detruit, renvoie la mediane des mesures valides (plus robuste qu'une
    moyenne face a une detection ponctuelle aberrante) ou None si pas assez de mesures."""
    locator_args = argparse.Namespace(
        image_topic=args.image_topic, camera_info_topic=args.camera_info_topic,
        marker_id=args.marker_id, marker_size=args.marker_size, dict=args.dict,
        publish_debug=True,
    )
    if locator_args.camera_info_topic is None:
        locator_args.camera_info_topic = locator_args.image_topic.replace("image_raw", "camera_info")

    node = ArucoCartonLocator(locator_args)
    readings = []
    t0 = time.time()
    last_seen_time = None
    while time.time() - t0 < window_s:
        rclpy.spin_once(node, timeout_sec=0.2)
        if node.last_distance is not None and node.last_distance_time != last_seen_time:
            last_seen_time = node.last_distance_time
            readings.append(node.last_distance)
    node.destroy_node()

    print(f"[MESURE] {len(readings)} lectures valides sur {window_s:.1f}s "
          f"(min={min(readings):.3f}m max={max(readings):.3f}m)" if readings else
          "[MESURE] aucune lecture valide.", flush=True)

    if len(readings) < MIN_READINGS:
        return None
    return statistics.median(readings)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target-distance", type=float, required=True,
                         help="Distance camera->marqueur visee APRES la marche, en metres. "
                              "A calibrer une premiere fois en comparant a une mesure au "
                              "double-decimetre reelle (meme logique que PINCH_X=0.216 "
                              "calibre pour 14.6cm bassin->podium) -- ne pas deviner.")
    parser.add_argument("--cm-per-rep", type=float, default=CM_PER_REP_DEFAULT,
                         help=f"Distance parcourue par repetition de marche, en metres "
                              f"(defaut {CM_PER_REP_DEFAULT}m, calibre le 2026-08-28/31 -- "
                              "voir docstring module).")
    parser.add_argument("--forward", type=float, default=walk_mod.DEFAULT_FORWARD_MPS)
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--turn", type=float, default=0.0,
                         help="Correction laterale constante, calibree separement -- "
                              "PAS calculee automatiquement ici (cf. docstring module).")
    parser.add_argument("--max-reps", type=int, default=10,
                         help="Plafond de securite -- refuse d'executer un plan qui demande "
                              "plus de repetitions que ca (probable erreur de mesure/calibrage).")
    parser.add_argument("--image-topic", default=DEFAULT_IMAGE_TOPIC)
    parser.add_argument("--camera-info-topic", default=None)
    parser.add_argument("--marker-id", type=int, default=DEFAULT_MARKER_ID)
    parser.add_argument("--marker-size", type=float, required=True)
    parser.add_argument("--dict", type=int, default=DEFAULT_DICT)
    parser.add_argument("--dry-run", action="store_true",
                         help="Mesure et calcule le plan, n'envoie AUCUNE commande de marche.")
    parser.add_argument("--no-confirm", action="store_true",
                         help="Pas de pause interactive entre repetitions -- jamais au premier essai.")
    args = parser.parse_args()

    rclpy.init()

    try:
        print("[ETAPE] Mesure de la distance actuelle (camera)...", flush=True)
        current = _measure_distance(args)
        if current is None:
            print("[ERREUR] Pas assez de detections fiables -- verifie que le marqueur est "
                  "visible, bien eclaire, et que le node camera + camera_info tournent. Arret.",
                  flush=True)
            return

        print(f"[INFO] Distance actuelle mesuree : {current:.3f}m -- cible : "
              f"{args.target_distance:.3f}m", flush=True)

        remaining = current - args.target_distance
        if remaining <= 0:
            print(f"[INFO] Deja a la distance cible ou plus pres ({current:.3f}m <= "
                  f"{args.target_distance:.3f}m) -- aucune marche necessaire.", flush=True)
            return

        reps_needed = math.ceil(remaining / args.cm_per_rep)
        print(f"[INFO] Distance restante : {remaining:.3f}m -- {reps_needed} repetition(s) "
              f"a {args.cm_per_rep:.3f}m/rep (forward={args.forward} duree={args.duration}s)",
              flush=True)

        if reps_needed > args.max_reps:
            print(f"[ERREUR] {reps_needed} repetitions depasse le plafond de securite "
                  f"--max-reps={args.max_reps} -- probable erreur de mesure/calibrage. "
                  "Arret, rien envoye au robot.", flush=True)
            return

        if args.dry_run:
            print(f"[dry-run] Plan : {reps_needed} x marche.marcher("
                  f"forward={args.forward}, turn={args.turn}, duration={args.duration}). "
                  "Rien envoye.", flush=True)
            return

        node = rclpy.create_node("walk_to_camera_distance_real")
        try:
            for i in range(reps_needed):
                _checkpoint(
                    f"repetition {i + 1}/{reps_needed} -- forward={args.forward}m/s "
                    f"turn={args.turn} duree={args.duration}s.",
                    not args.no_confirm,
                )
                ok = walk_mod.marcher(node, args.forward, 0.0, args.turn, args.duration,
                                       dry_run=False)
                if not ok:
                    print(f"[ERREUR] echec de la marche (repetition {i + 1}/{reps_needed}) "
                          "-- arret.", flush=True)
                    return
        finally:
            node.destroy_node()

        print("[ETAPE] Mesure de la distance finale (camera)...", flush=True)
        final = _measure_distance(args)
        if final is None:
            print("[AVERTISSEMENT] Marche terminee mais impossible de remesurer la "
                  "distance finale (marqueur non detecte apres la marche -- verifier "
                  "manuellement).", flush=True)
        else:
            print(f"[INFO] Distance finale mesuree : {final:.3f}m (cible etait "
                  f"{args.target_distance:.3f}m, ecart={final - args.target_distance:+.3f}m)",
                  flush=True)
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
