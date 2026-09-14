import argparse
import os
import sys

import rclpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import marche as walk_mod
import levee as lift_mod
import levee_pivot as pivot_mod
from lever import Lever


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--forward", type=float, default=walk_mod.DEFAULT_FORWARD_MPS,
                         help="Vitesse avant en VRAIS m/s (BodyVelCmd) -- PAS l'ancienne "
                              "echelle -1..1 du stick manette. Defaut prudent, jamais "
                              "valide sur ce robot avant le 2026-08-28.")
    parser.add_argument("--lateral", type=float, default=0.0, help="Vitesse laterale en m/s.")
    parser.add_argument("--turn", type=float, default=0.0,
                         help="Vitesse de lacet en rad/s (yaw_velocity) -- pas la meme "
                              "echelle/signe que l'ancien --turn du stick manette.")
    parser.add_argument("--duration", type=float, default=1.0, help="Duree de marche en secondes.")
    parser.add_argument("--walk-repeat", type=int, default=1,
                         help="Nombre de fois a repeter la phase marche avant la levee -- "
                              "chaque repetition est un cycle complet independant (pas de "
                              "vitesse cumulee). Calibre le 2026-08-28 pour CE setup precis : "
                              "--forward 0.45 --duration 3.0 --walk-repeat 6 depuis la "
                              "position de depart d'origine ameme le robot au contact du "
                              "podium (~188cm). Voir le docstring du module pour l'anomalie "
                              "duration=2.0 observee ce jour-la. Ne lancer qu'a partir de la "
                              "position de depart, jamais depuis une position deja proche du "
                              "podium.")
    parser.add_argument("--rate-hz", type=float, default=walk_mod.RATE_HZ)
    parser.add_argument("--walk-motion-state", type=str, default=walk_mod.WALK_MOTION_STATE)
    parser.add_argument("--only-phase", choices=["marche", "levee", "pivot"], default=None,
                         help="marche = s'arrete apres le retour en lower_body_balance "
                              "(pour mesurer/valider la distance au podium avant la "
                              "levee). levee = saute la marche (robot deja place a la "
                              "main), enchaine directement levee.py (approche/serrage/levee "
                              "SEULS, pas de pivot/depose). pivot = 2026-09-14, saute la "
                              "marche, enchaine directement levee_pivot.py (sequence COMPLETE "
                              "prise+pivot+depose) -- voir --angle-deg. None (par defaut) = "
                              "marche PUIS levee.py (comportement d'origine) sauf si "
                              "--with-pivot est passe (marche PUIS levee_pivot.py, sequence "
                              "COMPLETE marche+prise+pivot+depose en un seul appel).")
    parser.add_argument("--with-pivot", action="store_true",
                         help="2026-09-14 -- n'a d'effet que si --only-phase n'est PAS "
                              "precise (sequence complete) : utilise levee_pivot.py (prise+"
                              "pivot+depose) au lieu de levee.py (prise seule) apres la "
                              "marche. PREMIERE COMBINAISON marche+pivot+depose EN UN SEUL "
                              "APPEL jamais testee sur ce robot -- valider chaque brique "
                              "separement d'abord (--only-phase marche, puis --only-phase "
                              "pivot seul robot deja en place) avant d'utiliser ce flag, "
                              "jamais au premier essai d'une session.")
    parser.add_argument("--lift-only-phase", choices=["approche", "serrage", "levee"], default=None,
                         help="transmis a levee -- pour valider la levee pas a "
                              "pas apres la marche. Ignore si --with-pivot ou --only-phase "
                              "pivot (levee_pivot.py a son propre --only-phase interne, pas "
                              "expose ici -- lancer levee_pivot.py directement pour tester "
                              "ses phases separement).")
    parser.add_argument("--angle-deg", type=float, default=None,
                         help="2026-09-14, transmis a levee_pivot.py si --only-phase pivot ou "
                              "--with-pivot -- angle de pivot (deg), CARTON EN MAIN. None = "
                              "defaut prudent de levee_pivot.py (20deg) -- NE PAS remonter "
                              "sans validation progressive, voir sa docstring.")
    parser.add_argument("--pinch-z", type=float, default=0.106,
                         help="2026-09-09 : podium remesure a 0.8m, ajuste apres 1er essai reel "
                              "(\"leve les bras un peu trop\" a +0.126) -- voir levee.py pour "
                              "le detail du calcul.")
    parser.add_argument("--pinch-yaw-offset", type=float, default=0.0)
    parser.add_argument("--wrist-rotation-deg", type=float, default=lift_mod.ELBOW_YAW_ROTATION_DEG,
                         help="Transmis a levee -- pivot de l'avant-bras (deg), 0 pour desactiver.")
    parser.add_argument("--walk-stance-scale", type=float, default=lift_mod.WALK_STANCE_SCALE,
                         help="Transmis a levee -- echelle de flexion des genoux, 0 pour desactiver.")
    parser.add_argument("--skip-motion-state", action="store_true",
                         help="Suppose que le robot est deja dans le bon etat pour chaque "
                              "phase (marche: deja en etat de marche ; levee: deja en "
                              "lower_body_balance).")
    parser.add_argument("--dry-run", action="store_true",
                         help="N'envoie rien au robot -- affiche juste ce qui serait fait.")
    parser.add_argument("--confirm", action="store_true",
                         help="Pause interactive (Entree) entre chaque phase -- desactivee par "
                              "defaut depuis l'acceleration du 2026-09-09 (sequence complete "
                              "deja validee en continu le 2026-09-01). A utiliser pour un "
                              "premier essai apres un changement de code.")
    args = parser.parse_args()

    node = None
    if not args.dry_run:
        rclpy.init()
        node = rclpy.create_node("orchestrateur")

    try:
        marche_effectuee = False
        if args.only_phase in (None, "marche"):
            for i in range(args.walk_repeat):
                if args.walk_repeat > 1:
                    print(f"[ETAPE] repetition marche {i + 1}/{args.walk_repeat}", flush=True)
                    walk_mod._checkpoint(
                        f"repetition {i + 1}/{args.walk_repeat} -- forward={args.forward}m/s "
                        f"duree={args.duration}s. Verifie la distance/l'espace restant avant "
                        f"de continuer.",
                        args.confirm and not args.dry_run,
                    )
                ok = walk_mod.marcher(
                    node, args.forward, args.lateral, args.turn, args.duration,
                    rate_hz=args.rate_hz, walk_motion_state=args.walk_motion_state,
                    dry_run=args.dry_run, skip_motion_state=args.skip_motion_state,
                )
                if not ok and not args.dry_run:
                    print(f"[ERREUR] echec de la marche (repetition {i + 1}/{args.walk_repeat}, "
                          "bascule d'etat) -- arret.", flush=True)
                    return
                marche_effectuee = True

        if args.only_phase == "marche":
            print("[INFO] --only-phase marche : arret ici, robot cense etre en "
                  "lower_body_balance. Mesurer au ruban la distance bassin->bord podium, "
                  "puis relancer avec --only-phase levee si elle est proche de 14.6cm "
                  "(PINCH_X=0.216 valide pour cette distance) -- sinon ajuster "
                  "--forward/--duration et recommencer.", flush=True)
            return

        # 2026-09-14 : branche vers levee_pivot.py (prise+pivot+depose) au lieu de
        # levee.py (prise seule) si --only-phase pivot OU --with-pivot (sequence
        # complete). Namespace construit depuis les defauts propres de
        # levee_pivot._build_arg_parser() (angle_deg=20 prudent, pivot_duration=4.0,
        # etc.) -- seuls les champs deja partages avec orchestrateur.py sont
        # ecrases, le reste garde les valeurs par defaut deja validees de
        # levee_pivot.py (ne PAS les redupliquer ici, une seule source de verite).
        use_pivot = args.only_phase == "pivot" or args.with_pivot
        if use_pivot:
            pivot_args = pivot_mod._build_arg_parser().parse_args([])
            pivot_args.pinch_z = args.pinch_z
            pivot_args.pinch_yaw_offset = args.pinch_yaw_offset
            pivot_args.wrist_rotation_deg = args.wrist_rotation_deg
            pivot_args.walk_stance_scale = args.walk_stance_scale
            pivot_args.skip_motion_state = args.skip_motion_state
            pivot_args.dry_run = args.dry_run
            pivot_args.no_confirm = not args.confirm
            pivot_args.only_phase = None  # sequence complete prise+pivot+depose
            if args.angle_deg is not None:
                pivot_args.angle_deg = args.angle_deg
        else:
            lift_args = argparse.Namespace(
                pinch_z=args.pinch_z, pinch_yaw_offset=args.pinch_yaw_offset,
                wrist_rotation_deg=args.wrist_rotation_deg, walk_stance_scale=args.walk_stance_scale,
                skip_motion_state=args.skip_motion_state, only_phase=args.lift_only_phase,
                dry_run=args.dry_run, no_confirm=not args.confirm,
            )

        lever = None
        if not args.dry_run:
            # marche() se termine deja en lower_body_balance quand elle reussit -- ne
            # revalider l'etat que si la marche a ete sautee (only_phase="levee"/"pivot"
            # ou walk_repeat=0), pour eviter un 2e appel bloquant redondant (jusqu'a 1s
            # d'attente d'un message d'etat frais) juste avant de lever les bras.
            if not args.skip_motion_state and not marche_effectuee:
                ok = lift_mod.ensure_motion_state(node, "lower_body_balance",
                                                   timeout=lift_mod.MOTION_STATE_TIMEOUT)
                if not ok:
                    print("[ERREUR] impossible de passer en lower_body_balance avant la "
                          "levee -- arret.", flush=True)
                    return
            lever = Lever(node)

        if use_pivot:
            pivot_mod.run_lift_and_pivot(node, lever, pivot_args)
        else:
            lift_mod.run_lift_sequence(node, lever, lift_args)
    finally:
        if node is not None:
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
