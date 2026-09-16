# Package — séquence bras réelle (viser exclu)

Copie autonome, **simplifiée et sans commentaires**, des fichiers utilisés
pour la séquence lever → pivoter → poser → revenir sur le vrai PM01, telle
que validée le 2026-09-15. Ce dossier est un instantané pour archivage/
partage — les fichiers "source de vérité" (annotés, historique complet des
essais réels) restent dans `tools/joint_angle_commander/` et
`tools/robot_arm_ik/`.

**Simplifications faites ici (vérifiées numériquement identiques, mêmes
angles à 1e-8 près) :**
- Tous les commentaires et l'essentiel des docstrings retirés.
- Les rampes cartésiennes dupliquées (ajustement vertical, approche
  horizontale, levée, pré-relâchement, translation arrière — 5 boucles
  quasi identiques) factorisées en une seule fonction `cartesian_ramp()`
  dans `levee.py`.
- `lift_carton.py` réduit de 828 à ~100 lignes : ne garde que ce qu'utilise
  la séquence bras (chaînes cinématiques, IK, `ease`) — retire tout le code
  simulation-seule (marche, jambes, scène MuJoCo, cymbales, CLI de test).
  **Ce n'est donc plus le même fichier que celui partagé avec la simu**
  (contrairement à la version précédente de ce package) — juste un extrait.

Schéma explicatif (fichiers, nodes, topics) : voir le lien fourni séparément
(la structure des imports n'a pas changé, seul le contenu des fichiers a
été nettoyé).

## Fichiers

| Fichier | Réel / partagé | Rôle |
|---|---|---|
| `levee_pivot.py` | réel uniquement | **Point d'entrée.** Séquence complète en un seul process/Lever : approche → serrage → levée → pivot du buste (carton en main) → dépose → dépivot → retour Q_HOME. |
| `levee.py` | réel uniquement | Constantes géométriques (`PINCH_X`, `PINCH_Y`, `WAYPOINT_Q_LEFT`...) et fonctions partagées (`move_arms`, `cartesian_ramp`, `_publish`...) importées par `levee_pivot.py`. Équivalent simu = `lift.py` (fichier différent). |
| `motion_state.py` | réel uniquement | `ensure_motion_state()` : bascule le robot en `lower_body_balance` via `/motion/set_motion_state`, confirme via `/motion/motion_state`. |
| `pivot_real.py` | réel uniquement | Mécanisme de rotation du buste (gains `WAIST_KP`/`WAIST_KD`, index `WAIST_JOINT_INDEX`) — importé par `levee_pivot.py`. |
| `lever.py` | partagé (fichier identique à la simu) | Classe `Lever` : encapsule `/motion/joint_override_command`, republication continue. `lift.py` côté simu importe exactement ce même fichier. |
| `lift_carton.py` | **extrait**, plus identique à la simu | Géométrie du bras : `solve_arm_ik` (cinématique inverse), `forward_kinematics`, `mirror_left_to_right`, `ease`. Version réduite du fichier partagé — voir note ci-dessus. |

## Lancer

Sur Nezha (192.168.0.163), après avoir sourcé l'environnement ROS2 (voir
`COMMENT_LANCER_ROBOT_REEL.txt` dans `tools/joint_angle_commander/`) :

```bash
python3 levee_pivot.py --no-confirm --pinch-x 0.30 --angle-deg 20
```
