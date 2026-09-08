Sequence PM01 : marche vers le carton, detection vision, levee (pivot buste desactive)
========================================================================================

sim_capture.mp4          -- video vue PREMIERE PERSONNE (camera fantome pelvis, ~45s)
sim_capture_external.mp4 -- video vue EXTERIEURE FIXE (camera spectatrice sur le
                             cote, on voit le robot en entier -- demande utilisateur
                             "vue exterieure, on voit le robot bien"). ATTENTION :
                             sur cette capture le robot est retombe (z=0.19m constate
                             apres coup) quelques secondes apres "lift termine :
                             success=True" dans les logs -- visible normalement vers
                             la fin de la video, pas encore diagnostique/corrige.

Les deux videos capturees le 2026-08-20.

Ce dossier est une COPIE des fichiers au moment de cette capture, pas des liens --
la source vivante reste dans ros_ws/src/ et tools/vision/, a modifier la-bas si besoin.

ros_nodes/  -- noeuds ROS (package virtual_gamepad_ros)
  chef_node.py    orchestrateur : stand -> walk_to -> lift (pivot() dispo mais
                   plus appelee dans run_sequence(), cf commentaire dedans)
  stand.py        Action Stand (transition pd_stand)
  walk_to.py      Action WalkTo (marche boucle ouverte, forward/turn/duration)
  lift.py         Action Lift (flexion genoux + approche/serrage/levee/maintien/relachement)
  pivot.py        Action Pivot (buste seul, PAS le bassin -- desactivee dans chef_node.py)
  field_topics.py mapping boutons/sticks gamepad -> topics

interfaces/ -- definitions des Actions ROS (package virtual_gamepad_interfaces)
  WalkTo.action, Stand.action, Lift.action, Pivot.action

launch/virtual_gamepad.launch.py -- lance walk_to+stand+lift+pivot+camera+chef
  en une commande (voir prerequis dans l'en-tete du fichier : run.sh +
  run_mujoco.sh a lancer A PART, ROS_DOMAIN_ID=69 a exporter explicitement).

vision/ -- pipeline camera fantome + detection ArUco (hors package ROS)
  pelvis_camera_sim.py       rendu MuJoCo offscreen, camera fantome pelvis
  carton_pose_publisher.py   detection ArUco + pose robot-frame -> /vision/carton_pose_base
  shadow_scene.py            assemble la scene MuJoCo temporaire pour le rendu
                              (pelvis_cam ET spectator_cam -- camera exterieure
                              fixe ajoutee le 20/08, cf EXTERNAL_CAMERA_* dedans)
  record_external_view.py    script autonome (pas un node ROS) : enregistre une
                              video MP4 depuis spectator_cam. Usage :
                              python3 record_external_view.py [duree_s] [sortie.mp4]
  aruco_carton_test.py       outil de debug/validation (visualisation Foxglove)
  aruco_carton_marker_*.png  marqueur imprimable colle sur le carton
