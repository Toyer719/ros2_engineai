"""Publie des commandes brutes sur le canal LCM virtual_gamepad/gamepad_keys.

Indices GamepadKeys.digital_states (voir xbox_gamepad_bridge.py / virtual_gamepad_input_adapter.cc) :
    LB, RB, A, B, X, Y, BACK, START = 0, 1, 2, 3, 4, 5, 6, 7
    CROSS_X_UP, CROSS_X_DOWN, CROSS_Y_LEFT, CROSS_Y_RIGHT = 8, 9, 10, 11

Indices GamepadKeys.analog_states :
    LT, RT, LEFT_STICK_X, LEFT_STICK_Y, RIGHT_STICK_X, RIGHT_STICK_Y = 0, 1, 2, 3, 4, 5

Combinaisons de boutons -> action (source : Guide_PM01_FR.pdf, "Telecommande > Historique
des touches", machine a etats pm01_release.yaml) :
    LB + START        -> Idle / Repos
    LB + RB           -> passive (amorti)
    LB + A            -> PD (pd_stand, debout) -- confirme ce jour via test_lb_a.py
    LB + B            -> marche a demarche humanoide (walk) -- confirme ce jour
    LB + Y            -> marche sur terrain accidente (rl_terrain, cf. marche.py reel)
    LB + croix bas    -> demarche d'equilibre des membres inferieurs (lower_body_balance)
                         [V1.0.12 seulement]
    LB + X            -> marche a demarche mecanique
    RB + B            -> danse
    LB + croix gauche -> service de transmission directe des articulations (dev avance --
                         probablement le mode reellement utilise par lever.py/lift.py, a
                         verifier : distinct de lower_body_balance ci-dessus)
    RB + START        -> service de marche (dev avance)
    START + croix haut -> pd (s'asseoir au sol, PD)
    START + X         -> recuperation depuis position ventrale vers debout
    START + croix bas -> s'asseoir au sol
    LB + BACK         -> mode combat [V1.0.10 seulement]

Chaque combo est un PULSE (appui bref puis relache), pas un maintien -- publier a ~50Hz
en continu tant qu'on veut garder le controle (timeout adaptateur = 200ms sans message,
cf. virtual_gamepad_input_adapter.h::kInputTimeout).
"""
import lcm
from lcm_msgs.data import GamepadKeys

lc = lcm.LCM("udpm://239.255.76.67:7667")
msg = GamepadKeys()

msg.digital_states[LB] = 1
msg.digital_states[B] = 1
msg.digital_states[LB] = 0
msg.digital_states[B] = 0

msg.analog_states[LEFT_STICK_X] = 0.6

# 3. Relacher
msg.analog_states[LEFT_STICK_X] = 0.0
