"""Point d'entree : cree le Conductor et lance la sequence."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.chef import Conductor


def main():
    conductor = Conductor(rate_hz=20)

    # Le carton/podium est en (2.2, 0.0) -- on vise un standoff devant, pas le podium
    # lui-meme (voir tools/choreo_native/, meme logique : CARTON_X - standoff).
    # walk_to() est en version simplifiee (boucle ouverte, pousse fixe) -- voir
    # core/chef.py::Conductor.walk_to().
    conductor.run_sequence(target_x=1.7, target_y=0.0, duration=8.0)


if __name__ == "__main__":
    main()
