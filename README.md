# Vegetarian Vampire

This project implements a solver for the [Vegetarian Vampire](http://www.qcfdesign.com/wiki/DesktopDungeons/index.php?title=Vegetarian_Vampire) puzzle sub from the game [Desktop Dungeons](http://www.desktopdungeons.net)

## Getting Started

Download files veg_vampire.py and run_this.py. For now you need to modify run_this.py to suit your puzzle.

## Limitations

Currently only regular plant cutting in melee is supported. The following are **not supported**:
- ENDISWAL
- PISORFing the vampire
- WEYTWUT
- WONAFYT
- pushing the plants as a half-dragon or as a hero with knockback and consecrated strike
- transmutation seal


### Prerequisites

Python3 and numpy.

## Running the tests

I have implemented some tests for the code. They can be run as follows:

python3 test_Puzzle.py -v
python3 test_Solution_and_Set.py -v
python3 test_veg_vampire.py -v

## License
[GNU Affero General Public License](https://www.gnu.org/licenses/agpl.html)

