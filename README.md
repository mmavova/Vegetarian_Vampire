# Vegetarian Vampire

This project implements a solver for the [Vegetarian Vampire](http://www.qcfdesign.com/wiki/DesktopDungeons/index.php?title=Vegetarian_Vampire) puzzle sub from the game [Desktop Dungeons](http://www.desktopdungeons.net).

## Limitations

Currently only regular plant cutting in melee is supported. The following moves are **not supported** at the moment:
- ENDISWAL
- PISORFing the vampire
- WEYTWUT
- WONAFYT
- pushing the plants as a half-dragon or as a hero with knockback and consecrated strike
- transmutation seal

The following moves are possible or might be possible, but are also not supported:
- technically, you can also endiswal behind the vampire, and then move him via knockback and finally break a wall this way,
  but we do not have to consider this, since you already have ENDISWAL.
- patches the teddy technically could teleport you god knows where (kill the vampire or use Yendor/Cracked Amulet), but it is probably unimportant.
- Wicked Guitar can make the Vampire more durable. Wand of Binding can help to make the Vampire even stronger.
- I assume Mass09 Ledger is useless in this sub. Did not really try.
- Titan Guitar also can be used to move the Vampire, I suppose.
- I think you cannot get wall crunchers into this sub.

## Getting Started

Download files veg_vampire.py and run_this.py. For now you need to modify run_this.py to suit your puzzle.

### Prerequisites

Python3 and numpy.

## Running the tests

I have implemented some tests for the code. They can be run as follows:

```
python3 test_Puzzle.py -v
python3 test_Solution_and_Set.py -v
python3 test_veg_vampire.py -v
```

## License
[GNU Affero General Public License](https://www.gnu.org/licenses/agpl.html)

