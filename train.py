"""Dispatch standalone final-recipe training stages. No experiments run by default."""
import argparse
import subprocess
import sys
from pathlib import Path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['base', 'cache', 'refiner', 'reclassifier'])
    # Parse only the stage here, so `base --help` reaches the stage parser.
    args = parser.parse_args(sys.argv[1:2])
    rest = sys.argv[2:]
    names = dict(base='train_base.py', cache='cache_predictions.py', refiner='train_refiner.py', reclassifier='train_reclassifier.py')
    raise SystemExit(subprocess.call([sys.executable, str(Path(__file__).parent/'training'/names[args.stage]), *rest]))
