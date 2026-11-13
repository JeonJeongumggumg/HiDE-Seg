"""Run the unchanged submitted inference entry point with separately supplied weights."""
import subprocess
import sys
from pathlib import Path

if __name__ == '__main__':
    root = Path(__file__).resolve().parent
    if not any(a in ('-h','--help') for a in sys.argv[1:]):
        required = [root/'src/models'/f'fold{fold}'/name for fold in range(5)
                    for name in ('base.pt','refiner.pt','reclassifier.joblib')]
        missing = [str(p.relative_to(root)) for p in required if not p.is_file()]
        if missing:
            print('This release contains code and documentation only. Model weights are not published.\n'
                  'If you have the submitted weights, place them under src/models/fold0..fold4.\n'
                  'See docs/TRAINING.md for retraining and packaging new checkpoints.\n'
                  'Missing: '+', '.join(missing), file=sys.stderr)
            raise SystemExit(2)
    raise SystemExit(subprocess.call([sys.executable, str(root/'src/predict.py'), *sys.argv[1:]]))
