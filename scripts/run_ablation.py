"""Run a controlled temporal-only TFT vs spatial-GNN+TFT comparison."""
import argparse, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--max-zones', type=int, default=None)
args = parser.parse_args()
base = [sys.executable, '-m', 'src.training.train', '--epochs', str(args.epochs)]
if args.max_zones is not None:
    base += ['--max-zones', str(args.max_zones)]
for extra in [
    ['--no-spatial', '--run-name', 'tft_no_spatial'],
    ['--run-name', 'st_tft'],
]:
    cmd = base + extra
    print('\n>>>', ' '.join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)
print('\nCompare results/metrics/tft_no_spatial_test_metrics.json and st_tft_test_metrics.json')
