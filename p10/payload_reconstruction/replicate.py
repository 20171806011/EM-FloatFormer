"""Run exactly the frozen strong_large trial on a new independent seed range."""
import argparse
from pathlib import Path
import sys

import eval_rebuild

# Reassigned on module import so fork and spawn workers use the replication range.
eval_rebuild.SEED_START = 981120040000

if __name__ == '__main__':
    output_option = argparse.ArgumentParser(add_help=False)
    output_option.add_argument('--out', type=Path)
    args, _ = output_option.parse_known_args()
    if args.out is None:
        sys.argv.extend(['--out', 'rebuild_replication.json'])
    elif args.out.name == 'rebuild_raw.json':
        raise ValueError('The replication requires a distinct output file')
    eval_rebuild.main()
