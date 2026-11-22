#!/usr/bin/env python3
"""Apply the approved POST-COEF evidence-only rule to immutable Git commits."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from core070_postfit_p1_receipts import postcoef_ref_compatible

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('repo', type=Path)
parser.add_argument('source')
parser.add_argument('head')
args = parser.parse_args()
if postcoef_ref_compatible(args.repo, args.source, args.head):
    print('POSTCOEF_REF_COMPATIBLE')
else:
    print('POSTCOEF_REF_INCOMPATIBLE')
    sys.exit(1)
