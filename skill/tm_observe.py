#!/usr/bin/env python3
# tm_observe.py
"""
Feed a batch of observations to this agent's Tsetlin Machine.

Usage:
  python tm_observe.py \\
    --workspace /path/to/workspace \\
    --agent agent_a \\
    --X '[[1,0,1,...],[0,1,0,...]]' \\
    --y '[0,1,0,...]' \\
    [--epochs 50] \\
    [--noise-cols '6,7,8,9,10,11'] \\
    [--noise-rate 0.45]

Output (stdout, JSON):
  {"ok": true, "n_samples": 500, "new_in_batch": 2}
  {"ok": false, "error": "..."}

Notes:
  --X and --y are JSON arrays (list of lists, and list of ints).
  --noise-cols is a comma-separated list of column indices to apply noise to.
  If --noise-cols is given, noise is applied BEFORE adding to the buffer.
  This is how each agent simulates seeing only its own sensor zone clearly.
"""

import argparse
import json
import sys
import numpy as np

try:
    import tm_lib
except ImportError:
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    import tm_lib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--agent",     required=True)
    parser.add_argument("--X",         required=True, help="JSON array of feature vectors")
    parser.add_argument("--y",         required=True, help="JSON array of labels")
    parser.add_argument("--epochs",    type=int, default=50)
    parser.add_argument("--noise-cols", default="", help="Comma-separated column indices for noise")
    parser.add_argument("--noise-rate", type=float, default=0.45)
    args = parser.parse_args()

    try:
        X_new = np.array(json.loads(args.X), dtype=np.uint32)
        y_new = np.array(json.loads(args.y), dtype=np.uint32)
    except (json.JSONDecodeError, ValueError) as e:
        print(tm_lib.result_json(False, error=f"Failed to parse --X or --y: {e}"))
        sys.exit(1)

    # Apply noise if specified
    if args.noise_cols:
        noisy_cols = [int(c.strip()) for c in args.noise_cols.split(",") if c.strip()]
        X_new = tm_lib.apply_noise(X_new, noisy_cols, args.noise_rate)

    # Load existing buffer and append
    X_buf, y_buf = tm_lib.load_buffer(args.workspace, args.agent)
    X_buf += X_new.tolist()
    y_buf += y_new.tolist()
    tm_lib.save_buffer(args.workspace, args.agent, X_buf, y_buf)

    X_all = np.array(X_buf, dtype=np.uint32)
    y_all = np.array(y_buf, dtype=np.uint32)
    n_features = X_all.shape[1]

    # Load or create TM and retrain
    tm, _ = tm_lib.load_or_create_tm(args.workspace, args.agent, n_features)
    tm.fit(X_all, y_all, epochs=args.epochs)
    tm_lib.save_tm(tm, args.workspace, args.agent)

    print(tm_lib.result_json(True, n_samples=len(X_all), new_in_batch=len(X_new)))


if __name__ == "__main__":
    main()
