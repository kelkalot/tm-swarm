#!/usr/bin/env python3
# tm_status.py
"""
Report this agent's current TM status.

Usage:
  python tm_status.py --workspace /path/to/workspace --agent agent_a

Output (stdout, JSON):
  {"ok": true, "agent": "agent_a", "fitted": true, "n_train_samples": 500,
   "state_file_bytes": 11382, "buffer_file_bytes": 42000}
"""

import argparse
import sys
import os

try:
    import tm_lib
except ImportError:
    sys.path.insert(0, os.path.dirname(__file__))
    import tm_lib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--agent",     required=True)
    args = parser.parse_args()

    X_buf, y_buf = tm_lib.load_buffer(args.workspace, args.agent)
    state_file = tm_lib.state_path(args.workspace, args.agent)
    buf_file   = tm_lib.buffer_path(args.workspace, args.agent)
    fitted = os.path.exists(state_file)

    print(tm_lib.result_json(
        True,
        agent=args.agent,
        fitted=fitted,
        n_train_samples=len(X_buf),
        state_file_exists=fitted,
        state_file_bytes=os.path.getsize(state_file) if fitted else 0,
        buffer_file_bytes=os.path.getsize(buf_file) if os.path.exists(buf_file) else 0,
    ))


if __name__ == "__main__":
    main()
