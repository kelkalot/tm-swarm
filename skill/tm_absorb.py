#!/usr/bin/env python3
# tm_absorb.py
"""
Absorb a knowledge packet from a peer agent.

Usage:
  python tm_absorb.py \\
    --workspace /path/to/workspace \\
    --agent agent_b \\
    --packet '<json_string_from_sessions_send>' \\
    [--epochs 150]

Output (stdout, JSON):
  {"ok": true, "absorbed_from": "agent_a", "total_samples": 1000}
  {"ok": false, "error": "..."}

The --packet value is the exact string from the "packet" field in tm_share.py output.
After absorption, the TM is reset and retrained on all accumulated data.
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
    parser.add_argument("--packet",    required=True, help="JSON string from tm_share.py output")
    parser.add_argument("--epochs",    type=int, default=150)
    args = parser.parse_args()

    # Load world schema
    schema = tm_lib.load_schema(args.workspace)
    n_features = tm_lib.n_binary_features(schema)

    try:
        sender_id, X_peer, y_peer, meta = tm_lib.unpack_packet(args.packet, n_features)
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        print(tm_lib.result_json(False, error=f"Failed to parse packet: {e}"))
        sys.exit(1)

    # Add peer's data to this agent's buffer
    X_buf, y_buf = tm_lib.load_buffer(args.workspace, args.agent)
    X_buf += X_peer.tolist()
    y_buf += y_peer.tolist()
    tm_lib.save_buffer(args.workspace, args.agent, X_buf, y_buf)

    X_all = np.array(X_buf, dtype=np.uint32)
    y_all = np.array(y_buf, dtype=np.uint32)

    # Reset TM and retrain from scratch on all data
    # (Reset ensures imported knowledge is fully integrated, not skewed by prior training)
    from pyTsetlinMachine.tm import MultiClassTsetlinMachine
    tm = MultiClassTsetlinMachine(
        tm_lib.TM_N_CLAUSES, tm_lib.TM_T, tm_lib.TM_S,
        boost_true_positive_feedback=1,
    )
    tm.fit(X_all, y_all, epochs=args.epochs)
    tm_lib.save_tm(tm, args.workspace, args.agent)

    print(tm_lib.result_json(
        True,
        absorbed_from=sender_id,
        total_samples=len(X_all),
        peer_samples_added=len(X_peer),
    ))


if __name__ == "__main__":
    main()
