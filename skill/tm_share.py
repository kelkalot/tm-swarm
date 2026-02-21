#!/usr/bin/env python3
# tm_share.py
"""
Generate a knowledge packet from this agent's TM.
The output JSON should be sent to peers via sessions_send.

Usage:
  python tm_share.py \\
    --workspace /path/to/workspace \\
    --agent agent_a \\
    [--n 500]

Output (stdout, JSON):
  {"ok": true, "packet": "<json_string_to_send_via_sessions_send>", "packet_bytes": 8537}
  {"ok": false, "error": "not fitted yet"}

The "packet" field is a compact JSON string (~8KB for 500 samples).
Send it as-is via sessions_send. The receiving agent passes it to tm_absorb.py.

Example sessions_send message:
  "tm_knowledge_packet:" + result["packet"]
"""

import argparse
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
    parser.add_argument("--n",         type=int, default=500, help="Number of synthetic samples")
    args = parser.parse_args()

    # Load world schema to know n_features
    schema = tm_lib.load_schema(args.workspace)
    n_features = tm_lib.n_binary_features(schema)

    tm, fitted = tm_lib.load_or_create_tm(args.workspace, args.agent, n_features)
    if not fitted:
        print(tm_lib.result_json(False, error="Agent not fitted yet. Call tm_observe first."))
        sys.exit(1)

    # Generate synthetic predictions
    X_rand = np.random.randint(0, 2, (args.n, n_features)).astype(np.uint32)
    y_pred = tm.predict(X_rand).astype(np.uint32)

    # Get current accuracy for metadata (from buffer size as proxy)
    X_buf, _ = tm_lib.load_buffer(args.workspace, args.agent)

    packet_str = tm_lib.pack_packet(
        sender_id=args.agent,
        X=X_rand,
        y=y_pred,
        metadata={"n_train_samples": len(X_buf)},
    )

    print(tm_lib.result_json(
        True,
        packet=packet_str,
        packet_bytes=len(packet_str),
        n_synthetic=args.n,
    ))


if __name__ == "__main__":
    main()
