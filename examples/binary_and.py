# examples/binary_and.py
"""
Example 1: 2-agent binary AND.
The simplest demonstration. Matches the proven MVE results.

Agent A: sees features 0-5 cleanly, features 6-11 with 45% noise.
Agent B: sees features 6-11 cleanly, features 0-5 with 45% noise.
Truth:   feature[2] AND feature[9].

Expected results:
  Pre-share:  Agent A ~0.83, Agent B ~0.86
  Post-share: Agent A ~0.99, Agent B ~0.98
"""

import sys
import numpy as np
sys.path.insert(0, "..")   # find tm_collective from examples/

from tm_collective import WorldSchema, TMNode, Collective
from tm_collective.strategies import SyntheticDataStrategy, AllToAll, OnceOnlyTrigger
from tm_collective import evaluation

SEED = 42
N_FEATURES = 12
N_ROUNDS = 10
OBS_PER_ROUND = 100
SHARE_AT_ROUND = 5
N_TEST = 2000

np.random.seed(SEED)

# ── Schema: 12 boolean features ───────────────────────────────────────────────
schema = WorldSchema.from_dict({
    "version": 1,
    "description": "12 boolean features, truth = f[2] AND f[9]",
    "features": [{"id": i, "name": f"f{i}", "encoder": "boolean"} for i in range(N_FEATURES)]
})

# ── Test set (fixed, clean) ───────────────────────────────────────────────────
np.random.seed(SEED + 9999)
X_test = np.random.randint(0, 2, (N_TEST, N_FEATURES)).astype(np.uint32)
y_test = (X_test[:, 2] & X_test[:, 9]).astype(np.uint32)
np.random.seed(SEED)

# ── Agents ────────────────────────────────────────────────────────────────────
sharing = SyntheticDataStrategy(
    n_synthetic=500,
    retrain_epochs=150,
    mode="perturb",
    rate_mode="graduated",
)

node_a = TMNode(
    "agent_a", schema,
    noisy_features=[f"f{i}" for i in range(6, 12)],
    noise_rate=0.45, n_clauses=80, T=20, s=3.9,
    sharing=sharing, epochs_per_round=50,
)
node_b = TMNode(
    "agent_b", schema,
    noisy_features=[f"f{i}" for i in range(0, 6)],
    noise_rate=0.45, n_clauses=80, T=20, s=3.9,
    sharing=sharing, epochs_per_round=50,
)

# ── Collective ────────────────────────────────────────────────────────────────
collective = Collective(
    schema=schema,
    nodes={"agent_a": node_a, "agent_b": node_b},
    topology=AllToAll(),
    trigger=OnceOnlyTrigger(at_round=SHARE_AT_ROUND),
)

# ── Run ───────────────────────────────────────────────────────────────────────
print("Example 1: 2-agent binary AND\n")
print(f"{'Round':>6}  {'agent_a':>10}  {'agent_b':>10}")
print("-" * 35)

for round_i in range(1, N_ROUNDS + 1):
    X_new = np.random.randint(0, 2, (OBS_PER_ROUND, N_FEATURES)).astype(np.uint32)
    y_new = (X_new[:, 2] & X_new[:, 9]).astype(np.uint32)

    result = collective.step(X_new, y_new, X_test, y_test)
    acc = result["accuracies"]
    note = "  <<< SHARE >>>" if result["sharing_events"] else ""
    print(f"{round_i:>6}  {acc['agent_a']:>10.3f}  {acc['agent_b']:>10.3f}{note}")

# ── Summary ───────────────────────────────────────────────────────────────────
summary = collective.summary()
evaluation.print_summary_table(summary, share_round=SHARE_AT_ROUND)

share_rounds = [e["round"] for e in collective._share_events]
evaluation.plot_accuracy(
    history=collective._history,
    share_rounds=share_rounds,
    title="2-Agent Collective TM: feature[2] AND feature[9]",
    save_path="binary_and_plot.png",
    agent_labels={"agent_a": "Agent A (f0-5 clean)", "agent_b": "Agent B (f6-11 clean)"},
)
