# examples/ring_propagation.py
"""
Example 3: 4-agent ring — knowledge diffuses hop by hop.

4 agents arranged in a directed ring: A → B → C → D → A.
Each agent sends its knowledge to the next agent in the ring.

Round 1 of sharing: each agent absorbs from its predecessor.
Round 2 of sharing: knowledge has traveled 2 hops (C now has A's knowledge, etc.)

Truth: f[2] AND f[9] (same as Example 1).

Agent assignment (each agent has 1/4 of the feature space clean):
  A: f[0-2] clean, f[3-11] noisy  → sees relevant f[2] cleanly
  B: f[3-5] clean, rest noisy     → sees neither relevant feature cleanly
  C: f[6-8] clean, rest noisy     → sees neither relevant feature cleanly  
  D: f[9-11] clean, rest noisy    → sees relevant f[9] cleanly

A and D can each learn one half of the AND alone.
B and C are nearly helpless alone.
After ring propagation:
  Round 1: B gets A's knowledge, C gets B's (partial), D gets C's (partial)
  Round 2: C gets B's improved knowledge, D gets C's, A gets D's
  Knowledge converges across all 4 agents within 2-3 rounds.
"""

import sys, numpy as np
sys.path.insert(0, "..")

from tm_collective import WorldSchema, TMNode, Collective
from tm_collective.strategies import SyntheticDataStrategy, RingTopology, FixedRoundTrigger
from tm_collective import evaluation

SEED = 42
N_FEATURES = 12
N_ROUNDS = 12
SHARE_EVERY = 4
N_TEST = 2000

np.random.seed(SEED)

schema = WorldSchema.from_dict({
    "version": 1,
    "description": "Ring propagation: 4 agents",
    "features": [{"id": i, "name": f"f{i}", "encoder": "boolean"} for i in range(N_FEATURES)]
})

np.random.seed(SEED + 9999)
X_test = np.random.randint(0, 2, (N_TEST, N_FEATURES)).astype(np.uint32)
y_test = (X_test[:, 2] & X_test[:, 9]).astype(np.uint32)
np.random.seed(SEED)

sharing = SyntheticDataStrategy(
    n_synthetic=500,
    retrain_epochs=200,
    mode="perturb",
    rate_mode="graduated",
)

nodes = {
    "A": TMNode("A", schema, noisy_features=[f"f{i}" for i in range(3,  12)],
                noise_rate=0.45, sharing=sharing, epochs_per_round=50),
    "B": TMNode("B", schema, noisy_features=[f"f{i}" for i in list(range(0,3))+list(range(6,12))],
                noise_rate=0.45, sharing=sharing, epochs_per_round=50),
    "C": TMNode("C", schema, noisy_features=[f"f{i}" for i in list(range(0,6))+list(range(9,12))],
                noise_rate=0.45, sharing=sharing, epochs_per_round=50),
    "D": TMNode("D", schema, noisy_features=[f"f{i}" for i in range(0, 9)],
                noise_rate=0.45, sharing=sharing, epochs_per_round=50),
}

collective = Collective(
    schema=schema,
    nodes=nodes,
    topology=RingTopology(),        # A→B→C→D→A
    trigger=FixedRoundTrigger(every_n_rounds=SHARE_EVERY),
)

print("Example 3: Ring Topology — Knowledge Diffusion\n")
print(f"{'Round':>6}  {'A':>8}  {'B':>8}  {'C':>8}  {'D':>8}")
print("-" * 46)

for r in range(1, N_ROUNDS + 1):
    X_new = np.random.randint(0, 2, (100, N_FEATURES)).astype(np.uint32)
    y_new = (X_new[:, 2] & X_new[:, 9]).astype(np.uint32)
    result = collective.step(X_new, y_new, X_test, y_test)
    acc = result["accuracies"]
    note = "  <<RING SHARE>>" if result["sharing_events"] else ""
    print(f"{r:>6}  {acc['A']:>8.3f}  {acc['B']:>8.3f}  {acc['C']:>8.3f}  {acc['D']:>8.3f}{note}")

summary = collective.summary()
evaluation.print_summary_table(summary)
evaluation.plot_accuracy(
    history=collective._history,
    share_rounds=sorted({e["round"] for e in collective._share_events}),
    title="Ring Topology: Knowledge Diffuses Hop-by-Hop",
    save_path="ring_propagation_plot.png",
    agent_labels={
        "A": "A (f[2] visible)", "B": "B (blind)", 
        "C": "C (blind)",         "D": "D (f[9] visible)"
    },
)
