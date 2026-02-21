# examples/sensor_fusion.py
"""
Example 2: Sensor fusion with thermometer encoding.

3 features: temperature (continuous), motion (bool), category (onehot).
Truth: temperature > 30 AND motion = True.

Agent North: sees temperature cleanly, motion+category with 45% noise.
Agent South: sees motion cleanly, temperature+category with 45% noise.
Agent East:  sees category cleanly, temperature+motion with 45% noise.
             (Category is irrelevant to truth — East is a weaker agent.)

Agent East benefits from sharing because North and South teach it the
temperature AND motion pattern even though East can't observe them cleanly.
"""

import sys, numpy as np
sys.path.insert(0, "..")

from tm_collective import WorldSchema, TMNode, Collective
from tm_collective.strategies import SyntheticDataStrategy, AllToAll, OnceOnlyTrigger
from tm_collective import evaluation

SEED = 42
N_ROUNDS = 10
SHARE_AT_ROUND = 5
N_TEST = 1000

np.random.seed(SEED)

schema = WorldSchema.from_dict({
    "version": 1,
    "description": "Sensor fusion: temp, motion, category",
    "features": [
        {"id": 0, "name": "temp",     "encoder": "thermometer", "thresholds": [10, 20, 30, 40]},
        {"id": 1, "name": "motion",   "encoder": "boolean"},
        {"id": 2, "name": "category", "encoder": "onehot", "vocab": ["low", "med", "high"]},
    ]
})
print("Schema n_binary:", schema.n_binary)  # 4 + 1 + 3 = 8

def generate_obs(n, seed=None):
    rng = np.random.RandomState(seed)
    obs_list, labels = [], []
    for _ in range(n):
        temp   = rng.uniform(0, 50)
        motion = bool(rng.randint(0, 2))
        cat    = rng.choice(["low", "med", "high"])
        obs_list.append({"temp": temp, "motion": motion, "category": cat})
        labels.append(int(temp > 30 and motion))
    X = schema.encode_batch(obs_list)
    y = np.array(labels, dtype=np.uint32)
    return X, y

# Fixed test set
np.random.seed(SEED + 9999)
X_test, y_test = generate_obs(N_TEST, seed=SEED + 9999)
np.random.seed(SEED)

sharing = SyntheticDataStrategy(n_synthetic=500, retrain_epochs=200)

node_north = TMNode("north", schema, noisy_features=["motion", "category"],
                    noise_rate=0.45, sharing=sharing, epochs_per_round=50)
node_south = TMNode("south", schema, noisy_features=["temp", "category"],
                    noise_rate=0.45, sharing=sharing, epochs_per_round=50)
node_east  = TMNode("east",  schema, noisy_features=["temp", "motion"],
                    noise_rate=0.45, sharing=sharing, epochs_per_round=50)

collective = Collective(
    schema=schema,
    nodes={"north": node_north, "south": node_south, "east": node_east},
    topology=AllToAll(),
    trigger=OnceOnlyTrigger(at_round=SHARE_AT_ROUND),
)

print("\nExample 2: Sensor Fusion with Thermometer Encoding\n")
print(f"{'Round':>6}  {'north':>8}  {'south':>8}  {'east':>8}")
print("-" * 40)

for r in range(1, N_ROUNDS + 1):
    X_new, y_new = generate_obs(100, seed=r)
    result = collective.step(X_new, y_new, X_test, y_test)
    acc = result["accuracies"]
    note = "  <<SHARE>>" if result["sharing_events"] else ""
    print(f"{r:>6}  {acc['north']:>8.3f}  {acc['south']:>8.3f}  {acc['east']:>8.3f}{note}")

summary = collective.summary()
evaluation.print_summary_table(summary, share_round=SHARE_AT_ROUND)
evaluation.plot_accuracy(
    history=collective._history,
    share_rounds=[e["round"] for e in collective._share_events],
    title="Sensor Fusion: temp>30 AND motion  (thermometer + boolean + onehot)",
    save_path="sensor_fusion_plot.png",
    agent_labels={"north": "North (temp clean)", "south": "South (motion clean)",
                  "east": "East (category clean, weakest)"},
)
