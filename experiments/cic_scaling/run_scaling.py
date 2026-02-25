#!/usr/bin/env python3
"""
run_scaling.py — Scaling experiment: 3/6/9 agents × fixed/graduated synthetic sharing.

Reuses prepared data (thresholds, test sets) from cic_cross_env.
Re-samples per-agent training data from 2017 CSVs for each agent count.
Outputs results and comparison table/plot.
"""
import os
import sys
import json
import time
import glob
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from pyTsetlinMachine.tm import MultiClassTsetlinMachine
from tm_collective.strategies.sharing import SyntheticDataStrategy
from tm_collective.knowledge_packet import KnowledgePacket

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CROSS_ENV_DIR = os.path.join(SCRIPT_DIR, "..", "cic_cross_env")
DATA_DIR = os.path.join(CROSS_ENV_DIR, "data", "prepared")
RAW_2017_DIR = os.path.join(SCRIPT_DIR, "..", "..", "experiments_cic", "data", "2017")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
PLOTS_DIR = os.path.join(SCRIPT_DIR, "plots")

# TM hyperparameters (same as cic_cross_env)
TM_CLAUSES = 2000
TM_T = 50.0
TM_S = 10.0
TRAIN_EPOCHS = 100
ABSORB_EPOCHS = 100
N_SYNTHETIC = 5000
N_SEEDS = 5
GRADUATED_RATES = [0.05, 0.15, 0.30, 0.50]

NORMAL_PER_AGENT = 2000
ATTACK_PER_TYPE = 500

# Column mapping from cic_cross_env/prepare_data.py
COLUMN_MAP = [
    ("flow_duration",             "Flow Duration"),
    ("total_fwd_packets",         "Total Fwd Packets"),
    ("total_bwd_packets",         "Total Backward Packets"),
    ("total_length_fwd_packets",  "Total Length of Fwd Packets"),
    ("total_length_bwd_packets",  "Total Length of Bwd Packets"),
    ("fwd_packet_length_max",     "Fwd Packet Length Max"),
    ("fwd_packet_length_mean",    "Fwd Packet Length Mean"),
    ("bwd_packet_length_max",     "Bwd Packet Length Max"),
    ("bwd_packet_length_mean",    "Bwd Packet Length Mean"),
    ("flow_bytes_per_s",          "Flow Bytes/s"),
    ("flow_packets_per_s",        "Flow Packets/s"),
    ("flow_iat_mean",             "Flow IAT Mean"),
    ("flow_iat_max",              "Flow IAT Max"),
    ("fwd_iat_mean",              "Fwd IAT Mean"),
    ("bwd_iat_mean",              "Bwd IAT Mean"),
    ("fwd_psh_flags",             "Fwd PSH Flags"),
    ("fwd_urg_flags",             "Fwd URG Flags"),
    ("fin_flag_count",            "FIN Flag Count"),
    ("syn_flag_count",            "SYN Flag Count"),
    ("rst_flag_count",            "RST Flag Count"),
    ("psh_flag_count",            "PSH Flag Count"),
    ("ack_flag_count",            "ACK Flag Count"),
    ("avg_fwd_segment_size",      "Avg Fwd Segment Size"),
    ("avg_bwd_segment_size",      "Avg Bwd Segment Size"),
    ("init_win_bytes_forward",    "Init_Win_bytes_forward"),
    ("init_win_bytes_backward",   "Init_Win_bytes_backward"),
]

NUMERIC_FEATURES = [c[0] for c in COLUMN_MAP]

def normalize_label_2017(raw):
    s = raw.strip().lower()
    if s == "benign": return "normal"
    if s == "ddos": return "ddos"
    if s == "dos hulk": return "dos_hulk"
    if s == "dos goldeneye": return "dos_goldeneye"
    if s == "dos slowloris": return "dos_slowloris"
    if s == "dos slowhttptest": return "dos_slowhttptest"
    if s == "bot": return "bot"
    if s == "portscan": return "portscan"
    if s == "infiltration": return "infiltration"
    if "brute force" in s: return "web_bruteforce"
    if "xss" in s: return "web_xss"
    if "sql injection" in s: return "web_sqli"
    if "patator" in s or "heartbleed" in s: return None
    return None

# Agent attack assignments per configuration
AGENT_CONFIGS = {
    3: {
        "A": ["ddos", "dos_hulk", "dos_goldeneye"],
        "B": ["dos_slowloris", "dos_slowhttptest", "bot"],
        "C": ["portscan", "web_bruteforce", "web_xss", "web_sqli", "infiltration"],
    },
    6: {
        "A": ["ddos", "dos_hulk"],
        "B": ["dos_goldeneye", "dos_slowloris"],
        "C": ["dos_slowhttptest", "bot"],
        "D": ["portscan", "web_bruteforce"],
        "E": ["web_xss", "web_sqli"],
        "F": ["infiltration"],
    },
    9: {
        "A": ["ddos"],
        "B": ["dos_hulk"],
        "C": ["dos_goldeneye"],
        "D": ["dos_slowloris"],
        "E": ["dos_slowhttptest"],
        "F": ["bot"],
        "G": ["portscan"],
        "H": ["web_bruteforce", "web_xss"],
        "I": ["web_sqli", "infiltration"],
    },
}

ALL_ATTACKS = sorted(set(
    a for cfg in AGENT_CONFIGS.values()
    for attacks in cfg.values()
    for a in attacks
))

BOLD = "\033[1m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"


# ── Data loading ──────────────────────────────────────────────────────────

def load_2017_raw():
    """Load CIC-IDS2017 CSVs into a dataframe with internal column names."""
    csv_files = sorted(glob.glob(os.path.join(RAW_2017_DIR, "*.csv")))
    print(f"\nLoading CIC-IDS2017 ({len(csv_files)} files)...")
    frames = []
    for fpath in csv_files:
        print(f"  {os.path.basename(fpath)}...", end=" ")
        df = pd.read_csv(fpath, low_memory=False, encoding='latin-1')
        df.columns = [c.strip() for c in df.columns]
        rename = {}
        for internal, c2017 in COLUMN_MAP:
            if c2017 in df.columns:
                rename[c2017] = internal
        if "Label" in df.columns:
            rename["Label"] = "_label"
        df = df.rename(columns=rename)
        keep = [c for c in NUMERIC_FEATURES if c in df.columns] + ["_label"]
        df = df[[c for c in keep if c in df.columns]]
        if "_label" in df.columns:
            df["_label"] = df["_label"].astype(str).str.strip()
            df["attack_label"] = df["_label"].apply(normalize_label_2017)
            df = df.dropna(subset=["attack_label"])
            df = df.drop(columns=["_label"])
        for feat in NUMERIC_FEATURES:
            if feat in df.columns:
                df[feat] = pd.to_numeric(df[feat], errors="coerce").fillna(0)
        print(f"{len(df)} rows")
        frames.append(df)
    result = pd.concat(frames, ignore_index=True)
    print(f"  Total: {len(result)} rows")
    return result


def load_thresholds():
    """Load pre-computed thresholds from cic_cross_env."""
    with open(os.path.join(DATA_DIR, "feature_meta.json")) as f:
        meta = json.load(f)
    return meta["thresholds"]


def binarize_df(df, thresholds):
    """Binarize using thermometer encoding."""
    n = len(df)
    n_bits = sum(len(t) for t in thresholds.values())
    X = np.zeros((n, n_bits), dtype=np.int32)
    col_idx = 0
    for feat, thresh_vals in thresholds.items():
        vals = pd.to_numeric(df[feat], errors="coerce").fillna(0).values.astype(float)
        vals = np.where(np.isfinite(vals), vals, 0)
        for t in thresh_vals:
            X[:, col_idx] = (vals > t).astype(np.int32)
            col_idx += 1
    return X


def load_test_data(setting):
    fname = "test_within.json" if setting == "within" else "test_cross.json"
    with open(os.path.join(DATA_DIR, fname)) as f:
        d = json.load(f)
    return (np.array(d["X"], dtype=np.int32),
            np.array(d["y"], dtype=np.int32), d["labels"])


def prepare_agent_data(df_2017_train, thresholds, agent_config, rng):
    """Build per-agent binarized training sets from raw training data."""
    normal_pool = df_2017_train[df_2017_train["attack_label"] == "normal"]
    agent_data = {}

    for agent_id, attacks in agent_config.items():
        n_normal = min(NORMAL_PER_AGENT, len(normal_pool))
        agent_normal = normal_pool.sample(n=n_normal, random_state=rng)
        attack_frames = []
        for attack in attacks:
            adf = df_2017_train[df_2017_train["attack_label"] == attack]
            n = min(ATTACK_PER_TYPE, len(adf))
            if n == 0:
                continue
            attack_frames.append(
                adf.sample(n=n, random_state=rng, replace=(n > len(adf)))
            )
        agent_df = pd.concat([agent_normal] + attack_frames, ignore_index=True)
        agent_df = agent_df.sample(frac=1, random_state=rng).reset_index(drop=True)

        X = binarize_df(agent_df, thresholds)
        y = (agent_df["attack_label"] != "normal").astype(int).values
        labels = agent_df["attack_label"].tolist()
        agent_data[agent_id] = (X, y, attacks, labels)

    return agent_data


# ── Evaluation ────────────────────────────────────────────────────────────

def evaluate_agent(tm, X_test, y_test, labels, seen_attacks):
    preds = tm.predict(X_test)
    overall_acc = float(np.mean(preds == y_test))
    attack_recalls = {}
    for atype in sorted(set(labels)):
        mask = np.array([l == atype for l in labels])
        if mask.sum() == 0:
            continue
        if atype == "normal":
            attack_recalls[atype] = float(np.mean(preds[mask] == 0))
        else:
            attack_recalls[atype] = float(np.mean(preds[mask] == 1))
    unseen = {k: v for k, v in attack_recalls.items()
              if k not in seen_attacks and k != "normal"}
    return overall_acc, attack_recalls, unseen


class StandaloneAgent:
    """Wrapper to make standalone experiment code compatible with Sharing Strategies."""
    def __init__(self, tm, X_train, y_train, schema_mock, agent_id):
        self.tm = tm
        self.X_buffer = [X_train]
        self.y_buffer = [y_train]
        self.X_own_buffer = [X_train]
        self.y_own_buffer = [y_train]
        self.schema = schema_mock
        self.agent_id = agent_id
        self.round_i = 1
        self.last_accuracy = 0.0
        self._fitted = True
        self.n_observations = len(X_train)

    def _reset_tm(self):
        """Recreate the TM to clear its state before retraining."""
        self.tm = MultiClassTsetlinMachine(
            TM_CLAUSES, TM_T, TM_S, number_of_state_bits=8
        )


# ── Main experiment ───────────────────────────────────────────────────────

def run_condition(condition, seed, eval_settings, agent_data, n_bits, agent_ids):
    """Run one condition. Returns {setting: {agent_id: {overall, unseen_vals}}}."""
    rng = np.random.RandomState(seed)
    np.random.seed(seed)

    # Train individual agents
    agents = {}
    agent_attacks = {}
    for agent_id in agent_ids:
        X_train, y_train, attacks, _labels = agent_data[agent_id]
        tm = MultiClassTsetlinMachine(TM_CLAUSES, TM_T, TM_S, number_of_state_bits=8)
        for _ in range(TRAIN_EPOCHS):
            tm.fit(X_train, y_train, epochs=1, incremental=True)
        wrapper = StandaloneAgent(tm, X_train, y_train, type("MockSchema", (), {"n_binary": n_bits}), agent_id)
        agents[agent_id] = wrapper
        agent_attacks[agent_id] = attacks

    rate_mode = "graduated" if condition == "synthetic_graduated" else "fixed"
    strategy = SyntheticDataStrategy(
        n_synthetic=N_SYNTHETIC, 
        retrain_epochs=ABSORB_EPOCHS,
        mode="perturb",
        rate_mode=rate_mode,
        absorption="full"
    )

    # Generate synthetic data
    packets = {}
    for agent_id in agent_ids:
        packets[agent_id] = strategy.generate(agents[agent_id])

    # All-to-all sharing
    for agent_id in agent_ids:
        for peer_id in agent_ids:
            if peer_id != agent_id:
                strategy.absorb(agents[agent_id], packets[peer_id])

    # Evaluate
    results = {}
    for setting, (X_test, y_test, labels) in eval_settings.items():
        results[setting] = {}
        for agent_id in agent_ids:
            overall, per_attack, unseen = evaluate_agent(
                agents[agent_id].tm, X_test, y_test, labels, agent_attacks[agent_id]
            )
            agents[agent_id].last_accuracy = overall
            results[setting][agent_id] = {
                "overall": overall, "per_attack": per_attack, "unseen": unseen,
            }
    return results


def compute_summary(all_results, agent_ids):
    """Compute mean/std of unseen detection across seeds.

    Aggregation: for each seed, average unseen detection across all agents
    (each agent's unseen is the mean across its unseen attack types).
    Then compute mean ± std across the per-seed averages.
    """
    seed_within, seed_cross = [], []
    for seed_results in all_results.values():
        # Per-agent mean unseen for this seed
        w_agent_means, c_agent_means = [], []
        for agent_id in agent_ids:
            w_unseen = seed_results.get("within", {}).get(agent_id, {}).get("unseen", {})
            c_unseen = seed_results.get("cross", {}).get(agent_id, {}).get("unseen", {})
            if w_unseen:
                w_agent_means.append(np.mean(list(w_unseen.values())))
            if c_unseen:
                c_agent_means.append(np.mean(list(c_unseen.values())))
        seed_within.append(np.mean(w_agent_means) if w_agent_means else 0)
        seed_cross.append(np.mean(c_agent_means) if c_agent_means else 0)

    seed_within = np.array(seed_within)
    seed_cross = np.array(seed_cross)
    seed_degrad = seed_within - seed_cross
    return (seed_within.mean(), seed_within.std(),
            seed_cross.mean(), seed_cross.std(),
            seed_degrad.mean(), seed_degrad.std())


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Load test data from existing cic_cross_env
    eval_settings = {}
    for setting in ["within", "cross"]:
        X_test, y_test, labels = load_test_data(setting)
        eval_settings[setting] = (X_test, y_test, labels)
        print(f"  {setting}-env test: {len(X_test)} samples")

    n_bits = eval_settings["within"][0].shape[1]
    print(f"  Features: {n_bits} bits")

    # Load thresholds and 2017 raw data once
    thresholds = load_thresholds()
    df_2017 = load_2017_raw()

    # 80/20 train/test split (same seed as prepare_data.py)
    from sklearn.model_selection import train_test_split
    df_2017_train, _ = train_test_split(
        df_2017, test_size=0.2, random_state=42, stratify=df_2017["attack_label"]
    )
    print(f"  2017 training pool: {len(df_2017_train)} rows")

    # Run all configurations
    summary_table = {}  # (condition, n_agents) -> (w_mean, w_std, c_mean, c_std, d_mean, d_std)

    for n_agents, agent_config in sorted(AGENT_CONFIGS.items()):
        agent_ids = sorted(agent_config.keys())
        print(f"\n{'='*70}")
        print(f"  AGENT COUNT: {n_agents}")
        print(f"{'='*70}")
        for agent_id, attacks in agent_config.items():
            print(f"    {agent_id}: {attacks}")

        # Prepare training data for this agent config
        rng_prep = np.random.RandomState(42)
        agent_data = prepare_agent_data(df_2017_train, thresholds, agent_config, rng_prep)
        for agent_id in agent_ids:
            X, y, attacks, _ = agent_data[agent_id]
            print(f"    Agent {agent_id}: {len(X)} samples ({sum(y)} attacks)")

        for condition in ["synthetic", "synthetic_graduated"]:
            print(f"\n  -- {condition.upper()} ({n_agents} agents) --")
            all_results = {}
            for seed in range(N_SEEDS):
                t0 = time.time()
                result = run_condition(
                    condition, seed, eval_settings, agent_data, n_bits, agent_ids
                )
                dt = time.time() - t0
                all_results[str(seed)] = result

                # Quick per-seed summary
                w_vals = [np.mean(list(result["within"][a]["unseen"].values()))
                          for a in agent_ids if result["within"][a]["unseen"]]
                c_vals = [np.mean(list(result["cross"][a]["unseen"].values()))
                          for a in agent_ids if result["cross"][a]["unseen"]]
                w_avg = np.mean(w_vals) if w_vals else 0
                c_avg = np.mean(c_vals) if c_vals else 0
                print(f"    Seed {seed}: within={w_avg:.3f}, cross={c_avg:.3f} ({dt:.0f}s)")

            # Save per-config results
            results_path = os.path.join(
                RESULTS_DIR, f"results_{n_agents}agents_{condition}.json"
            )
            with open(results_path, "w") as f:
                json.dump(all_results, f, indent=2)
            print(f"    Saved: {results_path}")

            # Compute summary
            s = compute_summary(all_results, agent_ids)
            summary_table[(condition, n_agents)] = s
            print(f"    Summary: within={s[0]:.3f}±{s[1]:.3f}, "
                  f"cross={s[2]:.3f}±{s[3]:.3f}, Δ={s[4]:.3f}±{s[5]:.3f}")

    # ── Print comparison table ────────────────────────────────────────────
    print(f"\n\n{BOLD}{'='*80}")
    print("  SCALING COMPARISON: Unseen Attack Detection")
    print(f"{'='*80}{RESET}\n")
    print(f"{'':20s} {'Within-env':>16s}  {'Cross-env':>16s}  {'Degradation':>16s}")
    print(f"{'':20s} {'mean ± std':>16s}  {'mean ± std':>16s}  {'mean ± std':>16s}")
    print(f"{'─'*72}")

    for condition_label, condition_key in [("Fixed rate", "synthetic"),
                                           ("Graduated rate", "synthetic_graduated")]:
        print(f"\n{BOLD}{condition_label}{RESET}")
        for n_agents in [3, 6, 9]:
            key = (condition_key, n_agents)
            if key in summary_table:
                w_m, w_s, c_m, c_s, d_m, d_s = summary_table[key]
                print(f"  {n_agents} agents          "
                      f"{w_m:5.1%} ± {w_s:4.1%}    "
                      f"{c_m:5.1%} ± {c_s:4.1%}    "
                      f"{d_m:5.1%} ± {d_s:4.1%}")

    # ── Save summary as JSON ─────────────────────────────────────────────
    summary_json = {}
    for (cond, n), (wm, ws, cm, cs, dm, ds) in summary_table.items():
        summary_json[f"{cond}_{n}agents"] = {
            "within_mean": wm, "within_std": ws,
            "cross_mean": cm, "cross_std": cs,
            "degradation_mean": dm, "degradation_std": ds,
        }
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(summary_json, f, indent=2)

    # ── Generate comparison plot ──────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    agent_counts = [3, 6, 9]
    styles = {
        "synthetic": {"color": "#2196F3", "label": "Fixed rate (0.15)", "marker": "o"},
        "synthetic_graduated": {"color": "#4CAF50", "label": "Graduated rate", "marker": "s"},
    }

    for cond_key, style in styles.items():
        cross_means = []
        cross_stds = []
        within_means = []
        for n in agent_counts:
            key = (cond_key, n)
            if key in summary_table:
                wm, ws, cm, cs, dm, ds = summary_table[key]
                cross_means.append(cm)
                cross_stds.append(cs)
                within_means.append(wm)

        ax.errorbar(agent_counts, cross_means, yerr=cross_stds,
                     color=style["color"], marker=style["marker"],
                     linewidth=2.5, markersize=10, capsize=6, capthick=2,
                     label=f'{style["label"]} (cross-env)',
                     linestyle="-")
        ax.errorbar(agent_counts, within_means,
                     color=style["color"], marker=style["marker"],
                     linewidth=2.5, markersize=10,
                     label=f'{style["label"]} (within-env)',
                     linestyle="--", alpha=0.5)

    ax.set_xlabel("Number of Agents", fontsize=14, color="black")
    ax.set_ylabel("Avg Unseen Attack Detection Rate", fontsize=14, color="black")
    ax.set_title("Cross-Environment Transfer vs. Agent Count",
                 fontsize=16, fontweight="bold", color="black", pad=15)
    ax.set_xticks(agent_counts)
    ax.set_ylim(0, 1.05)
    ax.tick_params(colors="black")
    for spine in ax.spines.values():
        spine.set_color("black")
    ax.legend(fontsize=11, loc="lower right", facecolor="white",
              edgecolor="black", labelcolor="black")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, "scaling_comparison.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\n  Plot saved: {plot_path}")
    plt.close()

    print("\nDone!")


if __name__ == "__main__":
    main()
