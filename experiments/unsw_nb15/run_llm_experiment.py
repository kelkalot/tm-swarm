#!/usr/bin/env python3
"""
run_llm_experiment.py — LLM-driven collective intrusion detection on UNSW-NB15.

Each agent is driven by qwen3:8b via Ollama. The LLM:
  1. Reads raw network flow descriptions (as text)
  2. Extracts boolean features according to a schema  
  3. Calls tm_observe, tm_evaluate, tm_share, tm_absorb via bash tool calls
  4. Reports accuracy before and after knowledge sharing

This demonstrates the novel LLM+TM combination:
  - LLMs give TMs access to unstructured data (text → boolean features)
  - TMs give LLMs persistent, shareable learning across sessions
"""
import os
import sys
import json
import time
import subprocess
import argparse
import numpy as np
import ollama

# Add project root to path
PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, PROJECT_ROOT)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data", "prepared")
SKILL_DIR = os.path.join(PROJECT_ROOT, "skill")

# ── Terminal formatting ────────────────────────────────────────────────────
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
RESET = "\033[0m"

# ── Agent attack assignments ──────────────────────────────────────────────
AGENT_ATTACKS = {
    "A": ["dos", "fuzzers", "exploits"],
    "B": ["backdoor", "shellcode", "reconnaissance"],
    "C": ["worms", "analysis", "generic"],
}

# ── Feature schema for LLM to extract ────────────────────────────────────
FEATURE_NAMES = [
    "high_duration",          # flow lasted > 1 second
    "very_high_duration",     # flow lasted > 10 seconds
    "high_src_bytes",         # > 10KB from source
    "very_high_src_bytes",    # > 100KB from source
    "high_dst_bytes",         # > 10KB from destination
    "low_src_ttl",            # TTL < 32 (unusual)
    "high_src_packets",       # > 10 packets from source
    "high_dst_packets",       # > 10 packets from destination
    "high_src_load",          # high source bandwidth
    "high_dst_load",          # high destination bandwidth
    "tcp_protocol",           # TCP protocol
    "udp_protocol",           # UDP protocol
    "syn_no_ack",             # SYN without ACK (scan pattern)
    "connection_established", # full connection established
    "http_service",           # HTTP service
    "dns_service",            # DNS service
    "ftp_service",            # FTP service
    "ssh_service",            # SSH service
    "high_conn_rate",         # many connections in short time
    "same_src_dst",           # source IP == destination IP
]
N_FEATURES = len(FEATURE_NAMES)


def load_raw_flows():
    """Load pre-prepared data and create text descriptions of flows."""
    test_path = os.path.join(DATA_DIR, "test_data.json")
    with open(test_path) as f:
        test_data = json.load(f)

    # Load feature metadata to understand thresholds
    meta_path = os.path.join(DATA_DIR, "feature_meta.json")
    with open(meta_path) as f:
        meta = json.load(f)

    return test_data, meta


def generate_flow_descriptions(n_per_agent=50, rng=None):
    """Generate realistic text descriptions of network flows for each agent.

    Returns dict: agent_id -> list of (description_text, binary_label, attack_type)
    """
    if rng is None:
        rng = np.random.RandomState(42)

    # Load the full parquet to get raw values for realistic descriptions
    import pandas as pd
    parquet_path = os.path.join(SCRIPT_DIR, "data", "Network-Flows", "UNSW_Flow.parquet")
    df = pd.read_parquet(parquet_path)
    df["attack_label"] = df["attack_label"].str.lower().str.strip()

    agent_flows = {}

    for agent_id, attacks in AGENT_ATTACKS.items():
        flows = []

        # Sample normal flows
        normal_df = df[df["attack_label"] == "normal"].sample(
            n=n_per_agent // 2, random_state=rng)
        for _, row in normal_df.iterrows():
            desc = format_flow_description(row)
            flows.append((desc, 0, "normal"))

        # Sample attack flows (from this agent's assigned types)
        n_per_attack = max(1, (n_per_agent - n_per_agent // 2) // len(attacks))
        for attack in attacks:
            attack_df = df[df["attack_label"] == attack]
            n = min(n_per_attack, len(attack_df))
            sampled = attack_df.sample(n=n, random_state=rng)
            for _, row in sampled.iterrows():
                desc = format_flow_description(row)
                flows.append((desc, 1, attack))

        rng.shuffle(flows)
        agent_flows[agent_id] = flows

    # Full test set (all attack types)
    test_flows = []
    normal_test = df[df["attack_label"] == "normal"].sample(n=30, random_state=rng)
    for _, row in normal_test.iterrows():
        test_flows.append((format_flow_description(row), 0, "normal"))

    for attack in set(a for attacks in AGENT_ATTACKS.values() for a in attacks):
        attack_df = df[df["attack_label"] == attack]
        n = min(10, len(attack_df))
        sampled = attack_df.sample(n=n, random_state=rng)
        for _, row in sampled.iterrows():
            test_flows.append((format_flow_description(row), 1, attack))

    rng.shuffle(test_flows)
    return agent_flows, test_flows


def format_flow_description(row):
    """Convert a raw UNSW-NB15 row into a natural language description."""
    return (
        f"Network flow: {row['protocol']} from {row['source_ip']}:{row['source_port']} "
        f"to {row['destination_ip']}:{row['destination_port']} | "
        f"Duration: {row['dur']:.3f}s | "
        f"Bytes sent: {row['sbytes']}, received: {row['dbytes']} | "
        f"Packets: {row['spkts']} sent, {row['dpkts']} received | "
        f"TTL: src={row['sttl']}, dst={row['dttl']} | "
        f"State: {row['state']} | Service: {row['service']} | "
        f"Load: src={row['sload']:.0f} bps, dst={row['dload']:.0f} bps | "
        f"Jitter: src={row['sjit']:.3f}, dst={row['djit']:.3f} | "
        f"TCP RTT: {row['tcprtt']:.4f}s | "
        f"Mean pkt size: src={row['smeansz']}, dst={row['dmeansz']} | "
        f"Connections to same service from src: {row['ct_srv_src']}, to dst: {row['ct_srv_dst']}"
    )


# ── LLM Feature Extraction ──────────────────────────────────────────────

EXTRACTION_SYSTEM_PROMPT = f"""You are a network security analyst. Your job is to extract boolean features from network flow descriptions.

For each flow description, you must output EXACTLY {N_FEATURES} binary values (0 or 1) as a JSON array.

The features to extract are (in order):
{chr(10).join(f"  {i}: {name}" for i, name in enumerate(FEATURE_NAMES))}

Rules:
- Output ONLY a JSON array of {N_FEATURES} integers (0 or 1), nothing else
- No explanation, no markdown, just the array
- Example: [1,0,0,1,0,0,1,0,0,0,1,0,0,1,0,0,0,0,0,0]
"""


def extract_features_llm(model, flow_description):
    """Use Ollama LLM to extract boolean features from a flow description."""
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": flow_description},
        ],
        options={"temperature": 0.0, "num_predict": 100},
    )

    text = response["message"]["content"].strip()

    # Parse the JSON array from the response
    # Handle cases where model wraps in markdown or adds explanation
    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    # Find array in text
    start = text.find("[")
    end = text.rfind("]")
    if start >= 0 and end > start:
        text = text[start:end+1]

    try:
        features = json.loads(text)
        if isinstance(features, list) and len(features) == N_FEATURES:
            return [int(bool(v)) for v in features]
    except json.JSONDecodeError:
        pass

    # Fallback: try to extract numbers
    import re
    nums = re.findall(r'[01]', text)
    if len(nums) >= N_FEATURES:
        return [int(nums[i]) for i in range(N_FEATURES)]

    # Last resort: return zeros
    return [0] * N_FEATURES


# ── LLM Agent Class ──────────────────────────────────────────────────────

class LLMAgent:
    """An agent driven by a local LLM that processes network flows."""

    def __init__(self, agent_id, model, workspace, skill_dir):
        self.agent_id = agent_id
        self.model = model
        self.workspace = workspace
        self.skill_dir = skill_dir

    def log(self, msg):
        color = {"A": CYAN, "B": GREEN, "C": MAGENTA}.get(self.agent_id, "")
        print(f"  {color}[{self.agent_id}]{RESET} {msg}")

    def _write_data(self, X, y, prefix="data"):
        """Write X, y to files in workspace, return paths."""
        x_path = os.path.join(self.workspace, f"{prefix}_{self.agent_id}_X.json")
        y_path = os.path.join(self.workspace, f"{prefix}_{self.agent_id}_y.json")
        with open(x_path, "w") as f:
            json.dump(X if isinstance(X, list) else X.tolist(), f)
        with open(y_path, "w") as f:
            json.dump(y if isinstance(y, list) else y.tolist(), f)
        return x_path, y_path

    def extract_and_observe(self, flows):
        """Use LLM to extract features from text flows, then call tm_observe."""
        X = []
        y = []

        for i, (desc, label, attack_type) in enumerate(flows):
            features = extract_features_llm(self.model, desc)
            X.append(features)
            y.append(label)
            if (i + 1) % 10 == 0:
                self.log(f"  Extracted {i+1}/{len(flows)} features...")

        self.log(f"Extracted {len(X)} feature vectors ({sum(y)} attacks)")

        # Write to files to avoid shell arg length limits
        x_path, y_path = self._write_data(X, y, "train")

        cmd = (
            f'python {self.skill_dir}/tm_observe.py '
            f'--workspace {self.workspace} --agent {self.agent_id} '
            f'--X "$(cat {x_path})" --y "$(cat {y_path})" '
            f'--epochs 80'
        )
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                           cwd=self.workspace)
        if r.returncode == 0:
            result = json.loads(r.stdout.strip())
            self.log(f"  tm_observe: {result}")
        else:
            self.log(f"  ❌ tm_observe failed: {r.stderr[:300]}")

        return X, y

    def evaluate(self, X_test, y_test):
        """Call tm_evaluate with test data."""
        x_path, y_path = self._write_data(X_test, y_test, "test")

        cmd = (
            f'python {self.skill_dir}/tm_evaluate.py '
            f'--workspace {self.workspace} --agent {self.agent_id} '
            f'--X "$(cat {x_path})" --y "$(cat {y_path})"'
        )
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                           cwd=self.workspace)
        if r.returncode == 0:
            result = json.loads(r.stdout.strip())
            self.log(f"  Accuracy: {result.get('accuracy', '?')}")
            return result.get("accuracy", 0)
        else:
            self.log(f"  ❌ tm_evaluate failed: {r.stderr[:300]}")
            return 0

    def share(self, n_synthetic=500):
        """Generate a knowledge packet."""
        cmd = (
            f"python {self.skill_dir}/tm_share.py "
            f"--workspace {self.workspace} --agent {self.agent_id} --n {n_synthetic}"
        )
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                           cwd=self.workspace)
        if r.returncode == 0:
            result = json.loads(r.stdout.strip())
            if result.get("ok"):
                pkt_path = os.path.join(self.workspace,
                                        f"packet_from_{self.agent_id}.json")
                with open(pkt_path, "w") as f:
                    f.write(result["packet"])
                self.log(f"  📤 Packet: {result['packet_bytes']} bytes")
                return pkt_path
        self.log(f"  ❌ tm_share failed: {r.stderr[:200] if r.returncode != 0 else 'no ok'}")
        return None

    def absorb(self, packet_path, sender_id, epochs=100):
        """Absorb a peer's knowledge packet."""
        cmd = (
            f'python {self.skill_dir}/tm_absorb.py '
            f'--workspace {self.workspace} --agent {self.agent_id} '
            f'--packet "$(cat {packet_path})" --epochs {epochs}'
        )
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                           cwd=self.workspace)
        if r.returncode == 0:
            result = json.loads(r.stdout.strip())
            self.log(f"  📥 Absorbed from {sender_id}: "
                     f"+{result.get('peer_samples_added', '?')} samples")
            return True
        else:
            self.log(f"  ❌ tm_absorb failed: {r.stderr[:300]}")
            return False


def main():
    parser = argparse.ArgumentParser(description="LLM-driven UNSW-NB15 experiment")
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--flows-per-agent", type=int, default=40,
                        help="Number of flows per agent for training")
    parser.add_argument("--test-flows", type=int, default=60,
                        help="Number of test flows (total)")
    args = parser.parse_args()

    import tempfile
    workspace = tempfile.mkdtemp(prefix="miniclaw_ids_")

    # Copy world schema to workspace
    schema = {
        "version": 1,
        "description": f"LLM-extracted network flow features ({N_FEATURES} boolean)",
        "features": [
            {"id": i, "name": name, "encoder": "boolean"}
            for i, name in enumerate(FEATURE_NAMES)
        ]
    }
    with open(os.path.join(workspace, "world_schema.json"), "w") as f:
        json.dump(schema, f, indent=2)

    print(f"\n{BOLD}{'='*60}")
    print("  LLM-Driven Intrusion Detection on UNSW-NB15")
    print(f"{'='*60}{RESET}")
    print(f"  Model: {args.model}")
    print(f"  Agents: {len(AGENT_ATTACKS)}")
    print(f"  Flows per agent: {args.flows_per_agent}")
    print(f"  Test flows: {args.test_flows}")
    print(f"  Workspace: {workspace}")
    print(f"  Features: {N_FEATURES} (LLM-extracted from raw flow text)")
    print()

    # ── Generate text flow descriptions from raw UNSW-NB15 data ──────────
    print(f"{BOLD}Generating flow descriptions from UNSW-NB15...{RESET}")
    rng = np.random.RandomState(42)
    agent_flows, test_flows = generate_flow_descriptions(
        n_per_agent=args.flows_per_agent, rng=rng
    )

    # Limit test flows
    test_flows = test_flows[:args.test_flows]

    print(f"  Agent flows: {', '.join(f'{k}={len(v)}' for k, v in agent_flows.items())}")
    print(f"  Test flows: {len(test_flows)}")
    print(f"\n  Sample flow:\n    {DIM}{agent_flows['A'][0][0][:200]}...{RESET}")

    # ── Create agents ────────────────────────────────────────────────────
    agents = {}
    for agent_id in AGENT_ATTACKS:
        agents[agent_id] = LLMAgent(agent_id, args.model, workspace, SKILL_DIR)

    # ── Phase 1: LLM extracts features and trains TMs ────────────────────
    print(f"\n{BOLD}{'='*60}")
    print("  PHASE 1: LLM FEATURE EXTRACTION + TRAINING")
    print(f"{'='*60}{RESET}\n")

    t0_total = time.time()
    for agent_id, agent in agents.items():
        agent.log(f"Processing {len(agent_flows[agent_id])} flows "
                   f"(attacks: {AGENT_ATTACKS[agent_id]})")
        t0 = time.time()
        agent.extract_and_observe(agent_flows[agent_id])
        dt = time.time() - t0
        agent.log(f"  ⏱ {dt:.1f}s")

    # ── Extract test features ────────────────────────────────────────────
    print(f"\n{BOLD}Extracting test features via LLM...{RESET}")
    t0 = time.time()
    X_test = []
    y_test = []
    labels_test = []
    for desc, label, attack_type in test_flows:
        features = extract_features_llm(args.model, desc)
        X_test.append(features)
        y_test.append(label)
        labels_test.append(attack_type)
    dt = time.time() - t0
    print(f"  Extracted {len(X_test)} test vectors in {dt:.1f}s")

    # ── Phase 2: Pre-share evaluation ────────────────────────────────────
    print(f"\n{BOLD}{'='*60}")
    print("  PHASE 2: PRE-SHARE EVALUATION")
    print(f"{'='*60}{RESET}\n")

    pre_accuracies = {}
    for agent_id, agent in agents.items():
        acc = agent.evaluate(X_test, y_test)
        pre_accuracies[agent_id] = acc

    # ── Phase 3: Knowledge sharing ───────────────────────────────────────
    print(f"\n{BOLD}{'='*60}")
    print("  PHASE 3: KNOWLEDGE SHARING")
    print(f"{'='*60}{RESET}\n")

    # Generate packets
    packets = {}
    for agent_id, agent in agents.items():
        pkt_path = agent.share(n_synthetic=500)
        if pkt_path:
            packets[agent_id] = pkt_path

    # All-to-all absorption
    for agent_id, agent in agents.items():
        for sender_id, pkt_path in packets.items():
            if sender_id != agent_id:
                agent.absorb(pkt_path, sender_id, epochs=100)

    # ── Phase 4: Post-share evaluation ───────────────────────────────────
    print(f"\n{BOLD}{'='*60}")
    print("  PHASE 4: POST-SHARE EVALUATION")
    print(f"{'='*60}{RESET}\n")

    post_accuracies = {}
    for agent_id, agent in agents.items():
        acc = agent.evaluate(X_test, y_test)
        post_accuracies[agent_id] = acc

    # ── Summary ──────────────────────────────────────────────────────────
    dt_total = time.time() - t0_total

    print(f"\n{BOLD}{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}{RESET}\n")

    for agent_id in agents:
        pre = pre_accuracies[agent_id]
        post = post_accuracies[agent_id]
        delta = post - pre
        color = GREEN if delta > 0.01 else (YELLOW if delta >= 0 else RED)
        sign = "+" if delta >= 0 else ""
        print(f"  Agent {agent_id} ({', '.join(AGENT_ATTACKS[agent_id])}): "
              f"{pre:.3f} → {post:.3f}  {color}({sign}{delta:.3f}){RESET}")

    avg_pre = np.mean(list(pre_accuracies.values()))
    avg_post = np.mean(list(post_accuracies.values()))
    delta = avg_post - avg_pre
    color = GREEN if delta > 0 else RED
    print(f"\n  Average: {avg_pre:.3f} → {avg_post:.3f}  "
          f"{color}({'+' if delta >= 0 else ''}{delta:.3f}){RESET}")
    print(f"  Total time: {dt_total:.0f}s")
    print(f"\n  Workspace: {workspace}")
    print(f"  Clean up: rm -rf {workspace}")


if __name__ == "__main__":
    main()
