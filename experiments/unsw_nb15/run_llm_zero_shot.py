#!/usr/bin/env python3
"""
run_llm_zero_shot.py — LLM-driven zero-shot onboarding experiment.

Tests whether a completely blank Agent D can be onboarded by absorbing
knowledge from Agent A and Agent B, where A and B are driven by an LLM
(qwen3:8b) that extracted features from raw text describing network flows.

This combines the LLM unstructured capability with the zero-shot onboarding capability.
"""
import os
import sys
import json
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
    # Agent D has no assigned attacks (starts blank)
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
EXTRACTION_SYSTEM_PROMPT = f"""You are a network security analyst. Your job is to extract boolean features from network flow descriptions.

For each flow description, you must output EXACTLY {N_FEATURES} binary values (0 or 1) as a JSON array.

The features to extract are (in order):
{chr(10).join(f"  {i}: {name}" for i, name in enumerate(FEATURE_NAMES))}

Rules:
- Output ONLY a JSON array of {N_FEATURES} integers (0 or 1), nothing else
- No explanation, no markdown, just the array
- Example: [1,0,0,1,0,0,1,0,0,0,1,0,0,1,0,0,0,0,0,0]
"""

# ── Helpers ──────────────────────────────────────────────────────────────

def generate_flow_descriptions(n_per_agent=50, rng=None):
    """Generate realistic text descriptions of network flows for agents A and B."""
    if rng is None:
        rng = np.random.RandomState(42)

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
            flows.append((format_flow_description(row), 0, "normal"))

        # Sample attack flows
        n_per_attack = max(1, (n_per_agent - n_per_agent // 2) // len(attacks))
        for attack in attacks:
            attack_df = df[df["attack_label"] == attack]
            n = min(n_per_attack, len(attack_df))
            sampled = attack_df.sample(n=n, random_state=rng)
            for _, row in sampled.iterrows():
                flows.append((format_flow_description(row), 1, attack))

        rng.shuffle(flows)
        agent_flows[agent_id] = flows

    # Full test set (all attack types from A and B + normal)
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
    """Convert raw UNSW-NB15 row into natural language description."""
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

def extract_features_llm(model, flow_description):
    """Use Ollama LLM to extract boolean features."""
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": flow_description},
        ],
        options={"temperature": 0.0, "num_predict": 100},
    )

    text = response["message"]["content"].strip()
    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

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

    import re
    nums = re.findall(r'[01]', text)
    if len(nums) >= N_FEATURES:
        return [int(nums[i]) for i in range(N_FEATURES)]
    return [0] * N_FEATURES

# ── LLMAgent ─────────────────────────────────────────────────────────────

class LLMAgent:
    def __init__(self, agent_id, model, workspace, skill_dir):
        self.agent_id = agent_id
        self.model = model
        self.workspace = workspace
        self.skill_dir = skill_dir

    def log(self, msg):
        color = {"A": CYAN, "B": GREEN, "C": MAGENTA, "D": YELLOW}.get(self.agent_id, "")
        print(f"  {color}[{self.agent_id}]{RESET} {msg}")

    def _write_data(self, X, y, prefix="data"):
        x_path = os.path.join(self.workspace, f"{prefix}_{self.agent_id}_X.json")
        y_path = os.path.join(self.workspace, f"{prefix}_{self.agent_id}_y.json")
        with open(x_path, "w") as f:
            json.dump(X if isinstance(X, list) else X.tolist(), f)
        with open(y_path, "w") as f:
            json.dump(y if isinstance(y, list) else y.tolist(), f)
        return x_path, y_path

    def extract_and_observe(self, flows):
        X, y = [], []
        for i, (desc, label, _) in enumerate(flows):
            features = extract_features_llm(self.model, desc)
            X.append(features)
            y.append(label)
            if (i + 1) % 10 == 0:
                self.log(f"  Extracted {i+1}/{len(flows)} features...")

        self.log(f"Extracted {len(X)} feature vectors ({sum(y)} attacks)")
        x_path, y_path = self._write_data(X, y, "train")

        cmd = (
            f'python {self.skill_dir}/tm_observe.py '
            f'--workspace {self.workspace} --agent {self.agent_id} '
            f'--X "$(cat {x_path})" --y "$(cat {y_path})" '
            f'--epochs 80'
        )
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=self.workspace)
        if r.returncode == 0:
            result = json.loads(r.stdout.strip())
            self.log(f"  tm_observe: {result}")
        else:
            self.log(f"  ❌ tm_observe failed: {r.stderr[:300]}")
        return X, y

    def share(self, n_synthetic=500):
        cmd = (
            f"python {self.skill_dir}/tm_share.py "
            f"--workspace {self.workspace} --agent {self.agent_id} --n {n_synthetic}"
        )
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=self.workspace)
        if r.returncode == 0:
            result = json.loads(r.stdout.strip())
            if result.get("ok"):
                pkt_path = os.path.join(self.workspace, f"packet_from_{self.agent_id}.json")
                with open(pkt_path, "w") as f:
                    f.write(result["packet"])
                self.log(f"  📤 Packet: {result['packet_bytes']} bytes")
                return pkt_path
        self.log(f"  ❌ tm_share failed: {r.stderr[:200] if r.returncode != 0 else 'no ok'}")
        return None

    def absorb(self, packet_path, sender_id, epochs=100):
        cmd = (
            f'python {self.skill_dir}/tm_absorb.py '
            f'--workspace {self.workspace} --agent {self.agent_id} '
            f'--packet "$(cat {packet_path})" --epochs {epochs}'
        )
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=self.workspace)
        if r.returncode == 0:
            result = json.loads(r.stdout.strip())
            self.log(f"  📥 Absorbed from {sender_id}: +{result.get('peer_samples_added', '?')} samples")
            return True
        else:
            self.log(f"  ❌ tm_absorb failed: {r.stderr[:300]}")
            return False

# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LLM-driven zero-shot onboarding")
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--flows-per-agent", type=int, default=30)
    parser.add_argument("--test-flows", type=int, default=50)
    args = parser.parse_args()

    import tempfile
    workspace = tempfile.mkdtemp(prefix="miniclaw_ids_")

    schema = {
        "version": 1,
        "description": f"LLM-extracted network flow features ({N_FEATURES} boolean)",
        "features": [{"id": i, "name": name, "encoder": "boolean"} for i, name in enumerate(FEATURE_NAMES)]
    }
    with open(os.path.join(workspace, "world_schema.json"), "w") as f:
        json.dump(schema, f, indent=2)

    print(f"\n{BOLD}{'='*60}")
    print("  LLM-Driven Zero-Shot Onboarding on UNSW-NB15")
    print(f"{'='*60}{RESET}")
    print("  Agent A: DoS, Fuzzers, Exploits")
    print("  Agent B: Backdoor, Shellcode, Reconnaissance")
    print("  Agent D: starts BLANK (0 real training data)")
    print(f"  Model: {args.model}")
    print(f"  Workspace: {workspace}\n")

    # 1. Flow gen
    print(f"[{BOLD}1/4{RESET}] Generating flow descriptions...")
    rng = np.random.RandomState(42)
    agent_flows, test_flows = generate_flow_descriptions(args.flows_per_agent, rng)
    test_flows = test_flows[:args.test_flows]
    print(f"  Agent A flows: {len(agent_flows['A'])}, Agent B flows: {len(agent_flows['B'])}")

    # Agents
    agents = {
        "A": LLMAgent("A", args.model, workspace, SKILL_DIR),
        "B": LLMAgent("B", args.model, workspace, SKILL_DIR),
        "D": LLMAgent("D", args.model, workspace, SKILL_DIR),
    }

    # 2. Extract & Train
    print(f"\n[{BOLD}2/4{RESET}] Training base Collective (Agent A and Agent B)...")
    for agent_id in ["A", "B"]:
        agents[agent_id].log(f"Processing {len(agent_flows[agent_id])} flows...")
        agents[agent_id].extract_and_observe(agent_flows[agent_id])

    # 3. Share packets
    print(f"\n[{BOLD}3/4{RESET}] Generating Synthetic Knowledge Packets...")
    packets = {}
    for agent_id in ["A", "B"]:
        pkt_path = agents[agent_id].share(n_synthetic=1000)
        if pkt_path: packets[agent_id] = pkt_path

    # 4. Onboard D
    print(f"\n[{BOLD}4/4{RESET}] Onboarding Agent D...")
    for sender_id, pkt_path in packets.items():
        agents["D"].absorb(pkt_path, sender_id, epochs=100)

    # Evaluate D on test set
    print(f"\n{BOLD}Evaluating Agent D...{RESET}")
    X_test, y_test = [], []
    for desc, label, _ in test_flows:
        features = extract_features_llm(args.model, desc)
        X_test.append(features)
        y_test.append(label)

    x_path, y_path = agents["D"]._write_data(X_test, y_test, "test")
    cmd = (
        f'python {SKILL_DIR}/tm_evaluate.py '
        f'--workspace {workspace} --agent D '
        f'--X "$(cat {x_path})" --y "$(cat {y_path})"'
    )
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=workspace)
    if r.returncode == 0:
        result = json.loads(r.stdout.strip())
        print(f"\n  {YELLOW}Agent D (Zero-shot onboarded) Results:{RESET}")
        print(f"    Overall accuracy: {result.get('accuracy', 0):.3f}")
    else:
        print(f"\n  ❌ Evaluation failed: {r.stderr[:300]}")

    print(f"\n  Clean up: rm -rf {workspace}")


if __name__ == "__main__":
    main()
