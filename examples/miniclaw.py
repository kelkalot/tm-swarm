#!/usr/bin/env python
# examples/miniclaw.py
"""
MiniClaw — A lightweight OpenClaw simulator for testing TM skill scripts
with local LLMs via Ollama.

Spawns N agents, each driven by a local LLM (qwen3:8b by default).
Each agent gets SKILL.md injected as system prompt and can:
  - bash: run any command (tm_observe, tm_evaluate, tm_share, tm_absorb, tm_status)
  - sessions_list: discover peer agents
  - sessions_send: send messages (knowledge packets) to peers

Usage:
  python examples/miniclaw.py                          # 2 agents, default
  python examples/miniclaw.py --n-agents 4 --rounds 8  # 4 agents, 8 rounds
  python examples/miniclaw.py --model llama3.1:8b       # different model

Requires: pip install ollama pyTsetlinMachine numpy
"""

from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
from collections import defaultdict
import numpy as np

try:
    import ollama
except ImportError:
    print("ERROR: 'ollama' package not found. Install with: pip install ollama")
    sys.exit(1)


# ── ANSI colors for agent output ──────────────────────────────────────────────
COLORS = ["\033[94m", "\033[92m", "\033[93m", "\033[95m",
          "\033[96m", "\033[91m", "\033[97m", "\033[33m"]
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"


def color_for(idx: int) -> str:
    return COLORS[idx % len(COLORS)]


# ── Tool definitions for Ollama ───────────────────────────────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Execute a bash command and return its stdout. Use this for ALL tm_* script calls.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The bash command to execute"}
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "sessions_list",
            "description": "List all active peer agent IDs you can send messages to.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "sessions_send",
            "description": "Send a message to a peer agent.",
            "parameters": {
                "type": "object",
                "properties": {
                    "session_id": {"type": "string", "description": "The peer agent ID"},
                    "message": {"type": "string", "description": "The message string to send"}
                },
                "required": ["session_id", "message"]
            }
        }
    },
]


# ── MiniClaw Agent ────────────────────────────────────────────────────────────

class MiniClawAgent:
    """One LLM-driven agent with a tool-calling loop."""

    def __init__(self, agent_id, agent_idx, model, workspace, skill_dir,
                 all_agent_ids, inbox, inbox_lock, noise_cols, noise_rate,
                 system_prompt, max_tool_rounds=20):
        self.agent_id = agent_id
        self.agent_idx = agent_idx
        self.model = model
        self.workspace = workspace
        self.skill_dir = skill_dir
        self.all_agent_ids = all_agent_ids
        self.inbox = inbox
        self.inbox_lock = inbox_lock
        self.noise_cols = noise_cols
        self.noise_rate = noise_rate
        self.system_prompt = system_prompt
        self.max_tool_rounds = max_tool_rounds
        self.color = color_for(agent_idx)

    def log(self, msg, dim=False):
        prefix = f"{self.color}{BOLD}[{self.agent_id}]{RESET}"
        style = DIM if dim else ""
        print(f"{prefix} {style}{msg}{RESET}", flush=True)

    def execute_tool(self, tool_call) -> str:
        fn = tool_call.function
        name = fn.name
        args = fn.arguments or {}

        if name == "bash":
            cmd = args.get("command", "echo 'no command'")
            self.log(f"  🔧 $ {cmd[:120]}", dim=True)
            try:
                r = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                                   timeout=120, cwd=self.workspace)
                out = r.stdout.strip()
                if r.returncode != 0:
                    out = f"ERROR (exit {r.returncode}): {r.stderr.strip()}"
                if len(out) > 4000:
                    out = out[:4000] + "\n...(truncated)"
                self.log(f"  → {out[:200]}", dim=True)
                return out
            except subprocess.TimeoutExpired:
                return "ERROR: Command timed out after 120s"

        elif name == "sessions_list":
            peers = [a for a in self.all_agent_ids if a != self.agent_id]
            self.log(f"  🔧 sessions_list → {peers}", dim=True)
            return json.dumps(peers)

        elif name == "sessions_send":
            sid = args.get("session_id", "")
            message = args.get("message", "")
            if sid not in self.all_agent_ids:
                return json.dumps({"ok": False, "error": f"Unknown session: {sid}"})
            with self.inbox_lock:
                self.inbox[sid].append({"from": self.agent_id, "message": message})
            self.log(f"  📤 → {sid} ({len(message)} chars)")
            return json.dumps({"ok": True, "delivered_to": sid})

        return f"Unknown tool: {name}"

    def check_inbox(self):
        with self.inbox_lock:
            msgs = list(self.inbox[self.agent_id])
            self.inbox[self.agent_id].clear()
        return msgs

    def run_turn(self, user_message: str) -> str:
        """Run one agentic turn with tool-calling loop."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message},
        ]

        for _ in range(self.max_tool_rounds):
            try:
                response = ollama.chat(
                    model=self.model,
                    messages=messages,
                    tools=TOOLS,
                    think=False,
                    options={"num_predict": 2048, "temperature": 0.1},
                )
            except Exception as e:
                self.log(f"  ❌ Ollama error: {e}")
                return f"Error: {e}"

            msg = response["message"]
            tool_calls = msg.get("tool_calls")

            if not tool_calls:
                content = msg.get("content", "").strip()
                if content:
                    self.log(f"  💬 {content[:200]}")
                return content

            messages.append(msg)
            for tc in tool_calls:
                result = self.execute_tool(tc)
                messages.append({"role": "tool", "content": result})

        self.log("  ⚠ Max tool rounds reached")
        return "(max tool rounds)"


# ── Data generation ───────────────────────────────────────────────────────────

def generate_data(n, n_features=12, seed=None):
    rng = np.random.RandomState(seed)
    X = rng.randint(0, 2, (n, n_features)).astype(int)
    y = (X[:, 2] & X[:, 9]).tolist()
    return X.tolist(), y


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MiniClaw: LLM-driven TM agent test")
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--n-agents", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--test-samples", type=int, default=200)
    parser.add_argument("--share-round", type=int, default=None)
    args = parser.parse_args()

    if args.share_round is None:
        args.share_round = args.rounds

    workspace = tempfile.mkdtemp(prefix="miniclaw_")
    skill_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "skill")

    print(f"\n{BOLD}╔══════════════════════════════════════════════════╗")
    print("║  MiniClaw — LLM-Driven TM Collective Learning    ║")
    print(f"╚══════════════════════════════════════════════════╝{RESET}")
    print(f"  Model:     {args.model}")
    print(f"  Agents:    {args.n_agents}")
    print(f"  Rounds:    {args.rounds} (share at round {args.share_round})")
    print(f"  Workspace: {workspace}")
    print(f"  Skills:    {skill_dir}")
    print()

    # Copy world schema
    schema = {"version": 1,
              "features": [{"id": i, "name": f"f{i}", "encoder": "boolean"} for i in range(12)]}
    with open(os.path.join(workspace, "world_schema.json"), "w") as f:
        json.dump(schema, f)

    # Generate test set — save to file so LLM can reference it
    X_test, y_test = generate_data(args.test_samples, seed=999)
    test_data_path = os.path.join(workspace, "test_data.json")
    with open(test_data_path, "w") as f:
        json.dump({"X": X_test, "y": y_test}, f)

    # Setup agents: each sees its zone cleanly, rest is noisy
    n_features = 12
    agent_ids = [chr(ord('A') + i) for i in range(args.n_agents)]
    features_per_agent = n_features // args.n_agents

    agents = []
    inbox = defaultdict(list)
    inbox_lock = threading.Lock()

    for i, agent_id in enumerate(agent_ids):
        clean_start = i * features_per_agent
        clean_end = clean_start + features_per_agent
        noisy_cols = [c for c in range(n_features) if c < clean_start or c >= clean_end]
        noise_cols_str = ",".join(map(str, noisy_cols))
        peer_ids = [a for a in agent_ids if a != agent_id]

        c = color_for(i)
        print(f"  {c}{BOLD}{agent_id}{RESET}: clean=f{clean_start}-f{clean_end-1}, noisy=[{noise_cols_str}]")

        # Build system prompt — very directive to force tool usage
        sys_prompt = f"""You are agent {agent_id}. You MUST use the bash tool to run commands. NEVER describe what you would do — ALWAYS call the tool.

CRITICAL RULES:
- Your agent ID: {agent_id}
- Workspace: {workspace}
- Skill scripts: {skill_dir}
- Your noisy columns: {noise_cols_str}
- Your noise rate: 0.45
- To observe: bash tool with: python {skill_dir}/tm_observe.py --workspace {workspace} --agent {agent_id} --X '<DATA>' --y '<LABELS>' --noise-cols '{noise_cols_str}' --noise-rate 0.45 --epochs 50
- To evaluate: bash tool with: python {skill_dir}/tm_evaluate.py --workspace {workspace} --agent {agent_id} --X '<DATA>' --y '<LABELS>'
- To share: bash tool with: python {skill_dir}/tm_share.py --workspace {workspace} --agent {agent_id} --n 500
- To absorb: bash tool with: python {skill_dir}/tm_absorb.py --workspace {workspace} --agent {agent_id} --packet '<PACKET>' --epochs 150
- To check status: bash tool with: python {skill_dir}/tm_status.py --workspace {workspace} --agent {agent_id}
- Peers: {json.dumps(peer_ids)}

IMPORTANT: Data files are stored in {workspace}. When given a data file path, use `cat` to read it, then pass the contents to the scripts.
Use sessions_send to send knowledge packets to peers. Format: tm_knowledge_packet:<packet_json>

You MUST call tools. Do NOT just describe steps. Execute them NOW."""

        agent = MiniClawAgent(
            agent_id=agent_id, agent_idx=i, model=args.model,
            workspace=workspace, skill_dir=skill_dir,
            all_agent_ids=agent_ids, inbox=inbox, inbox_lock=inbox_lock,
            noise_cols=noise_cols_str, noise_rate=0.45,
            system_prompt=sys_prompt,
        )
        agents.append(agent)

    print()

    # ── Observation rounds ────────────────────────────────────────────────
    for round_i in range(1, args.rounds + 1):
        print(f"\n{BOLD}{'='*60}")
        print(f"  ROUND {round_i}/{args.rounds}")
        print(f"{'='*60}{RESET}\n")

        # Save round data to files
        X_round, y_round = generate_data(args.samples, seed=round_i * 100)
        round_data_path = os.path.join(workspace, f"round_{round_i}_data.json")
        with open(round_data_path, "w") as f:
            json.dump({"X": X_round, "y": y_round}, f)

        for agent in agents:
            agent.log(f"📊 Round {round_i}: Observing {args.samples} samples")

            # Give the LLM a very concrete, actionable instruction
            user_msg = (
                f"Round {round_i}. Run these 2 commands using the bash tool:\n\n"
                f"COMMAND 1 — Observe training data:\n"
                f"python {skill_dir}/tm_observe.py "
                f"--workspace {workspace} --agent {agent.agent_id} "
                f"--X \"$(python -c \"import json; d=json.load(open('{round_data_path}')); print(json.dumps(d['X']))\")\" "
                f"--y \"$(python -c \"import json; d=json.load(open('{round_data_path}')); print(json.dumps(d['y']))\")\" "
                f"--noise-cols '{agent.noise_cols}' --noise-rate 0.45 --epochs 50\n\n"
                f"COMMAND 2 — Evaluate accuracy:\n"
                f"python {skill_dir}/tm_evaluate.py "
                f"--workspace {workspace} --agent {agent.agent_id} "
                f"--X \"$(python -c \"import json; d=json.load(open('{test_data_path}')); print(json.dumps(d['X']))\")\" "
                f"--y \"$(python -c \"import json; d=json.load(open('{test_data_path}')); print(json.dumps(d['y']))\")\" \n\n"
                f"Run COMMAND 1 first, then COMMAND 2. Report the accuracy from COMMAND 2."
            )

            agent.run_turn(user_msg)

        # ── Sharing ───────────────────────────────────────────────────────
        if round_i == args.share_round:
            print(f"\n{BOLD}{'='*60}")
            print("  SHARING ROUND")
            print(f"{'='*60}{RESET}\n")

            # Pre-share: LLM evaluates each agent's current accuracy
            for agent in agents:
                agent.log("📊 Pre-share evaluation")
                eval_msg = (
                    f"Evaluate your current accuracy before sharing. Run:\n"
                    f"python {skill_dir}/tm_evaluate.py --workspace {workspace} "
                    f"--agent {agent.agent_id} "
                    f"--X \"$(python -c \"import json; d=json.load(open('{test_data_path}')); print(json.dumps(d['X']))\")\" "
                    f"--y \"$(python -c \"import json; d=json.load(open('{test_data_path}')); print(json.dumps(d['y']))\")\" "
                )
                agent.run_turn(eval_msg)

            # Orchestrator handles packet generation + transfer (like sessions_send)
            # This is what OpenClaw does behind the scenes — opaque transport
            packets = {}  # agent_id -> packet file path
            for agent in agents:
                agent.log("📤 Generating knowledge packet...")
                share_cmd = (
                    f"python {skill_dir}/tm_share.py "
                    f"--workspace {workspace} --agent {agent.agent_id} --n 500"
                )
                r = subprocess.run(share_cmd, shell=True, capture_output=True,
                                   text=True, cwd=workspace)
                if r.returncode != 0:
                    agent.log(f"  ❌ tm_share failed: {r.stderr.strip()}")
                    continue
                result = json.loads(r.stdout.strip())
                if not result.get("ok"):
                    agent.log(f"  ❌ tm_share error: {result.get('error')}")
                    continue

                pkt_path = os.path.join(workspace, f"packet_from_{agent.agent_id}.json")
                with open(pkt_path, "w") as f:
                    f.write(result["packet"])
                packets[agent.agent_id] = pkt_path
                agent.log(f"  ✅ Packet saved ({result['packet_bytes']} bytes)")

            # Orchestrator delivers packets to peers (all-to-all)
            for agent in agents:
                peer_packets = {aid: p for aid, p in packets.items() if aid != agent.agent_id}
                if not peer_packets:
                    agent.log("📭 No packets to absorb")
                    continue

                for sender_id, pkt_path in peer_packets.items():
                    agent.log(f"📥 Absorbing packet from {sender_id}...")

                    absorb_msg = (
                        f"You received a knowledge packet from {sender_id} "
                        f"(delivered via sessions_send). Run these commands:\n\n"
                        f"COMMAND 1 — Absorb the packet:\n"
                        f"python {skill_dir}/tm_absorb.py --workspace {workspace} "
                        f"--agent {agent.agent_id} "
                        f"--packet \"$(cat {pkt_path})\" --epochs 150\n\n"
                        f"COMMAND 2 — Evaluate your new accuracy:\n"
                        f"python {skill_dir}/tm_evaluate.py --workspace {workspace} "
                        f"--agent {agent.agent_id} "
                        f"--X \"$(python -c \"import json; d=json.load(open('{test_data_path}')); print(json.dumps(d['X']))\")\" "
                        f"--y \"$(python -c \"import json; d=json.load(open('{test_data_path}')); print(json.dumps(d['y']))\")\" \n\n"
                        f"Run COMMAND 1 first, then COMMAND 2. Report your accuracy."
                    )

                    agent.run_turn(absorb_msg)

    # ── Final evaluation ──────────────────────────────────────────────────
    print(f"\n{BOLD}{'='*60}")
    print("  FINAL EVALUATION")
    print(f"{'='*60}{RESET}\n")

    for agent in agents:
        agent.log("📈 Final evaluation")
        eval_msg = (
            f"Final check. Run this command:\n"
            f"python {skill_dir}/tm_evaluate.py --workspace {workspace} "
            f"--agent {agent.agent_id} "
            f"--X \"$(python -c \"import json; d=json.load(open('{test_data_path}')); print(json.dumps(d['X']))\")\" "
            f"--y \"$(python -c \"import json; d=json.load(open('{test_data_path}')); print(json.dumps(d['y']))\")\" "
        )
        agent.run_turn(eval_msg)

    print(f"\n{BOLD}Done! Workspace: {workspace}{RESET}")
    print(f"Clean up with: rm -rf {workspace}\n")


if __name__ == "__main__":
    main()
