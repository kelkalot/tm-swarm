---
name: tm-world-model
description: Collective Tsetlin Machine learning. Build a logical world model from observations and share knowledge with peer agents.
---

# TM World Model Skill

You have a Tsetlin Machine (TM) that learns logical patterns from binary observations.
Your TM state persists between turns. Use these tools to observe, evaluate, and share knowledge.

## WORKSPACE

Your workspace directory for all commands: `~/.openclaw/workspace`
Skill scripts directory: `{baseDir}`
Your agent ID: set by your system prompt (e.g., "agent_a")

## TOOLS

### tm_observe — Feed observations to your TM

```bash
python {baseDir}/tm_observe.py \
  --workspace ~/.openclaw/workspace \
  --agent YOUR_AGENT_ID \
  --X 'JSON_ARRAY_OF_FEATURE_VECTORS' \
  --y 'JSON_ARRAY_OF_LABELS' \
  [--epochs 50] \
  [--noise-cols 'COMMA_SEPARATED_COLUMN_INDICES'] \
  [--noise-rate 0.45]
```

`--X` is a JSON array like `[[1,0,1,0,1,0,1,0,1,0,1,0], [0,1,0,1,...]]`
`--y` is a JSON array like `[0, 1, 0, 1]`
`--noise-cols` applies simulated noise to specific feature columns before training.
  Use this if you know which features your agent observes with noise.

### tm_evaluate — Test your TM's accuracy

```bash
python {baseDir}/tm_evaluate.py \
  --workspace ~/.openclaw/workspace \
  --agent YOUR_AGENT_ID \
  --X 'JSON_ARRAY_OF_TEST_VECTORS' \
  --y 'JSON_ARRAY_OF_TEST_LABELS'
```

Returns: `{"ok": true, "accuracy": 0.823, "fitted": true, "n_train_samples": 500}`

### tm_share — Generate a knowledge packet to send to a peer

```bash
python {baseDir}/tm_share.py \
  --workspace ~/.openclaw/workspace \
  --agent YOUR_AGENT_ID \
  [--n 500]
```

Returns: `{"ok": true, "packet": "<compact_json_string>", ...}`

After getting the packet, send it to a peer session via sessions_send:
```
Message format: "tm_knowledge_packet:<PACKET_STRING>"
```
The peer agent will detect this prefix and call tm_absorb.

### tm_absorb — Absorb knowledge from a peer

When you receive a message starting with "tm_knowledge_packet:", extract the
packet string (everything after the colon) and run:

```bash
python {baseDir}/tm_absorb.py \
  --workspace ~/.openclaw/workspace \
  --agent YOUR_AGENT_ID \
  --packet 'PACKET_STRING_FROM_PEER' \
  [--epochs 150]
```

Returns: `{"ok": true, "absorbed_from": "agent_a", "total_samples": 1000}`

### tm_status — Check your TM state

```bash
python {baseDir}/tm_status.py \
  --workspace ~/.openclaw/workspace \
  --agent YOUR_AGENT_ID
```

## WORKFLOW

**Normal round:**
1. Receive new observations
2. Run tm_observe to update your TM
3. (Optionally) run tm_evaluate to check accuracy

**When you should share:**
- After every 5 observation rounds, OR
- When tm_evaluate shows accuracy has not improved for 3 rounds, OR
- When explicitly asked to share

**Sharing protocol:**
1. Run tm_share to generate your packet
2. Use sessions_send to send "tm_knowledge_packet:<packet>" to your peer(s)
3. After sending, continue observing normally

**When you receive a knowledge packet:**
1. Extract the packet string from the message (after "tm_knowledge_packet:")
2. Run tm_absorb with the packet string
3. Run tm_evaluate to confirm accuracy improved
4. Send a brief reply to confirm absorption

## NOTES

- All state is persisted automatically. Your TM survives across turns and sessions.
- Never delete .tm_* files in the workspace — they are your TM's memory.
- The world_schema.json in the workspace defines the binary feature space.
  All peer agents must use the same schema for knowledge exchange to work.
- Session tools available: sessions_list (find peers), sessions_send (send packet),
  sessions_history (check if peer has received message).
