# tm_collective/collective.py
"""
Collective: orchestrates N TMNode instances.

Runs a round loop:
  1. Generate new observations (provided by caller)
  2. Each agent observes the batch (applying its own noise profile)
  3. Each agent evaluates on the test set
  4. Check trigger policy for each agent — if firing, generate and send knowledge
  5. Receiving agents absorb knowledge packets from senders
  6. Record history and return round results

Usage:
  collective = Collective(
      schema=schema,
      nodes={"agent_a": TMNode(...), "agent_b": TMNode(...)},
      topology=AllToAll(),
      trigger=FixedRoundTrigger(5),
  )
  for X_batch, y_batch in stream:
      result = collective.step(X_batch, y_batch, X_test, y_test)
      print(result)
"""

from __future__ import annotations
import numpy as np

from tm_collective.world_schema import WorldSchema
from tm_collective.tm_node import TMNode
from tm_collective.knowledge_packet import KnowledgePacket
from tm_collective.strategies.topology import TopologyPolicy, AllToAll
from tm_collective.strategies.trigger import TriggerPolicy, FixedRoundTrigger


class Collective:
    """
    Args:
        schema:    WorldSchema shared by all agents
        nodes:     dict mapping agent_id → TMNode
        topology:  TopologyPolicy instance
        trigger:   TriggerPolicy instance
    """

    def __init__(
        self,
        schema: WorldSchema,
        nodes: dict[str, TMNode],
        topology: TopologyPolicy | None = None,
        trigger: TriggerPolicy | None = None,
    ):
        self.schema = schema
        self.nodes = nodes
        self.topology = topology or AllToAll()
        self.trigger = trigger or FixedRoundTrigger(5)
        self._round = 0
        self._history: dict[str, list[float]] = {aid: [] for aid in nodes}
        self._share_events: list[dict] = []

    def step(
        self,
        X_clean: np.ndarray,
        y: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> dict:
        """
        Execute one round:
          1. All agents observe X_clean (each applies own noise internally)
          2. All agents evaluate on X_test
          3. Trigger check → sharing events if needed
          4. Return round summary

        Args:
            X_clean: (n, schema.n_binary) uint32 — clean batch for this round
            y:       (n,) uint32 — ground truth labels for this batch
            X_test:  (n_test, schema.n_binary) uint32 — fixed clean test set
            y_test:  (n_test,) uint32 — test labels

        Returns:
            dict with keys:
              round_i, accuracies, sharing_events, all_accuracies_history
        """
        self._round += 1

        # ── Observe ──────────────────────────────────────────────────────────
        for node in self.nodes.values():
            node.observe_batch(X_clean, y)

        # ── Evaluate ─────────────────────────────────────────────────────────
        accuracies = {}
        for aid, node in self.nodes.items():
            acc = node.evaluate(X_test, y_test)
            accuracies[aid] = acc
            self._history[aid].append(acc)

        # ── Trigger check + sharing ───────────────────────────────────────────
        sharing_events = []
        packets: dict[str, KnowledgePacket] = {}

        for aid, node in self.nodes.items():
            if self.trigger.should_share(aid, self._round, node.accuracy_history()):
                pkt = node.generate_knowledge()
                packets[aid] = pkt

        for sender_id, pkt in packets.items():
            peer_ids = self.topology.get_peers(sender_id, list(self.nodes.keys()))
            for peer_id in peer_ids:
                result = self.nodes[peer_id].absorb_knowledge(pkt)
                event = {
                    "round": self._round,
                    "from": sender_id,
                    "to": peer_id,
                    **result,
                }
                sharing_events.append(event)
                self._share_events.append(event)

        # Re-evaluate agents that absorbed knowledge this round
        absorbed_agents = {e["to"] for e in sharing_events}
        for aid in absorbed_agents:
            acc = self.nodes[aid].evaluate(X_test, y_test)
            accuracies[aid] = acc
            self._history[aid][-1] = acc  # update this round's entry

        return {
            "round_i": self._round,
            "accuracies": accuracies,
            "sharing_events": sharing_events,
            "all_accuracies_history": {k: list(v) for k, v in self._history.items()},
        }

    def run(
        self,
        n_rounds: int,
        obs_per_round: int,
        n_test: int,
        generate_fn,
        truth_fn,
        verbose: bool = True,
    ) -> dict:
        """
        Convenience method: run a full experiment autonomously.

        Args:
            n_rounds:      total rounds
            obs_per_round: new observations per round
            n_test:        test set size (generated once at start)
            generate_fn:   callable() → (X_clean, y) for one round
            truth_fn:      callable(X) → y for test set generation
            verbose:       print round-by-round table

        Returns:
            history dict with all accuracies and sharing events
        """
        np.random.seed(None)
        X_test = np.random.randint(0, 2, (n_test, self.schema.n_binary)).astype(np.uint32)
        y_test = truth_fn(X_test)

        all_results = []
        agent_ids = list(self.nodes.keys())

        if verbose:
            header = f"{'Round':>6}  " + "  ".join(f"{a:>10}" for a in agent_ids)
            print(header)
            print("-" * len(header))

        for r in range(1, n_rounds + 1):
            X_batch, y_batch = generate_fn()
            result = self.step(X_batch, y_batch, X_test, y_test)
            all_results.append(result)

            if verbose:
                accs = result["accuracies"]
                share_note = " <<SHARE>>" if result["sharing_events"] else ""
                row = f"{r:>6}  " + "  ".join(f"{accs.get(a, 0.0):>10.3f}" for a in agent_ids)
                print(row + share_note)

        return {
            "n_rounds": n_rounds,
            "history": {k: list(v) for k, v in self._history.items()},
            "share_events": self._share_events,
        }

    def summary(self) -> dict:
        """Return pre/post-share accuracy averages for all agents."""
        share_rounds = sorted({e["round"] for e in self._share_events})
        if not share_rounds:
            return {"no_sharing_occurred": True}
        first_share = share_rounds[0]

        result = {}
        for aid, hist in self._history.items():
            pre  = [h for i, h in enumerate(hist, 1) if i < first_share]
            post = [h for i, h in enumerate(hist, 1) if i >= first_share]
            result[aid] = {
                "pre_share_avg":  float(np.mean(pre))  if pre  else None,
                "post_share_avg": float(np.mean(post)) if post else None,
                "final": hist[-1] if hist else None,
            }
        return result
