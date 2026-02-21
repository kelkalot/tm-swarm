# tm_collective/strategies/topology.py
"""
Topology policies: who each agent shares with.

AllToAll  — every agent shares with every other agent
Ring      — agent i sends to agent (i+1) % N (directed ring)
Star      — one hub agent receives from all and sends to all
Gossip    — each agent randomly selects fan_out peers each round
"""

from __future__ import annotations
import random


class TopologyPolicy:
    def get_peers(self, agent_id: str, all_agent_ids: list[str]) -> list[str]:
        raise NotImplementedError


class AllToAll(TopologyPolicy):
    """Every agent shares with every other agent. O(N²) messages per round."""
    def get_peers(self, agent_id, all_agent_ids):
        return [a for a in all_agent_ids if a != agent_id]


class RingTopology(TopologyPolicy):
    """
    Directed ring: agents sorted alphabetically. Agent i sends to agent (i+1) % N.
    Knowledge propagates hop-by-hop: after K sharing rounds, knowledge has
    traveled K hops around the ring.
    """
    def get_peers(self, agent_id, all_agent_ids):
        ids = sorted(all_agent_ids)
        idx = ids.index(agent_id)
        return [ids[(idx + 1) % len(ids)]]


class StarTopology(TopologyPolicy):
    """
    Hub-and-spoke. Hub shares with everyone; spokes only share with hub.
    Efficient for a trusted aggregator scenario.

    Args:
        hub_id: agent ID of the hub node
    """
    def __init__(self, hub_id: str):
        self.hub_id = hub_id

    def get_peers(self, agent_id, all_agent_ids):
        if agent_id == self.hub_id:
            return [a for a in all_agent_ids if a != self.hub_id]
        return [self.hub_id]


class GossipTopology(TopologyPolicy):
    """
    Each agent randomly selects fan_out peers per round.
    Approximates epidemic broadcast: with fan_out=2 and N agents,
    knowledge reaches all agents in O(log N) rounds with high probability.

    Args:
        fan_out: how many peers to select per agent per round
        seed:    optional random seed for reproducibility
    """
    def __init__(self, fan_out: int = 2, seed: int | None = None):
        self.fan_out = fan_out
        self._rng = random.Random(seed)

    def get_peers(self, agent_id, all_agent_ids):
        candidates = [a for a in all_agent_ids if a != agent_id]
        k = min(self.fan_out, len(candidates))
        return self._rng.sample(candidates, k)
