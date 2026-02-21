# tm_collective/strategies/__init__.py
from tm_collective.strategies.sharing import SyntheticDataStrategy, ClauseTransferStrategy
from tm_collective.strategies.topology import AllToAll, RingTopology, StarTopology, GossipTopology
from tm_collective.strategies.trigger import FixedRoundTrigger, PlateauTrigger, OnceOnlyTrigger

__all__ = [
    "SyntheticDataStrategy", "ClauseTransferStrategy",
    "AllToAll", "RingTopology", "StarTopology", "GossipTopology",
    "FixedRoundTrigger", "PlateauTrigger", "OnceOnlyTrigger",
]
