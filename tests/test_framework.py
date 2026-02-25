import pytest
import numpy as np
import warnings
from unittest.mock import MagicMock, patch
import matplotlib.pyplot as plt

from tm_collective.strategies.sharing import SyntheticDataStrategy, ClauseTransferStrategy
from tm_collective.knowledge_packet import KnowledgePacket
from tm_collective.tm_node import TMNode
from tm_collective.world_schema import WorldSchema
from skill.tm_lib import pack_packet, unpack_packet

# --- Setup Fixtures ---

@pytest.fixture
def mock_schema():
    schema = MagicMock(spec=WorldSchema)
    schema.n_binary = 10
    return schema

@pytest.fixture
def dummy_data():
    X = np.random.randint(0, 2, (100, 10)).astype(np.uint32)
    y = np.random.randint(0, 2, 100).astype(np.uint32)
    return X, y

@pytest.fixture
def mock_tm_node(mock_schema, dummy_data):
    X, y = dummy_data
    
    node = MagicMock(spec=TMNode)
    node.agent_id = "test_agent"
    node.round_i = 1
    node.schema = mock_schema
    node._fitted = True
    node.last_accuracy = 0.8
    node.n_observations = 100
    
    # Setup buffers
    node.X_buffer = [X]
    node.y_buffer = [y]
    node.X_own_buffer = [X]
    node.y_own_buffer = [y]
    
    # Setup TM
    node.tm = MagicMock()
    # Predict returns deterministic output for testing (alternating 0s and 1s)
    node.tm.predict.side_effect = lambda X_test: np.array([i % 2 for i in range(len(X_test))])
    
    return node

# --- Tests for SyntheticDataStrategy ---

def test_generate_perturb_mode(mock_tm_node):
    strategy = SyntheticDataStrategy(n_synthetic=20, mode="perturb", rate_mode="fixed", flip_rate_min=0.0)
    packet = strategy.generate(mock_tm_node)
    
    assert packet.metadata["mode"] == "perturb"
    assert packet.X.shape == (20, 10)
    assert len(packet.y) == 20
    assert packet.metadata["fitted"] is True
    
    # Check class balance (predict mock alternates 0s and 1s, so should be 50/50 target)
    assert np.sum(packet.y == 0) == 10
    assert np.sum(packet.y == 1) == 10

def test_generate_random_fallback(mock_tm_node):
    # Empty buffers to trigger fallback
    mock_tm_node.X_buffer = []
    mock_tm_node.X_own_buffer = []
    
    strategy = SyntheticDataStrategy(n_synthetic=20, mode="perturb")
    packet = strategy.generate(mock_tm_node)
    
    # Should fall back to random if no data pool
    assert packet.X.shape == (20, 10)
    assert len(packet.y) == 20

def test_generate_random_mode(mock_tm_node):
    strategy = SyntheticDataStrategy(n_synthetic=20, mode="random")
    packet = strategy.generate(mock_tm_node)
    
    assert packet.metadata["mode"] == "random"
    assert packet.X.shape == (20, 10)

def test_graduated_rate_sampling():
    strategy = SyntheticDataStrategy(rate_mode="graduated", flip_rate_min=0.1, flip_rate_max=0.4)
    rate1 = strategy._sample_flip_rate()
    rate2 = strategy._sample_flip_rate()
    
    assert 0.1 <= rate1 <= 0.4
    assert 0.1 <= rate2 <= 0.4
    # Highly unlikely to be exactly equal
    assert rate1 != rate2

def test_fixed_rate_sampling():
    strategy = SyntheticDataStrategy(rate_mode="fixed", flip_rate_min=0.1)
    rate1 = strategy._sample_flip_rate()
    rate2 = strategy._sample_flip_rate()
    
    assert rate1 == 0.1
    assert rate2 == 0.1

def test_hybrid_absorption(mock_tm_node):
    # Packet with 5 normal (0) and 5 attack (1) samples
    X_peer = np.ones((10, 10), dtype=np.uint32)
    y_peer = np.array([0]*5 + [1]*5, dtype=np.uint32)
    packet = KnowledgePacket("peer", 1, X_peer, y_peer, {"fitted": True})
    
    strategy = SyntheticDataStrategy(absorption="hybrid")
    
    # Pre-absorption state
    initial_buffer_len = len(mock_tm_node.X_buffer)
    
    result = strategy.absorb(mock_tm_node, packet)
    
    assert result["absorption_mode"] == "hybrid"
    
    # Buffer should have grown by 2 (X_peer_attacks, X_local_normals)
    assert len(mock_tm_node.X_buffer) == initial_buffer_len + 2
    
    # X_own_buffer should NOT be modified by absorb
    assert len(mock_tm_node.X_own_buffer) == 1
    
    # Verify the appended peer data only contains attacks (y=1)
    peer_data_appended = mock_tm_node.X_buffer[-2]
    peer_labels_appended = mock_tm_node.y_buffer[-2]
    assert len(peer_labels_appended) == 5
    assert np.all(peer_labels_appended == 1)
    
    # Verify the appended local data only contains normals (y=0)
    local_data_appended = mock_tm_node.X_buffer[-1]
    local_labels_appended = mock_tm_node.y_buffer[-1]
    assert len(local_labels_appended) == 5
    assert np.all(local_labels_appended == 0)

def test_full_absorption(mock_tm_node):
    X_peer = np.ones((10, 10), dtype=np.uint32)
    y_peer = np.array([0]*5 + [1]*5, dtype=np.uint32)
    packet = KnowledgePacket("peer", 1, X_peer, y_peer, {"fitted": True})
    
    strategy = SyntheticDataStrategy(absorption="full")
    initial_buffer_len = len(mock_tm_node.X_buffer)
    
    result = strategy.absorb(mock_tm_node, packet)
    
    assert result["absorption_mode"] == "full"
    
    # Buffer should have grown by 1 (the full packet)
    assert len(mock_tm_node.X_buffer) == initial_buffer_len + 1
    assert len(mock_tm_node.X_buffer[-1]) == 10
    
    # X_own_buffer unchanged
    assert len(mock_tm_node.X_own_buffer) == 1

# --- Test ClauseTransferStrategy Deprecation ---

def test_clause_transfer_deprecation():
    with pytest.warns(DeprecationWarning, match="ClauseTransferStrategy is deprecated"):
        ClauseTransferStrategy(top_k=10)

# --- Test tm_lib y-encoding bug ---

def test_pack_unpack_packet_multidigit_labels():
    X = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.uint32)
    # Using labels >= 10 to test the bug fix
    y = np.array([10, 15], dtype=np.uint32)
    
    # Pack
    packed = pack_packet("agent1", X, y)
    
    # Unpack
    sender_id, X_unpacked, y_unpacked, meta = unpack_packet(packed, n_features=3)
    
    assert sender_id == "agent1"
    np.testing.assert_array_equal(X, X_unpacked)
    np.testing.assert_array_equal(y, y_unpacked)
    
def test_unpack_packet_legacy_v1_format():
    X = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.uint32)
    y = np.array([0, 1], dtype=np.uint32)
    
    import json
    v1_packed = json.dumps({
        "v": 1,
        "type": "tm_knowledge_packet",
        "sender": "agent1",
        "X": ["101", "010"],
        "y": "01",
        "meta": {}
    })
    
    sender_id, X_unpacked, y_unpacked, meta = unpack_packet(v1_packed, n_features=3)
    
    assert sender_id == "agent1"
    np.testing.assert_array_equal(X, X_unpacked)
    np.testing.assert_array_equal(y, y_unpacked)

# --- Test evaluation.py axvline duplicate fix ---

def test_plot_accuracy_axvline():
    from tm_collective.evaluation import plot_accuracy
    
    # We just want to make sure it runs without crashing and checks the label logic
    # We patch ax.axvline to inspect its calls
    with patch('matplotlib.axes.Axes.axvline') as mock_axvline:
        plt.figure()
        
        histories = {"agent1": [0.5, 0.6, 0.7]}
        # Try plotting with 2 sharing events
        plot_accuracy(histories, share_rounds=[1, 2], title="Test", save_path="/tmp/test_plot.png")
        
        # Should be called exactly twice, once for each share round
        assert mock_axvline.call_count == 2
        
        # Output calls
        calls = mock_axvline.call_args_list
        
        # First call should have label="Sharing event"
        assert calls[0].kwargs.get('label') == "Sharing event"
        
        # Second call should have label=None
        assert calls[1].kwargs.get('label') is None
        
        plt.close('all')
