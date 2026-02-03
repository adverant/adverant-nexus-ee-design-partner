"""
Tests for PCB Graph Encoder module.

Tests the GNN-based encoder that converts PCB state to embeddings.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add paths
TEST_DIR = Path(__file__).parent
GAMING_AI_DIR = TEST_DIR.parent
if str(GAMING_AI_DIR) not in sys.path:
    sys.path.insert(0, str(GAMING_AI_DIR))

from conftest import MockPCBState, TORCH_AVAILABLE

# Import module under test
from pcb_graph_encoder import (
    PCBGraph, GraphNode, GraphEdge, NodeType, EdgeType, PCBGraphEncoder
)


class TestPCBGraph:
    """Tests for PCBGraph dataclass."""

    def test_empty_graph(self):
        """Test creating an empty graph."""
        graph = PCBGraph(nodes=[], edges=[])
        assert graph.num_nodes == 0
        assert graph.num_edges == 0

    def test_graph_with_nodes(self):
        """Test graph with nodes only."""
        nodes = [
            GraphNode(
                node_id="n1",
                node_type=NodeType.COMPONENT,
                x=10.0,
                y=20.0,
                rotation=0.0,
                layer="F.Cu",
                features={"width": 5.0, "height": 3.0}
            ),
            GraphNode(
                node_id="n2",
                node_type=NodeType.VIA,
                x=30.0,
                y=40.0,
                rotation=0.0,
                layer="F.Cu",
                features={"diameter": 0.8, "drill": 0.4}
            ),
        ]
        graph = PCBGraph(nodes=nodes, edges=[])

        assert graph.num_nodes == 2
        assert graph.num_edges == 0

    def test_graph_with_edges(self):
        """Test graph with nodes and edges."""
        nodes = [
            GraphNode(
                node_id="n1",
                node_type=NodeType.COMPONENT,
                x=10.0, y=20.0, rotation=0.0, layer="F.Cu",
                features={}
            ),
            GraphNode(
                node_id="n2",
                node_type=NodeType.PAD,
                x=15.0, y=25.0, rotation=0.0, layer="F.Cu",
                features={}
            ),
        ]
        edges = [
            GraphEdge(
                edge_id="e1",
                source_id="n1",
                target_id="n2",
                edge_type=EdgeType.TRACE,
                features={"width": 0.25, "length": 7.07}
            ),
        ]
        graph = PCBGraph(nodes=nodes, edges=edges)

        assert graph.num_nodes == 2
        assert graph.num_edges == 1

    def test_get_node_features(self):
        """Test extracting node feature matrix."""
        nodes = [
            GraphNode(
                node_id="n1",
                node_type=NodeType.COMPONENT,
                x=10.0, y=20.0, rotation=45.0, layer="F.Cu",
                features={}
            ),
            GraphNode(
                node_id="n2",
                node_type=NodeType.VIA,
                x=30.0, y=40.0, rotation=0.0, layer="B.Cu",
                features={}
            ),
        ]
        graph = PCBGraph(nodes=nodes, edges=[])
        features = graph.get_node_features()

        assert features.shape[0] == 2
        assert features.shape[1] == PCBGraphEncoder.NODE_FEATURE_DIM

    def test_get_edge_features(self):
        """Test extracting edge feature matrix."""
        nodes = [
            GraphNode("n1", NodeType.COMPONENT, 0.0, 0.0, 0.0, "F.Cu", {}),
            GraphNode("n2", NodeType.PAD, 10.0, 0.0, 0.0, "F.Cu", {}),
        ]
        edges = [
            GraphEdge("e1", "n1", "n2", EdgeType.TRACE, {"width": 0.25}),
        ]
        graph = PCBGraph(nodes=nodes, edges=edges)
        features = graph.get_edge_features()

        assert features.shape[0] == 1
        assert features.shape[1] == PCBGraphEncoder.EDGE_FEATURE_DIM

    def test_get_adjacency(self):
        """Test getting adjacency matrix."""
        nodes = [
            GraphNode("n1", NodeType.COMPONENT, 0.0, 0.0, 0.0, "F.Cu", {}),
            GraphNode("n2", NodeType.PAD, 10.0, 0.0, 0.0, "F.Cu", {}),
            GraphNode("n3", NodeType.VIA, 20.0, 0.0, 0.0, "F.Cu", {}),
        ]
        edges = [
            GraphEdge("e1", "n1", "n2", EdgeType.TRACE, {}),
            GraphEdge("e2", "n2", "n3", EdgeType.NET, {}),
        ]
        graph = PCBGraph(nodes=nodes, edges=edges)

        source_idx, target_idx = graph.get_adjacency()

        assert len(source_idx) == 2
        assert len(target_idx) == 2

    def test_from_pcb_state(self, mock_pcb_state):
        """Test creating graph from PCB state."""
        graph = PCBGraph.from_pcb_state(mock_pcb_state)

        # Should have nodes for components, vias, zones
        expected_min_nodes = (
            len(mock_pcb_state.components) +
            len(mock_pcb_state.vias) +
            len(mock_pcb_state.zones)
        )
        assert graph.num_nodes >= expected_min_nodes // 2  # Allow for filtering


class TestNodeType:
    """Tests for NodeType enumeration."""

    def test_all_node_types_exist(self):
        """Verify all expected node types exist."""
        expected_types = [
            'COMPONENT', 'PAD', 'VIA', 'ZONE', 'TRACE_POINT', 'BOARD_EDGE'
        ]
        for type_name in expected_types:
            assert hasattr(NodeType, type_name)

    def test_node_type_values(self):
        """Test node type integer values are unique."""
        values = [nt.value for nt in NodeType]
        assert len(values) == len(set(values))


class TestEdgeType:
    """Tests for EdgeType enumeration."""

    def test_all_edge_types_exist(self):
        """Verify all expected edge types exist."""
        expected_types = [
            'TRACE', 'NET', 'SPATIAL', 'LAYER_TRANSITION', 'CLEARANCE_VIOLATION'
        ]
        for type_name in expected_types:
            assert hasattr(EdgeType, type_name)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestPCBGraphEncoder:
    """Tests for PCBGraphEncoder neural network."""

    def test_encoder_initialization(self, pcb_graph_encoder):
        """Test encoder initializes correctly."""
        import torch
        assert isinstance(pcb_graph_encoder, torch.nn.Module)
        assert hasattr(pcb_graph_encoder, 'node_encoder')
        assert hasattr(pcb_graph_encoder, 'edge_encoder')
        assert hasattr(pcb_graph_encoder, 'gnn_layers')

    def test_encoder_forward(self, pcb_graph_encoder, mock_pcb_state):
        """Test forward pass through encoder."""
        import torch
        graph = PCBGraph.from_pcb_state(mock_pcb_state)

        node_features = torch.tensor(graph.get_node_features())
        edge_features = torch.tensor(graph.get_edge_features())
        source_idx, target_idx = graph.get_adjacency()
        edge_index = torch.tensor([source_idx, target_idx], dtype=torch.long)

        if graph.num_nodes == 0:
            pytest.skip("Empty graph generated")

        output = pcb_graph_encoder.forward(
            node_features,
            edge_index,
            edge_features if graph.num_edges > 0 else None,
        )

        # Output should be (1, hidden_dim)
        assert output.shape[0] == 1
        assert output.shape[1] == pcb_graph_encoder.hidden_dim

    def test_encode_graph_method(self, pcb_graph_encoder, mock_pcb_state):
        """Test the encode_graph convenience method."""
        graph = PCBGraph.from_pcb_state(mock_pcb_state)

        if graph.num_nodes == 0:
            pytest.skip("Empty graph generated")

        embedding = pcb_graph_encoder.encode_graph(graph)

        assert embedding.dim() == 2
        assert embedding.shape[1] == pcb_graph_encoder.hidden_dim

    def test_encoder_gradient_flow(self, pcb_graph_encoder, mock_pcb_state):
        """Test gradients flow through encoder."""
        import torch
        graph = PCBGraph.from_pcb_state(mock_pcb_state)

        if graph.num_nodes == 0:
            pytest.skip("Empty graph generated")

        embedding = pcb_graph_encoder.encode_graph(graph)
        loss = embedding.sum()
        loss.backward()

        # Check gradients exist
        for param in pcb_graph_encoder.parameters():
            if param.requires_grad:
                assert param.grad is not None

    def test_encoder_deterministic(self, pcb_graph_encoder, mock_pcb_state):
        """Test encoder produces same output for same input."""
        import torch
        pcb_graph_encoder.eval()
        graph = PCBGraph.from_pcb_state(mock_pcb_state)

        if graph.num_nodes == 0:
            pytest.skip("Empty graph generated")

        with torch.no_grad():
            emb1 = pcb_graph_encoder.encode_graph(graph)
            emb2 = pcb_graph_encoder.encode_graph(graph)

        assert torch.allclose(emb1, emb2)

    def test_encoder_different_graphs(self, pcb_graph_encoder):
        """Test encoder produces different outputs for different graphs."""
        import torch
        pcb_graph_encoder.eval()

        state1 = MockPCBState(num_components=5, violations=100)
        state2 = MockPCBState(num_components=15, violations=500)

        graph1 = PCBGraph.from_pcb_state(state1)
        graph2 = PCBGraph.from_pcb_state(state2)

        if graph1.num_nodes == 0 or graph2.num_nodes == 0:
            pytest.skip("Empty graph generated")

        with torch.no_grad():
            emb1 = pcb_graph_encoder.encode_graph(graph1)
            emb2 = pcb_graph_encoder.encode_graph(graph2)

        # Embeddings should be different
        assert not torch.allclose(emb1, emb2)

    def test_encoder_batch_processing(self, pcb_graph_encoder):
        """Test processing multiple graphs."""
        import torch
        pcb_graph_encoder.eval()

        embeddings = []
        for i in range(5):
            state = MockPCBState(num_components=5 + i, violations=100 + i * 50)
            graph = PCBGraph.from_pcb_state(state)
            if graph.num_nodes > 0:
                with torch.no_grad():
                    emb = pcb_graph_encoder.encode_graph(graph)
                embeddings.append(emb)

        if len(embeddings) < 2:
            pytest.skip("Not enough valid graphs")

        batch = torch.cat(embeddings, dim=0)
        assert batch.shape[0] == len(embeddings)
        assert batch.shape[1] == pcb_graph_encoder.hidden_dim
