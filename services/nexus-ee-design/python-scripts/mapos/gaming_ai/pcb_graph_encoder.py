"""
PCB Graph Encoder - Graph Neural Network for PCB State Representation

This module implements an edge-based Graph Neural Network (GNN) that encodes
PCB designs as graphs, inspired by AlphaChip's circuit placement network.

Architecture:
- Nodes: Components, pads, vias (with position, type, properties)
- Edges: Traces, nets, spatial proximity (with width, length, layer)
- Message Passing: Edge-conditioned convolution for PCB-specific relationships
- Global Pooling: Attention-based aggregation for board-level representation

References:
- AlphaChip: https://github.com/google-research/circuit_training
- GraphSAGE: https://arxiv.org/abs/1706.02216
- Edge Convolution: https://arxiv.org/abs/1801.07829
"""

import math
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set, Any, Union
from enum import Enum, auto

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Create placeholder classes for type hints
    class nn:
        class Module:
            pass


class NodeType(Enum):
    """Types of nodes in the PCB graph."""
    COMPONENT = 0       # IC, resistor, capacitor, etc.
    PAD = 1            # Component pad
    VIA = 2            # Through-hole via
    ZONE_ANCHOR = 3    # Representative point of copper zone
    TEST_POINT = 4     # Test point


class EdgeType(Enum):
    """Types of edges in the PCB graph."""
    TRACE = 0          # Copper trace connection
    NET = 1            # Logical net connection
    SPATIAL = 2        # Spatial proximity (for clearance)
    THERMAL = 3        # Thermal connection
    COMPONENT_PIN = 4  # Component to pad connection


@dataclass
class GraphNode:
    """A node in the PCB graph."""
    node_id: str
    node_type: NodeType
    x: float
    y: float
    layer: str
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_feature_vector(self, board_width: float = 250.0, board_height: float = 85.0) -> np.ndarray:
        """Convert node to feature vector."""
        # Normalize position to [0, 1]
        norm_x = self.x / board_width
        norm_y = self.y / board_height

        # One-hot encode node type (5 types)
        type_onehot = np.zeros(5)
        type_onehot[self.node_type.value] = 1.0

        # Layer encoding (front=1, back=-1, inner=0)
        layer_enc = 1.0 if 'F.' in self.layer else (-1.0 if 'B.' in self.layer else 0.0)

        # Property-based features
        rotation = self.properties.get('rotation', 0.0) / 360.0
        pin_count = min(self.properties.get('pin_count', 1) / 200.0, 1.0)  # Normalize
        is_power = 1.0 if self.properties.get('is_power', False) else 0.0
        is_ground = 1.0 if self.properties.get('is_ground', False) else 0.0
        diameter = self.properties.get('diameter', 0.0) / 2.0  # Normalize to ~0-1

        return np.array([
            norm_x, norm_y,           # Position (2)
            *type_onehot,             # Node type one-hot (5)
            layer_enc,                # Layer encoding (1)
            rotation,                 # Rotation (1)
            pin_count,                # Pin count (1)
            is_power,                 # Power net flag (1)
            is_ground,                # Ground net flag (1)
            diameter,                 # Via/pad diameter (1)
        ], dtype=np.float32)


@dataclass
class GraphEdge:
    """An edge in the PCB graph."""
    source_id: str
    target_id: str
    edge_type: EdgeType
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_feature_vector(self) -> np.ndarray:
        """Convert edge to feature vector."""
        # One-hot encode edge type (5 types)
        type_onehot = np.zeros(5)
        type_onehot[self.edge_type.value] = 1.0

        # Edge properties
        width = self.properties.get('width', 0.25) / 5.0  # Normalize trace width
        length = min(self.properties.get('length', 0.0) / 100.0, 1.0)  # Normalize length
        same_layer = 1.0 if self.properties.get('same_layer', True) else 0.0
        is_differential = 1.0 if self.properties.get('differential', False) else 0.0
        impedance = self.properties.get('impedance', 50.0) / 120.0  # Normalize impedance

        return np.array([
            *type_onehot,             # Edge type one-hot (5)
            width,                    # Trace width (1)
            length,                   # Trace length (1)
            same_layer,               # Same layer flag (1)
            is_differential,          # Differential pair flag (1)
            impedance,                # Impedance (1)
        ], dtype=np.float32)


@dataclass
class PCBGraph:
    """
    Graph representation of a PCB design.

    This is the input format for the PCBGraphEncoder neural network.
    Can be constructed from a PCBState or loaded from cache.
    """
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    board_width: float = 250.0
    board_height: float = 85.0
    layer_count: int = 10
    metadata: Dict[str, Any] = field(default_factory=dict)

    _node_id_to_idx: Dict[str, int] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        """Build node ID to index mapping."""
        self._node_id_to_idx = {node.node_id: i for i, node in enumerate(self.nodes)}

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_edges(self) -> int:
        return len(self.edges)

    def get_node_features(self) -> np.ndarray:
        """Get node feature matrix [num_nodes, node_feature_dim]."""
        if not self.nodes:
            return np.zeros((0, 13), dtype=np.float32)
        return np.stack([
            node.to_feature_vector(self.board_width, self.board_height)
            for node in self.nodes
        ])

    def get_edge_features(self) -> np.ndarray:
        """Get edge feature matrix [num_edges, edge_feature_dim]."""
        if not self.edges:
            return np.zeros((0, 10), dtype=np.float32)
        return np.stack([edge.to_feature_vector() for edge in self.edges])

    def get_edge_index(self) -> np.ndarray:
        """Get edge index tensor [2, num_edges]."""
        if not self.edges:
            return np.zeros((2, 0), dtype=np.int64)

        sources = []
        targets = []
        for edge in self.edges:
            if edge.source_id in self._node_id_to_idx and edge.target_id in self._node_id_to_idx:
                sources.append(self._node_id_to_idx[edge.source_id])
                targets.append(self._node_id_to_idx[edge.target_id])

        return np.array([sources, targets], dtype=np.int64)

    def get_adjacency_matrix(self) -> np.ndarray:
        """Get dense adjacency matrix [num_nodes, num_nodes]."""
        adj = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
        edge_index = self.get_edge_index()
        for i in range(edge_index.shape[1]):
            src, tgt = edge_index[0, i], edge_index[1, i]
            adj[src, tgt] = 1.0
            adj[tgt, src] = 1.0  # Undirected
        return adj

    @classmethod
    def from_pcb_state(cls, pcb_state: Any, include_spatial: bool = True,
                       spatial_threshold: float = 5.0) -> 'PCBGraph':
        """
        Construct PCBGraph from a PCBState object.

        Args:
            pcb_state: PCBState object from pcb_state.py
            include_spatial: Whether to add spatial proximity edges
            spatial_threshold: Distance threshold for spatial edges (mm)

        Returns:
            PCBGraph ready for neural network processing
        """
        nodes = []
        edges = []

        # Detect power and ground nets
        power_nets = {name for name in pcb_state.nets.keys()
                      if any(p in name.upper() for p in ['+', 'VDD', 'VCC', 'VIN', 'VBUS'])}
        ground_nets = {name for name in pcb_state.nets.keys()
                       if 'GND' in name.upper() or 'VSS' in name.upper()}

        # Add component nodes
        for ref, comp in pcb_state.components.items():
            nodes.append(GraphNode(
                node_id=f"comp_{ref}",
                node_type=NodeType.COMPONENT,
                x=comp.x,
                y=comp.y,
                layer=comp.layer,
                properties={
                    'rotation': comp.rotation,
                    'footprint': comp.footprint,
                    'pin_count': _estimate_pin_count(comp.footprint),
                    'reference': ref,
                }
            ))

        # Add via nodes
        for i, via in enumerate(pcb_state.vias):
            is_power = via.net_name in power_nets
            is_ground = via.net_name in ground_nets
            nodes.append(GraphNode(
                node_id=f"via_{i}",
                node_type=NodeType.VIA,
                x=via.x,
                y=via.y,
                layer=via.layers[0],  # Primary layer
                properties={
                    'diameter': via.diameter,
                    'drill': via.drill,
                    'net_name': via.net_name,
                    'is_power': is_power,
                    'is_ground': is_ground,
                }
            ))

        # Add zone anchor nodes (representative points)
        for zone_key, zone in pcb_state.zones.items():
            center_x = (zone.bounds[0] + zone.bounds[2]) / 2
            center_y = (zone.bounds[1] + zone.bounds[3]) / 2
            is_power = zone.net_name in power_nets
            is_ground = zone.net_name in ground_nets
            nodes.append(GraphNode(
                node_id=f"zone_{zone_key}",
                node_type=NodeType.ZONE_ANCHOR,
                x=center_x,
                y=center_y,
                layer=zone.layer,
                properties={
                    'net_name': zone.net_name,
                    'clearance': zone.clearance,
                    'is_power': is_power,
                    'is_ground': is_ground,
                }
            ))

        # Create graph instance to get node ID mapping
        graph = cls(nodes=nodes, edges=[])

        # Add trace edges
        # Group traces by net to find connections
        net_traces: Dict[str, List[Any]] = {}
        for trace in pcb_state.traces:
            if trace.net_name not in net_traces:
                net_traces[trace.net_name] = []
            net_traces[trace.net_name].append(trace)

        # For each net, find connected nodes
        for net_name, traces in net_traces.items():
            connected_node_ids = set()

            # Find vias on this net
            for node in nodes:
                if node.node_type == NodeType.VIA:
                    if node.properties.get('net_name') == net_name:
                        connected_node_ids.add(node.node_id)
                elif node.node_type == NodeType.ZONE_ANCHOR:
                    if node.properties.get('net_name') == net_name:
                        connected_node_ids.add(node.node_id)

            # Create edges between all nodes on this net
            node_list = list(connected_node_ids)
            total_length = sum(
                math.sqrt((t.end_x - t.start_x)**2 + (t.end_y - t.start_y)**2)
                for t in traces
            )
            avg_width = sum(t.width for t in traces) / len(traces) if traces else 0.25

            for i in range(len(node_list)):
                for j in range(i + 1, len(node_list)):
                    edges.append(GraphEdge(
                        source_id=node_list[i],
                        target_id=node_list[j],
                        edge_type=EdgeType.NET,
                        properties={
                            'net_name': net_name,
                            'width': avg_width,
                            'length': total_length / max(1, len(node_list) - 1),
                            'same_layer': True,
                            'is_power': net_name in power_nets,
                            'is_ground': net_name in ground_nets,
                        }
                    ))

        # Add spatial proximity edges
        if include_spatial:
            for i, node_i in enumerate(nodes):
                for j in range(i + 1, len(nodes)):
                    node_j = nodes[j]
                    dist = math.sqrt((node_i.x - node_j.x)**2 + (node_i.y - node_j.y)**2)

                    if dist < spatial_threshold:
                        # Check if same layer
                        same_layer = (node_i.layer == node_j.layer or
                                      'Cu' not in node_i.layer or 'Cu' not in node_j.layer)

                        edges.append(GraphEdge(
                            source_id=node_i.node_id,
                            target_id=node_j.node_id,
                            edge_type=EdgeType.SPATIAL,
                            properties={
                                'distance': dist,
                                'length': dist,
                                'same_layer': same_layer,
                            }
                        ))

        # Extract board dimensions from state if available
        board_width = 250.0
        board_height = 85.0

        if pcb_state.components:
            max_x = max(c.x for c in pcb_state.components.values())
            max_y = max(c.y for c in pcb_state.components.values())
            board_width = max(board_width, max_x + 10)
            board_height = max(board_height, max_y + 10)

        return cls(
            nodes=nodes,
            edges=edges,
            board_width=board_width,
            board_height=board_height,
            layer_count=10,
            metadata={
                'component_count': len(pcb_state.components),
                'via_count': len(pcb_state.vias),
                'trace_count': len(pcb_state.traces),
                'zone_count': len(pcb_state.zones),
                'net_count': len(pcb_state.nets),
                'state_id': getattr(pcb_state, 'state_id', 'unknown'),
            }
        )

    def get_hash(self) -> str:
        """Get hash of graph structure for caching."""
        content = f"{self.num_nodes}_{self.num_edges}_{self.board_width}_{self.board_height}"
        for node in sorted(self.nodes, key=lambda n: n.node_id):
            content += f"_{node.node_id}_{node.x:.2f}_{node.y:.2f}"
        return hashlib.md5(content.encode()).hexdigest()[:12]


def _estimate_pin_count(footprint: str) -> int:
    """Estimate pin count from footprint name."""
    footprint_upper = footprint.upper()

    # Extract number from common patterns
    import re

    # QFP-64, LQFP-144, etc.
    match = re.search(r'(QFP|TQFP|LQFP|BGA|LGA)[-_]?(\d+)', footprint_upper)
    if match:
        return int(match.group(2))

    # SOT-23, SOIC-8, etc.
    match = re.search(r'(SOT|SOIC|SOP|SSOP|TSSOP|MSOP)[-_]?(\d+)', footprint_upper)
    if match:
        return int(match.group(2))

    # DIP-8, etc.
    match = re.search(r'DIP[-_]?(\d+)', footprint_upper)
    if match:
        return int(match.group(1))

    # 0402, 0603, 0805 - passives have 2 pins
    if re.match(r'^0[0-9]{3}$', footprint_upper):
        return 2

    # Default
    return 4


if TORCH_AVAILABLE:

    class EdgeConvLayer(nn.Module):
        """
        Edge-conditioned graph convolution layer.

        Implements message passing where edge features modulate the messages:
        m_ij = MLP([h_i || h_j || e_ij])
        h_i' = h_i + Aggregate({m_ij : j in N(i)})
        """

        def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int, dropout: float = 0.1):
            super().__init__()
            self.node_dim = node_dim
            self.edge_dim = edge_dim
            self.hidden_dim = hidden_dim

            # Message MLP: [source_node || target_node || edge] -> message
            self.message_mlp = nn.Sequential(
                nn.Linear(node_dim * 2 + edge_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            )

            # Update MLP: [node || aggregated_messages] -> updated_node
            self.update_mlp = nn.Sequential(
                nn.Linear(node_dim + hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, node_dim),
            )

            # Residual connection
            self.residual = nn.Linear(node_dim, node_dim) if node_dim != hidden_dim else nn.Identity()

        def forward(self, node_features: torch.Tensor, edge_features: torch.Tensor,
                    edge_index: torch.Tensor) -> torch.Tensor:
            """
            Forward pass of edge convolution.

            Args:
                node_features: [num_nodes, node_dim]
                edge_features: [num_edges, edge_dim]
                edge_index: [2, num_edges] source and target node indices

            Returns:
                Updated node features [num_nodes, node_dim]
            """
            num_nodes = node_features.size(0)
            num_edges = edge_index.size(1)

            if num_edges == 0:
                return node_features

            # Get source and target node features
            source_idx = edge_index[0]  # [num_edges]
            target_idx = edge_index[1]  # [num_edges]

            source_features = node_features[source_idx]  # [num_edges, node_dim]
            target_features = node_features[target_idx]  # [num_edges, node_dim]

            # Compute messages
            message_input = torch.cat([source_features, target_features, edge_features], dim=-1)
            messages = self.message_mlp(message_input)  # [num_edges, hidden_dim]

            # Aggregate messages per node (mean aggregation)
            aggregated = torch.zeros(num_nodes, self.hidden_dim, device=node_features.device)
            counts = torch.zeros(num_nodes, 1, device=node_features.device)

            # Scatter add for aggregation
            aggregated.scatter_add_(0, target_idx.unsqueeze(-1).expand(-1, self.hidden_dim), messages)
            counts.scatter_add_(0, target_idx.unsqueeze(-1), torch.ones_like(target_idx.unsqueeze(-1).float()))

            # Mean aggregation (avoid division by zero)
            counts = counts.clamp(min=1)
            aggregated = aggregated / counts

            # Update nodes
            update_input = torch.cat([node_features, aggregated], dim=-1)
            updated = self.update_mlp(update_input)

            # Residual connection
            return updated + self.residual(node_features)


    class GlobalAttentionPooling(nn.Module):
        """
        Attention-based global graph pooling.

        Computes attention weights for each node and aggregates to global representation.
        """

        def __init__(self, hidden_dim: int):
            super().__init__()
            self.hidden_dim = hidden_dim

            # Attention mechanism
            self.attention = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, 1),
            )

            # Final transformation
            self.transform = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            )

        def forward(self, node_features: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
            """
            Global pooling with attention.

            Args:
                node_features: [num_nodes, hidden_dim]
                batch: [num_nodes] batch assignment (optional, for batched graphs)

            Returns:
                Global graph representation [batch_size, hidden_dim]
            """
            # Compute attention scores
            attention_scores = self.attention(node_features)  # [num_nodes, 1]

            if batch is None:
                # Single graph
                attention_weights = F.softmax(attention_scores, dim=0)
                global_repr = (attention_weights * node_features).sum(dim=0, keepdim=True)
            else:
                # Batched graphs - compute softmax per graph
                batch_size = batch.max().item() + 1

                # Compute softmax per batch
                attention_weights = torch.zeros_like(attention_scores)
                for b in range(batch_size):
                    mask = (batch == b)
                    attention_weights[mask] = F.softmax(attention_scores[mask], dim=0)

                # Weighted sum per batch
                weighted = attention_weights * node_features
                global_repr = torch.zeros(batch_size, self.hidden_dim, device=node_features.device)
                global_repr.scatter_add_(0, batch.unsqueeze(-1).expand(-1, self.hidden_dim), weighted)

            return self.transform(global_repr)


    class PCBGraphEncoder(nn.Module):
        """
        Graph Neural Network encoder for PCB designs.

        Architecture:
        1. Node/Edge encoders: Project raw features to hidden dimension
        2. GNN layers: Edge-conditioned message passing
        3. Global pooling: Attention-based graph-level representation

        Inspired by AlphaChip's placement network and GNN-based EDA tools.
        """

        NODE_FEATURE_DIM = 13  # From GraphNode.to_feature_vector()
        EDGE_FEATURE_DIM = 10  # From GraphEdge.to_feature_vector()

        def __init__(
            self,
            hidden_dim: int = 256,
            num_layers: int = 6,
            dropout: float = 0.1,
            use_layer_norm: bool = True,
        ):
            """
            Initialize PCB Graph Encoder.

            Args:
                hidden_dim: Hidden dimension for all layers
                num_layers: Number of GNN message passing layers
                dropout: Dropout rate
                use_layer_norm: Whether to use layer normalization
            """
            super().__init__()
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers

            # Node encoder
            self.node_encoder = nn.Sequential(
                nn.Linear(self.NODE_FEATURE_DIM, hidden_dim),
                nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity(),
                nn.ReLU(),
            )

            # Edge encoder
            self.edge_encoder = nn.Sequential(
                nn.Linear(self.EDGE_FEATURE_DIM, hidden_dim),
                nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity(),
                nn.ReLU(),
            )

            # GNN layers
            self.gnn_layers = nn.ModuleList([
                EdgeConvLayer(hidden_dim, hidden_dim, hidden_dim, dropout)
                for _ in range(num_layers)
            ])

            # Global pooling
            self.global_pool = GlobalAttentionPooling(hidden_dim)

            # Final projection
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity(),
                nn.ReLU(),
            )

        def forward(
            self,
            node_features: torch.Tensor,
            edge_features: torch.Tensor,
            edge_index: torch.Tensor,
            batch: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            """
            Encode PCB graph to fixed-size representation.

            Args:
                node_features: [num_nodes, NODE_FEATURE_DIM]
                edge_features: [num_edges, EDGE_FEATURE_DIM]
                edge_index: [2, num_edges]
                batch: [num_nodes] batch assignment for batched graphs

            Returns:
                Graph representation [batch_size, hidden_dim]
            """
            # Encode nodes and edges
            h_nodes = self.node_encoder(node_features)
            h_edges = self.edge_encoder(edge_features)

            # Message passing
            for layer in self.gnn_layers:
                h_nodes = layer(h_nodes, h_edges, edge_index)

            # Global pooling
            graph_repr = self.global_pool(h_nodes, batch)

            # Final projection
            return self.output_proj(graph_repr)

        def encode_graph(self, pcb_graph: PCBGraph) -> torch.Tensor:
            """
            Convenience method to encode a PCBGraph object.

            Args:
                pcb_graph: PCBGraph object

            Returns:
                Graph embedding [1, hidden_dim]
            """
            # Convert to tensors
            node_features = torch.tensor(pcb_graph.get_node_features(), dtype=torch.float32)
            edge_features = torch.tensor(pcb_graph.get_edge_features(), dtype=torch.float32)
            edge_index = torch.tensor(pcb_graph.get_edge_index(), dtype=torch.long)

            # Handle empty graph
            if node_features.size(0) == 0:
                return torch.zeros(1, self.hidden_dim)

            # Forward pass
            with torch.no_grad():
                return self(node_features, edge_features, edge_index)

        def encode_batch(self, graphs: List[PCBGraph]) -> torch.Tensor:
            """
            Encode a batch of PCBGraph objects.

            Args:
                graphs: List of PCBGraph objects

            Returns:
                Batch of graph embeddings [batch_size, hidden_dim]
            """
            if not graphs:
                return torch.zeros(0, self.hidden_dim)

            # Concatenate all graphs with batch indices
            all_node_features = []
            all_edge_features = []
            all_edge_indices = []
            batch_indices = []

            node_offset = 0
            for batch_idx, graph in enumerate(graphs):
                node_feat = graph.get_node_features()
                edge_feat = graph.get_edge_features()
                edge_idx = graph.get_edge_index()

                all_node_features.append(node_feat)
                all_edge_features.append(edge_feat)
                all_edge_indices.append(edge_idx + node_offset)
                batch_indices.extend([batch_idx] * graph.num_nodes)

                node_offset += graph.num_nodes

            # Stack tensors
            node_features = torch.tensor(np.concatenate(all_node_features, axis=0), dtype=torch.float32)
            edge_features = torch.tensor(np.concatenate(all_edge_features, axis=0), dtype=torch.float32)
            edge_index = torch.tensor(np.concatenate(all_edge_indices, axis=1), dtype=torch.long)
            batch = torch.tensor(batch_indices, dtype=torch.long)

            with torch.no_grad():
                return self(node_features, edge_features, edge_index, batch)


else:
    # Fallback implementation without PyTorch
    class PCBGraphEncoder:
        """Fallback PCB Graph Encoder (requires PyTorch for full functionality)."""

        NODE_FEATURE_DIM = 13
        EDGE_FEATURE_DIM = 10

        def __init__(self, hidden_dim: int = 256, **kwargs):
            self.hidden_dim = hidden_dim
            import warnings
            warnings.warn("PyTorch not available. PCBGraphEncoder will use random embeddings.")

        def encode_graph(self, pcb_graph: PCBGraph) -> np.ndarray:
            """Return random embedding (placeholder)."""
            # Use graph hash for deterministic random
            np.random.seed(int(pcb_graph.get_hash(), 16) % (2**31))
            return np.random.randn(1, self.hidden_dim).astype(np.float32)

        def encode_batch(self, graphs: List[PCBGraph]) -> np.ndarray:
            """Return random embeddings for batch."""
            return np.stack([self.encode_graph(g).squeeze() for g in graphs])


if __name__ == '__main__':
    # Test the encoder
    print("PCB Graph Encoder Test")
    print("=" * 60)

    # Create a test graph manually
    nodes = [
        GraphNode("comp_U1", NodeType.COMPONENT, 50.0, 40.0, "F.Cu",
                  {'rotation': 0, 'footprint': 'LQFP-144', 'pin_count': 144}),
        GraphNode("comp_C1", NodeType.COMPONENT, 55.0, 42.0, "F.Cu",
                  {'rotation': 0, 'footprint': '0603', 'pin_count': 2}),
        GraphNode("via_0", NodeType.VIA, 52.0, 41.0, "F.Cu",
                  {'diameter': 0.8, 'drill': 0.4, 'net_name': 'GND', 'is_ground': True}),
        GraphNode("via_1", NodeType.VIA, 48.0, 39.0, "F.Cu",
                  {'diameter': 0.8, 'drill': 0.4, 'net_name': '+3V3', 'is_power': True}),
    ]

    edges = [
        GraphEdge("comp_U1", "via_0", EdgeType.NET, {'net_name': 'GND', 'width': 0.25}),
        GraphEdge("comp_C1", "via_0", EdgeType.NET, {'net_name': 'GND', 'width': 0.25}),
        GraphEdge("comp_U1", "via_1", EdgeType.NET, {'net_name': '+3V3', 'width': 0.5}),
        GraphEdge("comp_U1", "comp_C1", EdgeType.SPATIAL, {'distance': 5.4, 'same_layer': True}),
    ]

    graph = PCBGraph(nodes=nodes, edges=edges)

    print(f"Nodes: {graph.num_nodes}")
    print(f"Edges: {graph.num_edges}")
    print(f"Node features shape: {graph.get_node_features().shape}")
    print(f"Edge features shape: {graph.get_edge_features().shape}")
    print(f"Edge index shape: {graph.get_edge_index().shape}")

    if TORCH_AVAILABLE:
        print("\nTesting PyTorch encoder...")
        encoder = PCBGraphEncoder(hidden_dim=256, num_layers=4)

        # Count parameters
        num_params = sum(p.numel() for p in encoder.parameters())
        print(f"Model parameters: {num_params:,}")

        # Encode graph
        embedding = encoder.encode_graph(graph)
        print(f"Embedding shape: {embedding.shape}")
        print(f"Embedding norm: {embedding.norm().item():.4f}")

        # Test batch encoding
        embeddings = encoder.encode_batch([graph, graph])
        print(f"Batch embedding shape: {embeddings.shape}")
    else:
        print("\nPyTorch not available, using fallback encoder")
        encoder = PCBGraphEncoder(hidden_dim=256)
        embedding = encoder.encode_graph(graph)
        print(f"Embedding shape: {embedding.shape}")
