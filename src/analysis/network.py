"""
Network analysis module for token co-occurrence visualization.

Uses NetworkX for graph construction and Plotly for visualization.
Nodes represent tokens, edges represent co-occurrences weighted by frequency.
"""

import networkx as nx
import plotly.graph_objects as go
from collections import defaultdict
from typing import List, Dict, Tuple, Set, Optional
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import NETWORK_MIN_COOCCURRENCE, NETWORK_TOP_NODES, PLOTS_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CoOccurrenceNetwork:
    """Network analyzer for token co-occurrence graphs."""
    
    def __init__(self, min_cooccurrence: int = NETWORK_MIN_COOCCURRENCE):
        """
        Initialize co-occurrence network analyzer.
        
        Args:
            min_cooccurrence: Minimum co-occurrence frequency for edges
        """
        self.min_cooccurrence = min_cooccurrence
        self.graph = nx.Graph()
    
    def build_cooccurrence_matrix(self, tokens: List[List[str]], window_size: int = 5) -> Dict[Tuple[str, str], int]:
        """
        Build co-occurrence matrix from tokenized texts.
        
        Args:
            tokens: List of token lists (each inner list is a preprocessed text)
            window_size: Window size for co-occurrence (tokens within N positions)
            
        Returns:
            Dictionary mapping (token1, token2) tuples to co-occurrence counts
        """
        cooccurrence = defaultdict(int)
        
        for token_list in tokens:
            # Consider tokens within window_size of each other as co-occurring
            for i in range(len(token_list)):
                for j in range(i + 1, min(i + window_size + 1, len(token_list))):
                    token1, token2 = token_list[i], token_list[j]
                    # Use sorted tuple to ensure (a, b) == (b, a)
                    pair = tuple(sorted([token1, token2]))
                    cooccurrence[pair] += 1
        
        return dict(cooccurrence)
    
    def build_graph(
        self,
        tokens: List[List[str]],
        min_cooccurrence: Optional[int] = None,
        top_n_nodes: Optional[int] = None,
    ) -> nx.Graph:
        """
        Build NetworkX graph from token co-occurrences.
        
        Args:
            tokens: List of token lists
            min_cooccurrence: Minimum co-occurrence frequency (overrides instance default)
            top_n_nodes: Limit to top N nodes by degree
            
        Returns:
            NetworkX Graph object
        """
        if min_cooccurrence is None:
            min_cooccurrence = self.min_cooccurrence
        
        logger.info(f"Building co-occurrence network (min_cooccurrence={min_cooccurrence})")
        
        # Build co-occurrence matrix
        cooccurrence = self.build_cooccurrence_matrix(tokens)
        
        # Create graph
        G = nx.Graph()
        
        # Add edges with weights
        for (token1, token2), count in cooccurrence.items():
            if count >= min_cooccurrence:
                G.add_edge(token1, token2, weight=count)
        
        # Optionally limit to top N nodes by degree
        if top_n_nodes is not None and len(G.nodes()) > top_n_nodes:
            logger.info(f"Limiting to top {top_n_nodes} nodes by degree")
            degrees = dict(G.degree())
            top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:top_n_nodes]
            top_node_set = set(node for node, _ in top_nodes)
            G = G.subgraph(top_node_set).copy()
        
        self.graph = G
        
        logger.info(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        return G
    
    def get_network_statistics(self, G: Optional[nx.Graph] = None) -> Dict:
        """
        Get statistics about the network.
        
        Args:
            G: NetworkX graph (uses self.graph if None)
            
        Returns:
            Dictionary with network statistics
        """
        if G is None:
            G = self.graph
        
        if G.number_of_nodes() == 0:
            return {}
        
        degrees = dict(G.degree())
        
        stats = {
            'num_nodes': G.number_of_nodes(),
            'num_edges': G.number_of_edges(),
            'avg_degree': sum(degrees.values()) / len(degrees),
            'max_degree': max(degrees.values()),
            'min_degree': min(degrees.values()),
            'density': nx.density(G),
            'avg_clustering': nx.average_clustering(G),
        }
        
        # Try to compute other metrics if possible
        try:
            if nx.is_connected(G):
                stats['diameter'] = nx.diameter(G)
                stats['avg_path_length'] = nx.average_shortest_path_length(G)
        except:
            pass
        
        return stats
    
    def visualize(
        self,
        G: Optional[nx.Graph] = None,
        layout: str = "spring",
        node_size_factor: float = 10.0,
        edge_width_factor: float = 0.5,
        save_path: Optional[Path] = None,
    ) -> go.Figure:
        """
        Visualize network using Plotly.
        
        Args:
            G: NetworkX graph (uses self.graph if None)
            layout: Layout algorithm ('spring', 'circular', 'kamada_kawai')
            node_size_factor: Factor to scale node sizes
            edge_width_factor: Factor to scale edge widths
            save_path: Path to save plot (optional)
            
        Returns:
            Plotly Figure object
        """
        if G is None:
            G = self.graph
        
        if G.number_of_nodes() == 0:
            logger.warning("Graph is empty, cannot visualize")
            return None
        
        logger.info(f"Visualizing network with {G.number_of_nodes()} nodes")
        
        # Compute layout
        if layout == "spring":
            pos = nx.spring_layout(G, k=1, iterations=50)
        elif layout == "circular":
            pos = nx.circular_layout(G)
        elif layout == "kamada_kawai":
            try:
                pos = nx.kamada_kawai_layout(G)
            except:
                pos = nx.spring_layout(G)
        else:
            pos = nx.spring_layout(G)
        
        # Extract node and edge information
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        node_text = list(G.nodes())
        node_degrees = [G.degree(node) for node in G.nodes()]
        
        # Create edge traces
        edge_x = []
        edge_y = []
        edge_weights = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_weights.append(G[edge[0]][edge[1]].get('weight', 1))
        
        # Create edge trace
        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=[w * edge_width_factor for w in edge_weights for _ in range(3)], color='#888'),
            hoverinfo='none',
            mode='lines',
        )
        
        # Create node trace
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            textposition="middle center",
            marker=dict(
                size=[d * node_size_factor for d in node_degrees],
                color=node_degrees,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Degree"),
                line=dict(width=2, color='white'),
            ),
        )
        
        # Create figure
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title='Token Co-occurrence Network',
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                annotations=[
                    dict(
                        text=f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}",
                        showarrow=False,
                        xref="paper",
                        yref="paper",
                        x=0.005,
                        y=-0.002,
                        xanchor="left",
                        yanchor="bottom",
                        font=dict(size=12),
                    )
                ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            ),
        )
        
        # Save if path provided
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(str(save_path))
            logger.info(f"Saved network visualization to {save_path}")
        
        return fig
