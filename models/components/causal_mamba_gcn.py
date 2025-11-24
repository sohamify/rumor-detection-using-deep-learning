# models/components/causal_mamba_gcn.py
import torch
import torch.nn as nn
from mamba_ssm import Mamba
import dgl
import torch.nn.functional as F

class CausalMambaGCN(nn.Module):
    def __init__(self, hidden_dim=768, n_layers=3):
        super().__init__()
        self.mamba = Mamba(
            d_model=hidden_dim,
            d_state=16,
            d_conv=4,
            expand=2,
        )
        self.gcn_layers = nn.ModuleList([
            dgl.nn.GraphConv(hidden_dim, hidden_dim) for _ in range(n_layers)
        ])
        # Placeholder for NOTEARS-style linear projection
        self.notears_proj = nn.Linear(hidden_dim, hidden_dim)
        self.causal_head = nn.Linear(hidden_dim, 1)  # Causal influence score
        
    def forward(self, g, node_feats):
        # Step 1: Mamba over temporal sequence (ordered by timestamp)
        sorted_idx = torch.argsort(g.ndata['timestamp'])
        seq_feats = node_feats[sorted_idx]
        mamba_out = self.mamba(seq_feats)  # [N, D]
        
        # Step 2: GCN over graph structure
        h = mamba_out
        for gcn in self.gcn_layers:
            h = F.gelu(gcn(g, h))
        
        # Step 3: NOTEARS-style causal projection (simplified)
        causal_scores = torch.sigmoid(self.causal_head(h))  # [N, 1]
        
        # Return graph representation + causal DAG scores
        graph_repr = dgl.mean_nodes(g, 'h', h)
        return {
            "graph_feat": graph_repr,
            "causal_scores": causal_scores.squeeze(-1),
            "influencer_mask": causal_scores.squeeze(-1) > 0.7  # top influencers
        }

# Test
if __name__ == "__main__":
    from data.graph_builder import build_dgl_graph_from_sample
    g, _ = build_dgl_graph_from_sample("data/sample_data/sample_00001.json")
    model = CausalMambaGCN()
    out = model(g, g.ndata['feat'])
    print("Graph feature shape:", out["graph_feat"].shape)
    print("Top influencers:", out["influencer_mask"].sum().item())