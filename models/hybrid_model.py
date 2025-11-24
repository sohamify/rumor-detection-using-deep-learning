from models.components.causal_mamba_gcn import CausalMambaGCN

class HybridADMambaCLIP(nn.Module):
    def __init__(self):
        super().__init__()
        # ... existing encoders ...
        self.causal_mamba = CausalMambaGCN()
        # ... rest ...
    
    def forward(self, text, pixel_values=None, graph=None, missing_mask=None):
        h_text = self.text_encoder(text)
        h_img = ...  # your image encoder
        h_graph_out = self.causal_mamba(graph, graph.ndata['feat'])
        h_graph = h_graph_out["graph_feat"]
        
        # Use causal interpretability
        causal_influencers = h_graph_out["influencer_mask"]
        
        # Final fusion with TriSPrompt
        h_fused = self.trisprompt(h_text, h_img, h_graph, missing_mask)
        # ... rest of forward ...
        
        return {
            "anomaly_score": ...,
            "ternary_logits": ...,
            "causal_influencers": causal_influencers  # for explanation
        }