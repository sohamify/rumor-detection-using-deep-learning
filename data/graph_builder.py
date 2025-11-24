# data/graph_builder.py
import json
import torch
import dgl
from datetime import datetime
import os

def build_dgl_graph_from_sample(sample_json_path):
    """
    Converts a single sample JSON (with replies) → timestamped bidirectional DGL graph
    Exactly like CausalMamba paper (Zhan & Cheng, 2025)
    """
    with open(sample_json_path, 'r', encoding='utf-8') as f:
        sample = json.load(f)
    
    source_text = sample["source_text"]
    replies = sample.get("replies", [])
    
    # Build node list: [0] = source, [1:] = replies
    nodes = [{"text": source_text, "timestamp": "2025-01-01T00:00:00Z", "is_source": True}]
    node_texts = [source_text]
    timestamps = [datetime.fromisoformat("2025-01-01T00:00:00Z")]
    
    # DFS to assign parent → child edges
    edges_src = []
    edges_dst = []
    
    def add_reply(reply, parent_idx):
        idx = len(nodes)
        ts = reply.get("timestamp", datetime.now().isoformat())
        nodes.append({"text": reply["text"], "timestamp": ts, "is_source": False})
        node_texts.append(reply["text"])
        timestamps.append(datetime.fromisoformat(ts.replace("Z", "+00:00")))
        edges_src.append(parent_idx)
        edges_dst.append(idx)
        # Recurse
        for sub in reply.get("replies", []):
            add_reply(sub, idx)
    
    for reply in replies:
        add_reply(reply, 0)
    
    # Create bidirectional graph (for GCN)
    g = dgl.graph((edges_src + edges_dst, edges_dst + edges_src))
    
    # Node features: dummy text embedding (replace with SoLM later)
    g.ndata['feat'] = torch.randn(g.num_nodes(), 768)
    g.ndata['timestamp'] = torch.tensor([(ts - timestamps[0]).total_seconds() for ts in timestamps])
    g.ndata['is_source'] = torch.zeros(g.num_nodes(), dtype=torch.bool)
    g.ndata['is_source'][0] = True
    
    return g, node_texts

# Test
if __name__ == "__main__":
    g, texts = build_dgl_graph_from_sample("data/sample_data/sample_00001.json")
    print(f"Graph: {g.num_nodes()} nodes, {g.num_edges()} edges")
    print(f"Source node: {texts[0][:100]}")