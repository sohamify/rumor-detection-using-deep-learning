# models/components/text_encoder.py
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class SoLMEncoder(nn.Module):
    def __init__(self, freeze=True):
        super().__init__()
        # Replace with actual SoLM when available
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-large")
        self.model = AutoModel.from_pretrained("roberta-large")
        
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
                
        print("SoLM placeholder loaded (RoBERTa-large) — 1024-dim")

    def forward(self, text):
        inputs = self.tokenizer(
            text, padding=True, truncation=True, max_length=512, return_tensors="pt"
        ).to(next(self.model.parameters()).device)
        
        with torch.no_grad() if not self.model.training else torch.enable_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :]  # [CLS] token