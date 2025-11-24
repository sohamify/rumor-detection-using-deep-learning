# models/components/prompts.py
import torch
import torch.nn as nn

class TriSPrompt(nn.Module):
    def __init__(self, dim=1024, prompt_len=20):
        super().__init__()
        # 3 modality-aware prompts
        self.ma_prompts = nn.Parameter(torch.randn(3, prompt_len, dim))
        # 8 missing modality prompts (000 to 111)
        self.mm_prompts = nn.Parameter(torch.randn(8, prompt_len, dim))
        # 1 mutual-view prompt
        self.mv_prompt = nn.Parameter(torch.randn(prompt_len, dim))
        self.proj = nn.Linear(dim, dim)
        nn.init.xavier_uniform_(self.ma_prompts)
        nn.init.xavier_uniform_(self.mm_prompts)

    def forward(self, h_text=None, h_img=None, h_graph=None, mask=None):
        B = h_text.size(0) if h_text is not None else (h_img.size(0) if h_img is not None else h_graph.size(0))
        device = h_text.device if h_text is not None else h_img.device
        
        # Base missing mask prompt
        mask_idx = int("".join(str(int(m)) for m in mask.tolist()), 2) if mask is not None else 0
        prompt = self.mm_prompts[mask_idx].expand(B, -1, -1)
        
        # Add present modality prompts
        if mask[0] and h_text is not None: prompt += self.ma_prompts[0]
        if mask[1] and h_img is not None:  prompt += self.ma_prompts[1]
        if mask[2] and h_graph is not None: prompt += self.ma_prompts[2]
        
        prompt = prompt + self.mv_prompt
        
        return h_text + self.proj(prompt.mean(1))