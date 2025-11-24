# data/hybrid_rumor_loader.py
import json
import os
from torch.utils.data import Dataset
from PIL import Image
import torch
from transformers import CLIPProcessor

class HybridRumorDataset(Dataset):
    def __init__(self, data_dir="data/sample_data", transform=None):
        self.data_dir = data_dir
        self.samples = []
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.transform = transform
        
        for file in sorted(os.listdir(data_dir)):
            if file.endswith(".json"):
                with open(os.path.join(data_dir, file), "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.samples.append(data)
        print(f"Loaded {len(self.samples)} samples from {data_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Text
        text = sample["source_text"]
        
        # Image (if exists)
        pixel_values = None
        if sample["image_path"] and os.path.exists(os.path.join(self.data_dir, sample["image_path"])):
            img = Image.open(os.path.join(self.data_dir, sample["image_path"])).convert("RGB")
            pixel_values = self.processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0)
        
        # Graph placeholder (we'll build DGL later)
        num_nodes = len(sample["replies"]) + 1
        graph = torch.zeros(num_nodes, 128)  # dummy node features
        
        item = {
            "text": text,
            "pixel_values": pixel_values,
            "graph": graph,
            "missing_mask": torch.tensor(sample["missing_mask"], dtype=torch.float),
            "label_anomaly": torch.tensor(sample["label_anomaly"], dtype=torch.float),
            "label_ternary": torch.tensor(sample["label_ternary"], dtype=torch.long),
        }
        return item

# Test
if __name__ == "__main__":
    dataset = HybridRumorDataset()
    sample = dataset[0]
    print("Sample loaded:", sample["text"][:100])
    print("Has image:", sample["pixel_values"] is not None)
    print("Label:", sample["label_ternary"])