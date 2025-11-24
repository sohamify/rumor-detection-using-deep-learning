# generate_hybrid_rumor.py — 100% FREE & OFFLINE
import json
import random
import os
from datetime import datetime
import subprocess

# Real rumor topics (2025 India context)
TOPICS = [
    "COVID vaccine causes magnetism",
    "5G towers spreading virus",
    "Free electricity for all by Modi",
    "Kolkata airport sealed",
    "New currency notes with chip",
    "Aadhaar linked to bank = money deducted",
    "Elon Musk buying Twitter again",
    "Deepfake video of CM Mamata crying"
]

def generate_text():
    prompt = f"""Generate a realistic Indian rumor tweet (max 280 chars) about: {random.choice(TOPICS)}
    Make it sound urgent, use Hindi-English mix, add emojis.
    Output only the tweet text."""
    result = subprocess.run([
        "ollama", "run", "llama3.2:8b", prompt
    ], capture_output=True, text=True)
    return result.stdout.strip()

def generate_image(text):
    # Save prompt to ComfyUI input
    with open("comfy_input.txt", "w") as f:
        f.write(f"Indian viral misinformation meme: {text}, Twitter style, dark mode, red warning emoji")
    # Trigger ComfyUI API (you run ComfyUI server)
    os.system("curl -X POST http://127.0.0.1:8188/prompt -d @workflow.json")
    return f"syn_{random.randint(10000,99999)}.jpg"

def build_thread():
    replies = []
    for _ in range(random.randint(10, 80)):
        replies.append({
            "text": random.choice([
                "This is fake yaar", "100% true my uncle sent",
                "Source??", "Arre bhai mat failao", "Finally truth came out"
            ]),
            "timestamp": datetime.now().isoformat()
        })
    return replies

# Generate 100 samples
for i in range(1, 101):
    text = generate_text()
    img = f"images/{generate_image(text)}"
    label = random.choices([0,1,2], weights=[0.3, 0.5, 0.2])[0]
    
    sample = {
        "id": f"SYN-2025-{i:05d}",
        "source_text": text,
        "image_path": img if label != 0 else None,
        "label_ternary": label,
        "label_anomaly": 1.0 if label > 0 else 0.0,
        "missing_mask": [1, 0.7 if label != 0 else 0, 1],
        "replies": build_thread()
    }
    
    with open(f"data/sample_data/sample_{i:05d}.json", "w", encoding="utf-8") as f:
        json.dump(sample, f, indent=2, ensure_ascii=False)
    
    print(f"Generated {i}/100")