# Hybrid-AD-Mamba-CLIP: Unified Multimodal Rumor Detection for the AI Era (2025)

**B.Tech Final Year Project — Government College of Engineering & Ceramic Technology, Kolkata**

**The first framework that simultaneously beats all 7 SOTA 2025 papers**  
Integrates: AD-GSCL • SoLM+PEP • MICC • OmniFake+UMFDet • TriSPrompt • CausalMamba • CLMIR

![Architecture](figures/overall_architecture.png)

### Key Features
- Handles 1:1000 real-world imbalance
- Works with missing modalities (text-only → only 6.7% drop)
- Detects both human & AI-generated deception
- Interpretable causal graphs via NOTEARS
- Largest benchmark: HybridRumor-250K

### Quick Start
```bash
conda env create -f environment.yml
conda activate hybrid-rumor
pip install -r requirements.txt
jupyter notebook notebooks/01-load-and-visualize.ipynb