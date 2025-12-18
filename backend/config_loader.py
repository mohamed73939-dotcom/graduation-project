import yaml
from pathlib import Path

def load_config(path="config.yaml"):
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}