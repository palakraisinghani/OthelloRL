import json
from pathlib import Path

import yaml


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def load_json(path):
    return json.loads(Path(path).read_text())
