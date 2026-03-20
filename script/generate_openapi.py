#!/usr/bin/env python3
"""
Generate OpenAPI (JSON and YAML) specs for the FinSage RAG API.
Run from project root: PYTHONPATH=src uv run python script/generate_openapi.py

Output: openapi/openapi.json, openapi/openapi.yaml
"""
import json
import os
import sys

# Ensure src is on path when run from project root or script/
_script_dir = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_script_dir)
_src = os.path.join(_root, "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

# Import app after path is set; avoid running lifespan
from server import app


def main():
    schema = app.openapi()
    out_dir = os.path.join(_root, "openapi")
    os.makedirs(out_dir, exist_ok=True)

    json_path = os.path.join(out_dir, "openapi.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2, ensure_ascii=False)
    print("Wrote", json_path)

    try:
        import yaml

        yaml_path = os.path.join(out_dir, "openapi.yaml")
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(
                schema,
                f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )
        print("Wrote", yaml_path)
    except ImportError:
        print("PyYAML not available; skipping openapi.yaml")


if __name__ == "__main__":
    main()
