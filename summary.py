import os, tarfile, json, textwrap, pathlib, re

ARCHIVE = "/mnt/data/ruido-project.tar.gz"
EXTRACT_DIR = "/mnt/data/ruido-project"

# Extract
os.makedirs(EXTRACT_DIR, exist_ok=True)
with tarfile.open(ARCHIVE, "r:gz") as tar:
    tar.extractall(EXTRACT_DIR)

# Collect high-level structure
def list_dir(root, max_depth=2):
    root = pathlib.Path(root)
    rows = []
    for p in root.rglob("*"):
        depth = len(p.relative_to(root).parts)
        if depth <= max_depth:
            rows.append({
                "type": "dir" if p.is_dir() else "file",
                "path": str(p.relative_to(root))
            })
    return rows

structure = list_dir(EXTRACT_DIR, max_depth=3)

# Try to read common config files
def safe_read(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"<<error reading {path}: {e}>>"

files_to_peek = [
    "package.json", "package-lock.json", "pnpm-lock.yaml", "yarn.lock",
    ".nvmrc", ".node-version", "README.md", "next.config.js", "next.config.mjs",
    "tsconfig.json", "jsconfig.json", ".env", ".env.local", ".env.example",
    "vite.config.ts", "vite.config.js", "turbo.json", "apps/web/package.json",
    "apps/website/package.json", "dockerfile", "Dockerfile"
]

peek_results = {}
for rel in files_to_peek:
    p = os.path.join(EXTRACT_DIR, rel)
    if os.path.exists(p):
        peek_results[rel] = safe_read(p)

# Parse package.json if present
pkg = None
if "package.json" in peek_results:
    try:
        pkg = json.loads(peek_results["package.json"])
    except Exception as e:
        pkg = {"_error": str(e)}

# Find env var usages in next.config/ts files to guess required envs
required_envs = set()
env_pattern = re.compile(r"process\.env\.([A-Z0-9_]+)")
for root, dirs, files in os.walk(EXTRACT_DIR):
    for f in files:
        if f.endswith((".ts", ".tsx", ".js", ".mjs", ".cjs")) and not f.startswith("."):
            try:
                content = open(os.path.join(root, f), "r", encoding="utf-8", errors="ignore").read()
                for m in env_pattern.finditer(content):
                    required_envs.add(m.group(1))
            except Exception:
                pass

summary = {
    "extract_dir": EXTRACT_DIR,
    "has_package_json": "package.json" in peek_results,
    "package_manager_lock": [k for k in ["package-lock.json","pnpm-lock.yaml","yarn.lock"] if k in peek_results],
    "top_level_scripts": pkg.get("scripts", {}) if pkg else None,
    "engines": pkg.get("engines", {}) if pkg else None,
    "dependencies_count": len(pkg.get("dependencies", {})) if pkg else None,
    "devDependencies_count": len(pkg.get("devDependencies", {})) if pkg else None,
    "node_version_files": {k: peek_results[k] for k in [".nvmrc",".node-version"] if k in peek_results},
    "has_next_config": any(k in peek_results for k in ["next.config.js","next.config.mjs"]),
    "peeked_files": {k: peek_results[k][:2000] for k in peek_results},  # trim for display
    "required_envs_detected_count": len(required_envs),
    "required_envs_sample": sorted(list(required_envs))[:20]
}

import pandas as pd
from caas_jupyter_tools import display_dataframe_to_user

structure_df = pd.DataFrame(structure)
display_dataframe_to_user("Project structure (first few levels)", structure_df.head(200))

summary
