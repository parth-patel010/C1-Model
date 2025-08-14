import json, glob, os
from pathlib import Path

RAW_DIR = Path("data/raw")
OUT = Path("data/prepared/c1_html_instructions.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)

TEMPLATES = {
    "cafe": "Create a responsive cafe landing page with a hero, a 3x2 menu grid with prices, testimonials, and a contact form.",
    "portfolio": "Create a responsive personal portfolio with a sticky navbar, hero, projects grid, and contact section.",
    "menu": "Create a responsive restaurant menu page with category tabs (Starters, Mains, Desserts) and a footer.",
}

# Heuristic: infer instruction by filename keywords

def infer_instruction(fname: str) -> str:
    f = fname.lower()
    if "cafe" in f or "coffee" in f:
        return TEMPLATES["cafe"]
    if "portfolio" in f:
        return TEMPLATES["portfolio"]
    if "menu" in f:
        return TEMPLATES["menu"]
    return "Create a responsive landing page with hero, features grid, and contact form."

count = 0
with OUT.open("w", encoding="utf-8") as w:
    for path in sorted(glob.glob(str(RAW_DIR / "*.html"))):
        with open(path, "r", encoding="utf-8") as f:
            html = f.read().strip()
        ins = infer_instruction(os.path.basename(path))
        rec = {"instruction": ins, "input": "", "output": html}
        w.write(json.dumps(rec, ensure_ascii=False) + "\n")
        count += 1

    print(f"Wrote {count} samples -> {OUT}")