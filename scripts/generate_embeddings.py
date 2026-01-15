import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import os
import re

# Types: G, R, W, L, P, F, D, M, C (Dragon excluded)
TYPE_MAP = {
    "G": ("Grass", 0), "R": ("Fire", 1), "W": ("Water", 2), "L": ("Lightning", 3),
    "P": ("Psychic", 4), "F": ("Fighting", 5), "D": ("Darkness", 6), "M": ("Metal", 7),
    "C": ("Colorless", 8)
}

def clean_text(text, names_regex):
    if not text:
        return ""
    # 1. Lowercase everything for consistent matching
    text = text.lower()
    # 2. Replace Pokemon names with "pokemon"
    text = names_regex.sub("pokemon", text)
    # 3. Explicitly catch any remaining accented word
    text = text.replace("pokémon", "pokemon")
    # 4. Normalize type symbols [G] -> Grass, etc.
    for sym, (full, _) in TYPE_MAP.items():
        text = text.replace(f"[{sym.lower()}]", full.lower())
    return text

MECH_MAP = {
    "poison": ["poisoned", "poison"],
    "paralyze": ["paralyzed", "paralysis"],
    "sleep": ["asleep", "sleep"],
    "burn": ["burned", "burn"],
    "confuse": ["confused", "confusion"],
    "active": ["active"],
    "bench": ["bench", "benched"],
    "your": ["your", "you"],
    "opponent": ["opponent", "opponent's"]
}

def get_type_references(text):
    if not text:
        return [0.0] * 9
    text_lower = text.lower()
    refs = [0.0] * 9
    for sym, (full, idx) in TYPE_MAP.items():
        if f"[{sym}]" in text or full.lower() in text_lower:
            refs[idx] = 1.0
    return refs

def get_mechanic_references(text):
    if not text:
        return [0.0] * 9
    text_lower = text.lower()
    refs = [0.0] * 9
    keys = ["poison", "paralyze", "sleep", "burn", "confuse", "active", "bench", "your", "opponent"]
    for i, key in enumerate(keys):
        for keyword in MECH_MAP[key]:
            if keyword in text_lower:
                refs[i] = 1.0
                break
    return refs

def generate():
    # 1. Load database.json
    db_path = "database.json"
    if not os.path.exists(db_path):
        print(f"Error: {db_path} not found.")
        return

    with open(db_path, "r") as f:
        db_raw = json.load(f)

    # Unwrap and collect names
    db = []
    names = {"Pokémon", "Pokemon"}
    for item in db_raw:
        card = None
        if "Pokemon" in item:
            card = item["Pokemon"]
        elif "Trainer" in item:
            card = item["Trainer"]
        
        if card:
            db.append(card)
            names.add(card["name"])
            # Also add parts of the name if it's "Charizard ex"
            if " ex" in card["name"]:
                names.add(card["name"].replace(" ex", ""))
            if card["name"].startswith("Mega "):
                names.add(card["name"].replace("Mega ", ""))

    # Sort names by length descending to avoid partial matches
    sorted_names = sorted(list(names), key=len, reverse=True)
    names_regex = re.compile(r'\b(' + '|'.join(map(re.escape, sorted_names)) + r')\b', re.IGNORECASE)

    texts = set()
    texts.add("")

    for card in db:
        if "ability" in card and card["ability"]:
            texts.add(card["ability"]["effect"])
        if "attacks" in card:
            for atk in card["attacks"]:
                if atk.get("effect"):
                    texts.add(atk["effect"])
        if "effect" in card:
            texts.add(card["effect"])

    unique_texts = sorted(list(texts))
    print(f"Found {len(unique_texts)} unique texts.")
    
    # Calculate unique word count
    all_words = set()
    for t in unique_texts:
        all_words.update(re.findall(r'\b\w+\b', t.lower()))
    print(f"Unique word count: {len(all_words)}")

    # 2. Generate embeddings for cleaned texts
    print("Generating embeddings with SentenceTransformer (CPU)...")
    cleaned_texts = [clean_text(t, names_regex) for t in unique_texts]
    
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    embeddings = model.encode(cleaned_texts)

    # 3. Reduce to 64 dims
    print(f"Reducing dimensions from {embeddings.shape[1]} to 64...")
    n_components = min(64, len(unique_texts))
    pca = PCA(n_components=n_components)
    reduced_embeddings = pca.fit_transform(embeddings)
    
    variance_sum = np.sum(pca.explained_variance_ratio_)
    print(f"Cumulative explained variance (64 dims): {variance_sum:.2%}")

    if reduced_embeddings.shape[1] < 64:
        padding = np.zeros((reduced_embeddings.shape[0], 64 - reduced_embeddings.shape[1]))
        reduced_embeddings = np.hstack([reduced_embeddings, padding])

    # 4. Save mapping normalized_text -> {embedding, type_references, mechanic_references}
    os.makedirs("src/rl/generated", exist_ok=True)
    
    # Save names for Rust
    with open("src/rl/generated/names.json", "w") as f:
        json.dump(sorted_names, f, indent=2)

    mapping = {}
    for idx, original_text in enumerate(unique_texts):
        norm_text = cleaned_texts[idx]
        # Multiple original texts might map to the same normalized text
        # We only need one entry per unique normalized text
        if norm_text not in mapping:
            mapping[norm_text] = {
                "embedding": reduced_embeddings[idx].tolist(),
                "type_refs": get_type_references(original_text),
                "mech_refs": get_mechanic_references(original_text)
            }

    with open("src/rl/generated/embeddings.json", "w") as f:
        json.dump(mapping, f, indent=2)

    # 5. Evolutionary metadata
    print("Calculating card metadata...")
    card_metadata = {}
    
    evolves_to = {}
    evolves_from = {}
    
    for card in db:
        if "evolves_from" in card:
            name = card["name"]
            ef = card["evolves_from"]
            if ef:
                evolves_from[name] = ef
                if ef not in evolves_to:
                    evolves_to[ef] = []
                evolves_to[ef].append(name)

    def get_line_size(name):
        curr = name
        while curr in evolves_from:
            curr = evolves_from[curr]
        root = curr
        def max_depth(n):
            if n not in evolves_to:
                return 1
            return 1 + max(max_depth(child) for child in evolves_to[n])
        return max_depth(root)

    for item in db_raw:
        card = None
        if "Pokemon" in item:
            card = item["Pokemon"]
            name = card["name"]
            line_size = get_line_size(name)
            is_final_stage = name not in evolves_to
            card_metadata[card["id"]] = {
                "line_size": line_size,
                "is_final_stage": 2 if is_final_stage else 1
            }
        elif "Trainer" in item:
            card = item["Trainer"]
            card_metadata[card["id"]] = {
                "line_size": 0,
                "is_final_stage": 0
            }

    with open("src/rl/generated/card_metadata.json", "w") as f:
        json.dump(card_metadata, f, indent=2)

    print("Success! Generated files in src/rl/generated/")

if __name__ == "__main__":
    generate()
