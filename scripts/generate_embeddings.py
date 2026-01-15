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

    # 4. Create mapping normalized_text -> {embedding, type_refs, mech_refs}
    text_to_data = {}
    for idx, original_text in enumerate(unique_texts):
        norm_text = cleaned_texts[idx]
        if norm_text not in text_to_data:
            text_to_data[norm_text] = {
                "embedding": reduced_embeddings[idx].tolist(),
                "type_refs": get_type_references(original_text),
                "mech_refs": get_mechanic_references(original_text)
            }

    # 5. Consolidated Card Features (Mapping ID -> Data)
    print("Consolidating all features into card_features.json...")
    card_features = {}
    
    # Pre-calculate metadata maps
    evolves_to = {}
    evolves_from = {}
    for card in db:
        if "evolves_from" in card:
            name, ef = card["name"], card["evolves_from"]
            if ef:
                evolves_from[name] = ef
                evolves_to.setdefault(ef, []).append(name)

    def get_line_size(name):
        curr = name
        while curr in evolves_from: curr = evolves_from[curr]
        def max_dist(n):
            if n not in evolves_to: return 1
            return 1 + max(max_dist(c) for c in evolves_to[n])
        return max_dist(curr)

    for item in db_raw:
        card = item.get("Pokemon") or item.get("Trainer")
        if not card: continue
        
        cid = card["id"]
        
        # Helper to get text data safely
        def get_text_data(txt):
            norm = clean_text(txt, names_regex)
            return text_to_data.get(norm, {
                "embedding": [0.0]*64, "type_refs": [0.0]*9, "mech_refs": [0.0]*9
            })

        # Ability
        ability_data = get_text_data(card["ability"]["effect"] if card.get("ability") else "")
        
        # Attacks
        attacks = card.get("attacks", [])
        atk1_data = get_text_data(attacks[0]["effect"] if len(attacks) > 0 else "")
        atk2_data = get_text_data(attacks[1]["effect"] if len(attacks) > 1 else "")
        
        # Trainer Effect (Supporter/Item/Tool)
        trainer_data = get_text_data(card.get("effect", ""))

        # Metadata
        if "Pokemon" in item:
            line_size = get_line_size(card["name"])
            is_final = 2 if card["name"] not in evolves_to else 1
        else:
            line_size, is_final = 0, 0

        card_features[cid] = {
            "ability": ability_data,
            "atk1": atk1_data,
            "atk2": atk2_data,
            "supporter": trainer_data,
            "line_size": line_size,
            "is_final_stage": is_final
        }

    os.makedirs("src/rl/generated", exist_ok=True)
    with open("src/rl/generated/card_features.json", "w") as f:
        json.dump(card_features, f, indent=2)
    
    # Still save names for reference, though Rust won't need them for cleaning
    with open("src/rl/generated/names.json", "w") as f:
        json.dump(sorted_names, f, indent=2)

    # We can stop generating the old separate files to save space
    # with open("src/rl/generated/embeddings.json", "w") as f: 
    #    json.dump(text_to_data, f, indent=2)

    print("Success! Generated files in src/rl/generated/")

if __name__ == "__main__":
    generate()
