"""
Configuration for embedding generation.
"""

from pathlib import Path

# --- Embedding Dimensions ---
EMBEDDING_DIM = 128  # Output dimension after PCA (was 64)

# --- Model Configuration ---
SENTENCE_MODEL = "all-MiniLM-L6-v2"  # SentenceTransformer model

# --- Paths ---
_MODULE_DIR = Path(__file__).parent
_PROJECT_ROOT = _MODULE_DIR.parents[2]  # deckgym-core/

PATHS = {
    "raw_data": _MODULE_DIR / "data" / "raw",
    "pocket_db": _PROJECT_ROOT / "database.json",
    "output_dir": _PROJECT_ROOT / "src" / "rl" / "generated",
    "pca_model": _MODULE_DIR / "data" / "pca_model.pkl",
}

# --- Type Mapping (for text normalization) ---
TYPE_MAP = {
    "G": ("Grass", 0),
    "R": ("Fire", 1),
    "W": ("Water", 2),
    "L": ("Lightning", 3),
    "P": ("Psychic", 4),
    "F": ("Fighting", 5),
    "D": ("Darkness", 6),
    "M": ("Metal", 7),
    "C": ("Colorless", 8),
}

# --- Mechanic Keywords ---
MECH_MAP = {
    "poison": ["poisoned", "poison"],
    "paralyze": ["paralyzed", "paralysis"],
    "sleep": ["asleep", "sleep"],
    "burn": ["burned", "burn"],
    "confuse": ["confused", "confusion"],
    "active": ["active"],
    "bench": ["bench", "benched"],
    "your": ["your", "you"],
    "opponent": ["opponent", "opponent's"],
}
