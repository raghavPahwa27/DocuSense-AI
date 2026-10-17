"""
migrate_faiss_to_cosine.py
──────────────────────────
One-shot migration: convert the persisted FAISS index from IndexFlatL2 (L2
distance) to IndexFlatIP (inner-product / cosine similarity).

Strategy
────────
When every vector is L2-normalised to unit length, the inner product between
two vectors equals their cosine similarity:

    cosine(a, b) = a·b  (if ‖a‖ = ‖b‖ = 1)

So we:
  1. Extract all raw vectors from the existing index via reconstruct_n().
  2. L2-normalise them with faiss.normalize_L2() (in-place, float32).
  3. Build a new IndexFlatIP of the same dimension and add the normed vectors.
  4. Overwrite faiss_store/index.faiss with the new index.
  5. Leave index.pkl untouched — it carries only docstore metadata, which is
     index-type-agnostic.
"""

import sys
from pathlib import Path

import numpy as np
import faiss

FAISS_DIR  = Path(__file__).parent / "faiss_store"
INDEX_PATH = FAISS_DIR / "index.faiss"
BACKUP_PATH = FAISS_DIR / "index_l2_backup.faiss"


def main() -> None:
    if not INDEX_PATH.exists():
        print("faiss_store/index.faiss not found. Nothing to migrate.")
        sys.exit(1)

    # 1. Load existing index
    old_index = faiss.read_index(str(INDEX_PATH))
    n   = old_index.ntotal
    dim = old_index.d
    print(f"Loaded index: type={type(old_index).__name__}, "
          f"metric={old_index.metric_type}, n={n}, dim={dim}")

    if old_index.metric_type == faiss.METRIC_INNER_PRODUCT:
        print("Index is already IndexFlatIP (cosine). Nothing to do.")
        sys.exit(0)

    # 2. Extract raw vectors
    vectors = np.zeros((n, dim), dtype=np.float32)
    old_index.reconstruct_n(0, n, vectors)
    print(f"Extracted {n} vectors of dimension {dim}.")

    # 3. L2-normalise
    faiss.normalize_L2(vectors)
    print("Vectors L2-normalised.")

    # 4. Build IndexFlatIP
    new_index = faiss.IndexFlatIP(dim)
    new_index.add(vectors)
    print(f"New IndexFlatIP built with {new_index.ntotal} vectors.")

    # 5. Backup old index then save new one
    import shutil
    shutil.copy(INDEX_PATH, BACKUP_PATH)
    print(f"Old index backed up to {BACKUP_PATH.name}")

    faiss.write_index(new_index, str(INDEX_PATH))
    print(f"Saved new IndexFlatIP index to {INDEX_PATH}")

    # 6. Verification
    check = faiss.read_index(str(INDEX_PATH))
    assert check.metric_type == faiss.METRIC_INNER_PRODUCT, \
        "Saved index metric_type does not match METRIC_INNER_PRODUCT!"
    assert check.ntotal == n, "Vector count mismatch after save!"
    print(f"Verification passed — metric_type={check.metric_type} "
          f"(0=IP/cosine), n={check.ntotal}")


if __name__ == "__main__":
    main()
