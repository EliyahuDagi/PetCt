"""Ad-hoc: report enumerated patients across the pipeline roots and how many are
already fresh in the SSD cache. Run with the WSL torch venv. Temporary helper."""
import sys
sys.path.insert(0, "/mnt/c/Users/algo/VScodeProjects/PetCt/PetCt")

from src.training.dataset_index import enumerate_patients
from src.training.precache import is_fresh

ROOTS = [
    "/mnt/d/DeepTrainingData/Project/ACRIN 6668",
    "/mnt/d/DeepTrainingData/Project/PetCt/Bladder 13.11.25",
    "/mnt/d/DeepTrainingData/Project/PetCtNormal/PET_CT_NORMAL_2",
]
CACHE_DIR = "/mnt/c/DeepTrainingData/PetCT"

ps = enumerate_patients(ROOTS, missing_ok=True)
print("TOTAL across roots:", len(ps))
fresh = [p for p in ps if is_fresh(CACHE_DIR, p)]
print("already fresh:", len(fresh), "-> to-do:", len(ps) - len(fresh))
# per-root breakdown
from collections import Counter
c = Counter()
for p in ps:
    for r in ROOTS:
        if p.startswith(r):
            c[r] += 1
            break
for r in ROOTS:
    print(f"  {c[r]:4d}  {r}")
