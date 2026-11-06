import json
from collections import Counter
from dataset import CHIRALITY_TO_ID

JSON_PATH = "MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json" 

with open(JSON_PATH) as f:
    coco = json.load(f)

# 1. What fields does an annotation actually have?
ann0 = coco["annotations"][0]
print("Annotation keys:", list(ann0.keys()))
print("Has 'chirality' key?      ", "chirality" in ann0)
print("Has 'category_id' key?    ", "category_id" in ann0)

# 2. What are the categories?
print("\nCategories:", coco.get("categories"))

# 3. What would the current Dataset code produce?
current = Counter(
    CHIRALITY_TO_ID.get(a.get("chirality", "Unidentifiable"), 3)
    for a in coco["annotations"]
)
print("\nLabels via current dataset.py logic:", dict(current))

# 4. What does the raw data say?
if "category_id" in ann0:
    print("Raw category_id counts:           ", dict(Counter(a["category_id"] for a in coco["annotations"])))
if "chirality" in ann0:
    print("Raw chirality counts:             ", dict(Counter(a["chirality"] for a in coco["annotations"])))