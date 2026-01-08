
import os
from PIL import Image

folder = "tests/data/benchmark_hf_50/ai"
for f in sorted(os.listdir(folder)):
    if f.lower().endswith(('.jpg', '.jpeg', '.png')):
        path = os.path.join(folder, f)
        with Image.open(path) as img:
            print(f"{f[:30]:<30}: {img.size}")
