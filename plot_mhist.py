import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# File paths
ANNOTATIONS_PATH = "annotations.csv"
IMAGES_DIR = "images"

# Load data
df = pd.read_csv(ANNOTATIONS_PATH)

# Select exactly 2 HP and 2 SSA samples
hp_samples = df[df['Majority Vote Label'] == 'HP'].sample(2, random_state=42)
ssa_samples = df[df['Majority Vote Label'] == 'SSA'].sample(2, random_state=42)
samples = pd.concat([hp_samples, ssa_samples]).sample(frac=1, random_state=42).reset_index(drop=True)

# Plot
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
fig.suptitle("MHIST Dataset Examples (HP & SSA)", fontsize=16)

for i, row in samples.iterrows():
    img_path = os.path.join(IMAGES_DIR, row['Image Name'])
    img = Image.open(img_path).convert("RGB")
    
    ax = axes[i]
    ax.imshow(img)
    ax.set_title(f"Label: {row['Majority Vote Label']}", fontsize=14)
    ax.axis('off')

plt.tight_layout()
plt.savefig("mhist_samples.png", dpi=150, bbox_inches='tight')
print("Successfully generated mhist_samples.png")
