# AER850 Project 2  Osamah Al-amoudi (501101146)
# Step 5: Model Testing 

import json, numpy as np, matplotlib.pyplot as plt, tensorflow as tf
from pathlib import Path
from tensorflow.keras.preprocessing import image

IMG_SIZE = (500, 500)
MODEL_PATH = "p2_best_model.keras"
CLASS_MAP_JSON = "class_indices.json"

TEST_DIR = Path("Data") / "test"
REQ_FILES = {
    "crack":        "test_crack.jpg",
    "missing-head": "test_missinghead.jpg",
    "paint-off":    "test_paintoff.jpg",
}

SAVE_DIR = Path(__file__).parent

def load_and_preprocess(img_path: Path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    arr = image.img_to_array(img) / 255.0  
    return np.expand_dims(arr, axis=0)     # shape (1, H, W, C)





if __name__ == "__main__":
    
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_MAP_JSON) as f:
        info = json.load(f)
        inv = {int(k): v for k, v in info["inverse"].items()} 

   
    for cls, fname in REQ_FILES.items():
        p = TEST_DIR / cls / fname
        x = load_and_preprocess(p)
        probs = model.predict(x)
        pred_idx = probs.argmax(axis=1)[0]
        pred_name = inv[pred_idx]
        conf = float(probs[0][pred_idx]) * 100

        
        img = image.load_img(p, target_size=IMG_SIZE)
        plt.figure(figsize=(4, 4))
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"Actual: {cls}\nPredicted: {pred_name} ({conf:.1f}%)")
        plt.tight_layout()

       

        save_path = SAVE_DIR / f"prediction_{cls}.png"
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.show()

        print(f"Saved figure to: {save_path}")
        print(f"File: {p.name:18s} | Actual: {cls:13s} | Predicted: {pred_name:13s} | Confidence: {conf:5.1f}%")
