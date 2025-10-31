# AER850 Project 2  Osamah Al-amoudi (501101146)
# Steps 14:

import json, itertools, numpy as np, matplotlib.pyplot as plt, tensorflow as tf
from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Flatten,
                                     Dense, Dropout, BatchNormalization)
from sklearn.metrics import classification_report, confusion_matrix


RNG = 42
tf.keras.utils.set_random_seed(RNG)
IMG_SIZE = (500, 500)              # (width, height) as required
BATCH_SIZE = 32
DATA_DIR = Path("Data")
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR   = DATA_DIR / "valid"
TEST_DIR  = DATA_DIR / "test"
MODEL_PATH = "p2_best_model.keras"
CLASS_MAP_JSON = "class_indices.json"


SAVE_DIR = Path(__file__).parent if "__file__" in globals() else Path.cwd()

# Step 1: Data Processing 

def step1_data_processing():
   
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.15,
        zoom_range=0.15,
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.05
    )
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=True,
        seed=RNG
    )
    val_gen = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False
    )

   
    class_map = train_gen.class_indices            
    inv_map = {v: k for k, v in class_map.items()}
    with open(CLASS_MAP_JSON, "w") as f:
        json.dump({"indices": class_map, "inverse": inv_map}, f, indent=2)
    print("Class mapping:", class_map)

    return train_gen, val_gen


# Step 2: Neural Network Architecture Design  

def step2_build_model(num_classes: int):
    model = Sequential([
        Input(shape=(IMG_SIZE[1], IMG_SIZE[0], 3)),  # (H, W, C) for TF internals
        Conv2D(32, (3,3), activation="relu"), BatchNormalization(), MaxPooling2D((2,2)),
        Conv2D(64, (3,3), activation="relu"), BatchNormalization(), MaxPooling2D((2,2)),
        Conv2D(128,(3,3), activation="relu"), BatchNormalization(), MaxPooling2D((2,2)),
        Flatten(),
        Dense(128, activation="relu"), Dropout(0.3),
        Dense(num_classes, activation="softmax")     # 3 neurons for 3 classes
    ])
    return model


# Step 3: Hyperparameter Analysis  

def step3_compile_model(model):
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()
    # Record chosen HPs 
    hparams = {
        "img_size": IMG_SIZE, "batch_size": BATCH_SIZE,
        "conv_filters": [32, 64, 128],
        "kernel_size": (3,3),
        "dense_units": 128,
        "activations": {"conv":"relu", "dense":"relu", "final":"softmax"},
        "optimizer": "adam", "lr": 1e-4, "loss":"categorical_crossentropy"
    }
    with open("p2_hparams.json", "w") as f: json.dump(hparams, f, indent=2)
    return model


# Step 4: Model Evaluation (train/plots)

def step4_train_and_evaluate(model, train_gen, val_gen, epochs=25):
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1)
    ]
    hist = model.fit(train_gen, validation_data=val_gen, epochs=epochs, callbacks=callbacks)
    model.save(MODEL_PATH)

    # Accuracy/Loss curves 
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(hist.history["accuracy"], label="train")
    plt.plot(hist.history["val_accuracy"], label="valid")
    plt.title("Model Accuracy"); plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend()
    plt.subplot(1,2,2)
    plt.plot(hist.history["loss"], label="train")
    plt.plot(hist.history["val_loss"], label="valid")
    plt.title("Model Loss"); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
    plt.tight_layout()

   
    out_path = SAVE_DIR / "training_curves.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"Saved training curves to: {out_path}")

    return hist

if __name__ == "__main__":
    train_gen, val_gen = step1_data_processing()
    model = step2_build_model(num_classes=len(train_gen.class_indices))
    model = step3_compile_model(model)
    _ = step4_train_and_evaluate(model, train_gen, val_gen, epochs=25)

