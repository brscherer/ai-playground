import os
import shutil
import random

SOURCE_DIR = "PetImages"
BASE_DIR = "data"

SPLIT_RATIO = 0.8

for category in ["Cat", "Dog"]:
    images = os.listdir(os.path.join(SOURCE_DIR, category))
    
    random.shuffle(images)
    
    split_index = int(len(images) * SPLIT_RATIO)
    
    train_images = images[:split_index]
    val_images = images[split_index:]

    for split, img_list in [("train", train_images), ("val", val_images)]:
        target_dir = os.path.join(BASE_DIR, split, category)
        os.makedirs(target_dir, exist_ok=True)

        for img in img_list:
            src = os.path.join(SOURCE_DIR, category, img)
            dst = os.path.join(target_dir, img)

            try:
                shutil.copyfile(src, dst)
            except:
                pass