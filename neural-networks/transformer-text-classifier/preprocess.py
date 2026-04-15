import tensorflow as tf

IMG_SIZE = (150, 150)
BATCH_SIZE = 32

def load_data():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        "data/train",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        "data/val",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    normalization_layer = tf.keras.layers.Rescaling(1./255)

    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

    return train_ds, val_ds