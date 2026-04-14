from data_loader import load_data
from model import build_model

train_ds, val_ds = load_data()

model = build_model()

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5
)

model.save("model.h5")