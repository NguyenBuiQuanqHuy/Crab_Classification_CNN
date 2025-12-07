import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# =====================
# 1. Load dataset
# =====================
train_dir = "dataset/train"
test_dir = "dataset/test"
img_size = 128
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1/255,
    rotation_range=15,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1/255)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode="categorical"
)

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode="categorical"
)

# =====================
# 2. Build CNN model
# =====================
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation="relu", input_shape=(img_size, img_size, 3)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation="relu"),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128, (3,3), activation="relu"),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(train_data.num_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =====================
# 3. Train model
# =====================
history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=10
)

# =====================
# 4. Save model
# =====================
model.save("model.h5")
print("✔ Model đã được lưu vào model.h5")

# =====================
# 5. Plot Loss/Accuracy
# =====================
plt.plot(history.history["accuracy"], label="Train Acc")
plt.plot(history.history["val_accuracy"], label="Val Acc")
plt.legend()
plt.show()

