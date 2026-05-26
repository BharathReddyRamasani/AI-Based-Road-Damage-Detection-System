import os
import json
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import Callback
from sklearn.model_selection import train_test_split

IMG_SIZE = 224
LABEL_MAP = {0: 'crack', 1: 'manhole', 2: 'pothole'}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

def download_dataset():
    """Downloads the road damage dataset from Kaggle via kagglehub."""
    import kagglehub
    path = kagglehub.dataset_download("lorenzoarcioni/road-damage-dataset-potholes-cracks-and-manholes")
    return path

def load_metadata(path):
    """Loads annotations and returns a clean DataFrame with image paths and labels."""
    annotation_path = os.path.join(path, "data", "annotations_coco.json")
    with open(annotation_path, "r") as f:
        coco = json.load(f)
        
    df_images = pd.DataFrame(coco["images"])
    df_annotations = pd.DataFrame(coco["annotations"])
    df_categories = pd.DataFrame(coco["categories"])
    
    # Merge image and annotation data
    df = df_annotations.merge(df_images, left_on="image_id", right_on="id")
    category_map = df_categories.set_index("id")["name"].to_dict()
    df["label"] = df["category_id"].map(category_map)
    df["path"] = df["file_name"].apply(lambda x: os.path.normpath(os.path.join(path, "data", "images", x)))
    df["encoded_label"] = df["label"].map(INV_LABEL_MAP)
    
    # Drop rows with invalid images if any
    df = df.dropna(subset=["encoded_label", "path"])
    df["encoded_label"] = df["encoded_label"].astype(int)
    
    return df, LABEL_MAP

def load_image(path, label):
    """TF image loader function."""
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image / 255.0
    return image, label

def build_model(num_classes=3):
    """Builds the Keras CNN model exactly as specified in the notebook."""
    model = Sequential([
        Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        # Block 1
        Conv2D(32, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        
        # Block 2
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        
        # Block 3
        Conv2D(128, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(num_classes, activation="softmax")
    ])
    
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

class StreamlitTrainingCallback(Callback):
    """Custom Keras Callback to report training progress to Streamlit UI."""
    def __init__(self, epochs, progress_bar, status_text, metrics_placeholder):
        super().__init__()
        self.epochs = epochs
        self.progress_bar = progress_bar
        self.status_text = status_text
        self.metrics_placeholder = metrics_placeholder
        self.history = {'accuracy': [], 'loss': [], 'val_accuracy': [], 'val_loss': []}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # Save metrics
        for key in self.history:
            if key in logs:
                self.history[key].append(logs[key])
                
        # Update progress bar
        progress = (epoch + 1) / self.epochs
        self.progress_bar.progress(progress)
        
        # Format metric message
        metrics_str = (
            f"**Epoch {epoch + 1}/{self.epochs} completed**\n"
            f"- **Loss**: `{logs.get('loss', 0):.4f}` | **Accuracy**: `{logs.get('accuracy', 0):.4f}`\n"
            f"- **Val Loss**: `{logs.get('val_loss', 0):.4f}` | **Val Accuracy**: `{logs.get('val_accuracy', 0):.4f}`"
        )
        self.status_text.text(f"Training in progress... Epoch {epoch + 1} of {self.epochs}")
        self.metrics_placeholder.markdown(metrics_str)

def train_model_ui(df, model_save_path, epochs=10, batch_size=8, subset_size=None, callback_objs=None):
    """Trains the model and saves it. If subset_size is set, trains on a small fraction for speed."""
    if subset_size and subset_size < len(df):
        # Stratified sampling to maintain class ratios
        df_train_test, _ = train_test_split(
            df,
            train_size=subset_size,
            random_state=42,
            stratify=df["encoded_label"]
        )
    else:
        df_train_test = df

    train_df, test_df = train_test_split(
        df_train_test,
        test_size=0.20,
        random_state=42,
        stratify=df_train_test["encoded_label"]
    )
    
    # Create TF datasets
    train_ds = tf.data.Dataset.from_tensor_slices((train_df["path"], train_df["encoded_label"]))
    test_ds = tf.data.Dataset.from_tensor_slices((test_df["path"], test_df["encoded_label"]))
    
    train_ds = train_ds.map(load_image).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.map(load_image).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    model = build_model(num_classes=3)
    
    callbacks = []
    st_cb = None
    if callback_objs:
        p_bar, s_text, m_placeholder = callback_objs
        st_cb = StreamlitTrainingCallback(epochs, p_bar, s_text, m_placeholder)
        callbacks.append(st_cb)
        
    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=test_ds,
        callbacks=callbacks
    )
    
    # Save the model in Keras format
    model.save(model_save_path)
    
    return history, st_cb.history if st_cb else history.history

def predict_damage(model_path, image_bytes):
    """Loads the model and predicts the class of an image from raw bytes."""
    model = tf.keras.models.load_model(model_path)
    
    # Preprocess image
    image = tf.image.decode_image(image_bytes, channels=3)
    display_image = image.numpy()
    
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image / 255.0
    image = tf.expand_dims(image, axis=0)
    
    predictions = model.predict(image)[0]
    pred_class_idx = int(tf.argmax(predictions))
    
    label = LABEL_MAP[pred_class_idx]
    confidence = float(predictions[pred_class_idx])
    
    detailed_probs = {LABEL_MAP[i]: float(predictions[i]) for i in range(len(predictions))}
    
    return label, confidence, detailed_probs, display_image
