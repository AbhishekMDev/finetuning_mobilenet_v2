import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

IMG_SIZE = 224
BATCH_SIZE = 30

TRAIN_PATH = "data/split/train"
VAL_PATH = "data/split/val"
TEST_PATH = "data/split/test"

# Data augmentation pipeline
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
    layers.RandomZoom(height_factor=0.1, width_factor=0.1),
    layers.RandomContrast(0.2),
], name="data_augmentation")

def preprocess(image_batch, label_batch):
    try:
        image_batch = preprocess_input(image_batch)
    except Exception as e:
        print(f"Error preprocessing batch: {e}")
    return image_batch, label_batch

def augment(image_batch, label_batch):
    try:
        image_batch = data_augmentation(image_batch)
        image_batch = preprocess_input(image_batch)
    except Exception as e:
        print(f"Error augmenting batch: {e}")
    return image_batch, label_batch

def augment_fn(image_batch, label_batch):
    try:
        image_batch = data_augmentation(image_batch)
        image_batch = preprocess_input(image_batch)
    except Exception as e:
        print(f"Error augmenting batch: {e}")
    return image_batch, label_batch

def load_dataset(data_dir, do_augment=False, shuffle=False):
    try:
        dataset = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            label_mode='int',
            image_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE,
            shuffle=shuffle
        )
    except Exception as e:
        print(f"Error loading dataset from {data_dir}: {e}")
        raise e

    if do_augment:
        dataset = dataset.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        dataset = dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.cache()
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset

if __name__ == "__main__":
    try:
        # Changed augment=True to do_augment=True
        train_ds = load_dataset(TRAIN_PATH, do_augment=True, shuffle=True)
        val_ds = load_dataset(VAL_PATH, do_augment=False)
        test_ds = load_dataset(TEST_PATH, do_augment=False)
        
        # Verify the datasets
        for images, labels in train_ds.take(1):
            print("\nTraining batch:")
            print("Images shape:", images.shape)
            print("Labels shape:", labels.shape)
            print("Sample label values:", labels.numpy()[:5])  # First 5 labels
            
        print("\nDataset loaded successfully!")
        print(f"Training batches: {len(train_ds)}")
        print(f"Validation batches: {len(val_ds)}")
        print(f"Test batches: {len(test_ds)}")
        
    except Exception as e:
        print(f"\nError initializing datasets: {e}")
        import traceback
        traceback.print_exc()