import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

def load_and_preprocess_image(img_path, target_size=(224, 224)):
    """Loads and preprocesses an image."""
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

def compute_embedding(model, img_path):
    """Computes the embedding of an image using the given model."""
    img = load_and_preprocess_image(img_path)
    return model.predict(img).flatten()

def save_embeddings(model, images_dir, output_path):
    """
    Computes and saves embeddings for all images in a directory.
    
    Args:
        model: Pre-trained model to compute embeddings.
        images_dir (str): Path to the directory containing images.
        output_path (str): Path to save the embeddings.
    """
    embeddings = {}
    for file_name in os.listdir(images_dir):
        file_path = os.path.join(images_dir, file_name)
        if os.path.isfile(file_path):
            embedding = compute_embedding(model, file_path)
            embeddings[file_path] = embedding
    np.save(output_path, embeddings)
    print(f"Embeddings saved to {output_path}")

if __name__ == "__main__":
    # Paths
    model_save_path = "embedding_model.h5"  # Path to save the model
    images_dir = "./img/instagram"  # Path to the folder containing images
    embeddings_output_path = "image_embeddings.npy"  # Path to save computed embeddings

    # Create and save the model
    model = ResNet50(weights="imagenet", include_top=False, pooling="avg")
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

    # Compute and save embeddings
    save_embeddings(model, images_dir, embeddings_output_path)
