import streamlit as st
import matplotlib.pyplot as plt
from tensorflow.keras.applications.resnet50 import preprocess_input, ResNet50
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import GlobalMaxPooling2D
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
import numpy as np
import os
import cv2
from PIL import Image
import pickle
import tensorflow

# Add CSS for gray background
st.markdown(
    """
    <style>
    .main {
        background-color: #f0f0f0;
        padding: 20px;
        border-radius: 10px;
    }
    .stApp {
        background-color: #d3d3d3;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Main content inside styled div
st.markdown('<div class="main">', unsafe_allow_html=True)

# Load models and data functions
def initialize_classification_model():
    return load_model('best_model13cls.keras')

def initialize_recommendation_model():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False
    return tensorflow.keras.Sequential([base_model, GlobalMaxPooling2D()])

def load_image_data():
    dir = os.path.join(os.getcwd(), 'dataset')
    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    data_set = datagen.flow_from_directory(dir, target_size=(224, 224), batch_size=32, class_mode='sparse')
    labels = dict((v, k) for k, v in data_set.class_indices.items())
    return labels

def load_feature_data():
    try:
        feature_list = np.array(pickle.load(open("featurevector.pkl", "rb")))
        filenames = pickle.load(open("filenames.pkl", "rb"))
        st.write("Loaded feature vector and filenames successfully.")
        return feature_list, filenames
    except Exception as e:
        st.write(f"Error loading files: {e}")
        st.stop()

st.title("Image Classification and Recommendation System")
st.write("Upload an image to classify it and receive similar image recommendations.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    model_classification = initialize_classification_model()
    model_recommendation = initialize_recommendation_model()
    labels = load_image_data()
    feature_list, filenames = load_feature_data()

    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    st.write("")

    # Part 1: Image Classification
    st.write("Classifying the image...")
    img = load_img(uploaded_file, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    pred = model_classification.predict(img_array)
    pred_probabilities = pred[0]
    top_indices = pred_probabilities.argsort()[-3:][::-1]
    top_labels = [(labels[i], pred_probabilities[i]) for i in top_indices]

    # st.write("Top 3 Predictions:")
    # for label, probability in top_labels:
    #     st.write(f"{label}: {probability * 100:.2f}%")

    # Horizontal Bar Chart for Top Predictions
    st.write("Prediction Scores:")
    sorted_top_labels = sorted(top_labels, key=lambda x: x[1], reverse=True)
    chart_labels = [label for label, _ in sorted_top_labels]
    chart_scores = [score * 100 for _, score in sorted_top_labels]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(chart_labels, chart_scores, color='skyblue', height=0.4)

    for bar, score in zip(bars, chart_scores):
        ax.text(score + 1, bar.get_y() + bar.get_height()/2, f'{score:.2f}%', va='center')

    ax.set_title("Top 3 Predictions")
    ax.set_xlabel("Score (%)")
    ax.set_xlim(0, 100)
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    ax.invert_yaxis()
    st.pyplot(fig)

    # Part 2: Image Recommendation
    st.write("Finding similar images...")

    def save_uploaded_file(uploaded_file):
        if not os.path.exists('uploads'):
            os.makedirs('uploads')
        file_path = os.path.join('uploads', uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return file_path

    file_path = save_uploaded_file(uploaded_file)

    def extract_feature(img_path, model):
        img = cv2.imread(img_path)
        if img is None:
            st.write("Error: Could not read the image with OpenCV.")
            return None
        img = cv2.resize(img, (224, 224))
        expand_img = np.expand_dims(img, axis=0)
        pre_img = preprocess_input(expand_img)
        result = model.predict(pre_img).flatten()
        normalized = result / norm(result)
        return normalized

    features = extract_feature(file_path, model_recommendation)
    if features is not None:
        def recommend(features, feature_list):
            neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
            neighbors.fit(feature_list)
            _, indices = neighbors.kneighbors([features])
            return indices

        indices = recommend(features, feature_list)
        if indices is not None:
            st.write("Recommended Images:")
            col1, col2, col3, col4, col5 = st.columns(5)
            for i, col in enumerate([col1, col2, col3, col4, col5]):
                try:
                    image_path = filenames[indices[0][i]]
                    if image_path:
                        folder_name = os.path.basename(os.path.dirname(image_path))
                        with col:
                            st.image(image_path, caption=folder_name)
                    else:
                        st.write("Error: Invalid image path.")
                except IndexError:
                    st.write("Error: Not enough recommended images to display.")
    else:
        st.write("Feature extraction failed. No recommendations available.")

st.markdown('</div>', unsafe_allow_html=True)
