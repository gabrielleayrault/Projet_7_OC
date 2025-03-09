import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2
import os
import gdown

# ================================
# 1. Configuration du Dashboard
# ================================
st.set_page_config(page_title="Comparaison VGG16 vs ConvNeXt", layout="wide")

st.title("🐶 Classification de races de chiens : VGG16 vs ConvNeXt")
st.markdown("Ce dashboard compare les performances de **VGG16** et **ConvNeXt** sur le **Stanford Dogs Dataset**.")

# ================================
# 2. Téléchargement et Chargement des Modèles
# ================================
url_vgg16 = "https://drive.google.com/uc?id=1VMkwLvWVipr3w8muEcuRy2FEhsI4BSeN"
url_convnext = "https://drive.google.com/uc?id=1JKxBONzDkTwG7F_78se6YI54g0ILvmoK"

MODEL_PATH_VGG16 = "VGG16_tfdata_with_aug_best.keras"
MODEL_PATH_CONVNEXT = "best_model_convnext.keras"

def download_model(model_path, url):
    if not os.path.exists(model_path):
        st.info(f"Téléchargement du modèle {model_path} en cours...")
        gdown.download(url, model_path, quiet=False)

st.sidebar.header("📌 Sélection du modèle")

if st.sidebar.button("📥 Télécharger les modèles"):
    download_model(MODEL_PATH_VGG16, url_vgg16)
    download_model(MODEL_PATH_CONVNEXT, url_convnext)
    st.sidebar.success("📥 Modèles téléchargés avec succès !")

models_available = os.path.exists(MODEL_PATH_VGG16) and os.path.exists(MODEL_PATH_CONVNEXT)

if models_available:
    selected_model = st.sidebar.radio("Choisissez un modèle :", ["VGG16", "ConvNeXt"])

    @st.cache_resource
    def load_selected_model(model_name):
        if model_name == "VGG16":
            return load_model(MODEL_PATH_VGG16)
        else:
            return load_model(MODEL_PATH_CONVNEXT)

    if st.sidebar.button("🔄 Charger le modèle"):
        st.session_state.model = load_selected_model(selected_model)
        st.sidebar.success("✅ Modèle chargé avec succès !")

# ================================
# 3. Affichage des Matrices de Confusion & Courbes
# ================================
st.sidebar.subheader("📊 Affichage des résultats")
view_results = st.sidebar.checkbox("Afficher les matrices de confusion & courbes")

if view_results:
    tab1, tab2 = st.tabs(["📊 Matrices de confusion", "📈 Courbes d’apprentissage"])
    
    with tab1:
        st.subheader("📊 Matrices de confusion")
        conf_matrix_vgg16 = "confusion_matrix_VGG16.png"
        conf_matrix_convnext = "confusion_matrix_ConvNeXt.png"
        st.image(conf_matrix_vgg16 if selected_model == "VGG16" else conf_matrix_convnext, 
                 caption=f"Matrice de confusion - {selected_model}", use_column_width=True)

    with tab2:
        st.subheader("📈 Courbes d’apprentissage")
        learning_curve_vgg16 = "learning_curve_VGG16.png"
        learning_curve_convnext = "learning_curve_ConvNeXt.png"
        st.image(learning_curve_vgg16 if selected_model == "VGG16" else learning_curve_convnext, 
                 caption=f"Courbes d’apprentissage - {selected_model}", use_column_width=True)

# ================================
# 4. Exploration des Images du Dataset Test
# ================================
st.subheader("📂 Exploration des images du dataset de test")

TEST_IMAGES_PATH = "test_dataset"

sample_images = {
    "Maltese_dog": os.path.join(TEST_IMAGES_PATH, "n02085936-Maltese_dog"),
    "Afghan_hound": os.path.join(TEST_IMAGES_PATH, "n02088094-Afghan_hound"),
    "Irish_wolfhound": os.path.join(TEST_IMAGES_PATH, "n02090721-Irish_wolfhound"),
}

selected_race = st.selectbox("Choisissez une race :", list(sample_images.keys()))
race_folder = sample_images[selected_race]

if os.path.exists(race_folder):
    image_files = os.listdir(race_folder)
    selected_image = st.selectbox("📸 Sélectionnez une image :", image_files)
    image_path = os.path.join(race_folder, selected_image)
    st.image(image_path, caption=f"Image de {selected_race}", width=700)

# ================================
# 5. Visualisation Grad-CAM
# ================================
st.subheader("🔎 Visualisation Grad-CAM")

def generate_grad_cam(model, img_array, layer_name):
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs], 
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, np.argmax(predictions)]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = np.dot(conv_outputs, pooled_grads.numpy().T)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    return heatmap

if selected_model == "VGG16":
    last_conv_layer = "block5_conv3"
else:
    last_conv_layer = "top_conv"

if st.sidebar.button("Générer Grad-CAM") and uploaded_file is not None:
    heatmap = generate_grad_cam(st.session_state.model, img_array, last_conv_layer)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img_original)
    ax.imshow(cv2.resize(heatmap, (224, 224)), cmap="jet", alpha=0.5)
    plt.axis("off")
    st.pyplot(fig)

# ================================
# 6. Mode Accessibilité
# ================================
st.sidebar.subheader("♿ Accessibilité")
high_contrast = st.sidebar.checkbox("🎨 Mode High Contrast", value=False)
large_text = st.sidebar.checkbox("🔠 Agrandir le texte")

if high_contrast:
    plt.style.use('dark_background')
else:
    plt.style.use('default')

if large_text:
    st.markdown("""
        <style>
            body { font-size: 20px !important; }
            h1, h2, h3, h4 { font-size: 26px !important; }
        </style>
    """, unsafe_allow_html=True)

# ================================
# 7. Déploiement sur le Cloud
# ================================
st.sidebar.subheader("🚀 Déploiement sur Streamlit Cloud")
st.sidebar.markdown("""
1️⃣ **Créer un repo GitHub** contenant `dashboard.py` et les fichiers nécessaires.  
2️⃣ **Se connecter à [Streamlit Cloud](https://share.streamlit.io/)** et lier le repo.  
3️⃣ **Ajouter un fichier `requirements.txt`** avec :
```text
streamlit
tensorflow
numpy
matplotlib
seaborn
opencv-python
gdown