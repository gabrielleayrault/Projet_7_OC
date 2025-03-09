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
# 2. Téléchargement des modèles
# ================================

st.sidebar.header("📌 Sélection du modèle")

url_vgg16 = "https://drive.google.com/uc?id=1VMkwLvWVipr3w8muEcuRy2FEhsI4BSeN"
url_convnext = "https://drive.google.com/uc?id=1JKxBONzDkTwG7F_78se6YI54g0ILvmoK"

MODEL_PATH_VGG16 = "VGG16_tfdata_with_aug_best.keras"
MODEL_PATH_CONVNEXT = "best_model_convnext.keras"

def download_model(model_path, url):
    """Télécharge un modèle s'il n'existe pas."""
    if not os.path.exists(model_path):
        st.info(f"Téléchargement du modèle {model_path} en cours...")
        gdown.download(url, model_path, quiet=False)

# Bouton pour télécharger les modèles
if st.sidebar.button("📥 Télécharger les modèles"):
    download_model(MODEL_PATH_VGG16, url_vgg16)
    download_model(MODEL_PATH_CONVNEXT, url_convnext)

    # Vérification après téléchargement
    if os.path.exists(MODEL_PATH_VGG16) and os.path.exists(MODEL_PATH_CONVNEXT):
        st.sidebar.success("📥 Modèles téléchargés avec succès !")
    else:
        st.sidebar.error("❌ Erreur dans le téléchargement des modèles.")

# Vérifier si les modèles existent avant de les charger
models_available = os.path.exists(MODEL_PATH_VGG16) and os.path.exists(MODEL_PATH_CONVNEXT)

if models_available:
    # Sélection du modèle
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = "VGG16"

    selected_model = st.sidebar.radio("Choisissez un modèle :", ["VGG16", "ConvNeXt"])

    @st.cache_resource
    def load_selected_model(model_name):
        """Charge le modèle sélectionné."""
        if model_name == "VGG16":
            return load_model(MODEL_PATH_VGG16)
        else:
            return load_model(MODEL_PATH_CONVNEXT)

    # Bouton pour charger le modèle
    if st.sidebar.button("🔄 Charger le modèle"):
        model = load_selected_model(selected_model)
        st.sidebar.success("✅ Modèle chargé avec succès !")
else:
    st.sidebar.warning("⚠️ Téléchargez d'abord les modèles pour les sélectionner.")

# ================================
# 3. Onglets interactifs 
# ================================

st.sidebar.subheader("📌 Options d'affichage")
view_results = st.sidebar.checkbox("Afficher les matrices de confusion & courbes")
view_test_images = st.sidebar.checkbox("Afficher les images du dataset TEST")
view_distribution = st.sidebar.checkbox("📊 Afficher la distribution des images")
view_transformations = st.sidebar.checkbox("🎨 Voir les transformations d'images")

if view_results:
    tab1, tab2 = st.tabs(["📊 Matrices de confusion", "📈 Courbes d’apprentissage"])
    
    with tab1:
        st.subheader("📊 Matrices de confusion")
        conf_matrix_vgg16 = "confusion_matrix_VGG16_tfdata_with_aug_TEST.png"
        conf_matrix_convnext = "confusion_matrix_ConvNeXt.png"
        st.image(conf_matrix_vgg16 if selected_model == "VGG16" else conf_matrix_convnext, 
                 caption=f"Matrice de confusion - {selected_model}", use_column_width=True)

    with tab2:
        st.subheader("📈 Courbes d’apprentissage")
        learning_curve_vgg16 = "learning_curve_VGG16.png"
        learning_curve_convnext = "learning_curve_ConvNeXt.png"
        st.image(learning_curve_vgg16 if selected_model == "VGG16" else learning_curve_convnext, 
                 caption=f"Courbes d’apprentissage - {selected_model}", use_column_width=True)

if view_distribution:
    st.subheader("📊 Distribution des images par ensemble et par classe")
    st.image("distribution_images.png", caption="Répartition des images", use_column_width=True)

# ================================
# 4. Prédiction sur une image uploadée
# ================================

st.subheader("📤 Téléchargez une image de chien")
uploaded_file = st.file_uploader("Sélectionnez une image", type=["jpg", "png", "jpeg"])

CLASSES = ["Maltese_dog", "Afghan_hound", "Irish_wolfhound", "Scottish_deerhound", "Bernese_mountain_dog", "Samoyed", "Pomeranian"]

def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img, img_array

if uploaded_file and models_available:
    img_original = image.load_img(uploaded_file)
    img, img_array = preprocess_image(uploaded_file)

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions)
    confidence = predictions[0][predicted_class_index] * 100
    predicted_class = CLASSES[predicted_class_index]

    st.image(img_original, caption="Image originale", use_column_width=True)
    st.write(f"**Classe prédite : {predicted_class}**")
    st.write(f"**Confiance : {confidence:.2f}%**")

# ================================
# 5. Déploiement sur le Cloud
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
""")

# ================================
# 6. Mode accessibilité (WCAG)
# ================================
st.sidebar.subheader("♿ Accessibilité") 

high_contrast = st.sidebar.checkbox("🎨 Mode High Contrast")
large_text = st.sidebar.checkbox("🔠 Texte agrandi")

if high_contrast:
    plt.style.use('dark_background')
else:
    plt.style.use('default')

if large_text:
    st.markdown(
        """
        <style>
            body { font-size: 20px !important; }
            h1, h2, h3, h4 { font-size: 26px !important; }
        </style>
        """,
        unsafe_allow_html=True
    )