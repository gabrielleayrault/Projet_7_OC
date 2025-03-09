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
from PIL import Image

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

    # Charger le modèle uniquement si ce n'est pas déjà fait
    if "model" not in st.session_state or st.session_state.selected_model != selected_model:
        st.session_state.model = load_selected_model(selected_model)
        st.session_state.selected_model = selected_model  # Mettre à jour le modèle sélectionné
        st.sidebar.success(f"✅ Modèle {selected_model} chargé avec succès !")

    # Utilisation du modèle
    model = st.session_state.model

else:
    st.sidebar.warning("⚠️ Téléchargez d'abord les modèles pour les sélectionner.")

# ================================
# 3. Onglets interactifs 
# ================================

st.sidebar.subheader("📌 Options d'affichage")
view_results = st.sidebar.checkbox("Afficher les matrices de confusion & courbes")
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
# 5. Affichage des probabilités sous forme de graphique
# ================================

st.subheader("📊 Probabilités de classification")

if uploaded_file is not None:
    fig, ax = plt.subplots(figsize=(10, 5))  # Augmentation de la taille du graphique

    # Utilisation d’une palette plus claire et distincte
    colors = sns.color_palette("pastel", len(CLASSES))

    # Création du graphique
    sns.barplot(x=CLASSES, y=predictions[0], palette=colors, ax=ax)

    # Ajustement des textes et du format
    ax.set_xticks(range(len(CLASSES)))  # Définit explicitement les ticks
    ax.set_xticklabels(CLASSES, rotation=30, fontsize=12)
    ax.set_ylabel("Probabilité", fontsize=14)
    ax.set_title(f"Scores de confiance - {selected_model}", fontsize=16)

    st.pyplot(fig)  # Affichage dans Streamlit
else:
    st.warning("⚠️ Veuillez uploader une image pour voir les probabilités de classification.")

# ================================
# 6. Comparaison avant / après transformations
# ================================

if view_transformations:
    st.subheader("🎨 Transformations effectuées sur les images")

    uploaded_file = st.file_uploader("Téléchargez une image pour voir les transformations", type=["jpg", "png", "jpeg"], key="uploader_transform")

    def preprocess_image(image_path):
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        return img, img_array

    if uploaded_file is not None:
        img_original = Image.open(uploaded_file)
        img, img_array = preprocess_image(uploaded_file)

        col1, col2 = st.columns(2)
        with col1:
            st.image(img_original, caption="Image originale", use_container_width=True)
        with col2:
            st.image(img, caption="Image après transformation (redimensionnement, normalisation)", use_container_width=True)

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
""")

# ================================
# 8. Mode accessibilité (WCAG)
# ================================
st.sidebar.subheader("♿ Accessibilité")
high_contrast = st.sidebar.checkbox("🎨 Activer le mode High Contrast (graphique uniquement)", value=False)
large_text = st.sidebar.checkbox("🔠 Agrandir le texte")

if high_contrast:
        plt.style.use('dark_background')
else:
        plt.style.use('default')

# Appliquer une taille de texte normale par défaut
st.markdown("""
    <style>
        body { font-size: 16px !important; }  /* Taille normale */
        h1, h2, h3, h4 { font-size: 22px !important; }
    </style>
""", unsafe_allow_html=True)

# Si l'option "Agrandir le texte" est cochée, augmenter la taille
if large_text:
    st.markdown("""
        <style>
            body { font-size: 20px !important; }  /* Texte plus grand */
            h1, h2, h3, h4 { font-size: 26px !important; }
        </style>
    """, unsafe_allow_html=True)

# 2. Texte alternatif et instructions claires

explanations_mode = st.sidebar.checkbox("📝 Activer les explications simplifiées")

if explanations_mode:
    st.write("""
        **Comment utiliser cette section ?**  
        1️⃣ Cliquez sur "Parcourir" et sélectionnez une image d’un chien sur votre appareil.  
        2️⃣ Le modèle va analyser l’image et prédire la race du chien.  
        3️⃣ Vous verrez aussi un score indiquant la fiabilité de la prédiction.
    """)

# 3. Raccourci Clavier (Validation via "Entrée")
st.markdown("""
    <script>
    document.addEventListener("keydown", function(event) {
        if (event.key === "Enter") {
            document.querySelector("button").click();
        }
    });
    </script>
""", unsafe_allow_html=True)