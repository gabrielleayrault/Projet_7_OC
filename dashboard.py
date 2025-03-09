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

# Liens Google Drive 
url_vgg16 = "https://drive.google.com/uc?id=1VMkwLvWVipr3w8muEcuRy2FEhsI4BSeN"
url_convnext = "https://drive.google.com/uc?id=1JKxBONzDkTwG7F_78se6YI54g0ILvmoK"

MODEL_PATH_VGG16 = "VGG16_tfdata_with_aug_best.keras"
MODEL_PATH_CONVNEXT = "best_model_convnext.keras"

# Fonction pour télécharger le modèle si absent
def download_model(model_path, url):
    if not os.path.exists(model_path):
        st.info(f"Téléchargement du modèle {model_path} en cours...")
        gdown.download(url, model_path, quiet=False)

# Télécharger les modèles si nécessaire
download_model(MODEL_PATH_VGG16, url_vgg16)
download_model(MODEL_PATH_CONVNEXT, url_convnext)

# Vérification après téléchargement
if os.path.exists(MODEL_PATH_VGG16) and os.path.exists(MODEL_PATH_CONVNEXT):
    st.success("📥 Modèles téléchargés avec succès !")
else:
    st.error("❌ Erreur dans le téléchargement des modèles.")

# Charger le modèle après téléchargement
@st.cache_resource
def load_selected_model(model_name):
    if model_name == "VGG16":
        return load_model(MODEL_PATH_VGG16)
    else:
        return load_model(MODEL_PATH_CONVNEXT)

# ================================
# 1. Configuration du Dashboard
# ================================

st.set_page_config(page_title="Comparaison VGG16 vs ConvNeXt", layout="wide")

st.title("🐶 Classification de races de chiens : VGG16 vs ConvNeXt")
st.markdown("Ce dashboard compare les performances de **VGG16** et **ConvNeXt** sur le **Stanford Dogs Dataset**.")

# ================================
# 2. Chargement des modèles
# ================================

st.sidebar.header("📌 Sélection du modèle")

# Vérifier si la clé est bien initialisée avant d'utiliser st.sidebar.radio

if "selected_model" not in st.session_state:
    st.session_state.selected_model = "VGG16"  # Valeur par défaut

selected_model = st.sidebar.radio("Choisissez un modèle :", ["VGG16", "ConvNeXt"])

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
        if selected_model == "VGG16":
            st.image(conf_matrix_vgg16, caption="Matrice de confusion - VGG16", use_container_width=True)
        else:
            st.image(conf_matrix_convnext, caption="Matrice de confusion - ConvNeXt", use_container_width=True)

    with tab2:
        st.subheader("📈 Courbes d’apprentissage")
        learning_curve_vgg16 = "learning_curve_VGG16.png"
        learning_curve_convnext = "learning_curve_ConvNeXt.png"
        if selected_model == "VGG16":
            st.image(learning_curve_vgg16, caption="Courbes d’apprentissage - VGG16", use_container_width=True)
        else:
            st.image(learning_curve_convnext, caption="Courbes d’apprentissage - ConvNeXt", use_container_width=True)

if view_distribution:
    st.subheader("📊 Distribution des images par ensemble et par classe")
    st.image("distribution_images.png", caption="Répartition des images", use_container_width=True)

if view_test_images:
    st.subheader("📂 Exploration des images du dataset de test")

    TEST_IMAGES_PATH = "C:/Users/gabri/OneDrive/Bureau/COURS/Cours OPENCLASSROOMS/Projet 7/test-20250216T213517Z-001/test"
    sample_images = {
        "Maltese_dog": os.path.join(TEST_IMAGES_PATH, "n02085936-Maltese_dog"),
        "Afghan_hound": os.path.join(TEST_IMAGES_PATH, "n02088094-Afghan_hound"),
        "Irish_wolfhound": os.path.join(TEST_IMAGES_PATH, "n02090721-Irish_wolfhound"),
        "Scottish_deerhound": os.path.join(TEST_IMAGES_PATH, "n02092002-Scottish_deerhound"),
        "Bernese_mountain_dog": os.path.join(TEST_IMAGES_PATH, "n02107683-Bernese_mountain_dog"),
        "Samoyed": os.path.join(TEST_IMAGES_PATH, "n02111889-Samoyed"),
        "Pomeranian": os.path.join(TEST_IMAGES_PATH, "n02112018-Pomeranian"),
    }

    selected_race = st.selectbox("Choisissez une race :", list(sample_images.keys()))
    race_folder = sample_images[selected_race]
    
    if os.path.exists(race_folder):
        image_files = os.listdir(race_folder)
        selected_image = st.selectbox("📸 Sélectionnez une image :", image_files)
        image_path = os.path.join(race_folder, selected_image)
        st.image(image_path, caption=f"Image de {selected_race}", use_container_width=True)

# ================================
# 4. Comparaison avant / après transformations
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
        img_original = image.load_img(uploaded_file)
        img, img_array = preprocess_image(uploaded_file)

        col1, col2 = st.columns(2)
        with col1:
            st.image(img_original, caption="Image originale", use_container_width=True)
        with col2:
            st.image(img, caption="Image après transformation (redimensionnement, normalisation)", use_container_width=True)

# ================================
# 5. Prédiction sur une image uploadée
# ================================

CLASSES = [
    "Maltese_dog",
    "Afghan_hound",
    "Irish_wolfhound",
    "Scottish_deerhound",
    "Bernese_mountain_dog",
    "Samoyed",
    "Pomeranian"
]

st.subheader("📤 Téléchargez une image")
uploaded_file = st.file_uploader("Téléchargez une image de chien", type=["jpg", "png", "jpeg"])

def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img, img_array

if uploaded_file is not None:
    image_path = uploaded_file
    img_original = image.load_img(image_path)  # Charge l'image sans modification
    img, img_array = preprocess_image(image_path)  # Prétraitement pour la prédiction

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions)
    confidence = predictions[0][predicted_class_index] * 100
    predicted_class = CLASSES[predicted_class_index]

    # Affichage uniquement si une image a été uploadée
    st.image(img_original, caption="Image originale", use_container_width=True, clamp=True, output_format="PNG")
    st.write(f"**Modèle sélectionné : {selected_model}**")
    st.write(f"**Classe prédite : {predicted_class}**")
    st.write(f"**Confiance : {confidence:.2f}%**")
else:
    st.warning("⚠️ Veuillez uploader une image pour la prédiction.")

# ================================
# 6. Affichage des probabilités sous forme de graphique
# ================================

st.subheader("📊 Probabilités de classification")

if uploaded_file is not None:
    fig, ax = plt.subplots(figsize=(10, 5))  # Augmentation de la taille du graphique

    # Utilisation d’une palette plus claire et distincte
    colors = sns.color_palette("pastel", len(CLASSES))

    # Création du graphique
    sns.barplot(x=CLASSES, y=predictions[0], hue=CLASSES, palette=colors, ax=ax, legend=False)

    # Ajustement des textes et du format
    ax.set_xticks(range(len(CLASSES)))  # Définit explicitement les ticks
    ax.set_xticklabels(CLASSES, rotation=30, fontsize=12)
    ax.set_ylabel("Probabilité", fontsize=14)
    ax.set_title(f"Scores de confiance - {selected_model}", fontsize=16)

    st.pyplot(fig)  # Affichage dans Streamlit
else:
    st.warning("⚠️ Veuillez uploader une image pour voir les probabilités de classification.")

# ================================
# 7. Visualisation Grad-CAM
# ================================

# # Sélection de la dernière couche convolutive appropriée
# layer_names = [layer.name for layer in model.layers]

# if "block5_conv3" in layer_names:  # VGG16
#     last_conv_layer = "block5_conv3"
# elif "top_conv" in layer_names:  # ConvNeXt
#     last_conv_layer = "top_conv"
# elif "convnext_block4" in layer_names:  # Autre couche pour ConvNeXt
#     last_conv_layer = "convnext_block4"
# else:
#     last_conv_layer = layer_names[-3]  # Sélectionne une couche convolutive proche de la sortie

# # Vérification
# st.write(f"Couche utilisée pour Grad-CAM : {last_conv_layer}")

# def generate_grad_cam(model, image_array, layer_name):
#     """
#     Génère une heatmap Grad-CAM pour expliquer la prédiction d'un modèle.
    
#     - model : Modèle entraîné (VGG16 ou ConvNeXt)
#     - image_array : Image prétraitée sous forme de tableau numpy
#     - layer_name : Nom de la dernière couche convolutive
#     """
#     grad_model = tf.keras.models.Model(
#         inputs=[model.inputs], 
#         outputs=[model.get_layer(layer_name).output, model.output]
#     )

#     with tf.GradientTape() as tape:
#         conv_outputs, predictions = grad_model(image_array)
#         loss = predictions[:, np.argmax(predictions)]

#     grads = tape.gradient(loss, conv_outputs)

#     # Ajustement des dimensions de grads
#     if len(grads.shape) == 2:  # Cas spécial si le tenseur ne contient que (batch, channels)
#         pooled_grads = tf.reduce_mean(grads, axis=0)
#     else:
#         pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

#     conv_outputs = conv_outputs[0]
#     heatmap = np.dot(conv_outputs, pooled_grads.numpy().T)
#     heatmap = np.maximum(heatmap, 0)
#     heatmap /= np.max(heatmap)

#     return heatmap

# st.subheader("🔎 Interprétation des prédictions (Grad-CAM)")

# if uploaded_file is not None:
#     heatmap = generate_grad_cam(model, img_array, last_conv_layer)

#     # Superposition de la heatmap sur l'image d'origine
#     fig, ax = plt.subplots(figsize=(6, 6))
#     ax.imshow(img_original)  # Affichage de l'image originale
#     ax.imshow(cv2.resize(heatmap, (224, 224)), cmap="jet", alpha=0.5)  # Heatmap Grad-CAM
#     plt.axis("off")

#     st.pyplot(fig)
# else:
#     st.warning("⚠️ Veuillez uploader une image pour générer la heatmap Grad-CAM.")

# ================================
# 8. Déploiement sur le Cloud
# ================================

st.sidebar.subheader("🚀 Déploiement sur le Cloud")
st.sidebar.markdown("""
1️⃣ **Créer un repo GitHub** avec `dashboard.py` + les fichiers nécessaires.  
2️⃣ **Aller sur [Streamlit Cloud](https://share.streamlit.io/)** et connecter le repo.  
3️⃣ **Configurer le fichier `requirements.txt`** et ajouter : 
```text
streamlit
tensorflow
numpy
matplotlib
seaborn
opencv-python """)

# ================================
# 9. Mode accessibilité (WCAG)
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