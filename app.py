import os
import subprocess
import cv2
import numpy as np
import streamlit as st
import insightface
from insightface.app import FaceAnalysis

# 1. On s'assure que le modèle est téléchargé EN PREMIER
NOM_DU_MODELE = "inswapper_128.onnx"
if not os.path.exists(NOM_DU_MODELE):
    print("⏳ Téléchargement du modèle IA inswapper_128.onnx...")
    URL = "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx"
    subprocess.run(f"wget -q -O {NOM_DU_MODELE} {URL}", shell=True)

# 2. Ensuite seulement, on initialise l'application
app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))
swapper = insightface.model_zoo.get_model(NOM_DU_MODELE, providers=['CPUExecutionProvider'])

st.title("Application d'échange de visages")

source_file = st.file_uploader("Choisissez l'image source (visage)", type=["jpg", "jpeg", "png"])
target_file = st.file_uploader("Choisissez l'image cible", type=["jpg", "jpeg", "png"])

if source_file and target_file:
    st.write("Traitement en cours...")
    # Conversion des images
    source_bytes = np.asarray(bytearray(source_file.read()), dtype=np.uint8)
    source_cv = cv2.imdecode(source_bytes, 1.7)
    
    target_bytes = np.asarray(bytearray(target_file.read()), dtype=np.uint8)
    target_cv = cv2.imdecode(target_bytes, 1)
    
    faces = app.get(source_cv)
    if len(faces) > 0:
        source_face = faces[0]
        target_faces = app.get(target_cv)
        res = target_cv.copy()
        for face in target_faces:
            res = swapper.get(res, face, source_face, paste_back=True)
            
        res_rgb = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
        st.image(res_rgb, caption="Résultat de l'échange", use_column_width=True)
    else:
        st.error("Aucun visage détecté sur l'image source.")
