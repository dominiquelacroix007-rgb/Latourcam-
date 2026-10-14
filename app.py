import os
import subprocess
import cv2
import numpy as np
import gradio as gr
import insightface
from insightface.app import FaceAnalysis

# Téléchargement automatique du modèle de face swap s'il n'existe pas
NOM_DU_MODELE = "inswapper_128.onnx"
if not os.path.exists(NOM_DU_MODELE):
    print("⏳ Téléchargement du modèle IA inswapper_128.onnx...")
    URL = "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx"
    subprocess.run(f"wget -q -O {NOM_DU_MODELE} {URL}", shell=True)

# Initialisation de l'analyseur de visage
app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))
swapper = insightface.model_zoo.get_model(NOM_DU_MODELE, providers=['CPUExecutionProvider'])

def process_frame(source_image, target_image):
    # Logique d'échange de visage simplifiée
    if source_image is None or target_image is None:
        return target_image
    
    # Conversion des images pour OpenCV
    source_cv = cv2.cvtColor(np.array(source_image), cv2.COLOR_RGB2BGR)
    target_cv = cv2.cvtColor(np.array(target_image), cv2.COLOR_RGB2BGR)
    
    faces = app.get(source_cv)
    if len(faces) == 0:
        return target_image
    source_face = faces[0]
    
    target_faces = app.get(target_cv)
    res = target_cv.copy()
    for face in target_faces:
        res = swapper.get(res, face, source_face, paste_back=True)
        
    return cv2.cvtColor(res, cv2.COLOR_BGR2RGB)

# Interface Gradio
demo = gr.Interface(
    fn=process_frame,
    inputs=[gr.Image(type="pil", label="Visage Source"), gr.Image(type="pil", label="Image Cible")],
    outputs=gr.Image(type="pil", label="Résultat"),
    title="Application d'échange de visages"
)

if __name__ == "__main__":
    demo.launch()
