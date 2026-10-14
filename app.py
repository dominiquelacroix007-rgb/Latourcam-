import os
import subprocess
import cv2
import numpy as np
import gradio as gr
import insightface
from insightface.app import FaceAnalysis

# Téléchargement du modèle de face swap depuis Hugging Face au démarrage
MODEL_PATH = "inswapper_128.onnx"
if not os.path.exists(MODEL_PATH):
    print("⏳ Téléchargement du modèle IA inswapper_128 (554 Mo)...")
    url = "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx"
    subprocess.run(f"wget -q -O {MODEL_PATH} {url}", shell=True)

# Chargement du moteur (utilisation CPU par défaut sur Hugging Face Space gratuit)
# Si vous possédez un GPU sur votre Space, remplacez par ['CUDAExecutionProvider']
app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))
swapper = insightface.model_zoo.get_model(MODEL_PATH, providers=['CPUExecutionProvider'])

CLE_DE_SECURITE_SECRETE = "MON_APP_2026"

def traiter_trame(img_source, img_cible, faces_source, blending=1.0, det_thresh=0.6, det_size=640):
    app.prepare(ctx_id=0, det_size=(det_size, det_size), det_thresh=det_thresh)
    faces_cible = app.get(img_cible)
    if len(faces_cible) == 0:
        return img_cible

    resultat = img_cible.copy()
    source_face = faces_source[0]

    for target_face in faces_cible:
        swapped_face = swapper.get(resultat, target_face, source_face, paste_back=True)
        if blending < 1.0:
            resultat = cv2.addWeighted(swapped_face, blending, resultat, 1.0 - blending, 0)
        else:
            resultat = swapped_face
    return resultat

def face_swap_main(cle_api, photo_visage_source, flux_webcam, fichier_video, blending, det_thresh, det_size):
    if cle_api != CLE_DE_SECURITE_SECRETE:
        return None, None, "❌ Clé de sécurité invalide !"

    if photo_visage_source is None:
        return None, None, "⚠️ Veuillez fournir une photo du visage source."

    app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.5)
    img_source = cv2.cvtColor(photo_visage_source, cv2.COLOR_RGB2BGR)
    faces_source = app.get(img_source)
    if len(faces_source) == 0:
        return None, None, "❌ Aucun visage détecté sur la photo source."

    if flux_webcam is not None:
        img_cible = cv2.cvtColor(flux_webcam, cv2.COLOR_RGB2BGR)
        res = traiter_trame(img_source, img_cible, faces_source, blending, det_thresh, int(det_size))
        res_rgb = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
        return res_rgb, None, "🎥 Traitement direct webcam actif"

    if fichier_video is not None:
        cap = cv2.VideoCapture(fichier_video)
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output_path = "video_modifiee.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_swapped = traiter_trame(img_source, frame, faces_source, blending, det_thresh, int(det_size))
            out.write(frame_swapped)

        cap.release()
        out.release()
        return None, output_path, "✅ Vidéo traitée avec succès !"

    return None, None, "⚠️ Fournissez soit le flux webcam, soit un fichier vidéo."

with gr.Blocks(title="Réplique facEvo - Live & Vidéo") as demo:
    gr.Markdown("# 🎭 Alternative facEvo : Face Swap Direct Webcam & Vidéo")
    cle_input = gr.Textbox(label="Clé API / Sûreté", placeholder="Clé d'accès...", type="password")

    with gr.Row():
        with gr.Column():
            visage_source = gr.Image(label="1. Visage à appliquer (Source)")
            webcam_input = gr.Image(sources=["webcam"], streaming=True, label="2. Caméra en direct (Webcam)")
            video_input = gr.Video(label="OU Fichier Vidéo (Import MP4)")
            
            with gr.Group():
                gr.Markdown("### ⚙️ Paramètres d'optimisation vidéo")
                blending_slider = gr.Slider(minimum=0.0, maximum=1.0, value=1.0, step=0.05, label="Taux de mélange (Blending)")
                det_thresh_slider = gr.Slider(minimum=0.1, maximum=0.9, value=0.5, step=0.05, label="Seuil de détection")
                det_size_choice = gr.Dropdown(choices=["320", "640", "960"], value="320", label="Taille d'analyse")
            
            bouton_lancer = gr.Button("🚀 Lancer le traitement", variant="primary")

        with gr.Column():
            output_live = gr.Image(label="Résultat Direct Webcam")
            output_video = gr.Video(label="Résultat Fichier Vidéo")
            statut_output = gr.Textbox(label="Statut", interactive=False)

    webcam_input.change(
        fn=face_swap_main,
        inputs=[cle_input, visage_source, webcam_input, gr.State(None), blending_slider, det_thresh_slider, det_size_choice],
        outputs=[output_live, output_video, statut_output]
    )

    bouton_lancer.click(
        fn=face_swap_main,
        inputs=[cle_input, visage_source, webcam_input, video_input, blending_slider, det_thresh_slider, det_size_choice],
        outputs=[output_live, output_video, statut_output]
    )

demo.launch()
