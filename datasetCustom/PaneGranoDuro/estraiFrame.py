import os
import cv2

def estrai_frame(video_path, intervallo=0.25, min_frame=1, max_frame=10):
    # Ottieni il nome del file senza estensione per creare una cartella di output
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_folder = os.path.join(os.getcwd(), base_name)

    # Crea la cartella di output se non esiste
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Carica il video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Errore nell'apertura del video {video_path}")
        return

    # Ottieni il frame rate del video
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * intervallo)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    current_frame = 0
    extracted_frames = 0

    while cap.isOpened() and extracted_frames < max_frame:
        ret, frame = cap.read()
        if not ret:
            break

        # Controlla se l'attuale frame è a intervallo di 0.25 secondi
        if current_frame % frame_interval == 0:
            # Salva il frame
            frame_name = os.path.join(output_folder, f"frame_{current_frame}.jpg")
            cv2.imwrite(frame_name, frame)
            extracted_frames += 1

            # Interrompe se raggiunge il numero massimo di frame
            if extracted_frames >= max_frame:
                break

        current_frame += 1

    # Assicurati di chiudere il video
    cap.release()

    # Verifica se il numero di frame estratti è sufficiente
    if extracted_frames < min_frame:
        print(f"Numero di frame estratti ({extracted_frames}) è minore del minimo richiesto ({min_frame}).")
    else:
        print(f"Frame estratti con successo da {video_path}: {extracted_frames}")

# Esegui l'estrazione dei frame per tutti i file mp4 nella cartella corrente
current_folder = os.getcwd()
video_files = [f for f in os.listdir(current_folder) if f.endswith('.mp4')]

for video_file in video_files:
    video_path = os.path.join(current_folder, video_file)
    estrai_frame(video_path, intervallo=0.1, min_frame=10, max_frame=400)