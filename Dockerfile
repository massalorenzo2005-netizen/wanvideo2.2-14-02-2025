# Usa l'immagine UFFICIALE e PUBBLICA di RunPod (Corretta)
FROM runpod/worker-comfyui:latest

# Installiamo git (necessario per risolvere il warning dei log)
USER root
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Posizioniamoci nella cartella dei nodi personalizzati
# L'immagine ufficiale mette ComfyUI in /ComfyUI
WORKDIR /ComfyUI/custom_nodes/ComfyUI-WanVideoWrapper

# Copiamo i file dal tuo repository GitHub (il tuo fork di Kijai)
COPY . .

# Installiamo le dipendenze specifiche dei tuoi nodi
RUN pip install --no-cache-dir -r requirements.txt

# Torniamo alla cartella principale per il worker
WORKDIR /

