# Usa l'immagine specifica che hai indicato (Ufficiale RunPod 5090)
FROM runpod/comfyui:latest-5090

# Installiamo git (fondamentale per collegare i nodi correttamente)
USER root
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Cartella dei nodi per l'immagine 5090
# Posizioniamoci nel tuo fork di Kijai
WORKDIR /ComfyUI/custom_nodes/ComfyUI-WanVideoWrapper

# Copiamo i file dal tuo repository
COPY . .

# Installiamo le dipendenze specifiche (requirements.txt)
RUN pip install --no-cache-dir -r requirements.txt

# Torniamo alla base per l'avvio del container
WORKDIR /
