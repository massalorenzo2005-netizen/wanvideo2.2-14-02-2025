FROM runpod/comfyui:latest-5090

# 1. Installiamo git e rclone
USER root
RUN apt-get update && apt-get install -y git rclone && rm -rf /var/lib/apt/lists/*

# 2. Creiamo lo script di avvio intelligente (Auto-Download & Auto-Backup)
RUN echo '#!/bin/bash\n\
mkdir -p ~/.config/rclone\n\
echo "[gcs]\ntype = google cloud storage\nservice_account_credentials = $RCLONE_CONFIG_GCS_SERVICE_ACCOUNT_CREDENTIALS" > ~/.config/rclone/rclone.conf\n\
\n\
# Controllo se i modelli esistono su Google Cloud\n\
echo "[Elite] Controllo disponibilità su Google Mother Ship..."\n\
if rclone ls gcs:runpodwanvideo2214022026/checkpoints | grep "wan"; then\n\
    echo "[Elite] Modelli trovati su Google! Inizio download turbo..."\n\
    rclone copy gcs:runpodwanvideo2214022026/ /ComfyUI/models/ --progress --transfers 8\n\
else\n\
    echo "[Elite] Google è vuoto. Al termine del primo Job, caricheremo i file su Google per sempre."\n\
fi\n\
\n\
exec python3 -u /rp_handler.py' > /start_elite.sh && chmod +x /start_elite.sh

# 3. Nodi e dipendenze
WORKDIR /ComfyUI/custom_nodes/ComfyUI-WanVideoWrapper
COPY . .
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /
ENTRYPOINT ["/start_elite.sh"]
