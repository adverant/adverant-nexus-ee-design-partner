# KiCad Worker Container for MAPOS
# Includes KiCad CLI, pcbnew Python API, and Xvfb for headless operation
#
# Build: docker build -t adverant/kicad-worker:latest -f docker/mapos/Dockerfile .
# Run:   docker run -v /path/to/pcb:/data adverant/kicad-worker fill-zones /data/board.kicad_pcb

FROM ubuntu:22.04

LABEL org.opencontainers.image.title="MAPOS KiCad Worker"
LABEL org.opencontainers.image.description="KiCad headless worker for MAPOS PCB optimization"
LABEL org.opencontainers.image.vendor="Adverant Inc."
LABEL org.opencontainers.image.version="1.0.0"

# Prevent interactive prompts during install
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    wget \
    curl \
    xvfb \
    x11vnc \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Add KiCad PPA and install KiCad
RUN add-apt-repository -y ppa:kicad/kicad-8.0-releases \
    && apt-get update \
    && apt-get install -y kicad \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip3 install --no-cache-dir \
    kiauto \
    requests \
    psutil \
    fastapi \
    uvicorn \
    python-multipart

# Create working directory
WORKDIR /app

# Copy MAPOS scripts (path relative to build context at repo root)
COPY services/nexus-ee-design/python-scripts/mapos/ /app/

# Set environment variables
ENV DISPLAY=:99
ENV KICAD_HEADLESS=1
ENV PYTHONUNBUFFERED=1

# Create entrypoint script that starts Xvfb
RUN echo '#!/bin/bash\n\
Xvfb :99 -screen 0 1920x1080x24 &\n\
sleep 1\n\
exec "$@"' > /entrypoint.sh && chmod +x /entrypoint.sh

# Expose API port
EXPOSE 8080

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python3", "/app/kicad_api_server.py", "--host", "0.0.0.0", "--port", "8080"]
