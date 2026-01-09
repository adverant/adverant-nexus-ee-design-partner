# Nexus EE Design Partner - Production Dockerfile
#
# Multi-stage build for optimized production image
# with Node.js runtime and Python for KiCad automation

# =============================================================================
# Stage 1: Builder
# =============================================================================
FROM node:20-alpine AS builder

WORKDIR /app

# Install build dependencies
RUN apk add --no-cache python3 make g++

# Copy package files
COPY package*.json ./

# Install all dependencies (including dev)
RUN npm install

# Copy source code
COPY tsconfig.json ./
COPY src ./src

# Build TypeScript
RUN npm run build

# Prune dev dependencies
RUN npm prune --production

# =============================================================================
# Stage 2: Python Environment
# =============================================================================
FROM python:3.11-slim AS python-builder

WORKDIR /opt

# Install Python dependencies
RUN python -m venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"

COPY python-scripts/requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

# =============================================================================
# Stage 3: Production Runtime
# =============================================================================
FROM node:20-alpine AS production

# Build arguments for metadata
ARG BUILD_ID
ARG BUILD_TIMESTAMP
ARG GIT_COMMIT
ARG GIT_BRANCH
ARG VERSION

# Labels for image metadata
LABEL org.opencontainers.image.title="Nexus EE Design Partner" \
      org.opencontainers.image.description="End-to-end hardware/software development automation" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${GIT_COMMIT}" \
      org.opencontainers.image.vendor="Adverant Inc." \
      org.opencontainers.image.authors="engineering@adverant.ai" \
      com.adverant.build.id="${BUILD_ID}" \
      com.adverant.build.timestamp="${BUILD_TIMESTAMP}" \
      com.adverant.build.branch="${GIT_BRANCH}"

# Environment variables
ENV NODE_ENV=production \
    PORT=8080 \
    NEXUS_BUILD_ID="${BUILD_ID}" \
    NEXUS_BUILD_TIMESTAMP="${BUILD_TIMESTAMP}" \
    NEXUS_GIT_COMMIT="${GIT_COMMIT}" \
    NEXUS_VERSION="${VERSION}" \
    PYTHON_PATH="/opt/venv/bin/python3" \
    PYTHON_SCRIPTS_DIR="/app/python-scripts" \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Install runtime dependencies
RUN apk add --no-cache \
    python3 \
    py3-pip \
    tini \
    curl \
    && rm -rf /var/cache/apk/*

# Create non-root user
RUN addgroup -g 1001 nexus \
    && adduser -u 1001 -G nexus -s /bin/sh -D nexus

# Copy Python virtual environment
COPY --from=python-builder /opt/venv /opt/venv

# Copy built application
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/package*.json ./

# Copy Python scripts
COPY python-scripts ./python-scripts

# Create necessary directories
RUN mkdir -p /tmp/ee-design /data/projects /data/output /var/log/nexus-ee-design \
    && chown -R nexus:nexus /app /tmp/ee-design /data /var/log/nexus-ee-design

# Switch to non-root user
USER nexus

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Use tini as init process
ENTRYPOINT ["/sbin/tini", "--"]

# Start the application
CMD ["node", "dist/index.js"]