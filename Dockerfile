FROM python:3.13-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright \
    DEBIAN_FRONTEND=noninteractive

# Sys libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    fonts-liberation \
    libasound2 \
    libatk-bridge2.0-0 \
    libatk1.0-0 \
    libatspi2.0-0 \
    libdbus-1-3 \
    libdrm2 \
    libgbm1 \
    libgtk-3-0 \
    libnspr4 \
    libnss3 \
    libx11-6 \
    libx11-xcb1 \
    libxcb1 \
    libxcomposite1 \
    libxdamage1 \
    libxext6 \
    libxfixes3 \
    libxkbcommon0 \
    libxrandr2 \
    libxrender1 \
    libxshmfence1 \
    libcairo2 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    wget \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && python -m playwright install chromium

COPY app.py .

RUN mkdir -p /app/data

RUN useradd -m -u 10001 appuser \
 && chown -R appuser:appuser /app /ms-playwright

USER appuser
ENV PYTHONUNBUFFERED=1

CMD ["python", "app.py"]
