FROM python:3.11-slim

# Playwright system dependencies
RUN apt-get update && apt-get install -y \
    wget curl gnupg \
    libnss3 libatk-bridge2.0-0 libdrm2 libxkbcommon0 libgbm1 \
    libgtk-3-0 libxss1 libasound2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright Chromium and its dependencies
RUN crawl4ai-setup && python -m playwright install --with-deps chromium

COPY . .

# Output volume mount point
VOLUME ["/data/raw_markdown"]

ENV OUTPUT_DIR=/data/raw_markdown

CMD ["python", "scraper.py"]
