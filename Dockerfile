FROM python:3.10-slim

# Install system dependencies for PaddleOCR, OpenCV, and PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies (using CPU-only PyTorch to keep image small)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy the rest of the application code
COPY . .

# Expose Streamlit's default port
EXPOSE 8501

# Run Streamlit (headless mode is required for Docker)
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]