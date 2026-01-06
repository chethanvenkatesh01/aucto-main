# Base Image: Lightweight Python
FROM python:3.11-slim

# Set Working Directory
WORKDIR /app

# 1. Install Dependencies Layer (Cached)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. Copy Kernel Source Code
COPY . .

# 3. Create Persistence Files if they don't exist (to avoid permission errors)
RUN touch ados_ledger.db model_store.joblib

# Expose the API Port
EXPOSE 8000

# Ignite the Kernel
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
