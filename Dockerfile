# Dockerfile
FROM nvidia/cuda:12.1-base
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Run Django
CMD ["gunicorn", "framepack_api.wsgi:application", "--bind", "0.0.0.0:8000"]