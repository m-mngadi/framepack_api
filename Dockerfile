# Dockerfile
FROM nvidia/cuda:12.1-base
RUN apt-get update && apt-get install -y python3-pip
WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy project files
COPY . .

# Run Django
CMD ["gunicorn", "framepack_api.wsgi:application", "--bind", "0.0.0.0:8000"]
