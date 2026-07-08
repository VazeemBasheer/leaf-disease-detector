FROM python:3.11-slim

WORKDIR /app
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY app/ ./app/
COPY src/ ./src/
COPY models/ ./models/

ENV MODEL_PATH=/app/models/resnet18_leaf_best.pth
ENV CLASS_NAMES_PATH=/app/models/class_names.json

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
