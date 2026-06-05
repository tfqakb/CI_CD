FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN python -m pip install --upgrade pip --no-cache-dir --progress-bar off
RUN pip install --no-cache-dir --progress-bar off -r requirements.txt
EXPOSE 8000
CMD ["uvicorn",
"predict_model:app",
"--host",
"0.0.0.0",
"port",
"8000"]