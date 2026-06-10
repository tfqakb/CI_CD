FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN python -m pip install --upgrade pip --no-cache-dir --progress-bar off
RUN pip install --no-cache-dir --progress-bar off -r requirements.txt
EXPOSE 8000
CMD python predict_model.py