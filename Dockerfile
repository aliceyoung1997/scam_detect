FROM python:3.12.10

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY phishing_detector.pkl .
COPY predict.py .
COPY phishing_detector.pkl .
COPY preprocessed_data.pkl .


EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
