# docker build -t review_clf .
# docker run -p 8000:8000 review_clf
FROM python:3.12.2-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "src/api.py", "./runs/train1/best_catboost_model.cbm"]
