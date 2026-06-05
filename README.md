# News Sentiment Analysis API with FastAPI & Machine Learning

A FastAPI-based Machine Learning API to classify Indonesian stock news sentiment using TF-IDF vectorization and Logistic Regression.

---

## Folder Structure

```text
news-sentiment-analysis/
├── dataset/
│   ├── news_sentiment.csv          # training dataset
│   └── evaluation_result.csv       # evaluation output
├── docker-compose.yaml             # docker service definition
├── Dockerfile                      # container configuration
├── main.py                         # FastAPI application
├── model.pkl                       # trained model
├── README.md
├── requirements.txt
├── schema.py                       # request / response schema
└── train_model.py                  # model training process
```

---

## Machine Learning Flow

```text
load_dataset → split_train_test → tfidf_vectorization → train_model → evaluation → save_model → prediction_api
```

| Step                  | Description                            |
| --------------------- | -------------------------------------- |
| `load_dataset`        | Load sentiment dataset from CSV        |
| `split_train_test`    | Split dataset into train and test data |
| `tfidf_vectorization` | Convert text into TF-IDF features      |
| `train_model`         | Train Logistic Regression model        |
| `evaluation`          | Evaluate prediction performance        |
| `save_model`          | Store trained model as `model.pkl`     |
| `prediction_api`      | Serve predictions using FastAPI        |

---

## Supported Sentiment Labels

| Label      | Description        |
| ---------- | ------------------ |
| `positive` | Positive sentiment |
| `negative` | Negative sentiment |
| `neutral`  | Neutral sentiment  |

---

## Tech Stack

| Component        | Technology   |
| ---------------- | ------------ |
| Language         | Python 3.12  |
| API Framework    | FastAPI      |
| ML Library       | Scikit-learn |
| Data Processing  | Pandas       |
| Containerization | Docker       |
| API Server       | Uvicorn      |

---

## Setup & Running

### 1. Clone Repository

```bash
git clone https://github.com/gitaferdina/news-sentiment-analysis.git

cd news-sentiment-analysis
```

### 2. Download Dataset

Download dataset:

```text
https://www.kaggle.com/datasets/triagungj/cnbc-indonesia-stock-news-sentiment-dataset
```

Rename file:

```text
news_sentiment.csv
```

Move file into:

```text
dataset/
```

### 3. Build Container

```bash
docker compose up --build
```

### 4. Open API Documentation

```text
http://localhost:8000/docs
```

---

## Training Process

Training berjalan otomatis ketika container startup.

| Step | Process                      |
| ---- | ---------------------------- |
| 1    | Load dataset                 |
| 2    | Train / test split           |
| 3    | TF-IDF transformation        |
| 4    | Logistic Regression training |
| 5    | Model evaluation             |
| 6    | Save model                   |

---

## Evaluation Result

### evaluation_result.csv

Berisi detail hasil prediksi model.

| Column                | Description            |
| --------------------- | ---------------------- |
| `news`                | News text              |
| `actual_sentiment`    | Actual sentiment       |
| `predicted_sentiment` | Model prediction       |
| `is_correct`          | Prediction correctness |

---

## Docker Configuration

### docker-compose.yaml

```yaml
version: '3.9'

services:

  news-api:

    build: .

    container_name: news-sentiment-api

    ports:
      - "8000:8000"

    volumes:
      - .:/app

    command: >
      sh -c "
      python train_model.py &&
      uvicorn main:app --host 0.0.0.0 --port 8000
      "

    restart: always
```

---

## Run Without Docker

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Train Model

```bash
python train_model.py
```

### Run API

```bash
uvicorn main:app --reload
```

---

## Requirements

```text
fastapi
scikit-learn
pandas
numpy
uvicorn
joblib
docker
```
