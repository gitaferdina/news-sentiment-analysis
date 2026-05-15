import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report


def load_dataset(path):
    df = pd.read_csv(path)
    X = df["judul"].astype(str)
    y = df["sentimen"]

    return X, y


def split_dataset(X, y):
    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )


def build_pipeline():
    pipeline = Pipeline([("tfidf", TfidfVectorizer()),
                         ("model",LogisticRegression(max_iter=1000))])

    return pipeline


def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)

    return model


def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(classification_report(y_test, y_pred))

    print("Accuracy:", accuracy)

    result_df = pd.DataFrame({
        "news": X_test.values,
        "actual_sentiment": y_test.values,
        "predicted_sentiment": y_pred
    })

    result_df["is_correct"] = (result_df["actual_sentiment"] == result_df["predicted_sentiment"])
    result_df.to_csv("dataset/evaluation_result.csv", index=False)

    print("Evaluation CSV saved!")

    return y_pred


def save_model(model, path):
    joblib.dump(model, path)


def main():

    # Load
    X, y = load_dataset(
        "dataset/news_sentiment.csv"
    )

    # Split
    X_train, X_test, y_train, y_test = split_dataset(X, y)

    # Build model
    pipeline = build_pipeline()

    # Train
    train_model(
        pipeline,
        X_train,
        y_train
    )

    # Evaluate
    evaluate_model(
        pipeline,
        X_test,
        y_test
    )

    # Save
    save_model(
        pipeline,
        "model.pkl"
    )

    print("Training selesai!")


if __name__ == "__main__":
    main()