import sys

from app import app, init_app
from embed import embed_for_training
from train import train


TRAIN_DATA_PATH = "data/BiLSTM_train_data.csv"
VALIDATION_DATA_PATH = "data/BiLSTM_validation_data.csv"
EMBEDDING_PATH = "embeding/GoogleNews-vectors-negative300"
MODEL_PATH = "model"
RESPONSE_DATA_PATH = "data/response_data.csv"

PORT = 22370


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ["train", "serve"]:
        sys.exit("Expect argument [train|serve]")

    if sys.argv[1].lower() == "train":
        x_train, y_train, x_validation, y_validation, embeddings = embed_for_training(
            TRAIN_DATA_PATH, VALIDATION_DATA_PATH, EMBEDDING_PATH)
        model = train(x_train, y_train, x_validation, y_validation, embeddings)
        model.save(MODEL_PATH)
    if sys.argv[1].lower() == "serve":
        init_app(MODEL_PATH, RESPONSE_DATA_PATH)
        app.run(host="0.0.0.0", port=PORT)
