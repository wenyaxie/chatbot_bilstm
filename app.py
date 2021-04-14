from flask import jsonify, request, Flask
from pandas import read_csv
from tensorflow.keras.models import load_model

from embed import embed_for_request, embed_for_serving
from predict import predict


app = Flask(__name__)


def init_app(model_path, response_data_path):
    app.model = load_model(model_path)

    app.df, app.vocabs, app.candidates = embed_for_serving()

    resp_data = read_csv(response_data_path)
    app.response_map = {}
    for index, row in read_csv(response_data_path).iterrows():
        app.response_map[row["Question"]] = row["Response"]


@app.route("/", methods=["POST"])
def serve():
    body = request.get_json()
    if not body:
        return "Bad requerst: expected json body", 400

    question = body["question"]
    if not question:
        return "Bad request: expected key \"question\"", 400

    answer_count = 10
    if "answer_count" in body:
        try:
            answer_count = int(body["answer_count"])
        except ValueError:
            return "Bad request: \"answer_count\" is not a valid integer", 400

    left, right = embed_for_request(app.df, app.vocabs, question)
    answers = predict(app.model, app.candidates,
                      app.response_map, left, right, answer_count)
    return jsonify(answer="__SEP__".join([answer.strip() for answer in answers]))
