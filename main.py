from flask import Flask, jsonify, request
from flask_cors import CORS

from DataLoader import load_data
from PredictionModel import PredictionModel
from util.DateUtils import get_date_ranges, find_date_index, str_to_datetime

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

df = load_data()

@app.route("/api/ping", methods=["GET"])
def ping():
    return jsonify({})

#
@app.route("/api/forecast_unemployment", methods=['GET'])
def forecast():
    start = request.args.get('start')
    end = request.args.get('end')

    if start is None or end is None:
        return jsonify({"error": "start and end parameters are required"})

    predictor = PredictionModel(df, start, end, 10, 50)
    prediction = predictor.predict(6, 5, 1.1)

    #need to zip main data back to correct form
    original_data = list(zip(*predictor.training_df))[0]
    return jsonify({
        "base": {
            "index": get_date_ranges(df, start, end),
            "data": original_data
        },
        "forecast": prediction
    })


@app.route("/api/evaluate_model", methods=['GET'])
def evaluate_model():
    start = request.args.get('start')
    end = request.args.get('end')

    if start is None or end is None:
        return jsonify({"error": "start and end parameters are required"})

    predictor = PredictionModel(df, start, end, 10, 50)
    data = predictor.evaluate_model()
    return jsonify(data)


@app.route("/api/get_interest_and_inflation", methods=['GET'])
def get_interest_and_inflation():
    date = request.args.get('date')

    if date is None:
        return jsonify({"error": "date parameter is required"})

    i = find_date_index(df, str_to_datetime(date))

    inflation1 = df["data"][i][1]
    inflation2 = df["data"][i - 12][1]
    inflation = round((inflation1 - inflation2)/inflation2 * 100,2)
    interest = round(df["data"][i][2],2)

    return jsonify({"interest": interest, "inflation": inflation})


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
