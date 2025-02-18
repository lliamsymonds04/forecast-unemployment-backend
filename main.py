import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

from FormatData import format_data
from util.FormatDate import format_date
from PredictionModel import PredictionModel

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

df = format_data()

@app.route("/api/ping", methods=["GET"])
def ping():
    return jsonify({})

@app.route("/api/data/get_period", methods=['GET'])
def get_data_within_period():
    start = request.args.get('start')
    end = request.args.get('end')

    if start is None or end is None:
        return jsonify({"error": "start and end parameters are required"})

    period_df: pd.DataFrame = df.loc[(df.index >= start) & (df.index < end)]

    return jsonify(period_df.to_dict(orient='records'))

@app.route("/api/forecast_unemployment", methods=['GET'])
def forecast():
    start = request.args.get('start')
    end = request.args.get('end')

    if start is None or end is None:
        return jsonify({"error": "start and end parameters are required"})

    predictor = PredictionModel(df, start, end, 10, 50)
    prediction = predictor.predict(6, 5, 1.1)

    # return predictor.graph_predictions(prediction)
    data = predictor.graph_predictions(prediction)
    return jsonify(data.to_json(orient='records'))

@app.route("/api/evaluate_model", methods=['GET'])
def evaluate_model():
    start = request.args.get('start')
    end = request.args.get('end')

    if start is None or end is None:
        return jsonify({"error": "start and end parameters are required"})

    predictor = PredictionModel(df, start, end, 10, 50)
    # return predictor.evaluate_model()
    data = predictor.evaluate_model()
    return jsonify(data.to_json(orient='records'))



@app.route("/api/get_interest_and_inflation", methods=['GET'])
def get_interest_and_inflation():
    date = request.args.get('date')

    if date is None:
        return jsonify({"error": "date parameter is required"})

    formatted_date = format_date(date)
    row = df.loc[df.index <= formatted_date]
    inflation1 = row.iloc[-1,1]
    inflation2 = row.iloc[-12,1]
    inflation = round((inflation1 - inflation2)/inflation2 * 100,2)
    interest = round(row.iloc[-1, 2],2)

    return jsonify({"interest": interest, "inflation": inflation})



if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
