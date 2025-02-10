import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

from FormatData import format_data
from PredictionModel import PredictionModel

app = Flask(__name__)
CORS(app)

df = format_data()

@app.route('/')
def hello_world():  # put application's code here
    return jsonify(df.to_dict(orient='records'))

@app.route("/api/data/get_period", methods=['GET'])
def get_data_within_period():
    # data = request.get_json()
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

    return jsonify(prediction)



if __name__ == '__main__':
    app.run(debug=True)
