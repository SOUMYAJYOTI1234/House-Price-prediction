from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application


@app.route('/')
def index():
    """Landing page."""
    return render_template('index.html')


@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    """Prediction page - form + results."""
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            MedInc=float(request.form.get('MedInc')),
            HouseAge=float(request.form.get('HouseAge')),
            AveRooms=float(request.form.get('AveRooms')),
            AveBedrms=float(request.form.get('AveBedrms')),
            Population=float(request.form.get('Population')),
            AveOccup=float(request.form.get('AveOccup')),
            Latitude=float(request.form.get('Latitude')),
            Longitude=float(request.form.get('Longitude')),
        )

        pred_df = data.get_data_as_dataframe()
        pipeline = PredictPipeline()
        results = pipeline.predict(pred_df)

        # Format as dollar amount (in $100,000s - sklearn dataset uses this unit)
        predicted_price = results[0] * 100000

        return render_template(
            'home.html',
            results=f"${predicted_price:,.0f}"
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
