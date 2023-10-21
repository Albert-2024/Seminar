from flask import Flask, request, render_template,jsonify
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

app = Flask(__name__)

with open('prophet_model3.pkl', 'rb') as file:
    loaded_data = pickle.load(file)
    
@app.route('/')
def index():
    return render_template('index.html')
    
@app.route('/predict',methods=['POST'])
def predict():
    
    start_date = request.form['start-date']
    end_date = request.form['end-date']

    future_dates = pd.DataFrame({'ds': pd.to_datetime(pd.date_range(start=start_date, end=end_date))})

    forecast = loaded_data.predict(future_dates)

    print(forecast[['ds','yhat']])

    loaded_data.plot(forecast)
    
    data = forecast

    plt.plot(data['ds'], data['yhat'], label='Actual')
    plt.plot(forecast['ds'], forecast['yhat'], label='Predicted')

# Add labels and titlefrom matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title('Forecast')

# Show the plot
    plt.show()
    
    forecast_data = forecast[['ds', 'yhat']].to_dict(orient='records')
    return jsonify({'forecast': forecast_data})

if __name__ == '__main__':
    app.run(debug=True)