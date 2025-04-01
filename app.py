from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from models import train_models, predict_cost, detect_anomalies, segment_patients, detect_fraud
from data_preprocessing import preprocess_data
from sklearn.linear_model import LogisticRegression, LinearRegression
import matplotlib.dates as mdates

app = Flask(__name__)
# Load and preprocess data with Age
data = preprocess_data('dataset/visits.csv', 'dataset/patients.csv')


# Train Models
regressor, anomaly_model, cluster_model = train_models(data)

# Function to save and encode plots
def save_plot(fig, filename):
    img_path = f"static/{filename}.png"
    fig.savefig(img_path, bbox_inches='tight')
    plt.close(fig)
    return img_path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analysis')
def analysis():
    # Ensure necessary columns exist
    if not {'Age', 'Payment Status', 'Service Type', 'Date of Visit', 'Room Charges(daily rate)'}.issubset(data.columns):
        return "Error: Required columns missing from dataset.", 400

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))  # 2x2 layout

    # ✅ 1. Logistic Regression: Payment Status Analysis  
    data['Payment Status'] = data['Payment Status'].map({'Paid': 1, 'Unpaid': 0}).fillna(0)
    
    X_log = data[['Age', 'Treatment Cost', 'Medication Cost']].apply(pd.to_numeric, errors='coerce').fillna(0)
    y_log = data['Payment Status']

    log_reg = LogisticRegression()
    log_reg.fit(X_log, y_log)
    predictions = log_reg.predict_proba(X_log)[:, 1]  # Probability of being Paid

    sns.histplot(predictions, bins=20, kde=True, ax=axes[0, 0])
    axes[0, 0].set_title('Payment Probability Distribution')

    # ✅ 2. Month-wise Service Type Analysis  
    data['Date of Visit'] = pd.to_datetime(data['Date of Visit'], errors='coerce')
    data['Month'] = data['Date of Visit'].dt.strftime('%Y-%m')

    service_type_monthly = data.groupby(['Month', 'Service Type']).size().unstack().fillna(0)

    service_type_monthly.plot(kind='bar', stacked=True, ax=axes[0, 1])
    axes[0, 1].set_title('Month-wise Service Type Analysis')
    axes[0, 1].set_xticklabels(service_type_monthly.index, rotation=45)

    # ✅ 3. Linear Regression: Room Charges Month-wise  
    room_charges_monthly = data.groupby('Month')['Room Charges(daily rate)'].mean().reset_index()
    room_charges_monthly['Month'] = pd.to_datetime(room_charges_monthly['Month'])
    room_charges_monthly = room_charges_monthly.sort_values('Month')

    X_lin = (room_charges_monthly['Month'] - room_charges_monthly['Month'].min()).dt.days.values.reshape(-1, 1)
    y_lin = room_charges_monthly['Room Charges(daily rate)'].values

    lin_reg = LinearRegression()
    lin_reg.fit(X_lin, y_lin)
    y_pred = lin_reg.predict(X_lin)

    axes[1, 0].plot(room_charges_monthly['Month'], y_lin, marker='o', label='Actual')
    axes[1, 0].plot(room_charges_monthly['Month'], y_pred, linestyle='dashed', color='red', label='Predicted')
    axes[1, 0].set_title('Room Charges Trend (Linear Regression)')
    axes[1, 0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axes[1, 0].legend()

    # ✅ 4. Age-wise Billing Amount Analysis  
    sns.barplot(x=data['Age'], y=data['Treatment Cost'], ax=axes[1, 1])
    axes[1, 1].set_title('Age-wise Billing Amount')

    fig.tight_layout()
    analysis_plot = save_plot(fig, "analysis_plot")
    
    return render_template('analysis.html', analysis_plot=analysis_plot)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            features = request.json['features']

            # Ensure only the allowed features are passed
            prediction = predict_cost(features, regressor)

            # If prediction is NaN, return an error
            if np.isnan(prediction):
                return jsonify({'error': 'Prediction failed due to input error.'}), 400

            return jsonify({'predicted_cost': round(prediction, 2)})
        except Exception as e:
            return jsonify({'error': str(e)}), 400
    return render_template('prediction.html')  # Render prediction page for GET requests



@app.route('/anomaly_detection')
def anomaly():
    anomalies = detect_anomalies(data, anomaly_model)

    # If no anomalies are detected, show a message
    if anomalies.empty:
        return render_template('anomaly.html', anomaly_plot=None, anomalies=[])

    # Generate anomaly visualization
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(x=data['Treatment Cost'], y=data['Medication Cost'], label="Normal Data", alpha=0.5)
    sns.scatterplot(x=anomalies['Treatment Cost'], y=anomalies['Medication Cost'], color='red', label="Anomalies", marker='X', s=100, ax=ax)
    ax.set_title("Top Anomalies in Cost")

    anomaly_plot = save_plot(fig, "anomaly_plot")
    
    return render_template('anomaly.html', anomaly_plot=anomaly_plot, anomalies=anomalies.to_dict(orient='records'))

@app.route('/segmentation')
def segmentation():
    clusters = segment_patients(data, cluster_model)

    fig, ax = plt.subplots(figsize=(8, 5))
    clusters.plot(kind='pie', y='Treatment Cost', ax=ax, autopct='%1.1f%%', legend=True)
    ax.set_ylabel('')
    ax.set_title('Patient Segmentation')

    segmentation_plot = save_plot(fig, "segmentation_plot")

    return render_template('segmentation.html', segmentation_plot=segmentation_plot)

def save_plot(fig, filename):
    img_path = f"static/{filename}.png"
    fig.savefig(img_path, bbox_inches='tight')
    plt.close(fig)
    return img_path

@app.route('/fraud_detection')
def fraud_detection():
    fraud_data = detect_fraud(data.copy())

    # Ensure 'color' column is assigned for template
    fraud_data['color'] = fraud_data['Fraudulent'].apply(lambda x: 'red' if x == 'Yes' else 'green')

    # Generate fraud detection bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(x=fraud_data['Fraudulent'], palette={'No': 'green', 'Yes': 'red'}, ax=ax)
    ax.set_title("Fraudulent vs Legitimate Transactions")

    fraud_plot = save_plot(fig, "fraud_plot")

    return render_template('fraud_detection.html', fraud_plot=fraud_plot, fraud_data=fraud_data.to_dict(orient='records'))

@app.route('/export_fraud_data')
def export_fraud_data():
    fraud_data = detect_fraud(data.copy())

    export_path = 'dataset/fraud_data.csv'
    fraud_data.to_csv(export_path, index=False)

    return send_file(export_path, as_attachment=True)
if __name__ == '__main__':
    app.run(debug=True)
