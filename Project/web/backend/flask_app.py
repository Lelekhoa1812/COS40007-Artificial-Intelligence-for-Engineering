from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS ensuring data transmission
import numpy as np
import tensorflow as tf
import joblib
import pandas as pd

# Load the best model
model_per = tf.keras.models.load_model('best_ann_model_speed.h5')  # Best Model for performance
model_sec = joblib.load('model_top_sec.pkl')                       # Best Model for sector
model_site = joblib.load('model_top_site.pkl')                     # Best Model for site

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:8000"}})  # Allow requests from localhost:8000

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the request (as JSON)
        input_data = request.json

        ## PERFORMANCE
        # Extract features from input data and create a DataFrame with correct column names
        features_per = pd.DataFrame([[
            input_data['Maximum Number of Users in a Cell'],
            input_data['IRAT/Session Continuity to 2G'], 
            input_data['Secondary Cell Performance'],
            input_data['Carrier Number DL (earfcn)'],
            input_data['Inter Frequency Handover Success Rate (%)']
        ]], columns=[
            'Maximum Number of Users in a Cell', 
            'IRAT/Session Continuity to 2G', 
            'Secondary_Cell_Performance', 
            'Carrier Number DL (earfcn)', 
            'Inter Frequency Handover Success Rate (%)' 
        ])

        # Predict using the performance model
        prediction_per = model_per.predict(features_per)
        prediction_per_list = prediction_per.tolist()

        print(f"Performance Prediction: {prediction_per}")
        print(f"Performance Shape: {prediction_per.shape}")
        

        ## SECTOR
        # Use the first prediction value from performance prediction in the sector model
        traffic_volume_and_payload_metrics = prediction_per[0][0]  # Correct access

        # Extract features for sector prediction
        features_sec = pd.DataFrame([[
            traffic_volume_and_payload_metrics,  # Use performance prediction 
            input_data['IRAT/Session Continuity to 2G'], 
            input_data['Secondary Cell Performance'],
            input_data['Inter Frequency Handover Success Rate (%)'],
            input_data['Maximum Number of Users in a Cell'],
            input_data['Setup Success Rate'],
            input_data['Carrier Number DL (earfcn)'],
            input_data['DC_E_ERBS_EUTRANCELLFDD.pmPagDiscarded']
        ]], columns=[
            'Traffic_Volume_and_Payload_Metrics',
            'IRAT/Session Continuity to 2G',
            'Secondary_Cell_Performance',
            'Inter Frequency Handover Success Rate (%)',
            'Maximum Number of Users in a Cell',
            'Setup_Success_Rate',
            'Carrier Number DL (earfcn)',
            'DC_E_ERBS_EUTRANCELLFDD.pmPagDiscarded'
        ])

        # Predict using the sector model
        prediction_sec = model_sec.predict(features_sec)
        prediction_sec_list = prediction_sec.tolist()

        print(f"Sector Prediction: {prediction_sec}")
        print(f"Sector Shape: {prediction_sec.shape}")


        ## SITE
        # Extract features for site prediction
        features_site = pd.DataFrame([[
            input_data['Secondary Cell Performance'],
            input_data['IRAT/Session Continuity to 2G'],
            traffic_volume_and_payload_metrics,  # Use performance prediction 
            input_data['Maximum Number of Users in a Cell'],
            input_data['Inter Frequency Handover Success Rate (%)'],
            input_data['Setup Success Rate'],
            input_data['Carrier Number DL (earfcn)'],
            input_data['DC_E_ERBS_EUTRANCELLFDD.pmPagDiscarded']
        ]], columns=[
            'Secondary_Cell_Performance',
            'IRAT/Session Continuity to 2G',
            'Traffic_Volume_and_Payload_Metrics',
            'Maximum Number of Users in a Cell',
            'Inter Frequency Handover Success Rate (%)',
            'Setup_Success_Rate',
            'Carrier Number DL (earfcn)',
            'DC_E_ERBS_EUTRANCELLFDD.pmPagDiscarded'
        ])

        # Predict using the site model
        prediction_site = model_site.predict(features_site)
        prediction_site_list = prediction_site.tolist()

        print(f"Site Prediction: {prediction_site}")
        print(f"Site Shape: {prediction_site.shape}")
        
        # Return the predictions as a JSON response
        return jsonify({
            'Performance': prediction_per_list,
            'Sector Id': prediction_sec_list,
            'Site Id': prediction_site_list
        })

    except Exception as e:
        print("Error:", str(e))
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
