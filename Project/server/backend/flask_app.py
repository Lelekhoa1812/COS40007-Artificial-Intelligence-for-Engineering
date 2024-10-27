from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS ensuring data transmission
import numpy as np
import tensorflow as tf
import joblib
import pandas as pd

# Load the best model
model_per = tf.keras.models.load_model('best_ann_model_speed.h5')  # Best Model for performance
features_scaler = joblib.load('features_scaler.pkl')               # Scaler for performance
per_scaler = joblib.load('per_scaler.pkl')                         # Inverse-Scaler for performance
model_sec = joblib.load('model_top_sec.pkl')                       # Best Model for sector
model_site = joblib.load('model_top_site.pkl')                     # Best Model for site

app = Flask(__name__)
# CORS(app, resources={r"/*": {"origins": ["http://127.0.0.1:5501", "http://localhost:5000"]}})  # Allow requests from localhost:8000
CORS(app, resources={r"/*": {"origins": "http://localhost:8000"}})  # Allow requests from localhost:8000

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the request (as JSON)
        input_data = request.json

        ## PERFORMANCE
        # Extract features from input data and create a DataFrame with correct column names
        features = pd.DataFrame([[
            input_data['Carrier Number DL (earfcn)'],
            input_data['Inter Frequency Handover Success Rate (%)'],
            input_data['DC_E_ERBS_EUTRANCELLFDD.pmPagDiscarded'],
            input_data['Maximum Number of Users in a Cell'],
            input_data['IRAT/Session Continuity to 2G'], 
            input_data['Secondary_Cell_Performance'],
            input_data['Setup_Success_Rate'],
            input_data['Cell_Availability_and_Downtime'],
        ]], columns=[
            'Carrier Number DL (earfcn)', #
            'Inter Frequency Handover Success Rate (%)',#
            'DC_E_ERBS_EUTRANCELLFDD.pmPagDiscarded',
            'Maximum Number of Users in a Cell', #
            'IRAT/Session Continuity to 2G', #
            'Secondary_Cell_Performance', #
            'Setup_Success_Rate',
            'Cell_Availability_and_Downtime',
        ])

        # Normalize the features using the loaded scaler
        features_scaled = features_scaler.transform(features)
        per_features = features_scaled[:, [0, 1, 3, 4, 5]]

        # Predict using the performance model
        prediction_per = model_per.predict(per_features)
        prediction_per_list = prediction_per.tolist()

        print(f"Performance Prediction: {prediction_per}")
        print(f"Performance Shape: {prediction_per.shape}")
        

        ## SECTOR
        # Use the first prediction value from performance prediction in the sector model
        traffic_volume_and_payload_metrics = prediction_per[0][0]  # Correct access

        # Scale and convert back to DataFrame
        features_scaled = pd.DataFrame(
            features_scaler.transform(features),
            columns=features.columns
        )

        # Extract the current array without the last value
        sec_site_features = features_scaled.drop(['Cell_Availability_and_Downtime'], axis=1, errors='ignore')

        # Insert the new column at the second-to-last position
        sec_site_features.insert(
            loc=0,  # Insert at second-to-last position
            column='Traffic_Volume_and_Payload_Metrics',
            value=traffic_volume_and_payload_metrics
        )
        sector_desired_order = [
            'Traffic_Volume_and_Payload_Metrics',          # 6
            'IRAT/Session Continuity to 2G',               # 4
            'Secondary_Cell_Performance',                  # 5
            'Inter Frequency Handover Success Rate (%)',   # 1
            'Maximum Number of Users in a Cell',           # 3
            'Setup_Success_Rate',                          # 7
            'Carrier Number DL (earfcn)',                  # 0
            'DC_E_ERBS_EUTRANCELLFDD.pmPagDiscarded',     # 2
        ]

        # Reorder the columns
        sec_features = sec_site_features[sector_desired_order]

        # Predict using the sector model
        prediction_sec = model_sec.predict(sec_features)
        prediction_sec_list = prediction_sec.tolist()

        print(f"Sector Prediction: {prediction_sec}")
        print(f"Sector Shape: {prediction_sec.shape}")


        ## SITE
        site_desired_order = [
            'Secondary_Cell_Performance',
            'IRAT/Session Continuity to 2G',
            'Traffic_Volume_and_Payload_Metrics',
            'Maximum Number of Users in a Cell',
            'Inter Frequency Handover Success Rate (%)',
            'Setup_Success_Rate',
            'Carrier Number DL (earfcn)',
            'DC_E_ERBS_EUTRANCELLFDD.pmPagDiscarded'
        ]

        # Reorder the columns
        site_features = sec_site_features[site_desired_order]

        # Predict using the site model
        prediction_site = model_site.predict(site_features)
        prediction_site_list = prediction_site.tolist()

        print(f"Site Prediction: {prediction_site}")
        print(f"Site Shape: {prediction_site.shape}")

        prediction_value = prediction_per_list[0][0]  # gets the scalar value
        prediction_reshaped = np.array(prediction_value).reshape(1, -1)  # alternative reshape
        prediction_original = per_scaler.inverse_transform(prediction_reshaped)
        
        # Return the predictions as a JSON response
        return jsonify({
            'Normalized Performance': prediction_per_list,
            'Raw Performance': prediction_original.tolist(),
            'Sector Id': prediction_sec_list,
            'Site Id': prediction_site_list
        })

    except Exception as e:
        print("Error:", str(e))
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
