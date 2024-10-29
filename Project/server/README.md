# 5G Network Quality Prediction System

## Overview

This project leverages machine learning models to predict 5G network performance across different geographical zones, based on data features like carrier frequency, session continuity, handover success rate, and more. The predictions include performance scores, sector, and site IDs using pre-trained models for both performance and sector classification. The system is built using Flask for the backend and a responsive web interface for input and output visualization.

## Running Guide

### Backend Setup

1. **Navigate to Backend Folder**:
   ```bash
   cd backend
   ```

2. **Install Dependencies**:
   Install all the required Python libraries using `pip3`:
   ```bash
   pip3 install -r requirements.txt
   ```
3. **Run Live Web**:
   Open the frontend HTML file in your browser or host it on a local server.

4. **Modify CORS Configuration**:
   Ensure that the Flask backend allows requests from the live web interface by modifying the `CORS` setup:
   ```python
   CORS(app, resources={r"/*": {"origins": ["<<Live IP Address>>"]}})
   ```
   Adjust the IP address or domain to the one where your frontend is hosted. If you are developing locally, use `http://127.0.0.1:5501` or `http://localhost:5000`.

5. **Modify Frontend to Send Requests to Flask Backend**:
   In the frontend code, update the `fetch` request URL to match your Flask backend IP address:
   ```javascript
   fetch('http://<<Your Flask Backend IP Address>>/predict', {
       method: 'POST',
       headers: {
           'Content-Type': 'application/json',
       },
       body: JSON.stringify(inputData),
   })
   ```
   Replace `<<Your Flask Backend IP Address>>` with the actual address of your Flask server.
   
6. **Run Flask Application**:
   Start the Flask app by executing:
   ```bash
   python3 flask_app.py
   ```


### Dataset

- **Dataset Name**: `LTE_KPI.csv`
- **Description**: The dataset contains 5G network KPIs (Key Performance Indicators) such as throughput, handover success rate, carrier frequency, etc. It is used to predict network performance by sector and site.
- **Dataset Provider**: Ericsson Vietnam

### Important Notes:
- **Acknowledgment**: The dataset is a private and confidential industrial asset of Ericsson Vietnam. Unauthorized use is strictly prohibited.

### Group Members

- **BAO Nguyen**
- **Nguyen Hung Nguyen**
- **Trung Kien Nguyen**
- **Dang Khoa Le**
- **Saw Ko Ko Oo**

## Features

1. **Carrier Number DL (earfcn)**: Downlink carrier frequency.
2. **Inter Frequency Handover Success Rate (%)**: Percentage of successful handovers between different frequency bands.
3. **DC_E_ERBS_EUTRANCELLFDD.pmPagDiscarded**: Paging discarded requests.
4. **Maximum Number of Users in a Cell**: Highest number of users in a cell at one time.
5. **IRAT/Session Continuity to 2G**: Continuity of sessions when switching from LTE to 2G networks.
6. **Secondary Cell Performance**: A composite metric combining secondary cell performance in dual connectivity scenarios.
7. **Setup Success Rate**: Success rate for RRC setup, intra-frequency handover, and RAB setup.
8. **Cell Availability and Downtime**: The percentage of time a cell is available versus its downtime.
