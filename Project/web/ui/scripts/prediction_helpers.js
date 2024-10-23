// Send data to Flask API
// Function to compute composite features based on user input
function computeCompositeFeatures() {
    // Get values for 'Secondary Cell Performance' from input fields
    const scellVolumeDL = parseFloat(document.getElementById('scell_volume_dl_mac').value);
    const scellTrafficVolume = parseFloat(document.getElementById('scell_traffic_volume').value);

    // Compute 'Secondary Cell Performance'
    const secondaryCellPerformance = (scellVolumeDL + scellTrafficVolume) / 2;
    console.log("Computed Secondary Cell Performance:", secondaryCellPerformance);

    // Get values for 'Setup Success Rate' from input fields
    const rrcSetupSuccess = parseFloat(document.getElementById('ps_rrc_setup_success').value);
    const intraFreqHandoverSuccess = parseFloat(document.getElementById('intra_freq_handover').value);
    const rabSetupSuccess = parseFloat(document.getElementById('ps_rab_setup_success').value);

    // Compute 'Setup Success Rate'
    const setupSuccessRate = (rrcSetupSuccess + intraFreqHandoverSuccess + rabSetupSuccess) / 3;
    console.log("Computed Setup Success Rate:", setupSuccessRate);

    // Get values for 'Cell Availability and Downtime'
    const cellDowntime = parseFloat(document.getElementById('dce_erbs_cell_down').value);
    const cellAvailability = parseFloat(document.getElementById('cell_availability').value);

    // Compute 'Cell Availability and Downtime'
    const cellAvailabilityDowntime = (cellAvailability - cellDowntime);
    console.log("Computed Cell Availability and Downtime:", cellAvailabilityDowntime);

    return {
        secondaryCellPerformance: secondaryCellPerformance,
        setupSuccessRate: setupSuccessRate,
        cellAvailabilityDowntime: cellAvailabilityDowntime
    };
}

// Function to send data to Flask API when 'Predict Performance' button is clicked
function makePrediction() {
    // Compute composite features
    const computedValues = computeCompositeFeatures();

    // Gather all input data
    const inputData = {
        'Maximum Number of Users in a Cell': document.getElementById('max_users').value,
        'IRAT/Session Continuity to 2G': document.getElementById('irat_continuity').value,
        'Secondary Cell Performance': computedValues.secondaryCellPerformance,
        'Carrier Number DL (earfcn)': document.getElementById('carrier_number_dl').value,
        'Inter Frequency Handover Success Rate (%)': document.getElementById('inter_freq_handover').value,
        'DC_E_ERBS_EUTRANCELLFDD.pmPagDiscarded': document.getElementById('paging_discarded').value,
        'Setup Success Rate': computedValues.setupSuccessRate,
        'Cell Availability and Downtime': computedValues.cellAvailabilityDowntime
    };

    // Send the data to the Flask API
    fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(inputData),
    })
    .then(response => {
        // Log the raw response
        console.log(response);
        // Check if the response is OK
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        // Parse and return JSON response
        return response.json();
    })
    .then(data => {
        console.log("Data received:", data);       // Log the received data
        // Extract the 'Site Id' and 'Sector Id' from the response
        const performanceVal = data['Performance'][0];         
        const sectorId = data['Sector Id'][0]; 
        const siteId = data['Site Id'][0];            
        handlePredictionResponse(performanceVal, siteId, sectorId);          
    })
    .catch(error => console.error('Error:', error));
}

// Attach 'makePrediction' function to the button click event
document.getElementById('prediction-form').addEventListener('submit', function(event) {
    event.preventDefault();  // Prevent the form from being submitted in the traditional way
    makePrediction();        // Call the prediction function
});


// SAMPLE DATA INPUT USAGES: python3 -m http.server 8000

// TEST 1 (2C)
// Carrier Number DL (earfcn): 1399
// Inter Frequency Handover Success Rate (%): 98.89476539
// DC_E_ERBS_EUTRANCELLFDD.pmPagDiscarded: 0
// Maximum Number of Users in a Cell: 81
// IRAT/Session Continuity to 2G: 411
// Secondary Cell Performance:
//  - ScellVolume_DL_MAC: 4.68281275
//  - Scell Traffic Volume: 4.68281275
// Setup Success Rate:
//  - PS E-UTRAN RRC Setup Success Ratio: 99.93689594
//  - Intra Frequency Handover Success Rate: 98.29239427
//  - PS E-UTRAN RAB Setup Success Rate: 99.93458105
// Cell Availability and Downtime
//  - Cell Availability: 100
//  - DC_E_ERBS_EUTRANCELLFDD.pmCellDowntimeMan: 0

// TEST 2 (10B)
// Carrier Number DL (earfcn): 426
// Inter Frequency Handover Success Rate (%): 98.75357581
// DC_E_ERBS_EUTRANCELLFDD.pmPagDiscarded: 2
// Maximum Number of Users in a Cell: 71
// IRAT/Session Continuity to 2G: 167
// Secondary Cell Performance:
//  - ScellVolume_DL_MAC: 2.442353875
//  - Scell Traffic Volume: 2.442353875
// Setup Success Rate:
//  - PS E-UTRAN RRC Setup Success Ratio: 99.90471142
//  - Intra Frequency Handover Success Rate: 98.41644205
//  - PS E-UTRAN RAB Setup Success Rate: 99.83216405
// Cell Availability and Downtime
//  - Cell Availability: 100
//  - DC_E_ERBS_EUTRANCELLFDD.pmCellDowntimeMan: 0

// TEST 3 (10C)
// Carrier Number DL (earfcn): 426
// Inter Frequency Handover Success Rate (%): 99.25619835
// DC_E_ERBS_EUTRANCELLFDD.pmPagDiscarded: 7
// Maximum Number of Users in a Cell: 47
// IRAT/Session Continuity to 2G: 76
// Secondary Cell Performance:
//  - ScellVolume_DL_MAC: 2.6443275
//  - Scell Traffic Volume: 2.6443275
// Setup Success Rate:
//  - PS E-UTRAN RRC Setup Success Ratio: 99.89139251
//  - Intra Frequency Handover Success Rate: 98.63522088
//  - PS E-UTRAN RAB Setup Success Rate: 99.83580686


// Drop down raw features in a list of the composite features
function toggleDropdown(dropdownId, iconId) {
    var dropdown = document.getElementById(dropdownId);
    var icon = document.getElementById(iconId);

    // Toggle the visibility of the dropdown content
    if (dropdown.style.display === 'none') {
        dropdown.style.display = 'block';
        icon.src = 'pics/up.png'; // Change the icon to "up"
    } else {
        dropdown.style.display = 'none';
        icon.src = 'pics/down.png'; // Change the icon to "down"
    }
}

// Function to populate the result once obtain it
function handlePredictionResponse(performanceVal, siteId, sectorId) {
    // Update the result display in the HTML with the predicted values
    document.getElementById('predictedPerformance').textContent = performanceVal;
    document.getElementById('predictedSite').textContent = siteId;
    document.getElementById('predictedSector').textContent = sectorId;
    
    // Optionally, you can trigger other actions here like map updates or animations
    addPinToMap(siteId);                       // Handle the site prediction
    adjustNeedle(performanceVal);              // Rotate the needle based on the performance
}
