document.addEventListener('DOMContentLoaded', function () {
    var locationModal = document.getElementById("locationModal");
    var sectorModal = document.getElementById("sectorModal");

    var locationMarker = document.getElementById("locationMarker");
    var sectorInfo = document.getElementById("sectorInfo");

    var closeButtons = document.getElementsByClassName("close");

    // Open modal for location
    locationMarker.onclick = function () {
        locationModal.style.display = "block";
    };

    // Open modal for sector
    sectorInfo.onclick = function () {
        sectorModal.style.display = "block";
    };

    // Close modals
    for (var i = 0; i < closeButtons.length; i++) {
        closeButtons[i].onclick = function () {
            locationModal.style.display = "none";
            sectorModal.style.display = "none";
        };
    }

    // Close modal on clicking outside of it
    window.onclick = function (event) {
        if (event.target == locationModal) {
            locationModal.style.display = "none";
        } else if (event.target == sectorModal) {
            sectorModal.style.display = "none";
        }
    }

    // Function making prediction upon user input executing zonal prediction 
    // TODO: set proper alg for zonal prediction
    function makePrediction() {
        // Get user input values
        const dl_user_throughput = document.getElementById('dl_user_throughput').value;
        const ul_user_throughput = document.getElementById('ul_user_throughput').value;
        const average_dl_pdcp = document.getElementById('average_dl_pdcp').value;
        const average_ul_pdcp = document.getElementById('average_ul_pdcp').value;
        // Add more feature values if necessary
    
        // Prepare the data to send to the API
        const inputData = {
            'dl_user_throughput': parseFloat(dl_user_throughput),
            'ul_user_throughput': parseFloat(ul_user_throughput),
            'average_dl_pdcp': parseFloat(average_dl_pdcp),
            'average_ul_pdcp': parseFloat(average_ul_pdcp)
            // Add more features as needed
        };
    
        // Make a POST request to the Flask API // localhost run
        fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(inputData)
        })
        .then(response => response.json())
        .then(data => {
            // Display the predicted zone
            document.getElementById('predicted-zone').textContent = 'Predicted Zone: ' + data.zone;
    
            // Update the speedometer with the predicted performance
            drawSpeedometer(data.performance);
        })
        .catch(error => {
            console.error('Error:', error);
        });
    }
        
    // Draw a simple speedometer for the predicted performance
    // TODO: Work on the context of performance data
    function drawSpeedometer(performance) {
        const canvas = document.getElementById('speedometer');
        const context = canvas.getContext('2d');
    
        // Clear the canvas
        context.clearRect(0, 0, canvas.width, canvas.height);
    
        // Draw the speedometer arc
        context.beginPath();
        context.arc(150, 150, 100, Math.PI, 2 * Math.PI);
        context.lineWidth = 10;
        context.strokeStyle = '#000';
        context.stroke();
    
        // Draw the needle (performance score)
        const angle = Math.PI + (performance * Math.PI);  // Map performance (0 to 1) to an angle
        const needleLength = 90;
        const needleX = 150 + needleLength * Math.cos(angle);
        const needleY = 150 + needleLength * Math.sin(angle);
    
        context.beginPath();
        context.moveTo(150, 150);
        context.lineTo(needleX, needleY);
        context.lineWidth = 5;
        context.strokeStyle = '#f00';
        context.stroke();
    }    
});
