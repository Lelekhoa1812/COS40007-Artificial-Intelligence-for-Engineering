document.addEventListener('DOMContentLoaded', function () {
    var locationModal = document.getElementById("locationModal");
    var sectorModal = document.getElementById("sectorModal");
    var NRModal = document.getElementById("NRModal");
    var RFModal = document.getElementById("RFModal");
    var FMModal = document.getElementById("predictionModal");

    // Listen triggers for Location (map) and Sector images
    var locationMarker = document.getElementById("locationDemo");
    var sectorInfo = document.getElementById("sectorInfo");
    
    // Listen triggers for Neural Network and Random Forrest result
    var NR = document.getElementById("section2b1");
    var RF = document.getElementById("section2b2");

    // Listen for Find More
    var FM = document.getElementById("find-more");

    var closeButtons = document.getElementsByClassName("close");

    // Open modal for location
    locationMarker.onclick = function () {
        locationModal.style.display = "block";
    };

    // Open modal for sector
    sectorInfo.onclick = function () {
        sectorModal.style.display = "block";
    };

    // Open modal for NR
    NR.onclick = function () {
        NRModal.style.display = "block";
    };

    // Open modal for RF
    RF.onclick = function () {
        RFModal.style.display = "block";
    };

    // Open modal for FM
    FM.onclick = function () {
        FMModal.style.display = "block";
    };

    // Close modals
    for (var i = 0; i < closeButtons.length; i++) {
        closeButtons[i].onclick = function () {
            locationModal.style.display = "none";
            sectorModal.style.display = "none";
            NRModal.style.display = "none";
            RFModal.style.display = "none";
            FMModal.style.display = "none";
        };
    }

    // Close modal on clicking outside of it
    window.onclick = function (event) {
        if (event.target == locationModal) {
            locationModal.style.display = "none";
        } else if (event.target == sectorModal) {
            sectorModal.style.display = "none";
        } else if (event.target == NRModal) {
            NRModal.style.display = "none";
        } else if (event.target == RFModal) {
            RFModal.style.display = "none";
        } else if (event.target == FMModal) {
            FMModal.style.display = "none";
        }
    }
});