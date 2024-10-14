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
});
