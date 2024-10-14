document.addEventListener('DOMContentLoaded', function () {
    var siteMarkers = document.querySelectorAll(".site-marker");

    siteMarkers.forEach(function(marker) {
        marker.addEventListener("mouseover", function(event) {
            var tooltip = document.createElement("div");
            tooltip.className = "tooltip";

            // Create site ID label
            var siteLabel = document.createElement("p");
            siteLabel.innerText = marker.getAttribute("data-site");
            tooltip.appendChild(siteLabel);

            // Create image
            var siteImage = document.createElement("img");
            siteImage.src = "pics/" + marker.getAttribute("data-image");
            siteImage.style.width = "100px";  // Set image size
            siteImage.style.height = "100px";
            tooltip.appendChild(siteImage);

            // Append tooltip to body and position it
            document.body.appendChild(tooltip);
            tooltip.style.top = event.pageY + "px";
            tooltip.style.left = event.pageX + "px";
        });

        marker.addEventListener("mouseout", function() {
            var tooltip = document.querySelector(".tooltip");
            if (tooltip) {
                tooltip.remove();
            }
        });
    });
});
