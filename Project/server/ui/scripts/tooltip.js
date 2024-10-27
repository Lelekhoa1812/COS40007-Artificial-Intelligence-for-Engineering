document.addEventListener('DOMContentLoaded', function () {
    var siteMarkers = document.querySelectorAll(".site-marker");
    var formLabels = document.querySelectorAll("label[data-title]");
    var formLists = document.querySelectorAll("li[data-title]");
    var predictedPerformance = document.getElementById("predictedPerformance");

    // Function to create tooltip with text children
    function createTooltip(content, event) {
        var tooltip = document.createElement("div");
        tooltip.className = "tooltip";
        // Create text content (list or label description)
        var textContent = document.createElement("p");
        textContent.innerText = content;
        tooltip.appendChild(textContent);
        // Append tooltip to body and position it
        document.body.appendChild(tooltip);
        tooltip.style.top = event.pageY - 30 + "px";  // Adjust as needed
        tooltip.style.left = event.pageX + "px";

        return tooltip;
    }

    // Tooltip for form labels
    formLabels.forEach(function(label) {
        label.addEventListener("mouseover", function(event) {
            var tooltip = createTooltip(label.getAttribute("data-title"), event);
        });
        label.addEventListener("mouseout", function() {
            var tooltip = document.querySelector(".tooltip");
            if (tooltip) {
                tooltip.remove();
            }
        });
    });

    // Tooltip for form lists
    formLists.forEach(function(li) {
        li.addEventListener("mouseover", function(event) {
            var tooltip = createTooltip(li.getAttribute("data-title"), event);
        });
        li.addEventListener("mouseout", function() {
            var tooltip = document.querySelector(".tooltip");
            if (tooltip) {
                tooltip.remove();
            }
        });
    });

    // Tooltip for predicted performance
    predictedPerformance.addEventListener("mouseover", function(event) {
        var tooltip = createTooltip(predictedPerformance.getAttribute("data-title"), event);
    });
    predictedPerformance.addEventListener("mouseout", function() {
        var tooltip = document.querySelector(".tooltip");
        if (tooltip) {
            tooltip.remove();
        }
    });

    // Tooltips for site markers
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
            tooltip.style.top = event.pageY-140 + "px";
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