// Function to add pins to the location map
function addPinToMap(pin_id) {
    const locationMap = document.getElementById('locationOutput');
    const mapContainer = locationMap.parentElement;

    // Ensure the container has position relative for absolute positioning of pins
    mapContainer.style.position = 'relative';

    // Pin data
    const pins = [
        { site: "Site 1", image: "q1.jpg", top: 459, left: 217 },
        { site: "Site 2", image: "q1.jpg", top: 469, left: 210 },
        { site: "Site 3", image: "bt.jpeg", top: 464, left: 229 },
        { site: "Site 4", image: "q1.jpg", top: 456, left: 210 },
        { site: "Site 5", image: "bh.jpeg", top: 92, left: 406 },
        { site: "Site 6", image: "q12.jpg", top: 357, left: 71 },
        { site: "Site 7", image: "q1.jpg", top: 454, left: 221 },
        { site: "Site 8", image: "tb.jpg", top: 456, left: 181 },
        { site: "Site 9", image: "bd.jpg", top: 53, left: 152 },
        { site: "Site 10", image: "bd.jpg", top: 72, left: 157 }
    ];

    // Check if the pin_id is valid
    if (pin_id < 0 || pin_id >= pins.length) {
        console.error('Invalid pin_id');
        return;
    }

    const pin = pins[pin_id];

    const pinElement = document.createElement('div');
    pinElement.className = 'site-marker';
    pinElement.dataset.site = pin.site;
    pinElement.dataset.image = pin.image;
    pinElement.style.position = 'absolute';
    pinElement.style.top = `${pin.top}px`;
    pinElement.style.left = `${pin.left}px`;
    pinElement.style.width = '10px';
    pinElement.style.height = '10px';
    pinElement.style.borderRadius = '50%';
    pinElement.style.backgroundColor = 'red';
    pinElement.style.cursor = 'pointer';

    // Add click event listener to the pin
    pinElement.addEventListener('click', () => {
        console.log(`Clicked on ${pin.site}`);
        // You can add more functionality here, like showing the image
    });

    mapContainer.appendChild(pinElement);
}

// Example usage:
document.addEventListener('DOMContentLoaded', () => addPinToMap(0)); // Adds Site 1
// document.addEventListener('DOMContentLoaded', () => addPinToMap(5)); // Adds Site 6