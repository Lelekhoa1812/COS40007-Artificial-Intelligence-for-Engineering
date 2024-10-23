// Function to adjust the needle angle based on performance value
function adjustNeedle(performanceValue) {
    // Ensure the performance value is between 0 and 100
    performanceValue = Math.max(0, Math.min(100, performanceValue));

    // Calculate the angle based on the performance value
    // -90 degrees is the starting position (leftmost)
    // 90 degrees is the ending position (rightmost)
    const angle = -90 + (performanceValue * 1.8);

    // Get the needle element
    const needle = document.getElementById('needle');

    // Apply the rotation transform
    needle.setAttribute('transform', `rotate(${angle}, 100, 90)`);
}

