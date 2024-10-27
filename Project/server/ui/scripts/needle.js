// Function to adjust the needle angle based on performance value
function adjustNeedle(performanceValue) {
    performanceValue = performanceValue*100
    console.log(performanceValue)

    // Calculate the angle based on the performance value
    // -90 degrees is the starting position (leftmost)
    // 90 degrees is the ending position (rightmost)
    const angle = -90 + (performanceValue * 1.8);

    // Get the needle element
    const needle = document.getElementById('needle');

    // Apply the rotation transform
    needle.setAttribute('transform', `rotate(${angle}, 100, 90)`);
}