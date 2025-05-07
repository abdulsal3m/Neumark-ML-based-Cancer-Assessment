document.addEventListener("DOMContentLoaded", function() {
    const counterElement = document.getElementById("live-counter");
    if (!counterElement) {
        console.error("Counter element not found");
        return;
    }

    // Set the start date: January 1, 2025, 00:00:00 UTC
    const startDate = new Date(Date.UTC(2025, 0, 1, 0, 0, 0));
    const incrementIntervalSeconds = 17;

    function updateCounter() {
        const now = new Date();
        const diffSeconds = Math.floor((now - startDate) / 1000);

        // Calculate the current count
        const currentCount = Math.max(0, Math.floor(diffSeconds / incrementIntervalSeconds));

        // Format the number with commas
        counterElement.textContent = currentCount.toLocaleString();
    }

    // Initial update
    updateCounter();

    // Update the counter every 17 seconds
    setInterval(updateCounter, incrementIntervalSeconds * 1000);
});

