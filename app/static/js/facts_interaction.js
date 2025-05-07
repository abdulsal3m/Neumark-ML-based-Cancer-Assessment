document.addEventListener("DOMContentLoaded", function() {
    const factItems = document.querySelectorAll(".fact-item-v4");

    function setActiveFact(selectedItem) {
        factItems.forEach(item => {
            if (item === selectedItem) {
                // Delay adding 'active' class slightly if needed for complex sequences,
                // but for now, CSS transitions with delays should handle the sequence.
                item.classList.add("active");
            } else {
                item.classList.remove("active");
            }
        });
    }

    // Set the first item as active by default
    if (factItems.length > 0) {
        setActiveFact(factItems[0]);
    }

    factItems.forEach(item => {
        item.addEventListener("mouseover", function() {
            setActiveFact(item);
        });
        
        // Make the entire item focusable for keyboard navigation
        item.setAttribute("tabindex", "0");
        item.addEventListener("focus", function() {
            setActiveFact(item);
        });
        // Allow activation with Enter or Space key for accessibility
        item.addEventListener("keydown", function(event) {
            if (event.key === "Enter" || event.key === " ") {
                event.preventDefault(); // Prevent default space scroll
                setActiveFact(item);
            }
        });
    });
});

