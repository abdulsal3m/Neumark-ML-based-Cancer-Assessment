document.addEventListener("DOMContentLoaded", function() {
    const accordionItems = document.querySelectorAll(".accordion-item");

    accordionItems.forEach(item => {
        const trigger = item.querySelector(".accordion-trigger");
        const content = item.querySelector(".accordion-content");
        const arrow = trigger.querySelector(".accordion-arrow");

        if (!trigger || !content || !arrow) return;

        trigger.addEventListener("click", () => {
            const isActive = item.classList.contains("active");

            // Close all other items
            accordionItems.forEach(otherItem => {
                if (otherItem !== item && otherItem.classList.contains("active")) {
                    otherItem.classList.remove("active");
                    otherItem.querySelector(".accordion-content").style.maxHeight = null;
                    otherItem.querySelector(".accordion-arrow").textContent = "\u25BC"; // Down arrow
                }
            });

            // Toggle current item
            if (isActive) {
                item.classList.remove("active");
                content.style.maxHeight = null;
                arrow.textContent = "\u25BC"; // Down arrow
            } else {
                item.classList.add("active");
                content.style.maxHeight = content.scrollHeight + "px";
                arrow.textContent = "\u25B2"; // Up arrow
            }
        });
    });
});

