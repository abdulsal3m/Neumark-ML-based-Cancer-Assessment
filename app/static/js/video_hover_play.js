document.addEventListener("DOMContentLoaded", function() {
    const video = document.getElementById("homePageVideo");

    if (video) {
        video.addEventListener("mouseover", function() {
            // Play the video if it's not already playing due to user interaction
            if (video.paused) {
                video.play().catch(error => {
                    // Autoplay was prevented, usually by browser policy if not muted or no user interaction yet.
                    // Since it's muted and hover is a user interaction, this should generally work.
                    console.warn("Video play on hover prevented: ", error);
                });
            }
        });

        video.addEventListener("mouseout", function() {
            // Pause the video if it's playing and the user hasn't manually paused it via controls
            // This check helps avoid pausing if the user clicked play and then moused out while it was playing.
            // However, for strict hover-to-play, we might want to always pause on mouseout.
            // For now, let's pause it if it's playing.
            if (!video.paused) {
                video.pause();
            }
        });
    }
});

