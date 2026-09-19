// The overlay half of overlay.html. Loading this file is what makes the page the
// taller of its two screens; a controller that aborts the request never sees it.
document.getElementById("overlay").style.display = "block";
document.getElementById("readout").textContent = "overlay: shown";
