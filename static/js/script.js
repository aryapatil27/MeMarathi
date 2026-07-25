async function predictNews() {
    const news = document.getElementById("news").value;

    if (news.trim() === "") {
        alert("Please enter Marathi News.");
        return;
    }

    // Modern styled fallback while processing
    const categoryEl = document.getElementById("category");
    categoryEl.innerHTML = "Predicting...";
    categoryEl.style.color = "var(--text-muted)";

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                text: news
            })
        });

        const data = await response.json();
        
        // Return styling back to active state color
        categoryEl.style.color = "var(--success)";
        categoryEl.innerHTML = data.category;

    } catch (error) {
        categoryEl.style.color = "var(--danger)";
        categoryEl.innerHTML = "Unable to connect to server.";
    }
}

function clearText() {
    const categoryEl = document.getElementById("category");
    document.getElementById("news").value = "";
    categoryEl.style.color = "var(--text-muted)";
    categoryEl.innerHTML = "----------";
}