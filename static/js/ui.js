function showLoader() {
  document.getElementById("resultCard").classList.add("hidden");
  document.getElementById("loaderCard").classList.remove("hidden");
  document.getElementById("loaderCard").classList.add("fade-in");
}

function hideLoader() {
  document.getElementById("loaderCard").classList.add("hidden");
}

function renderPredictionResult(category) {
  hideLoader();
  const resultCard = document.getElementById("resultCard");
  const categoryEl = document.getElementById("category");
  const categoryEmojiEl = document.getElementById("categoryEmoji");

  const config = getCategoryConfig(category);

  categoryEl.textContent = category;
  categoryEl.style.color = config.color;
  categoryEmojiEl.textContent = config.emoji;
  resultCard.style.borderLeftColor = config.color;

  resultCard.classList.remove("hidden");
  resultCard.classList.add("fade-in");
}

function renderError(message) {
  hideLoader();
  const resultCard = document.getElementById("resultCard");
  const categoryEl = document.getElementById("category");
  const categoryEmojiEl = document.getElementById("categoryEmoji");

  categoryEl.textContent = message || "Error Predict News";
  categoryEl.style.color = "var(--danger)";
  categoryEmojiEl.textContent = "⚠️";
  resultCard.style.borderLeftColor = "var(--danger)";

  resultCard.classList.remove("hidden");
  resultCard.classList.add("fade-in");
}

function clearInputs() {
  const textarea = document.getElementById("news");
  const charCounter = document.getElementById("charCounter");
  const resultCard = document.getElementById("resultCard");

  textarea.value = "";
  updateCharCount("", charCounter);
  resultCard.classList.add("hidden");
  textarea.focus();
}