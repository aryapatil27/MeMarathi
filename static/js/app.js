document.addEventListener("DOMContentLoaded", () => {
  const textarea = document.getElementById("news");
  const charCounter = document.getElementById("charCounter");
  const predictBtn = document.getElementById("predictBtn");
  const clearBtn = document.getElementById("clearBtn");
  const sampleBtn = document.getElementById("sampleBtn");

  // Character counter trigger
  textarea.addEventListener("input", (e) => {
    updateCharCount(e.target.value, charCounter);
  });

  // Predict execution
  async function handlePredict() {
    const text = textarea.value.trim();
    if (!text) {
      alert("कृपया आधी बातमी लिहा किंवा Sample वापरून पहा.");
      return;
    }

    showLoader();

    try {
      const data = await fetchPrediction(text);
      renderPredictionResult(data.category);
    } catch (err) {
      renderError("कनेक्शनमध्ये अडचण आली.");
    }
  }

  // Button actions
  predictBtn.addEventListener("click", handlePredict);
  
  clearBtn.addEventListener("click", clearInputs);

  sampleBtn.addEventListener("click", () => {
    textarea.value = getRandomSampleNews();
    updateCharCount(textarea.value, charCounter);
  });

  // Ctrl + Enter Keyboard shortcut
  textarea.addEventListener("keydown", (e) => {
    if (e.ctrlKey && e.key === "Enter") {
      e.preventDefault();
      handlePredict();
    }
  });
});