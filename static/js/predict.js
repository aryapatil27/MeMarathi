async function fetchPrediction(text) {
  const response = await fetch("/predict", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ text: text })
  });

  if (!response.ok) {
    throw new Error("Server response was not OK");
  }

  return await response.json();
}