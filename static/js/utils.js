const categoryConfig = {
  "Sports": { emoji: "🏏", color: "#16A34A" },
  "Politics": { emoji: "🏛️", color: "#2563EB" },
  "Crime": { emoji: "🚨", color: "#EF4444" },
  "Tech": { emoji: "💻", color: "#9333EA" },
  "Education": { emoji: "🎓", color: "#0284C7" },
  "Health": { emoji: "🏥", color: "#059669" },
  "Travel": { emoji: "✈️", color: "#0891B2" },
  "Auto": { emoji: "🚗", color: "#EA580C" },
  "Fashion": { emoji: "👗", color: "#DB2777" },
  "Bhakti": { emoji: "🛕", color: "#D97706" },
  "International": { emoji: "🌐", color: "#4F46E5" },
  "Entertainment": { emoji: "🎬", color: "#7C3AED" }
};

function getCategoryConfig(categoryName) {
  return categoryConfig[categoryName] || { emoji: "📰", color: "#2563EB" };
}

function updateCharCount(text, element) {
  const count = text ? text.length : 0;
  element.textContent = `${count.toLocaleString()} character${count === 1 ? '' : 's'}`;
}