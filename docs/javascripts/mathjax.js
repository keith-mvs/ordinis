// MathJax Configuration for Ordinis Documentation
window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};

// Document ready handler
document$.subscribe(function() {
  // Initialize any custom functionality
  console.log("Ordinis Documentation loaded");

  // Add version badge if metadata available
  var versionElement = document.querySelector('.version-badge');
  if (versionElement) {
    versionElement.textContent = 'v0.2.0-dev';
  }
});
