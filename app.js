let userLat, userLng;

/* ───────── Navigation ───────── */
function goHome() {
  window.location.href = "index.html";
}

/* ───────── Hamburger Menu ───────── */
function toggleMenu() {
  document.getElementById("menu")?.classList.toggle("open");
  document.getElementById("menuOverlay")?.classList.toggle("open");
}

/* ───────── Location Handling ───────── */
function fetchUserLocation(cb) {
  navigator.geolocation.getCurrentPosition(
    pos => {
      userLat = pos.coords.latitude;
      userLng = pos.coords.longitude;

      localStorage.setItem("userLat", userLat);
      localStorage.setItem("userLng", userLng);

      cb?.();
    },
    () => {
      alert("Location access is required for routing.");
    }
  );
}

/* ───────── Analyze Flow (Index Page) ───────── */
function startAnalyze() {
  const file = document.getElementById("azFile");
  if (!file.files.length) {
    alert("Please upload an image.");
    return;
  }

  fetchUserLocation(() => {
    // Mock ML output
    const detected = [
      { type: "Plastic", pct: 50 },
      { type: "Metal", pct: 30 },
      { type: "Paper", pct: 20 }
    ];

    localStorage.setItem("analysisData", JSON.stringify(detected));
    window.location.href = "result.html";
  });
}

/* ───────── Facility Routing Data ───────── */
const facilities = {
  Metal: [
    { name: "Metal Factory", icon: "🏭", dx: 0.01, dy: 0.01 },
    { name: "Scrap Yard", icon: "🔩", dx: -0.01, dy: 0.008 },
    { name: "Recycling Plant", icon: "♻️", dx: 0.008, dy: -0.01 }
  ],
  Plastic: [
    { name: "Plastic Recycling Center", icon: "♻️", dx: 0.01, dy: 0.005 }
  ],
  Paper: [
    { name: "Paper Recycling Mill", icon: "📄", dx: -0.008, dy: 0.01 }
  ],
  "E-Waste": [
    { name: "Authorized E-Waste Center", icon: "💻", dx: 0.01, dy: -0.01 }
  ],
  Medical: [
    { name: "Medical Waste Facility", icon: "🏥", dx: -0.01, dy: -0.008 }
  ],
  Glass: [
    { name: "Glass Recycling Plant", icon: "🍾", dx: 0.007, dy: -0.009 }
  ],
  Organic: [
    { name: "Composting Unit", icon: "🌱", dx: -0.006, dy: 0.007 }
  ]
};

/* ───────── Result Page Logic ───────── */
function loadResults() {
  const storedData = localStorage.getItem("analysisData");
  const lat = parseFloat(localStorage.getItem("userLat"));
  const lng = parseFloat(localStorage.getItem("userLng"));

  if (!storedData || isNaN(lat) || isNaN(lng)) {
    alert("Missing analysis data or location.");
    window.location.href = "index.html";
    return;
  }

  userLat = lat;
  userLng = lng;

  const detected = JSON.parse(storedData);

  /* ---- TEXT OUTPUT ---- */
  document.getElementById("analysisText").innerHTML =
    `<p><strong>Detected Waste Types:</strong> ${detected.map(d => d.type).join(", ")}</p>
     <p><strong>Confidence Level:</strong> High (${Math.max(...detected.map(d => d.pct))}%)</p>`;

  /* ---- PIE CHART ---- */
  new Chart(document.getElementById("wasteChart"), {
    type: "doughnut",
    data: {
      labels: detected.map(d => d.type),
      datasets: [{
        data: detected.map(d => d.pct),
        backgroundColor: [
          "#94a3b8", "#78909c", "#bcaaa4",
          "#80cbc4", "#ef9a9a", "#a5d6a7", "#b39ddb"
        ],
        borderWidth: 0
      }]
    },
    options: {
      plugins: {
        legend: {
          position: "bottom"
        }
      }
    }
  });

  /* ---- MAP ---- */
  const map = L.map("map").setView([userLat, userLng], 13);

  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    attribution: "© OpenStreetMap"
  }).addTo(map);

  // User location
  L.marker([userLat, userLng])
    .addTo(map)
    .bindPopup("📍 Your Location")
    .openPopup();

  // Facility pins
  detected.forEach(d => {
    (facilities[d.type] || []).forEach(f => {
      L.marker([userLat + f.dx, userLng + f.dy])
        .addTo(map)
        .bindPopup(`${f.icon} ${f.name}`);
    });
  });

  setTimeout(() => map.invalidateSize(), 200);
}
