let userLat;
let userLng;
let wasteChart;

function goHome() {
  window.location.href = "index.html";
}

function toggleMenu() {
  document.getElementById("menu")?.classList.toggle("open");
  document.getElementById("menuOverlay")?.classList.toggle("open");
}

function fetchUserLocation(cb) {
  navigator.geolocation.getCurrentPosition(
    pos => {
      userLat = pos.coords.latitude;
      userLng = pos.coords.longitude;

      localStorage.setItem("userLat", String(userLat));
      localStorage.setItem("userLng", String(userLng));

      cb?.({ lat: userLat, lng: userLng });
    },
    () => {
      alert("Location access is required.");
    }
  );
}

async function startAnalyze() {
  const fileInput = document.getElementById("azFile");
  const analyzeBtn = document.querySelector(".btn.btn-dark");

  if (!fileInput?.files.length) {
    alert("Please upload an image.");
    return;
  }

  analyzeBtn.disabled = true;
  analyzeBtn.textContent = "Analyzing...";

  fetchUserLocation(async ({ lat, lng }) => {
    try {
      const formData = new FormData();
      formData.append("image", fileInput.files[0]);
      formData.append("lat", String(lat));
      formData.append("lng", String(lng));

      const res = await fetch("/api/predict", {
        method: "POST",
        body: formData
      });

      if (!res.ok) {
        throw new Error(await res.text());
      }

      const prediction = await res.json();
      localStorage.setItem("predictionId", String(prediction.id));
      window.location.href = `result.html?id=${prediction.id}`;
    } catch (err) {
      alert(`Analyze failed: ${err.message || err}`);
      analyzeBtn.disabled = false;
      analyzeBtn.textContent = "Analyze Image";
    }
  });
}

async function loadResults() {
  const params = new URLSearchParams(window.location.search);
  const predictionId = params.get("id") || localStorage.getItem("predictionId");

  if (!predictionId) {
    alert("No prediction found. Please analyze an image first.");
    window.location.href = "index.html";
    return;
  }

  try {
    const res = await fetch(`/api/predictions/${predictionId}`);
    if (!res.ok) {
      throw new Error(await res.text());
    }

    const data = await res.json();

    document.getElementById("analysisText").innerHTML =
      `<p><strong>Detected Waste Types:</strong> ${data.items.map(d => d.label).join(", ")}</p>
       <p><strong>Top Prediction:</strong> ${data.top_label} (${data.top_pct}%)</p>`;

    if (wasteChart) {
      wasteChart.destroy();
    }

    wasteChart = new Chart(document.getElementById("wasteChart"), {
      type: "doughnut",
      data: {
        labels: data.items.map(d => d.label),
        datasets: [{
          data: data.items.map(d => d.pct),
          backgroundColor: ["#94a3b8", "#78909c", "#bcaaa4", "#80cbc4", "#ef9a9a"],
          borderWidth: 0
        }]
      },
      options: {
        plugins: {
          legend: { position: "bottom" }
        }
      }
    });

    const map = L.map("map").setView([data.user_lat, data.user_lng], 13);

    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      attribution: "© OpenStreetMap"
    }).addTo(map);

    data.pins.forEach(p => {
      const popupText = `${p.icon} ${p.name}`;
      const marker = L.marker([p.lat, p.lng]).addTo(map).bindPopup(popupText);
      if (p.pin_type === "user") {
        marker.openPopup();
      }
    });

    setTimeout(() => map.invalidateSize(), 200);
  } catch (err) {
    alert(`Failed to load results: ${err.message || err}`);
  }
}

function initSellPage() {
  fetchUserLocation(({ lat, lng }) => {
    document.getElementById("sellLocation").value = `${lat.toFixed(6)}, ${lng.toFixed(6)}`;
    document.getElementById("sellLat").value = String(lat);
    document.getElementById("sellLng").value = String(lng);
  });
}

async function submitSellListing() {
  const material = document.getElementById("sellMaterial").value;
  const weightKg = Number(document.getElementById("sellWeight").value);
  const price = Number(document.getElementById("sellPrice").value);
  const contact = document.getElementById("sellContact").value.trim();
  const lat = Number(document.getElementById("sellLat").value);
  const lng = Number(document.getElementById("sellLng").value);

  if (!material || !weightKg || !price || !contact || Number.isNaN(lat) || Number.isNaN(lng)) {
    alert("Please complete all fields with valid values.");
    return;
  }

  try {
    const res = await fetch("/api/listings", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        material,
        weight_kg: weightKg,
        price,
        contact,
        lat,
        lng
      })
    });

    if (!res.ok) {
      throw new Error(await res.text());
    }

    alert("Listing created successfully.");
    window.location.href = "buy.html";
  } catch (err) {
    alert(`Failed to create listing: ${err.message || err}`);
  }
}

async function loadMarketplace() {
  const container = document.getElementById("listingsGrid");
  if (!container) return;

  try {
    const res = await fetch("/api/listings");
    if (!res.ok) {
      throw new Error(await res.text());
    }

    const listings = await res.json();
    if (!listings.length) {
      container.innerHTML = "<p>No open listings yet.</p>";
      return;
    }

    container.innerHTML = listings
      .map(
        l => `
      <div class="listing">
        <div class="listing-body">
          <strong>${l.material}</strong>
          <p>${l.weight_kg}kg • ₹${l.price}</p>
          <p>Contact: ${l.contact}</p>
          <button class="listing-btn" onclick="buyListing(${l.id})">Buy Now</button>
        </div>
      </div>
    `
      )
      .join("");
  } catch (err) {
    container.innerHTML = `<p>Failed to load listings: ${err.message || err}</p>`;
  }
}

async function buyListing(listingId) {
  const buyerContact = prompt("Enter your contact number to confirm purchase:");
  if (!buyerContact || !buyerContact.trim()) return;

  try {
    const res = await fetch(`/api/listings/${listingId}/buy`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ buyer_contact: buyerContact.trim() })
    });

    if (!res.ok) {
      throw new Error(await res.text());
    }

    alert("Purchase successful.");
    loadMarketplace();
  } catch (err) {
    alert(`Purchase failed: ${err.message || err}`);
  }
}
