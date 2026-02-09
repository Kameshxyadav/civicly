let userLat;
let userLng;
let wasteChart;
let currentUser = null;

function goHome() {
  window.location.href = "index.html";
}

function toggleMenu() {
  document.getElementById("menu")?.classList.toggle("open");
  document.getElementById("menuOverlay")?.classList.toggle("open");
}

function getStoredLocation() {
  const lat = Number(localStorage.getItem("userLat"));
  const lng = Number(localStorage.getItem("userLng"));
  if (!Number.isFinite(lat) || !Number.isFinite(lng)) return null;
  return { lat, lng };
}

function locationErrorMessage(err) {
  if (!err) return "Location access is required.";
  switch (err.code) {
    case err.PERMISSION_DENIED:
      return "Location permission denied. Enable it and try again.";
    case err.POSITION_UNAVAILABLE:
      return "Location unavailable. Check your device settings and try again.";
    case err.TIMEOUT:
      return "Location request timed out. Please try again.";
    default:
      return "Location access is required.";
  }
}

function fetchUserLocation(cb, onError) {
  const cached = getStoredLocation();

  if (!navigator.geolocation) {
    if (cached) {
      cb?.(cached);
      return;
    }
    const msg = "Geolocation is not supported by this browser.";
    onError?.(msg);
    alert(msg);
    return;
  }

  navigator.geolocation.getCurrentPosition(
    pos => {
      userLat = pos.coords.latitude;
      userLng = pos.coords.longitude;

      localStorage.setItem("userLat", String(userLat));
      localStorage.setItem("userLng", String(userLng));

      cb?.({ lat: userLat, lng: userLng });
    },
    err => {
      if (cached) {
        alert("Using last known location. Enable location for better accuracy.");
        cb?.(cached);
        return;
      }
      const msg = locationErrorMessage(err);
      onError?.(msg);
      alert(msg);
    },
    {
      enableHighAccuracy: true,
      timeout: 10000,
      maximumAge: 60000
    }
  );
}

// ───────────────── Auth ─────────────────

async function checkAuth() {
  try {
    const res = await fetch("/api/me");
    if (!res.ok) return null;
    const data = await res.json();
    currentUser = data;
    return data;
  } catch {
    return null;
  }
}

async function updateTopbar() {
  await checkAuth();
  const topbarRight = document.getElementById("topbarRight");
  if (!topbarRight) return;

  if (currentUser) {
    topbarRight.innerHTML = `
      <span class="topbar-user">${currentUser.username}</span>
      <button class="notif-bell" onclick="toggleNotifications()">
        &#128276;
        ${currentUser.unread > 0 ? `<span class="notif-badge">${currentUser.unread}</span>` : ""}
      </button>
      <div class="notif-panel" id="notifPanel"></div>
      <button class="btn-link" onclick="logoutUser()">Logout</button>
    `;
  } else {
    topbarRight.innerHTML = `<a class="btn-link" href="login.html">Login</a>`;
  }

  // Update hamburger menu auth links
  const menuAuthLinks = document.getElementById("menuAuthLinks");
  if (menuAuthLinks) {
    if (currentUser) {
      menuAuthLinks.innerHTML = `<a href="#" onclick="logoutUser(); return false;">Logout (${currentUser.username})</a>`;
    } else {
      menuAuthLinks.innerHTML = `<a href="login.html">Login</a>`;
    }
  }
}

function getRedirectUrl() {
  const params = new URLSearchParams(window.location.search);
  return params.get("redirect") || "index.html";
}

async function loginUser() {
  const username = document.getElementById("loginUsername").value.trim();
  const password = document.getElementById("loginPassword").value;
  if (!username || !password) {
    alert("Please enter username and password.");
    return;
  }
  try {
    const res = await fetch("/api/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password })
    });
    if (!res.ok) {
      throw new Error(await res.text());
    }
    window.location.href = getRedirectUrl();
  } catch (err) {
    alert(`Login failed: ${err.message || err}`);
  }
}

async function registerUser() {
  const username = document.getElementById("regUsername").value.trim();
  const password = document.getElementById("regPassword").value;
  if (!username || !password) {
    alert("Please enter username and password.");
    return;
  }
  try {
    const res = await fetch("/api/register", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password })
    });
    if (!res.ok) {
      throw new Error(await res.text());
    }
    window.location.href = getRedirectUrl();
  } catch (err) {
    alert(`Registration failed: ${err.message || err}`);
  }
}

async function logoutUser() {
  await fetch("/api/logout", { method: "POST" });
  currentUser = null;
  window.location.href = "index.html";
}

async function checkLoginOrRedirect() {
  const user = await checkAuth();
  if (!user) {
    window.location.href = "login.html?redirect=" + encodeURIComponent(window.location.pathname);
  }
}

// ───────────────── Notifications ─────────────────

async function toggleNotifications() {
  const panel = document.getElementById("notifPanel");
  if (!panel) return;

  if (panel.classList.contains("open")) {
    panel.classList.remove("open");
    return;
  }

  try {
    const res = await fetch("/api/notifications");
    if (!res.ok) return;
    const notifs = await res.json();

    if (!notifs || !notifs.length) {
      panel.innerHTML = `<div class="notif-item">No notifications</div>`;
    } else {
      panel.innerHTML = notifs.map(n =>
        `<div class="notif-item ${n.is_read ? "" : "notif-unread"}">${n.message}</div>`
      ).join("");
    }
    panel.classList.add("open");

    // Mark all as read
    await fetch("/api/notifications/read", { method: "POST" });
    const badge = document.querySelector(".notif-badge");
    if (badge) badge.remove();
  } catch {
    // ignore
  }
}

// ───────────────── Analyze ─────────────────

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
  }, () => {
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = "Analyze Image";
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
      attribution: "\u00a9 OpenStreetMap"
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

async function reverseGeocode(lat, lng) {
  try {
    const res = await fetch(
      `https://nominatim.openstreetmap.org/reverse?lat=${lat}&lon=${lng}&format=json&zoom=10`,
      { headers: { "Accept-Language": "en" } }
    );
    if (!res.ok) return null;
    const data = await res.json();
    const a = data.address || {};
    const city = a.city || a.town || a.village || a.county || "";
    const state = a.state || "";
    const country = a.country || "";
    return [city, state, country].filter(Boolean).join(", ");
  } catch {
    return null;
  }
}

function initSellPage() {
  fetchUserLocation(async ({ lat, lng }) => {
    document.getElementById("sellLat").value = String(lat);
    document.getElementById("sellLng").value = String(lng);
    const locInput = document.getElementById("sellLocation");
    locInput.value = "Detecting location...";
    const place = await reverseGeocode(lat, lng);
    locInput.value = place || `${lat.toFixed(4)}, ${lng.toFixed(4)}`;
  }, msg => {
    const input = document.getElementById("sellLocation");
    if (input) input.value = msg;
  });
}

async function submitSellListing() {
  const material = document.getElementById("sellMaterial").value;
  const weightKg = Number(document.getElementById("sellWeight").value);
  const price = Number(document.getElementById("sellPrice").value);
  const contact = document.getElementById("sellContact").value.trim();
  const lat = Number(document.getElementById("sellLat").value);
  const lng = Number(document.getElementById("sellLng").value);
  const imageInput = document.getElementById("sellImage");

  if (!material || !weightKg || !price || !contact || Number.isNaN(lat) || Number.isNaN(lng)) {
    alert("Please complete all fields with valid values.");
    return;
  }

  const formData = new FormData();
  formData.append("material", material);
  formData.append("weight_kg", String(weightKg));
  formData.append("price", String(price));
  formData.append("contact", contact);
  formData.append("lat", String(lat));
  formData.append("lng", String(lng));
  if (imageInput?.files.length) {
    formData.append("image", imageInput.files[0]);
  }

  try {
    const res = await fetch("/api/listings", {
      method: "POST",
      body: formData
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

// ───────────────── Marketplace ─────────────────

async function loadMarketplace() {
  const container = document.getElementById("listingsGrid");
  if (!container) return;

  await checkAuth();

  try {
    const res = await fetch("/api/listings");
    if (!res.ok) {
      throw new Error(await res.text());
    }

    const listings = await res.json();
    if (!listings || !listings.length) {
      container.innerHTML = "<p>No open listings yet.</p>";
      return;
    }

    // Reverse-geocode all listings in parallel
    const locations = await Promise.all(
      listings.map(l => reverseGeocode(l.lat, l.lng))
    );

    container.innerHTML = listings
      .map((l, i) => {
        const isOwner = currentUser && currentUser.id === l.user_id;
        const actionBtn = isOwner
          ? `<button class="listing-btn" onclick="markSold(${l.id})">Mark as Sold</button>`
          : `<button class="listing-btn" onclick="buyListing(${l.id})">Buy Now</button>`;
        const locText = locations[i] || `${l.lat.toFixed(4)}, ${l.lng.toFixed(4)}`;
        return `
      <div class="listing">
        ${l.image_url ? `<img class="listing-img" src="${l.image_url}" alt="${l.material}">` : ""}
        <div class="listing-body">
          <strong>${l.material}</strong>
          <p>${l.weight_kg}kg &bull; &#8377;${l.price}</p>
          <p class="listing-location">${locText}</p>
          <p>Contact: ${l.contact}</p>
          ${actionBtn}
        </div>
      </div>
    `;
      })
      .join("");
  } catch (err) {
    container.innerHTML = `<p>Failed to load listings: ${err.message || err}</p>`;
  }
}

async function buyListing(listingId) {
  if (!currentUser) {
    alert("Please login to purchase.");
    window.location.href = "login.html?redirect=" + encodeURIComponent(window.location.pathname);
    return;
  }
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

async function markSold(listingId) {
  if (!confirm("Mark this listing as sold?")) return;
  try {
    const res = await fetch(`/api/listings/${listingId}/sold`, { method: "POST" });
    if (!res.ok) {
      throw new Error(await res.text());
    }
    alert("Listing marked as sold.");
    loadMarketplace();
  } catch (err) {
    alert(`Failed: ${err.message || err}`);
  }
}

// ───────────────── Admin ─────────────────

async function loadAdminPanel() {
  const user = await checkAuth();
  if (!user || user.role !== "admin") {
    alert("Admin access required.");
    window.location.href = "login.html";
    return;
  }
  await updateTopbar();

  // Load users
  try {
    const res = await fetch("/api/admin/users");
    if (!res.ok) throw new Error("Failed to load users");
    const users = await res.json();
    const tbody = document.getElementById("adminUsersBody");
    if (tbody && users) {
      tbody.innerHTML = users.map(u =>
        `<tr><td>${u.id}</td><td>${u.username}</td><td>${u.password}</td><td>${u.role}</td><td>${u.created_at}</td></tr>`
      ).join("");
    }
  } catch { /* ignore */ }

  // Load listings
  try {
    const res = await fetch("/api/admin/listings");
    if (!res.ok) throw new Error("Failed to load listings");
    const listings = await res.json();
    const tbody = document.getElementById("adminListingsBody");
    if (tbody && listings) {
      tbody.innerHTML = listings.map(l =>
        `<tr><td>${l.id}</td><td>${l.material}</td><td>${l.weight_kg}</td><td>${l.price}</td><td>${l.contact}</td><td>${l.status}</td><td>${l.user_id}</td><td>${l.created_at}</td></tr>`
      ).join("");
    }
  } catch { /* ignore */ }
}

// ───────────────── Heatmap ─────────────────

let heatmapInstance = null;
let heatLayer = null;
let allHeatmapData = [];

async function initHeatmap() {
  const mapEl = document.getElementById("heatmap");
  const hintEl = document.getElementById("heatmapHint");
  if (!mapEl) return;

  let points = [];
  try {
    const res = await fetch("/api/heatmap");
    if (!res.ok) throw new Error("Failed to load heatmap data");
    points = await res.json();
  } catch {
    if (hintEl) hintEl.textContent = "Could not load heatmap data.";
    return;
  }

  if (!points.length) {
    if (hintEl) hintEl.textContent = "No waste data yet. Analyze some images to populate the heatmap.";
    mapEl.style.display = "none";
    return;
  }

  allHeatmapData = points;

  const filterEl = document.getElementById("heatmapFilter");
  if (filterEl) {
    const labels = [...new Set(points.map(p => p.label))].sort();
    labels.forEach(label => {
      const opt = document.createElement("option");
      opt.value = label;
      opt.textContent = label;
      filterEl.appendChild(opt);
    });
    filterEl.addEventListener("change", () => updateHeatLayer(filterEl.value));
  }

  fetchUserLocation(
    ({ lat, lng }) => {
      if (hintEl) hintEl.style.display = "none";
      createHeatmap(lat, lng, points);
    },
    () => {
      const avgLat = points.reduce((s, p) => s + p.lat, 0) / points.length;
      const avgLng = points.reduce((s, p) => s + p.lng, 0) / points.length;
      if (hintEl) hintEl.textContent = "Showing all waste data (location unavailable).";
      createHeatmap(avgLat, avgLng, points);
    }
  );
}

function createHeatmap(centerLat, centerLng, points) {
  heatmapInstance = L.map("heatmap").setView([centerLat, centerLng], 12);

  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    attribution: "\u00a9 OpenStreetMap"
  }).addTo(heatmapInstance);

  const heatData = points.map(p => [p.lat, p.lng, p.count]);

  heatLayer = L.heatLayer(heatData, {
    radius: 25,
    blur: 15,
    maxZoom: 17,
    max: Math.max(...points.map(p => p.count), 1),
    gradient: {
      0.2: "#ffffb2",
      0.4: "#fecc5c",
      0.6: "#fd8d3c",
      0.8: "#f03b20",
      1.0: "#bd0026"
    }
  }).addTo(heatmapInstance);

  setTimeout(() => heatmapInstance.invalidateSize(), 200);
}

function updateHeatLayer(filterValue) {
  if (!heatmapInstance || !heatLayer) return;

  const filtered = filterValue === "all"
    ? allHeatmapData
    : allHeatmapData.filter(p => p.label === filterValue);

  const heatData = filtered.map(p => [p.lat, p.lng, p.count]);
  heatLayer.setLatLngs(heatData);
}

// ───────────────── Auto-init topbar ─────────────────

document.addEventListener("DOMContentLoaded", () => {
  updateTopbar();
});
