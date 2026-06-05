/**
 * popup.js — Cyberbully Shield Popup Logic
 */

const API_URL = 'http://localhost:5000/api';

const toggleEnabled = document.getElementById('toggleEnabled');
const thresholdSlider = document.getElementById('thresholdSlider');
const thresholdValue = document.getElementById('thresholdValue');
const statusDot = document.getElementById('statusDot');
const backendStatus = document.getElementById('backendStatus');
const pageDetections = document.getElementById('pageDetections');
const totalThreats = document.getElementById('totalThreats');
const totalScans = document.getElementById('totalScans');
const btnScan = document.getElementById('btnScan');
const btnDashboard = document.getElementById('btnDashboard');

// ── Load saved settings ──────────────────────────────────────────────
chrome.storage.local.get(['cbs_enabled', 'cbs_threshold', 'cbs_backend_online'], (result) => {
  toggleEnabled.checked = result.cbs_enabled !== false;
  if (result.cbs_threshold !== undefined) {
    thresholdSlider.value = Math.round(result.cbs_threshold * 100);
    thresholdValue.textContent = thresholdSlider.value + '%';
  }
  updateBackendUI(result.cbs_backend_online);
});

// ── Toggle protection ────────────────────────────────────────────────
toggleEnabled.addEventListener('change', () => {
  chrome.storage.local.set({ cbs_enabled: toggleEnabled.checked });
});

// ── Sensitivity slider ──────────────────────────────────────────────
thresholdSlider.addEventListener('input', () => {
  const val = thresholdSlider.value;
  thresholdValue.textContent = val + '%';
  chrome.storage.local.set({ cbs_threshold: val / 100 });
});

// ── Scan Now button ──────────────────────────────────────────────────
btnScan.addEventListener('click', () => {
  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    if (tabs[0]?.id) {
      chrome.tabs.sendMessage(tabs[0].id, { type: 'FORCE_SCAN' }, () => {
        btnScan.textContent = '✓ Scanning...';
        setTimeout(() => { btnScan.textContent = '⚡ Scan Now'; }, 2000);
      });
    }
  });
});

// ── Dashboard button ─────────────────────────────────────────────────
btnDashboard.addEventListener('click', () => {
  chrome.tabs.create({ url: 'http://localhost:5000/dashboard' });
});

// ── Get page stats from content script ───────────────────────────────
chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
  if (tabs[0]?.id) {
    chrome.tabs.sendMessage(tabs[0].id, { type: 'GET_PAGE_STATS' }, (response) => {
      if (response) {
        pageDetections.textContent = response.detections || 0;
      }
    });
  }
});

// ── Fetch global stats from API ──────────────────────────────────────
async function fetchGlobalStats() {
  try {
    const res = await fetch(`${API_URL}/stats`);
    const data = await res.json();
    totalThreats.textContent = data.threats_detected;
    totalScans.textContent = data.total_scans;
    updateBackendUI(true);
  } catch {
    updateBackendUI(false);
  }
}

function updateBackendUI(online) {
  if (online) {
    statusDot.className = 'status-indicator online';
    backendStatus.textContent = '🟢 Backend running on localhost:5000';
  } else {
    statusDot.className = 'status-indicator offline';
    backendStatus.textContent = '🔴 Backend offline — start the server';
  }
}

fetchGlobalStats();
