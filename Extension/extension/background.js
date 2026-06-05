/**
 * background.js — Cyberbully Shield Service Worker
 * Manages badge count and settings persistence.
 */

const API_URL = 'http://localhost:5000/api';

// ── Set default settings on install ──────────────────────────────────
chrome.runtime.onInstalled.addListener(() => {
  chrome.storage.local.set({
    cbs_enabled: true,
    cbs_threshold: 0.5
  });
  // Set initial badge
  chrome.action.setBadgeBackgroundColor({ color: '#6366f1' });
  chrome.action.setBadgeText({ text: '' });
});

// ── Listen for messages from content script ──────────────────────────
chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.type === 'UPDATE_COUNT') {
    const count = msg.count;
    const text = count > 0 ? String(count) : '';
    const color = count > 0 ? '#ef4444' : '#6366f1';

    if (sender.tab?.id) {
      chrome.action.setBadgeText({ text, tabId: sender.tab.id });
      chrome.action.setBadgeBackgroundColor({ color, tabId: sender.tab.id });
    }
  }
});

// ── Periodic health check to the backend ─────────────────────────────
async function checkBackendHealth() {
  try {
    const res = await fetch(`${API_URL}/health`, { method: 'GET' });
    const data = await res.json();
    return data.status === 'online';
  } catch {
    return false;
  }
}

// Check health every 30 seconds and store result
setInterval(async () => {
  const online = await checkBackendHealth();
  chrome.storage.local.set({ cbs_backend_online: online });
}, 30000);

// Initial check
checkBackendHealth().then(online => {
  chrome.storage.local.set({ cbs_backend_online: online });
});
