/**
 * content.js — Cyberbully Shield Content Script
 * Scans page text for cyberbullying and applies blur overlays on detected content.
 * Communicates with the local FastAPI backend at localhost:5000.
 */

(() => {
  'use strict';

  const API_URL = 'http://localhost:5000/api';
  const SCAN_INTERVAL_MS = 5000;       // Re-scan every 5 seconds for new content
  const MIN_TEXT_LENGTH = 8;            // Ignore very short text
  const PROCESSED_ATTR = 'data-cbs-scanned';

  let isEnabled = true;
  let threshold = 0.5;
  let pageDetections = 0;

  // ── Load settings from storage ─────────────────────────────────────
  chrome.storage.local.get(['cbs_enabled', 'cbs_threshold'], (result) => {
    if (result.cbs_enabled !== undefined) isEnabled = result.cbs_enabled;
    if (result.cbs_threshold !== undefined) threshold = result.cbs_threshold;
    if (isEnabled) startScanning();
  });

  chrome.storage.onChanged.addListener((changes) => {
    if (changes.cbs_enabled) {
      isEnabled = changes.cbs_enabled.newValue;
      if (isEnabled) startScanning();
    }
    if (changes.cbs_threshold) {
      threshold = changes.cbs_threshold.newValue;
    }
  });

  // ── Selectors for social media content ─────────────────────────────
  const CONTENT_SELECTORS = [
    // Twitter / X
    '[data-testid="tweetText"]',
    // YouTube
    '#content-text',
    'yt-formatted-string#content-text',
    // Reddit
    '[data-testid="comment"]',
    '.Comment .RichTextJSON-root',
    'shreddit-comment div[slot="comment"]',
    // Facebook
    '[data-ad-preview="message"]',
    'div[dir="auto"][style]',
    // Instagram
    'span._ap3a',
    // Generic - comment/post patterns
    '.comment-body',
    '.post-body',
    '.message-content',
    'article p',
    '.comment p',
  ];

  // ── Get text elements to scan ──────────────────────────────────────
  function getTextElements() {
    const elements = [];
    const selector = CONTENT_SELECTORS.join(', ');

    document.querySelectorAll(selector).forEach(el => {
      if (el.getAttribute(PROCESSED_ATTR)) return;
      const text = el.innerText?.trim();
      if (text && text.length >= MIN_TEXT_LENGTH) {
        elements.push(el);
      }
    });

    return elements;
  }

  // ── Send texts to backend for prediction ───────────────────────────
  async function predictBatch(texts) {
    try {
      const res = await fetch(`${API_URL}/predict_batch`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          texts: texts,
          source_url: window.location.href
        })
      });
      if (!res.ok) return null;
      const data = await res.json();
      return data.results;
    } catch (e) {
      // Backend offline — silently fail
      return null;
    }
  }

  // ── Apply blur overlay to a detected element ───────────────────────
  function censorElement(el, confidence) {
    // Wrap in a relative container if not already
    if (el.parentElement?.classList?.contains('cbs-wrapper')) return;

    const wrapper = document.createElement('span');
    wrapper.className = 'cbs-wrapper';
    wrapper.style.position = 'relative';
    wrapper.style.display = el.style.display === 'inline' ? 'inline' : 'block';

    el.parentNode.insertBefore(wrapper, el);
    wrapper.appendChild(el);

    // Apply blur
    el.classList.add('cbs-overlay');

    // Add warning badge
    const badge = document.createElement('span');
    badge.className = 'cbs-badge';
    badge.innerHTML = `<span class="cbs-badge-icon"></span> Blocked (${Math.round(confidence * 100)}%)`;
    badge.title = 'Click to reveal content';
    badge.addEventListener('click', (e) => {
      e.stopPropagation();
      el.classList.toggle('cbs-revealed');
      badge.innerHTML = el.classList.contains('cbs-revealed')
        ? '<span class="cbs-badge-icon"></span> Revealed'
        : `<span class="cbs-badge-icon"></span> Blocked (${Math.round(confidence * 100)}%)`;
    });
    wrapper.appendChild(badge);

    pageDetections++;
    updateBadgeCount();
  }

  // ── Update extension badge count ───────────────────────────────────
  function updateBadgeCount() {
    chrome.runtime.sendMessage({
      type: 'UPDATE_COUNT',
      count: pageDetections
    }).catch(() => {});
  }

  // ── Show toast notification ────────────────────────────────────────
  function showToast(count) {
    // Remove existing toast
    document.querySelectorAll('.cbs-toast').forEach(t => t.remove());

    const toast = document.createElement('div');
    toast.className = 'cbs-toast';
    toast.innerHTML = `
      <span class="icon">🛡️</span>
      <span><strong>${count}</strong> item${count > 1 ? 's' : ''} flagged as potential cyberbullying and censored.</span>
    `;
    document.body.appendChild(toast);

    setTimeout(() => {
      toast.style.opacity = '0';
      toast.style.transform = 'translateY(20px)';
      toast.style.transition = 'all 0.3s ease';
      setTimeout(() => toast.remove(), 300);
    }, 4000);
  }

  // ── Main scan cycle ────────────────────────────────────────────────
  async function scanPage() {
    if (!isEnabled) return;

    const elements = getTextElements();
    if (elements.length === 0) return;

    // Mark as processed immediately to avoid re-scanning
    const texts = elements.map(el => {
      el.setAttribute(PROCESSED_ATTR, '1');
      return el.innerText.trim();
    });

    const results = await predictBatch(texts);
    if (!results) return;

    let newDetections = 0;
    results.forEach((result, i) => {
      if (result.is_bullying && result.confidence >= threshold) {
        censorElement(elements[i], result.confidence);
        newDetections++;
      }
    });

    if (newDetections > 0) {
      showToast(newDetections);
    }
  }

  // ── Scanning loop ─────────────────────────────────────────────────
  let scanTimer = null;

  function startScanning() {
    // Initial scan after a short delay
    setTimeout(scanPage, 1500);
    // Periodic re-scan for dynamically loaded content
    if (scanTimer) clearInterval(scanTimer);
    scanTimer = setInterval(scanPage, SCAN_INTERVAL_MS);
  }

  // ── Listen for messages from popup/background ──────────────────────
  chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
    if (msg.type === 'GET_PAGE_STATS') {
      sendResponse({
        detections: pageDetections,
        url: window.location.href
      });
    }
    if (msg.type === 'FORCE_SCAN') {
      scanPage();
      sendResponse({ status: 'scanning' });
    }
  });

})();
