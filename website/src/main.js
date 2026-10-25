import './style.css';
import { events, getEvent, answerQuestion } from './scenario.js';

const $ = (selector) => document.querySelector(selector);
const state = { selected: 'visitor', tab: 'decision', corrected: false, replayed: false, conversations: {} };
const escapeHtml = (value) => String(value).replace(/[&<>"']/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[char]);
const event = () => getEvent(state.selected, state.corrected);

// Keep reading and keyboard order aligned with the responsive visual layout.
const mobileLayout = matchMedia('(max-width: 700px)');
function arrangeEventStream() {
  $('.product-shell').insertBefore($('.event-stream'), mobileLayout.matches ? $('.workbench') : $('.scenario-summary'));
}
mobileLayout.addEventListener('change', arrangeEventStream);
arrangeEventStream();

function alignDetectionBox() {
  const view = $('#camera-view');
  const { width, height } = view.getBoundingClientRect();
  const scale = Math.max(width / 1600, height / 1000);
  const [x, y, w, h] = event().scene === 'headlights' ? [1420, 630, 180, 280]
    : event().scene === 'courier' ? [820, 580, 100, 185] : [1320, 510, 110, 185];
  // Match the image's cover crop and 58% horizontal object position.
  const left = Math.max(2, x * scale + (width - 1600 * scale) * 0.58);
  const top = y * scale + (height - 1000 * scale) / 2;
  Object.assign($('#detection-box').style, {
    left: `${left}px`, top: `${top}px`,
    width: `${Math.min(w * scale, width - left - 2)}px`, height: `${h * scale}px`,
  });
}
new ResizeObserver(alignDetectionBox).observe($('#camera-view'));

function announce(text) { $('#story-announcement').textContent = text; }

function renderEvents() {
  const list = state.replayed ? [...events, getEvent('return', true)] : events;
  $('#event-list').innerHTML = list.map((item, index) => {
    const current = getEvent(item.id, state.corrected);
    return `<button type="button" class="event-card ${state.selected === item.id ? 'selected' : ''}" data-event="${item.id}" aria-pressed="${state.selected === item.id}" aria-label="${current.title}, ${current.time}, ${current.label}"><span class="event-thumbnail"><img src="/scenes/${current.scene}.svg" alt="" width="160" height="100" /><span class="event-number">0${index + 1}</span></span><span class="event-card-copy"><span class="event-time">${current.time}</span><strong>${current.short}</strong><span class="event-status ${current.tone}"><i class="tiny-dot"></i>${current.label}</span></span><span class="event-arrow" aria-hidden="true">↗</span></button>`;
  }).join('');
  $('#event-total').textContent = list.length;
  $('#quiet-total').textContent = state.replayed ? 3 : 2;
  $('#review-total').textContent = state.corrected ? 1 : 2;
  $('#corrected-total').hidden = !state.corrected;
}

function renderDecision() {
  const item = event();
  const correction = item.id === 'cleaner' ? state.corrected
    ? `<div class="correction-box corrected"><span class="eyebrow">A LITTLE WISER, THANKS TO YOU</span><p><strong>2 → 3 corrections</strong><span>Pattern trust: 0.67 → 0.75</span></p><button type="button" class="button small green-button" data-action="replay">See the next matching visit <span aria-hidden="true">→</span></button><button type="button" class="undo-button" data-action="undo">Undo my correction</button></div>`
    : `<div class="correction-box"><span class="eyebrow">YOU KNOW THE MISSING CONTEXT</span><p>This is your cleaner, arriving a little late.<span>Two past visits have already been reviewed.</span></p><button type="button" class="button small green-button" data-action="correct">This is the expected cleaner <span aria-hidden="true">✓</span></button></div>`
    : item.id === 'return' ? `<div class="memory-note"><span aria-hidden="true">↳</span><p><strong>Learned from your correction.</strong>3 reviews · trust 0.75 · matching activity</p></div>` : '';
  $('#panel-decision').innerHTML = `<span class="decision-badge ${item.tone}"><i class="tiny-dot"></i>${item.label}</span><h3 class="decision-title">${item.decision}</h3><p class="decision-reason">${item.reason}</p><div class="route"><span class="eyebrow">THE PATH TO THIS DECISION</span><div class="route-steps">${['Gate', 'Vision', 'Investigate'].map((label, index) => `<span class="${index <= item.route ? 'reached' : ''} ${index === item.route ? 'last' : ''}"><i>${index < item.route ? '✓' : `0${index + 1}`}</i>${label}</span>${index < 2 ? '<b aria-hidden="true">→</b>' : ''}`).join('')}</div><p>${item.routeLabel}</p></div>${correction}<button type="button" class="evidence-link" data-action="evidence">Follow the evidence <span aria-hidden="true">↗</span></button>`;
}

function renderEvidence() {
  const item = event();
  $('#evidence-count').textContent = item.trace.length;
  $('#panel-evidence').innerHTML = `<div class="evidence-heading"><span class="eyebrow">NOTHING HIDDEN, EVEN WHEN IT’S QUIET</span><h3>The decision trail.</h3><p>${item.camera} · ${item.time} · ${item.disposition}</p></div><ol class="evidence-timeline">${item.trace.map((step, index) => `<li><span class="trace-node">${index + 1}</span><details ${index === 0 ? 'open' : ''}><summary>${step.title}<span aria-hidden="true">+</span></summary><p>${step.detail}</p></details></li>`).join('')}</ol><div class="confidence-note"><span>Recorded confidence <strong>${item.confidence}</strong></span><p>Authored scenario value, not a measured benchmark.</p></div>`;
}

function renderConversation() {
  const item = event();
  const messages = state.conversations[state.selected] || [];
  $('#conversation').innerHTML = `<div class="chat-message assistant-message"><span class="message-label">WATCHWISE · ${item.short.toUpperCase()}</span><p>${item.answer}</p></div>${messages.map((message) => `<div class="chat-message ${message.role === 'user' ? 'user-message' : 'assistant-message'}"><span class="message-label">${message.role === 'user' ? 'YOU' : 'WATCHWISE'}</span><p>${escapeHtml(message.text)}</p></div>`).join('')}`;
  $('#question').value = '';
}

function setTab(tab, focus = false) {
  state.tab = tab;
  document.querySelectorAll('[data-tab]').forEach((button) => {
    const selected = button.dataset.tab === tab;
    button.setAttribute('aria-selected', String(selected));
    button.tabIndex = selected ? 0 : -1;
    $(`#panel-${button.dataset.tab}`).hidden = !selected;
    if (focus && selected) button.focus();
  });
}

function render() {
  const item = event();
  $('#scene').src = `/scenes/${item.scene}.svg`;
  $('#scene').alt = item.alt;
  $('#camera-view').dataset.scene = item.scene;
  $('#camera-name').textContent = item.camera;
  $('#camera-id').textContent = item.cameraId;
  $('#camera-time').textContent = item.stamp;
  $('#camera-kicker').textContent = item.kicker;
  $('#camera-title').textContent = item.headline;
  $('#camera-summary').textContent = item.summary;
  $('#detection-label').textContent = item.id === 'headlights' ? 'car · driveway' : item.id === 'courier' ? 'person · front entry' : item.id === 'visitor' ? 'person · rear gate' : 'person · service path';
  $('#detection-box').classList.toggle('benign', item.tone === 'green');
  renderEvents();
  renderDecision();
  renderEvidence();
  renderConversation();
  setTab(state.tab);
  alignDetectionBox();
}

function selectEvent(id, focusCard = false) {
  state.selected = id;
  render();
  if (focusCard) $(`[data-event="${id}"]`).focus({ preventScroll: true });
  announce(`${event().title} selected. ${event().label}.`);
}

$('#event-list').addEventListener('click', (e) => {
  const button = e.target.closest('[data-event]');
  if (button) selectEvent(button.dataset.event, true);
});

document.querySelector('.panel-tabs').addEventListener('click', (e) => {
  const button = e.target.closest('[data-tab]');
  if (button) setTab(button.dataset.tab);
});
document.querySelector('.panel-tabs').addEventListener('keydown', (e) => {
  const tabs = ['decision', 'evidence', 'ask'];
  const index = tabs.indexOf(state.tab);
  let next;
  if (e.key === 'ArrowRight') next = (index + 1) % 3;
  if (e.key === 'ArrowLeft') next = (index + 2) % 3;
  if (e.key === 'Home') next = 0;
  if (e.key === 'End') next = 2;
  if (next !== undefined) { e.preventDefault(); setTab(tabs[next], true); }
});

$('#panel-decision').addEventListener('click', (e) => {
  const action = e.target.closest('[data-action]')?.dataset.action;
  if (action === 'evidence') setTab('evidence', true);
  if (action === 'correct') {
    state.corrected = true;
    state.conversations = {};
    render();
    $('[data-action="replay"]').focus({ preventScroll: true });
    announce('Correction recorded. The cleaner pattern now has three corrections and trust 0.75. Explore the next matching visit.');
  }
  if (action === 'replay') {
    state.replayed = true;
    // Replaying changes global totals and filtered-event answers. Discard the
    // prior story's answers so every event reflects the current scenario state.
    state.conversations = {};
    selectEvent('return');
    $('#panel-decision').focus({ preventScroll: true });
    announce('Next Thursday: matching activity resolved from memory, without a model call.');
  }
  if (action === 'undo') {
    state.corrected = false;
    state.replayed = false;
    state.conversations = {};
    selectEvent('cleaner');
    $('[data-action="correct"]').focus({ preventScroll: true });
    announce('Correction undone. The pattern returns to two corrections.');
  }
});

$('#try-correction').addEventListener('click', () => {
  state.tab = 'decision';
  selectEvent('cleaner');
  const target = matchMedia('(max-width: 700px)').matches ? $('#panel-decision') : $('#experience');
  target.scrollIntoView({ behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth', block: 'start' });
  $(`[data-action="${state.corrected ? 'replay' : 'correct'}"]`).focus({ preventScroll: true });
});

$('#reset').addEventListener('click', () => {
  Object.assign(state, { selected: 'visitor', tab: 'decision', corrected: false, replayed: false, conversations: {} });
  render();
  announce('Story reset. Four original events, two resolved quietly, two for review.');
});

function ask(question) {
  const text = question.trim();
  if (!text) return;
  const messages = state.conversations[state.selected] ||= [];
  messages.push({ role: 'user', text }, { role: 'assistant', text: answerQuestion(text, event(), state.corrected, state.replayed) });
  // Keep the current scenario useful without allowing an unbounded DOM history.
  if (messages.length > 12) messages.splice(0, 2);
  renderConversation();
  const log = $('#conversation');
  log.scrollTop = log.scrollHeight;
}
document.querySelector('.suggestions').addEventListener('click', (e) => {
  const button = e.target.closest('[data-question]');
  if (button) ask(button.dataset.question);
});
$('#ask-form').addEventListener('submit', (e) => { e.preventDefault(); ask($('#question').value); $('#question').focus(); });

$('#copy-commands').addEventListener('click', async () => {
  const commands = 'watchwise doctor\nwatchwise init\nwatchwise check\nwatchwise serve';
  try {
    await navigator.clipboard.writeText(commands);
    $('#copy-commands').textContent = 'Copied ✓';
    $('#copy-status').textContent = 'Setup commands copied to clipboard.';
  } catch {
    // Selection provides a usable fallback when clipboard permission is denied.
    const range = document.createRange();
    range.selectNodeContents(document.querySelector('.terminal-body'));
    window.getSelection().removeAllRanges();
    window.getSelection().addRange(range);
    $('#copy-commands').textContent = 'Select + copy';
    $('#copy-status').textContent = 'Clipboard unavailable. Commands selected; use your device’s copy shortcut.';
  }
});

render();
