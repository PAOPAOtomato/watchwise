import '../style.css';
import './style.css';
import { initialLanguage, language, setLanguage, t } from './strings.js';
import { loadTrace, replayAt, timelineEntries, nextTime, learningLinks, renderTelegramHTML, safeFrameUrl } from './trace.js';

const $ = (id) => document.getElementById(id);
const node = (tag, text, className) => {
  const element = document.createElement(tag);
  if (text !== undefined && text !== null) element.textContent = String(text);
  if (className) element.className = className;
  return element;
};
const p = (text, className) => node('p', text, className);
const button = (text, click, className = 'record-button') => {
  const element = node('button', text, className); element.type = 'button'; element.addEventListener('click', click); return element;
};
const json = (value) => typeof value === 'string' ? value : JSON.stringify(value, null, 2);
const time = (at) => typeof at === 'string' ? at.slice(11, 19) : '—';
const stamp = (item) => `${time(item?.at)} · ${item?.id ?? ''}`;
const byId = (items, id) => items.find((item) => item.id === id);
const disposition = (key) => t(({ alerted: 'Alerted', escalated_human: 'Owner review', downgraded: 'Held back', suppressed: 'Suppressed' })[key] ?? 'Not decided');
const stage = (key) => t(({ gated: 'Rule / memory gate', face: 'Face recognition', screened: 'First look', investigated: 'Investigation', summarized: 'Whole-event review', ingested: 'Ingested' })[key] ?? 'Unknown');
const armingName = (key) => t(({ armed_home: 'Home', armed_away: 'Away', armed_night: 'Night', disarmed: 'Off' })[key] ?? 'Unknown');
const verdict = (key) => key === 'benign' ? t('Expected') : key === 'suspicious' ? t('Suspicious') : key ?? t('Unknown');
const proposalStatus = (key) => t(({ pending: 'Pending', applied: 'Applied', cancelled: 'Cancelled', expired: 'Expired', failed: 'Failed' })[key] ?? 'Unknown');
let data, trace, entries = [], clock = 0, selectedEvent = null, selectedDecision = null;
let playing = false, lastTick = null, lastRenderKey = '', snapshot;
const phoneNodes = new Map();

function localizedUI(persist = false) {
  document.documentElement.lang = language;
  document.title = t('Watchwise — Replay a run');
  document.querySelector('meta[name="description"]').content = t('Follow a Watchwise run from phone cards to evidence, corrections, and the daily digest.');
  document.querySelectorAll('[data-ui]').forEach((element) => { element.textContent = t(element.dataset.ui); });
  document.querySelectorAll('[data-home]').forEach((element) => { element.href = `/?lang=${language}`; });
  $('language').value = language;
  $('replay-phone').setAttribute('aria-label', t('Recorded messages'));
  if (persist) {
    try { localStorage.setItem('watchwise.language', language); } catch { /* Language still works without storage. */ }
    const url = new URL(location.href); url.searchParams.set('lang', language); history.replaceState(null, '', url);
  }
  if (trace) { renderSource(); renderTimeline(); renderMetrics(); setTime(clock, true); }
}
setLanguage(initialLanguage()); localizedUI();
$('language').addEventListener('change', (event) => { setLanguage(event.target.value); localizedUI(true); });

function renderSource() {
  $('replay-source').textContent = t(data.fixture ? 'Invented fixture · no photos · simulated, not a measured run.' : trace.meta.source === 'recorded' ? 'One recorded run on one setup.' : 'Simulated trace · not a measured run.');
  $('replay-meta').textContent = [trace.meta.title, trace.meta.machine, trace.meta.window.start, '→', trace.meta.window.end, trace.meta.timezone].filter(Boolean).join(' · ');
}

function frameFigure(frameId) {
  const frame = byId(trace.frames, frameId);
  const src = frame?.privacy === 'blurred' ? safeFrameUrl(frame, data.baseUrl) : null;
  const figure = node('figure', null, 'record-frame');
  const img = node('img');
  const caption = node('figcaption', t(src ? 'Exported event frame · people pixelated' : 'Illustration · no exported photo'));
  img.alt = t(src ? 'Pixelated event frame' : 'Illustrated camera scene');
  img.loading = 'lazy'; img.decoding = 'async';
  if (src) {
    // Natural pixels are the upper bound, even if the exported metadata is wrong.
    img.style.maxWidth = `min(100%, ${Number.isFinite(frame.width) && frame.width > 0 ? frame.width : 960}px)`;
    img.addEventListener('load', () => { img.style.maxWidth = `min(100%, ${Math.min(img.naturalWidth, frame.width || img.naturalWidth, 960)}px)`; });
    img.addEventListener('error', () => { img.remove(); caption.textContent = t('Frame unavailable'); }, { once: true });
  } else { img.className = 'illustration'; }
  img.src = src ?? '/scenes/visitor.svg';
  figure.append(img, caption); return figure;
}

function openEvent(id, seekFinal = false) {
  selectedEvent = id; selectedDecision = null;
  if (seekFinal) {
    const event = byId(trace.events, id);
    const decisions = trace.decisions.filter((decision) => decision.event === id);
    setTime(Math.max(event?.t ?? 0, ...decisions.map((decision) => decision.t)), true);
  } else setTime(clock, true);
}
function eventButton(id, text, seekFinal = true) {
  const control = button(text ?? `${t('Evidence')} · ${id}`, () => openEvent(id, seekFinal));
  control.dataset.openEvent = id; return control;
}
function renderTimeline() {
  const container = $('replay-timeline'); container.replaceChildren();
  const items = [...trace.events.map((item) => ({ kind: 'event', item })), ...trace.arming.map((item) => ({ kind: 'arming', item }))].sort((a, b) => a.item.t - b.item.t);
  for (const { kind, item } of items) {
    const control = button('', () => { pause(); kind === 'event' ? openEvent(item.id, true) : setTime(Math.max(0, item.t)); }, `timeline-item ${kind}`);
    control.dataset.t = item.t;
    control.append(node('time', time(item.at)), node('strong', kind === 'event' ? `${item.camera} · ${item.id}` : `${t('Arming')} · ${armingName(item.state)}`));
    control.append(node('span', kind === 'event' ? disposition(item.outcome?.disposition) : item.source));
    if (kind === 'event') control.dataset.event = item.id;
    else control.dataset.arming = item.state;
    container.append(control);
  }
}

function matchingAction(message, action) {
  return trace.owner_actions.find((entry) => entry.kind === 'button' && entry.on_message === message.message && entry.action === action.action &&
    ['event', 'proposal', 'state', 'name'].every((key) => action[key] == null || entry[key] === action[key]));
}
function renderPhone(state) {
  const parent = $('replay-phone');
  const following = playing && parent.scrollHeight - parent.scrollTop - parent.clientHeight < 100;
  const visible = new Set(state.messages.map((message) => message.message));
  for (const [id, element] of phoneNodes) if (!visible.has(id)) { element.remove(); phoneNodes.delete(id); }
  parent.querySelector('.phone-empty')?.remove();
  if (!state.messages.length) parent.append(p(t('No messages at this point.'), 'phone-empty'));
  for (const message of state.messages) {
    let element = phoneNodes.get(message.message);
    if (!element) { element = node('article', null, 'phone-message'); element.dataset.message = message.message; phoneNodes.set(message.message, element); parent.append(element); }
    const actions = message.ownerActions ?? [];
    const signature = `${language}:${message.id}:${actions.map((action) => action.id).join(',')}`;
    if (element.dataset.signature === signature) continue;
    element.dataset.signature = signature;
    const active = element.contains(document.activeElement) ? document.activeElement.dataset.action : null;
    element.replaceChildren();
    const header = node('div', null, 'message-meta');
    header.append(node('strong', 'Watchwise'), node('time', `${time(message.at)}${message.action === 'edited' ? ` · ${t('Edited')}` : ''}`));
    element.append(header);
    if (message.frame || message.has_photo || ['alert', 'spot_check', 'chat_photo'].includes(message.kind)) element.append(frameFigure(message.frame));
    const body = p(null, 'telegram-text'); body.append(renderTelegramHTML(message.text ?? '')); element.append(body);
    for (const row of message.buttons ?? []) {
      const buttons = node('div', null, 'message-buttons');
      for (const action of row) {
        const recorded = matchingAction(message, action);
        const control = button(action.label, () => { pause(); setTime(recorded.t, true); }, 'phone-button');
        control.dataset.recordedAction = recorded?.id ?? ''; control.dataset.action = action.action ?? 'link';
        control.disabled = !recorded;
        control.title = t(action.link ? 'Recording link omitted from trace' : recorded ? 'Recorded tap' : 'No tap recorded');
        if (recorded && recorded.t <= clock) control.classList.add('was-tapped');
        buttons.append(control);
      }
      element.append(buttons);
    }
    for (const action of actions) {
      const record = node('div', null, 'card-action'); record.dataset.ownerAction = action.id;
      record.append(node('strong', `${t('Recorded tap')} · ${time(action.at)}`), p(action.name ? `${action.action} · ${action.name}` : action.action));
      if (action.outcome?.reply) record.append(p(action.outcome.reply));
      element.append(record);
    }
    if (message.footer) element.append(p(message.footer, 'card-footer'));
    for (const event of [message.event, ...(message.members ?? [])].filter(Boolean)) element.append(eventButton(event, `${t('Evidence')} → ${event}`, false));
    if (active) Array.from(element.querySelectorAll('[data-action]')).find((item) => item.dataset.action === active)?.focus({ preventScroll: true });
  }
  if (following) parent.scrollTop = parent.scrollHeight;
  $('replay-toast').textContent = state.toast ?? '';
  $('replay-toast').hidden = !state.toast;
  const owner = $('replay-owner'); owner.replaceChildren();
  const notes = state.ownerActions.filter((action) => action.kind !== 'button');
  if (notes.length) owner.append(node('h3', t('Recorded actions')));
  for (const action of notes) {
    const record = node('article'); record.dataset.ownerAction = action.id;
    record.append(node('small', `${t('Owner')} · ${stamp(action)}`), p(action.text));
    if (action.reply) record.append(p(`${t('Reply')}: ${action.reply}`));
    owner.append(record);
  }
}

function field(parent, label, value) { if (value != null && value !== '') { const entry = node('div', null, 'record-field'); entry.append(node('h4', t(label)), p(value)); parent.append(entry); } }
function toolStep(step, evidence = false) {
  const details = node('details', null, 'tool-step');
  const key = step.evidence_id ?? step.tool;
  details.dataset.detailKey = `tool:${key}`;
  if (evidence) details.dataset.evidenceId = step.evidence_id;
  details.append(node('summary', `${step.evidence_id ? `${step.evidence_id} · ` : ''}${step.tool}`));
  details.append(node('h4', t('Arguments')), node('pre', json(step.arguments)), node('h4', t('Result')), node('pre', json(step.result)));
  if (step.error) details.append(node('h4', t('Error')), p(step.error));
  return details;
}
function renderEvidence() {
  const parent = $('replay-evidence'); parent.replaceChildren();
  const event = byId(trace.events, selectedEvent);
  if (!event) { parent.append(p(t('Select an event or a card to inspect its decisions.'), 'empty-note')); return; }
  parent.append(node('h3', `${event.camera} · ${event.id}`), p(stamp(event), 'record-meta'));
  const decisions = trace.decisions.filter((decision) => decision.event === event.id && decision.t <= clock).sort((a, b) => a.t - b.t);
  if (!decisions.length) { parent.append(p(t('No decision yet.'))); return; }
  const decision = byId(decisions, selectedDecision) ?? decisions.at(-1);
  const nav = node('div', null, 'decision-tabs'); nav.setAttribute('aria-label', t('Decision history'));
  for (const item of decisions) {
    const control = button(`${item.id} · ${stage(item.stage)}`, () => { selectedDecision = item.id; setTime(clock, true); });
    control.dataset.decision = item.id; control.setAttribute('aria-pressed', String(item.id === decision.id)); nav.append(control);
  }
  parent.append(nav, p(`${disposition(decision.disposition)} · ${time(decision.at)}`, 'verdict-line'));
  if (decision.history === 'latest_only') parent.append(p(t('Only the latest decision was recorded.'), 'notice'));
  field(parent, 'Reason', decision.reason); field(parent, 'Rationale', decision.rationale); field(parent, 'Summary', decision.summary);
  if (decision.screen) {
    const details = node('details', null, 'tool-step'); details.dataset.detailKey = `screen:${decision.id}`; details.append(node('summary', t('Model’s first look')));
    field(details, 'Verdict', verdict(decision.screen.verdict)); field(details, 'Reason', decision.screen.reason); field(details, 'Model', decision.screen.model); field(details, 'Error', decision.screen.error); parent.append(details);
  }
  const investigation = byId(trace.investigations, decision.investigation);
  if (investigation) {
    parent.append(node('h3', `${t('Investigation')} · ${investigation.id}`));
    if (investigation.finished?.t > clock || (!investigation.finished && ['running', 'pending'].includes(investigation.status))) parent.append(p(t('Investigation still in progress at this point.')));
    else {
      if (!investigation.finished) parent.append(p(t('Completion time not reported.'), 'record-meta'));
      field(parent, 'Conclusion', investigation.conclusion);
      for (const step of investigation.steps ?? []) parent.append(toolStep(step, true));
      for (const [key, title] of [['observations', 'Observations'], ['inferences', 'Inferences'], ['unknowns', 'Unknowns']]) {
        const values = investigation[key] ?? [];
        field(parent, title, values.length ? values.map((value) => typeof value === 'object' ? `${value.fact} [${value.evidence_id}]` : value).join('\n') : t('Not reported'));
      }
      field(parent, 'Error', investigation.error);
    }
  } else parent.append(p(t('No investigation attached to this decision.'), 'empty-note'));
  parent.append(node('h3', t('Event frame')), frameFigure(event.frame), p(t('The exported frame belongs to this event; it may differ from the moment shown on the phone.'), 'record-meta'));
}
function renderLearning() {
  const parent = $('replay-learning'); parent.replaceChildren();
  const corrections = trace.corrections.filter((item) => item.t <= clock);
  if (!corrections.length) { parent.append(p(t('No correction yet.'), 'empty-note')); return; }
  const links = learningLinks(trace, clock);
  for (const correction of corrections) {
    const item = node('article', null, 'learning-record');
    item.append(node('h3', `${correction.id} · ${verdict(correction.verdict)}`), eventButton(correction.event), p(stamp(correction), 'record-meta'));
    if (correction.note) item.append(p(correction.note));
    const undone = correction.undone && correction.undone.t <= clock;
    const effects = links.filter((link) => link.correction.id === correction.id);
    if (undone) item.append(p(t('Undone')));
    else if (!effects.length) item.append(p(t('This correction has no active effects in the export.')));
    for (const link of effects) {
      item.append(p(`↓ ${t(link.effect.effect)} · ${link.pattern?.id ?? link.effect.pattern}`, 'learning-arrow'));
      if (link.pattern) { field(item, 'Pattern', link.pattern.description); item.append(p(t('Pattern at export'), 'record-meta')); }
      if (!link.decisions.length) item.append(p(t('No later matching decision at this point.')));
      for (const decision of link.decisions) {
        item.append(p(t(link.heldBack.some((held) => held.id === decision.id) ? 'You taught it → memory held this event back.' : 'The pattern informed this decision.')));
        item.append(button(`${decision.event} · ${disposition(decision.disposition)} · ${stage(decision.stage)}`, () => { pause(); selectedEvent = decision.event; selectedDecision = decision.id; setTime(decision.t, true); }));
      }
    }
    parent.append(item);
  }
}
function jumpMessage(id) {
  const message = trace.messages.find((entry) => entry.message === id && entry.action === 'sent');
  if (!message) return;
  pause(); setTime(Math.max(clock, message.t), true);
  const target = phoneNodes.get(id); target?.scrollIntoView({ block: 'nearest', behavior: 'instant' }); target?.setAttribute('tabindex', '-1'); target?.focus({ preventScroll: true });
}
function renderDigests() {
  const parent = $('replay-digests'); parent.replaceChildren();
  const digests = trace.digests.filter((digest) => digest.t <= clock);
  if (!digests.length) { parent.append(p(t('No digest yet.'), 'empty-note')); return; }
  for (const digest of digests) {
    const item = node('article', null, 'digest-record');
    item.append(node('h3', `${t(digest.kind === 'scheduled' ? 'Scheduled digest' : 'Requested digest')} · ${digest.id}`), p(`${digest.covers?.start ?? ''} → ${digest.covers?.end ?? ''}`, 'record-meta'));
    if (!digest.sent) item.append(p(t('No message sent.')));
    for (const id of digest.messages ?? []) item.append(button(`${t('Open phone message')} · ${id}`, () => jumpMessage(id)));
    for (const eventId of digest.spot_checks ?? []) {
      const spot = node('div', null, 'spot-record'); spot.append(eventButton(eventId, `${t('Spot-check')} · ${eventId}`));
      const messages = trace.messages.filter((message) => message.kind === 'spot_check' && message.event === eventId && message.t <= clock);
      if (messages.length) spot.append(button(t('Open phone message'), () => jumpMessage(messages[0].message)));
      const answers = trace.owner_actions.filter((action) => action.t <= clock && action.t >= digest.t && action.event === eventId && action.outcome?.spot_check && messages.some((message) => message.message === action.on_message));
      if (!answers.length) spot.append(p(t('Awaiting recorded answer')));
      for (const answer of answers) spot.append(p(`${time(answer.at)} · ${answer.outcome.footer ?? answer.outcome.toast ?? answer.action}`));
      item.append(spot);
    }
    parent.append(item);
  }
}
function renderProposal(proposal) {
  const parent = node('article', null, 'proposal-record'); parent.dataset.proposal = proposal.id;
  const decided = proposal.decided && proposal.decided.t <= clock;
  const status = proposal.decided ? decided ? proposal.status : 'pending' : proposal.status;
  parent.append(node('h4', `${t('Proposal')} · ${proposal.id} · ${proposalStatus(status)}`), p(proposal.summary));
  if (decided || !proposal.decided) field(parent, 'Result', proposal.result);
  if (!proposal.decided && status !== 'pending') parent.append(p(t('Outcome at export'), 'record-meta'));
  parent.append(p(t('Only the recorded Confirm tap applies a proposal.'), 'record-meta'));
  const message = trace.messages.find((message) => message.buttons?.flat().some((action) => action.proposal === proposal.id));
  if (message) parent.append(button(t('Open phone message'), () => jumpMessage(message.message)));
  return parent;
}
function renderChat() {
  const parent = $('replay-conversations'); parent.replaceChildren();
  const turns = trace.conversations.flatMap((conversation) => conversation.turns ?? []).filter((turn) => turn.t <= clock).sort((a, b) => a.t - b.t);
  if (!turns.length) parent.append(p(t('No chat yet.'), 'empty-note'));
  const linked = new Set();
  for (const turn of turns) {
    const item = node('article', null, 'conversation-record'); item.dataset.turn = turn.id;
    item.append(p(stamp(turn), 'record-meta')); field(item, 'Question', turn.question); field(item, 'Answer', turn.answer);
    if (turn.about) item.append(eventButton(turn.about));
    if (turn.guard) field(item, 'Safety guard', turn.guard);
    if (turn.steps_recorded === false) item.append(p(t('Tool steps were not recorded.')));
    else if (turn.tools?.length) {
      const details = node('details', null, 'chat-tools'); details.dataset.detailKey = `chat:${turn.id}`; details.append(node('summary', t('Tool steps')));
      for (const [index, step] of turn.tools.entries()) { const tool = toolStep(step); tool.dataset.detailKey = `chat:${turn.id}:${index}`; details.append(tool); } item.append(details);
    }
    for (const id of turn.photos ?? []) { const event = byId(trace.events, id); item.append(eventButton(id), frameFigure(event?.frame)); }
    for (const id of turn.proposals ?? []) { const proposal = byId(trace.proposals, id); if (proposal && proposal.t <= clock) { item.append(renderProposal(proposal)); linked.add(id); } }
    parent.append(item);
  }
  for (const proposal of trace.proposals) if (proposal.t <= clock && !linked.has(proposal.id)) parent.append(renderProposal(proposal));
  for (const watch of trace.watches) if (watch.created?.t <= clock) {
    const item = node('article', null, 'watch-record'); item.append(node('h3', `${t('Watch')} · ${watch.id}`), p(watch.request), p(`${t('Watch state at export')}: ${watch.status}`, 'record-meta'));
    if (watch.matched) item.append(eventButton(watch.matched, `${t('Matched event')} · ${watch.matched}`)); parent.append(item);
  }
}

function metricRows(parent, values, fields, labelId, recordedLabels = false) {
  const list = node('dl', null, 'metric-values');
  list.setAttribute('aria-describedby', labelId);
  for (const [key, label] of fields) if (typeof values?.[key] === 'number' && Number.isFinite(values[key])) {
    const pair = node('div'); pair.append(node('dt', recordedLabels ? label : t(label)), node('dd', values[key])); list.append(pair);
  }
  parent.append(list);
}
function renderMetrics() {
  const parent = $('replay-metrics'); parent.replaceChildren();
  const counts = trace.metrics?.counts, timings = trace.metrics?.timings;
  if (!counts && !timings) { parent.append(p(t('No metrics reported.'))); return; }
  for (const [key, block, title] of [['counts', counts, 'Counts'], ['timings', timings, 'Timings']]) {
    if (!block) continue;
    const item = node('article', null, 'metric-block'); item.dataset.metricBlock = key;
    const labelId = `metric-label-${key}`; const label = p(block.label ?? t('Label not provided'), 'metric-label'); label.id = labelId;
    item.append(node('h3', t(title)), label);
    if (key === 'counts') {
      metricRows(item, block, [['events', 'Events'], ['sent_to_owner', 'Sent to owner'], ['held_back', 'Held back'], ['alerted_while_muted', 'Alerted while muted']], labelId);
      for (const [group, heading, fields] of [
        ['held_back_by_reason', 'Held back by reason', Object.keys(block.held_back_by_reason ?? {}).map((reason) => [reason, reason])],
        ['final_stage', 'Final decision stage', ['gated', 'face', 'screened', 'investigated', 'summarized', 'ingested'].map((key) => [key, stage(key)])],
        ['model_calls', 'Model calls', [['screenings', 'Screenings'], ['investigations', 'Investigations'], ['chat_answers', 'Chat answers']]],
        ['owner_answers', 'Owner answers', [['benign', 'Expected'], ['suspicious', 'Suspicious']]],
        ['spot_checks', 'Spot-check answers', [['answered', 'Answered'], ['right_to_hold_back', 'Right to hold back'], ['accuracy', 'Accuracy (as recorded)']]],
      ]) if (block[group]) { item.append(node('h4', t(heading))); metricRows(item, block[group], fields, labelId, group === 'held_back_by_reason'); }
    } else {
      for (const [group, heading] of [['first_card_after_event_start_s', 'Event start to first card (s)'], ['screen_ms', 'First look (ms)'], ['investigation_s', 'Investigation (s)'], ['chat_answer_s', 'Chat answer (s)']]) if (block[group]) {
        item.append(node('h4', t(heading))); metricRows(item, block[group], [['n', 'Samples'], ['median', 'Median'], ['max', 'Maximum']], labelId);
      }
    }
    parent.append(item);
  }
}

function setTime(value, force = false) {
  if (!trace) return;
  clock = Math.min(trace.meta.window.seconds, Math.max(0, Number(value) || 0));
  $('replay-clock').value = clock;
  // Display the trace's local clock; do not convert it to the visitor's timezone.
  const base = Date.parse(trace.meta.window.start.slice(0, 23) + 'Z');
  const local = Number.isFinite(base) ? new Date(base + clock * 1000).toISOString().slice(11, 19) : `${clock.toFixed(1)}s`;
  $('replay-time').textContent = `${local} / ${time(trace.meta.window.end)}`;
  $('replay-clock').setAttribute('aria-valuetext', local);
  $('replay-play').textContent = t(playing ? 'Pause' : 'Play'); $('replay-play').setAttribute('aria-pressed', String(playing));
  const position = entries.filter((item) => item.t <= clock).length;
  const key = `${position}:${selectedEvent}:${selectedDecision}:${language}`;
  if (!force && key === lastRenderKey) return;
  snapshot = replayAt(trace, clock);
  $('replay-arming').textContent = `${t('Arming')} · ${armingName(snapshot.arming?.state)}`;
  document.querySelectorAll('.timeline-item').forEach((item) => { item.classList.toggle('occurred', Number(item.dataset.t) <= clock); if (item.dataset.event) item.setAttribute('aria-pressed', String(item.dataset.event === selectedEvent)); });
  const open = new Set([...document.querySelectorAll('details[open][data-detail-key]')].map((item) => item.dataset.detailKey));
  renderPhone(snapshot); renderEvidence(); renderLearning(); renderDigests(); renderChat();
  document.querySelectorAll('details[data-detail-key]').forEach((item) => { item.open = open.has(item.dataset.detailKey); });
  lastRenderKey = `${position}:${selectedEvent}:${selectedDecision}:${language}`;
}
function pause() { playing = false; lastTick = null; $('replay-play').textContent = t('Play'); $('replay-play').setAttribute('aria-pressed', 'false'); }
function tick(now) {
  if (!playing) return;
  const elapsed = lastTick == null ? 0 : (now - lastTick) / 1000;
  lastTick = now;
  setTime(clock + elapsed * Number($('replay-speed').value));
  if (clock >= trace.meta.window.seconds) { pause(); return; }
  requestAnimationFrame(tick);
}
$('replay-play').addEventListener('click', () => {
  if (playing) { pause(); return; }
  if (clock >= trace.meta.window.seconds) setTime(0, true);
  playing = true; lastTick = null; setTime(clock); requestAnimationFrame(tick);
});
$('replay-step').addEventListener('click', () => { pause(); setTime(nextTime(trace, clock), true); });
$('replay-reset').addEventListener('click', () => { pause(); selectedEvent = null; selectedDecision = null; setTime(0, true); });
$('replay-clock').addEventListener('input', (event) => { pause(); setTime(event.target.value); });
$('replay-speed').addEventListener('change', () => { lastTick = null; });
document.addEventListener('visibilitychange', () => { if (document.hidden) pause(); });

async function start() {
  try {
    data = await loadTrace(); trace = data.trace; entries = timelineEntries(trace);
    $('replay-clock').max = trace.meta.window.seconds;
    $('replay-loading').hidden = true; $('replay-app').hidden = false;
    renderSource(); renderTimeline(); renderMetrics(); setTime(0, true);
  } catch {
    $('replay-loading').hidden = true;
    const error = $('replay-error'); error.hidden = false;
    const message = p(t('Unable to load this trace. Check its format, version, privacy setting, and data files.'));
    message.dataset.ui = 'Unable to load this trace. Check its format, version, privacy setting, and data files.';
    const retry = button(t('Try again'), () => location.reload()); retry.dataset.ui = 'Try again';
    error.append(message, retry);
  }
}
start();
