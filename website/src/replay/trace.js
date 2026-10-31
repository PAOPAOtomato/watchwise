// Trace 1.x reader. Only the fields used by the replay influence its behavior.
const arrays = ['cameras', 'events', 'decisions', 'investigations', 'frames', 'messages', 'owner_actions', 'corrections', 'patterns', 'conversations', 'proposals', 'watches', 'digests', 'arming'];
const sortTime = (items) => [...items].sort((a, b) => a.t - b.t);
const occurred = (items, t) => sortTime(items).filter((item) => Number.isFinite(item.t) && item.t <= t);

export function validateTrace(raw) {
  if (!raw || raw.format !== 'watchwise-trace') throw new Error('Unsupported trace format.');
  if (typeof raw.version !== 'string' || !/^1\.\d+$/.test(raw.version)) throw new Error('Unsupported trace version.');
  const window = raw.meta?.window;
  if (!window || !Number.isFinite(window.seconds) || window.seconds < 0 || !Number.isFinite(Date.parse(window.start)) || !Number.isFinite(Date.parse(window.end))) throw new Error('Invalid trace window.');
  if (!['blurred', 'none'].includes(raw.meta?.privacy?.frames)) throw new Error('This trace is not approved for public frames.');
  const trace = { format: raw.format, version: raw.version, meta: raw.meta, metrics: raw.metrics ?? {} };
  for (const field of arrays) {
    if (raw[field] != null && !Array.isArray(raw[field])) throw new Error(`Invalid trace collection: ${field}.`);
    trace[field] = (raw[field] ?? []).filter((item) => item && typeof item === 'object');
  }
  if (trace.frames.some((frame) => frame.privacy !== 'blurred')) throw new Error('This trace contains a frame without public privacy protection.');
  if (trace.meta.privacy.frames === 'none' && trace.frames.length) throw new Error('A no-frames trace contains frames.');
  return trace;
}

export async function loadTrace({ url = '/data/run-2026-10-30/trace.json', fixtureUrl = '/data/replay-fixture/trace.json', signal } = {}) {
  let response = await fetch(url, { signal, headers: { Accept: 'application/json' } });
  let fixture = false;
  let selected = url;
  if (response.status === 404) {
    await response.arrayBuffer(); // Finish the 404 so the browser doesn't abort it.
    response = await fetch(fixtureUrl, { signal, headers: { Accept: 'application/json' } });
    fixture = true;
    selected = fixtureUrl;
  }
  if (!response.ok) throw new Error(`Trace request failed (${response.status}).`);
  const trace = validateTrace(await response.json());
  return { trace, fixture, baseUrl: new URL('.', new URL(selected, location.href)).href };
}

export function replayAt(trace, t) {
  const ownerActions = occurred(trace.owner_actions, t);
  const phone = new Map();
  for (const entry of occurred(trace.messages, t)) {
    if (entry.action === 'sent' && !phone.has(entry.message)) phone.set(entry.message, entry);
    else if (entry.action === 'edited' && phone.has(entry.message)) phone.set(entry.message, entry);
  }
  const messages = [...phone.values()].map((entry) => {
    const actions = ownerActions.filter((action) => action.on_message === entry.message);
    const footers = actions.filter((action) => action.outcome?.footer != null);
    return { ...entry, ownerActions: actions, footer: footers.at(-1)?.outcome.footer ?? null };
  });
  return {
    t, messages, ownerActions,
    toast: ownerActions.filter((action) => action.outcome?.toast != null).at(-1)?.outcome.toast ?? null,
    decisions: occurred(trace.decisions, t),
    arming: occurred(trace.arming, t).at(-1) ?? null,
  };
}

export function timelineEntries(trace) {
  const result = [];
  const add = (type, item, time = item, id = item.id) => {
    if (Number.isFinite(time?.t)) result.push({ type, t: time.t, at: time.at, id: id ?? `${type}-${result.length}`, item });
  };
  for (const [field, type] of [['events', 'event'], ['arming', 'arming'], ['messages', 'message'], ['owner_actions', 'owner_action'], ['decisions', 'decision'], ['corrections', 'correction'], ['proposals', 'proposal'], ['digests', 'digest']]) {
    for (const item of trace[field]) add(type, item);
  }
  for (const conversation of trace.conversations) for (const turn of conversation.turns ?? []) add('conversation', turn);
  for (const proposal of trace.proposals) if (proposal.decided) add('proposal_outcome', proposal, proposal.decided, `${proposal.id}-outcome`);
  for (const investigation of trace.investigations) if (investigation.finished) add('investigation', investigation, investigation.finished);
  for (const correction of trace.corrections) if (correction.undone) add('undo', correction, correction.undone, `${correction.id}-undo`);
  return sortTime(result);
}

export function nextTime(trace, t) {
  return Math.min(timelineEntries(trace).find((entry) => entry.t > t)?.t ?? trace.meta.window.seconds, trace.meta.window.seconds);
}

export function learningLinks(trace, t = Infinity) {
  const patterns = new Map(trace.patterns.map((pattern) => [pattern.id, pattern]));
  return occurred(trace.corrections, t).flatMap((correction) => {
    if (correction.undone && correction.undone.t <= t) return [];
    return (correction.effects ?? []).flatMap((effect) => {
      const pattern = patterns.get(effect.pattern);
      if (!pattern) return [];
      const decisions = occurred(trace.decisions, t).filter((decision) => decision.t > correction.t && (decision.patterns ?? []).includes(pattern.id));
      const heldBack = decisions.filter((decision) => decision.stage === 'gated' && ['downgraded', 'suppressed'].includes(decision.disposition));
      return [{ correction, effect, pattern, decisions, heldBack }];
    });
  });
}

// Decode only entities, without ever handing the source to an HTML insertion API.
const decode = (text) => text.replace(/&(#x[\da-f]+|#\d+|amp|lt|gt|quot|apos|nbsp);/gi, (match, entity) => {
  const named = { amp: '&', lt: '<', gt: '>', quot: '"', apos: "'", nbsp: '\u00a0' };
  if (entity[0] !== '#') return named[entity.toLowerCase()] ?? match;
  const number = entity[1].toLowerCase() === 'x' ? parseInt(entity.slice(2), 16) : parseInt(entity.slice(1), 10);
  return number > 0 && number <= 0x10ffff && !(number >= 0xd800 && number <= 0xdfff) ? String.fromCodePoint(number) : '\ufffd';
});

export function renderTelegramHTML(text) {
  const fragment = document.createDocumentFragment();
  const stack = [{ tag: '', element: fragment }];
  const tokens = String(text ?? '').match(/<[^>]*>|[^<]+|</g) ?? [];
  for (const token of tokens) {
    const closing = token.match(/^<\s*\/\s*(b|i|code|a)\s*>$/i);
    if (closing) {
      const index = stack.map((entry) => entry.tag).lastIndexOf(closing[1].toLowerCase());
      if (index > 0) stack.length = index;
      continue;
    }
    const opening = token.match(/^<\s*(b|i|code|a)(\s[^>]*)?>$/i);
    if (opening) {
      const tag = opening[1].toLowerCase();
      const element = document.createElement(tag);
      if (tag === 'a') {
        const href = (opening[2] ?? '').match(/(?:^|\s)href\s*=\s*(?:"([^"]*)"|'([^']*)')/i);
        if (href) {
          const value = decode(href[1] ?? href[2]).trim();
          try {
            const url = new URL(value);
            if (['http:', 'https:'].includes(url.protocol)) {
              element.href = url.href;
              element.target = '_blank';
              element.rel = 'noopener noreferrer';
            }
          } catch { /* Unsafe or incomplete links remain readable text. */ }
        }
      }
      stack.at(-1).element.append(element);
      stack.push({ tag, element });
    } else if (!/^<[^>]+>$/.test(token)) {
      stack.at(-1).element.append(document.createTextNode(decode(token)));
    }
  }
  return fragment;
}

export function safeFrameUrl(frame, baseUrl) {
  if (!frame || frame.privacy !== 'blurred' || !/^frames\/[A-Za-z0-9_-]+\.jpg$/.test(frame.file ?? '')) return null;
  try {
    const base = new URL(baseUrl, location.href);
    const url = new URL(frame.file, base);
    return url.origin === location.origin && url.pathname.startsWith(new URL('frames/', base).pathname) ? url.href : null;
  } catch { return null; }
}
