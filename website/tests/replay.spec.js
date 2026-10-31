import { readFile, access } from 'node:fs/promises';
import { test as base, expect } from '@playwright/test';

const fixturePath = new URL('../public/data/replay-fixture/trace.json', import.meta.url);
const traceURL = '**/data/run-2026-10-30/trace.json';
const chronological = (items) => [...items].sort((a, b) => a.t - b.t);
const plain = (html) => html.replace(/<[^>]*>/g, '').replaceAll('&lt;', '<').replaceAll('&gt;', '>').replaceAll('&amp;', '&');

const test = base.extend({
  page: async ({ page }, use) => {
    const failures = [];
    page.on('pageerror', (error) => failures.push(error.message));
    page.on('console', (message) => {
      if (message.type() === 'error' && !message.location().url.endsWith('/data/run-2026-10-30/trace.json')) failures.push(`Console: ${message.text()}`);
    });
    page.on('requestfailed', (request) => failures.push(`${request.url()}: ${request.failure()?.errorText}`));
    page.on('response', (response) => {
      if (response.status() >= 400 && !response.url().endsWith('/data/run-2026-10-30/trace.json')) {
        failures.push(`HTTP ${response.status()}: ${response.url()}`);
      }
    });
    await use(page);
    expect(failures, 'Replay has no browser exceptions or failed assets').toEqual([]);
  },
});

async function fixture() {
  return JSON.parse(await readFile(fixturePath, 'utf8'));
}
async function open(page, trace, language = 'en') {
  await page.route(traceURL, (route) => route.fulfill({ json: trace }));
  await page.goto(`/replay.html?lang=${language}`);
  await expect(page.locator('#replay-clock')).toBeVisible();
}
async function seek(page, time) {
  await page.locator('#replay-clock').evaluate((input, value) => {
    input.value = String(value);
    input.dispatchEvent(new Event('input', { bubbles: true }));
  }, time);
  await expect(page.locator('#replay-clock')).toHaveValue(String(time));
}
function firstEditable(trace) {
  const edit = chronological(trace.messages).find((entry) => entry.action === 'edited');
  const sent = trace.messages.find((entry) => entry.message === edit.message && entry.action === 'sent');
  return { sent, edit };
}

// The source order is intentionally scrambled: t, not file position, drives playback.
test('messages replay in time order and edits preserve the original phone card', async ({ page }) => {
  const trace = await fixture();
  const { sent, edit } = firstEditable(trace);
  trace.messages.reverse();
  trace.decisions.reverse();
  trace.owner_actions.reverse();
  await open(page, trace);
  await expect(page.locator('#replay-phone [data-message]')).toHaveCount(0);
  await seek(page, sent.t);
  const card = page.locator(`#replay-phone [data-message="${sent.message}"]`);
  await expect(card).toContainText(plain(sent.text));
  await card.evaluate((element) => { element.dataset.identityCheck = 'original'; });
  await seek(page, edit.t);
  await expect(card).toHaveAttribute('data-identity-check', 'original');
  await expect(card).toContainText(plain(edit.text));
  await expect(card).toHaveCount(1);
  await seek(page, trace.meta.window.seconds);
  const expected = [...new Set(chronological(trace.messages).filter((entry) => entry.action === 'sent').map((entry) => entry.message))];
  expect(await page.locator('#replay-phone [data-message]').evaluateAll((cards) => cards.map((card) => card.dataset.message))).toEqual(expected);
  await seek(page, 0);
  await expect(page.locator('#replay-phone [data-message]')).toHaveCount(0);
  await page.locator('#replay-step').click();
  expect(Number(await page.locator('#replay-clock').inputValue())).toBeGreaterThan(0);
  await page.locator('#replay-speed').selectOption('60');
  await page.locator('#replay-play').click();
  await expect.poll(async () => Number(await page.locator('#replay-clock').inputValue())).toBeGreaterThan(sent.t);
  await page.locator('#replay-play').click();
  const paused = await page.locator('#replay-clock').inputValue();
  await page.waitForTimeout(100);
  await expect(page.locator('#replay-clock')).toHaveValue(paused);
});

test('the full event timeline includes held-back events and timed arming changes', async ({ page }) => {
  const trace = await fixture();
  await open(page, trace);
  for (const event of trace.events) await expect(page.locator(`[data-event="${event.id}"]`)).toBeVisible();
  await expect(page.locator('[data-arming]')).toHaveCount(trace.arming.length);
  await expect(page.locator('#replay-arming')).toContainText('Home');
  const change = chronological(trace.arming).find((entry) => entry.t > 0);
  await seek(page, change.t - 0.1);
  await expect(page.locator('#replay-arming')).toContainText('Home');
  await seek(page, change.t);
  await expect(page.locator('#replay-arming')).toContainText('Away');
  await seek(page, 0);
  await expect(page.locator('#replay-arming')).toContainText('Home');
});

test('the recorded owner tap appears on its card with the toast, footer and learning link', async ({ page }) => {
  const trace = await fixture();
  const action = chronological(trace.owner_actions).find((entry) => entry.kind === 'button' && entry.outcome?.footer && entry.outcome?.corrections?.length);
  const correction = trace.corrections.find((entry) => action.outcome.corrections.includes(entry.id) && entry.effects.length);
  const effect = correction.effects[0];
  const pattern = trace.patterns.find((entry) => entry.id === effect.pattern);
  const later = chronological(trace.decisions).find((entry) => entry.t > correction.t && entry.patterns.includes(pattern.id) && entry.stage === 'gated');
  await open(page, trace);
  await seek(page, action.t - 0.1);
  const card = page.locator(`#replay-phone [data-message="${action.on_message}"]`);
  await expect(card.locator(`[data-owner-action="${action.id}"]`)).toHaveCount(0);
  await card.locator(`[data-recorded-action][data-action="${action.action}"]`).first().click();
  await expect(page.locator('#replay-clock')).toHaveValue(String(action.t));
  await expect(card.locator(`[data-owner-action="${action.id}"]`)).toBeVisible();
  await expect(card).toContainText(action.outcome.footer);
  await expect(page.locator('#replay-toast')).toContainText(action.outcome.toast);
  await seek(page, later.t);
  await expect(page.locator('#replay-learning')).toContainText(pattern.description);
  await expect(page.locator('#replay-learning')).toContainText(later.event);
  const event = trace.events.find((entry) => entry.id === later.event);
  await page.locator(`[data-event="${event.id}"]`).click();
  await expect(page.locator('#replay-evidence')).toContainText(later.reason);
  // A held-back incident belongs on the timeline without fabricating an alert card.
  expect(trace.messages.filter((entry) => entry.kind === 'alert' && entry.event === event.id)).toHaveLength(0);
});

test('card evidence links decisions, investigation steps and the event illustration', async ({ page }, testInfo) => {
  const trace = await fixture();
  const decision = trace.decisions.find((entry) => entry.investigation);
  const investigation = trace.investigations.find((entry) => entry.id === decision.investigation);
  const message = trace.messages.find((entry) => entry.event === decision.event);
  await open(page, trace);
  await seek(page, Math.max(decision.t, investigation.finished.t, message.t));
  await page.locator(`#replay-phone [data-open-event="${decision.event}"]`).first().click();
  const panel = page.locator('#replay-evidence');
  await panel.locator(`[data-decision="${decision.id}"]`).click();
  await expect(panel).toContainText(decision.reason);
  for (const step of investigation.steps) {
    const details = panel.locator(`[data-evidence-id="${step.evidence_id}"]`);
    await details.locator('summary').click();
    await expect(details).toHaveAttribute('open', '');
    await expect(details).toContainText(step.tool);
    await expect(details).toContainText(typeof step.result === 'string' ? step.result : JSON.stringify(step.result));
  }
  await expect(panel).toContainText(/Illustration/i);
  await expect(panel.locator('img').first()).toHaveAttribute('src', /\/scenes\//);
  await page.screenshot({ path: testInfo.outputPath('replay-desktop.png'), fullPage: true });
});

test('open evidence follows later decisions until a visitor explicitly selects history', async ({ page }) => {
  const trace = await fixture();
  const { sent } = firstEditable(trace);
  const decisions = chronological(trace.decisions.filter((entry) => entry.event === sent.event));
  const latest = decisions.at(-1);
  await open(page, trace);
  await seek(page, sent.t);
  await page.locator(`#replay-phone [data-open-event="${sent.event}"]`).click();
  await expect(page.locator('#replay-evidence')).toContainText(decisions[0].reason);
  await seek(page, latest.t);
  await expect(page.locator('#replay-evidence')).toContainText(latest.reason);
  await expect(page.locator(`[data-decision="${latest.id}"]`)).toHaveAttribute('aria-pressed', 'true');
  await page.locator(`[data-decision="${decisions[0].id}"]`).click();
  await seek(page, trace.meta.window.seconds);
  await expect(page.locator(`[data-decision="${decisions[0].id}"]`)).toHaveAttribute('aria-pressed', 'true');
});

test('metrics use the supplied values and labels and ignore new fields', async ({ page }) => {
  const trace = await fixture();
  trace.version = '1.17';
  trace.future_section = { unsafe_copy: 'DO NOT RENDER THIS FUTURE FIELD' };
  trace.metrics.counts.events = 12345;
  trace.metrics.counts.future_rate = 777777;
  await open(page, trace);
  const metrics = page.locator('#replay-metrics');
  for (const name of ['counts', 'timings']) {
    await expect(metrics.locator(`[data-metric-block="${name}"]`)).toContainText(trace.metrics[name].label);
  }
  await expect(metrics.locator('[data-metric-block="counts"]')).toContainText('12345');
  await expect(metrics).not.toContainText('777777');
  await expect(page.locator('body')).not.toContainText('DO NOT RENDER THIS FUTURE FIELD');
  await expect(page.locator('body')).toContainText(/one (recorded )?run on one setup/i);
});

test('digest spot checks and chat tool evidence retain recorded answers and proposal outcomes', async ({ page }) => {
  const trace = await fixture();
  const proposal = trace.proposals.find((entry) => entry.decided);
  await open(page, trace);
  await seek(page, proposal.t);
  await expect(page.locator('#replay-conversations')).toContainText(proposal.summary);
  await expect(page.locator('#replay-conversations')).not.toContainText(proposal.result);
  await seek(page, trace.meta.window.seconds);
  for (const conversation of trace.conversations) {
    for (const turn of conversation.turns) {
      await expect(page.locator('#replay-conversations')).toContainText(turn.question);
      await expect(page.locator('#replay-conversations')).toContainText(turn.answer);
    }
  }
  await expect(page.locator('#replay-conversations')).toContainText(proposal.result);
  const tool = trace.conversations.flatMap((entry) => entry.turns).flatMap((turn) => turn.tools)[0];
  const toolGroup = page.locator('#replay-conversations .chat-tools').filter({ hasText: tool.tool }).first();
  await toolGroup.locator(':scope > summary').click();
  const toolDetails = toolGroup.locator('details').filter({ hasText: tool.tool }).first();
  await toolDetails.locator(':scope > summary').click();
  await expect(toolDetails).toHaveAttribute('open', '');
  await expect(toolDetails).toContainText(tool.result);
  for (const digest of trace.digests.filter((entry) => entry.sent)) {
    for (const event of digest.spot_checks) await expect(page.locator('#replay-digests')).toContainText(event);
  }
  const answer = trace.owner_actions.find((entry) => entry.outcome?.spot_check && entry.outcome?.footer);
  await expect(page.locator('#replay-digests')).toContainText(answer.outcome.footer);
  await seek(page, proposal.t);
  await expect(page.locator('#replay-conversations')).not.toContainText(proposal.result);
});

test('English and Chinese controls preserve trace text, time and language preference', async ({ page }) => {
  const trace = await fixture();
  const { sent } = firstEditable(trace);
  await open(page, trace);
  await seek(page, sent.t);
  await expect(page.locator('#replay-play')).toHaveAccessibleName(/Play/);
  await page.locator('#language').selectOption('zh-CN');
  await expect(page.locator('html')).toHaveAttribute('lang', 'zh-CN');
  await expect(page.locator('#replay-play')).toHaveAccessibleName(/播放/);
  await expect(page.locator('#replay-step')).toHaveAccessibleName(/[\u3400-\u9fff]/);
  await expect(page.locator('#replay-clock')).toHaveValue(String(sent.t));
  await expect(page.locator(`#replay-phone [data-message="${sent.message}"]`)).toContainText(plain(sent.text));
  await page.reload();
  await expect(page.locator('#language')).toHaveValue('zh-CN');
  await expect(page.locator('#replay-play')).toHaveAccessibleName(/播放/);
  await page.locator('#language').selectOption('en');
  await expect(page.locator('#replay-play')).toHaveAccessibleName(/Play/);
});

test('phone width, keyboard controls and reduced motion remain usable in both languages', async ({ page }, testInfo) => {
  const trace = await fixture();
  await page.setViewportSize({ width: 375, height: 812 });
  await open(page, trace);
  for (const language of ['en', 'zh-CN']) {
    await page.locator('#language').selectOption(language);
    await seek(page, 0);
    await page.locator('#replay-step').focus();
    await page.keyboard.press('Enter');
    expect(Number(await page.locator('#replay-clock').inputValue())).toBeGreaterThan(0);
    await expect(page.locator('#replay-step')).toBeFocused();
    await page.locator('#replay-clock').focus();
    await page.keyboard.press('End');
    await expect(page.locator('#replay-clock')).toHaveValue(String(trace.meta.window.seconds));
    expect(await page.evaluate(() => matchMedia('(prefers-reduced-motion: reduce)').matches)).toBe(true);
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    await expect(page.locator('#replay-phone')).toBeVisible();
    const phone = await page.locator('#replay-phone').boundingBox();
    expect(phone.width).toBeLessThanOrEqual(375);
    await page.screenshot({ path: testInfo.outputPath(`replay-phone-${language}.png`), fullPage: true });
  }
});

test('Telegram HTML is whitelisted and unsafe frame paths cannot request arbitrary assets', async ({ page }) => {
  const trace = await fixture();
  const message = chronological(trace.messages).find((entry) => entry.action === 'sent');
  message.text = '<b>Safe bold</b>\n<i>Safe italic</i> <code>&lt;camera&gt;</code> <a href="https://example.com/guide">Safe link</a><a href="javascript:window.__unsafeReplay=1">Unsafe link</a><img src="/unsafe-asset" onerror="window.__unsafeReplay=1"><script>window.__unsafeReplay=1</script><svg onload="window.__unsafeReplay=1"></svg>';
  message.frame = 'f-unsafe';
  message.has_photo = true;
  trace.meta.privacy.frames = 'blurred';
  trace.frames.push({ id: 'f-unsafe', event: message.event, file: '../unsafe-asset.jpg', width: 960, height: 540, privacy: 'blurred' });
  const unsafeRequests = [];
  page.on('request', (request) => { if (request.url().includes('unsafe-asset')) unsafeRequests.push(request.url()); });
  await open(page, trace);
  await seek(page, message.t);
  const card = page.locator(`#replay-phone [data-message="${message.message}"]`);
  await expect(card.locator('b')).toContainText('Safe bold');
  await expect(card.locator('i')).toContainText('Safe italic');
  await expect(card.locator('code')).toContainText('<camera>');
  await expect(card.getByRole('link', { name: 'Safe link', exact: true })).toHaveAttribute('href', 'https://example.com/guide');
  await expect(card.locator('script, svg, [onerror], [onload], a[href^="javascript:"]')).toHaveCount(0);
  expect(await page.evaluate(() => window.__unsafeReplay)).toBeUndefined();
  expect(unsafeRequests).toEqual([]);
});

for (const invalid of ['format', 'version', 'private frames']) {
  test(`unsupported ${invalid} fails visibly without substituting a fixture`, async ({ page }) => {
    const trace = await fixture();
    if (invalid === 'format') trace.format = 'another-format';
    if (invalid === 'version') trace.version = '2.0';
    if (invalid === 'private frames') trace.meta.privacy.frames = 'full';
    let fixtureRequests = 0;
    page.on('request', (request) => { if (request.url().endsWith('/data/replay-fixture/trace.json')) fixtureRequests++; });
    await page.route(traceURL, (route) => route.fulfill({ json: trace }));
    await page.goto('/replay.html');
    await expect(page.locator('#replay-error')).toBeVisible();
    expect(fixtureRequests).toBe(0);
    await expect(page.locator('#replay-phone [data-message]')).toHaveCount(0);
  });
}

test('a missing reviewed run uses the explicitly invented, photo-free fixture', async ({ page }) => {
  let reviewedExists = false;
  try { await access(new URL('../public/data/run-2026-10-30/trace.json', import.meta.url)); reviewedExists = true; } catch { /* Missing is the scenario under test. */ }
  test.skip(reviewedExists, 'The owner has supplied a reviewed trace; real-file fallback no longer applies.');
  const missing = await page.request.get('/data/run-2026-10-30/trace.json');
  expect(missing.status(), 'Missing static JSON must not become the HTML entry page').toBe(404);
  await page.goto('/replay.html?lang=en');
  await expect(page.locator('#replay-clock')).toBeVisible();
  await expect(page.locator('body')).toContainText(/invented fixture/i);
  await expect(page.locator('body')).toContainText(/no photos/i);
});
