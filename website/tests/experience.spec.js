import { test as base, expect } from '@playwright/test';

// Every story test also checks the production bundle and local scene assets.
const test = base.extend({
  page: async ({ page }, use) => {
    const failures = [];
    page.on('pageerror', (error) => failures.push(`Page: ${error.message}`));
    page.on('console', (message) => {
      if (message.type() === 'error') failures.push(`Console: ${message.text()}`);
    });
    page.on('requestfailed', (request) => failures.push(`Request: ${request.url()} ${request.failure()?.errorText}`));
    page.on('response', (response) => {
      if (response.status() >= 400) failures.push(`HTTP ${response.status()}: ${response.url()}`);
    });
    await page.goto('/');
    await expect(page.locator('#event-list button')).toHaveCount(4);
    await use(page);
    expect(failures, 'No browser errors or failed asset requests').toEqual([]);
  },
});

const selectEvent = (page, id) => page.locator(`[data-event="${id}"]`).click();
const tab = (page, name) => page.getByRole('tab', { name, exact: name !== 'Evidence' });

async function expectCounts(page, total, quiet, review) {
  await expect(page.locator('#event-total')).toHaveText(String(total));
  await expect(page.locator('#quiet-total')).toHaveText(String(quiet));
  await expect(page.locator('#review-total')).toHaveText(String(review));
}

async function correctCleaner(page) {
  await selectEvent(page, 'cleaner');
  await tab(page, 'Decision').click();
  await page.getByRole('button', { name: 'This is the expected cleaner' }).click();
}

test('each event keeps its scene, decision, evidence, and conversation coherent', async ({ page }, testInfo) => {
  await expect(page.getByRole('heading', { level: 1 })).toHaveText('An AI guard for the cameras you already have.', { useInnerText: true });
  await expect(page.locator('.scenario-label')).toHaveText('Illustrated scenario');
  await expectCounts(page, 4, 2, 2);

  const cases = [
    { id: 'headlights', camera: 'Driveway', headline: 'Just passing through.', decision: 'Headlights. Nothing more.', evidence: 'A familiar signature', answer: '20 corrections', count: 3 },
    { id: 'courier', camera: 'Front entry', headline: 'A delivery at the door.', decision: 'A parcel at the door.', evidence: 'Review received', answer: 'does not identify the courier', count: 3 },
    { id: 'cleaner', camera: 'Side entry', headline: 'Expected by you. Unclear to Watchwise.', decision: 'A familiar visit, still uncertain.', evidence: 'Camera health checked', answer: 'only two corrections', count: 4 },
    { id: 'visitor', camera: 'Side entry', headline: 'Someone at the side gate.', decision: 'Worth your attention.', evidence: 'Camera health checked', answer: 'continued presence', count: 4 },
  ];

  for (const item of cases) {
    await selectEvent(page, item.id);
    await expect(page.locator('[data-event][aria-pressed="true"]')).toHaveCount(1);
    await expect(page.locator(`[data-event="${item.id}"]`)).toHaveAttribute('aria-pressed', 'true');
    await expect(page.locator('#scene')).toHaveAttribute('src', `/scenes/${item.id}.svg`);
    await expect(page.locator('#scene')).toHaveAttribute('alt', /Illustrated/);
    await expect(page.locator('#camera-name')).toHaveText(item.camera);
    await expect(page.locator('#camera-title')).toHaveText(item.headline);
    await tab(page, 'Decision').click();
    await expect(page.locator('.decision-title')).toHaveText(item.decision);
    await page.getByRole('button', { name: 'Follow the evidence' }).click();
    await expect(tab(page, 'Evidence')).toHaveAttribute('aria-selected', 'true');
    await expect(page.locator('.evidence-timeline li')).toHaveCount(item.count);
    await expect(page.locator('.evidence-timeline summary').first()).toContainText(item.evidence);
    await expect(page.locator('.evidence-heading')).toContainText(item.camera);
    await page.locator('.evidence-timeline summary').last().click();
    await expect(page.locator('.evidence-timeline details').last()).toHaveAttribute('open', '');
    await tab(page, 'Ask Watchwise').click();
    await expect(page.locator('#conversation')).toContainText(item.answer);
    await expect(page.locator('#conversation .chat-message')).toHaveCount(1);
  }

  await tab(page, 'Decision').click();
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.screenshot({ path: testInfo.outputPath('desktop.png'), fullPage: true });
});

test('human correction changes the later matching event; undo and reset restore the original story', async ({ page }) => {
  await correctCleaner(page);
  await expect(page.locator('.decision-title')).toHaveText('Your context is now part of the record.');
  await expect(page.locator('.correction-box')).toContainText('2 → 3 corrections');
  await expect(page.locator('.correction-box')).toContainText('0.67 → 0.75');
  await expectCounts(page, 4, 2, 1);
  await tab(page, 'Evidence').click();
  await expect(page.locator('.evidence-timeline li')).toHaveCount(5);
  await expect(page.locator('.evidence-timeline li').last()).toContainText('Human correction recorded');
  await tab(page, 'Decision').click();
  await page.getByRole('button', { name: 'See the next matching visit' }).click();
  await expect(page.locator('#event-list button')).toHaveCount(5);
  await expect(page.locator('#camera-time')).toContainText('NEXT THU');
  await expect(page.locator('.decision-title')).toHaveText('A lesson put to work.');
  await expect(page.locator('.route')).toContainText('Resolved by learned memory');
  await expect(page.locator('.memory-note')).toContainText('3 reviews');
  await expectCounts(page, 5, 3, 1);
  await tab(page, 'Ask Watchwise').click();
  await expect(page.locator('#conversation')).toContainText('3 corrections, trust 0.75, and similarity 0.95');
  await page.getByRole('button', { name: 'What did you filter out?' }).click();
  await expect(page.locator('#conversation .assistant-message').last()).toContainText('later matching visit was also resolved by memory');

  await selectEvent(page, 'cleaner');
  await tab(page, 'Decision').click();
  await page.getByRole('button', { name: 'Undo my correction' }).click();
  await expect(page.locator('#event-list button')).toHaveCount(4);
  await expect(page.locator('.decision-title')).toHaveText('A familiar visit, still uncertain.');
  await expectCounts(page, 4, 2, 2);

  await correctCleaner(page);
  await page.getByRole('button', { name: 'See the next matching visit' }).click();
  await page.getByRole('button', { name: 'Reset story' }).click();
  await expect(page.locator('#event-list button')).toHaveCount(4);
  await expect(page.locator('[data-event="visitor"]')).toHaveAttribute('aria-pressed', 'true');
  await expect(page.locator('.decision-title')).toHaveText('Worth your attention.');
  await expectCounts(page, 4, 2, 2);
  await tab(page, 'Ask Watchwise').click();
  await expect(page.locator('#conversation .chat-message')).toHaveCount(1);
});

test('keyboard tabs, event selection, and teaching shortcut preserve usable focus', async ({ page }) => {
  await tab(page, 'Decision').focus();
  await page.keyboard.press('ArrowRight');
  await expect(tab(page, 'Evidence')).toBeFocused();
  await expect(page.locator('#panel-evidence')).toBeVisible();
  await expect(page.locator('#panel-decision')).toBeHidden();
  await page.keyboard.press('ArrowRight');
  await expect(tab(page, 'Ask Watchwise')).toBeFocused();
  await page.keyboard.press('ArrowRight');
  await expect(tab(page, 'Decision')).toBeFocused();
  await page.keyboard.press('ArrowLeft');
  await expect(tab(page, 'Ask Watchwise')).toBeFocused();
  await page.keyboard.press('Home');
  await expect(tab(page, 'Decision')).toBeFocused();
  await page.keyboard.press('End');
  await expect(tab(page, 'Ask Watchwise')).toBeFocused();
  await expect(page.locator('[role="tab"][tabindex="0"]')).toHaveCount(1);

  await page.locator('[data-event="headlights"]').focus();
  await page.keyboard.press('Enter');
  await expect(page.locator('[data-event="headlights"]')).toBeFocused();
  await expect(page.locator('#camera-title')).toHaveText('Just passing through.');
  await page.getByRole('button', { name: 'Teach Watchwise' }).click();
  await expect(page.locator('[data-event="cleaner"]')).toHaveAttribute('aria-pressed', 'true');
  await expect(page.getByRole('button', { name: 'This is the expected cleaner' })).toBeFocused();
  await page.keyboard.press('Enter');
  await expect(page.getByRole('button', { name: 'See the next matching visit' })).toBeFocused();
});

test('suggested and typed questions use the selected record and keep event-specific history', async ({ page }) => {
  await tab(page, 'Ask Watchwise').click();
  await page.getByRole('button', { name: 'Why did you alert me?' }).click();
  await expect(page.locator('#conversation .user-message').last()).toHaveText('YOUWhy did you alert me?');
  await expect(page.locator('#conversation .assistant-message').last()).toContainText('person remains near the rear gate');
  await page.getByRole('button', { name: 'What did you filter out?' }).click();
  await expect(page.locator('#conversation .assistant-message').last()).toContainText('courier by vision screening');
  await page.getByRole('button', { name: 'What evidence did you use?' }).click();
  await expect(page.locator('#conversation .assistant-message').last()).toContainText('An unexpected visitor:');
  await expect(page.locator('#conversation .assistant-message').last()).toContainText('0.87');

  await selectEvent(page, 'courier');
  await expect(page.locator('#conversation .chat-message')).toHaveCount(1);
  await page.getByRole('textbox', { name: 'Ask about the selected event' }).fill('Explain this decision');
  await page.getByRole('button', { name: 'Send question' }).click();
  await expect(page.locator('#conversation .assistant-message').last()).toContainText('carrying a parcel');
  await expect(page.locator('#question')).toHaveValue('');
  await expect(page.locator('#question')).toBeFocused();
  await selectEvent(page, 'visitor');
  await expect(page.locator('#conversation .user-message')).toHaveCount(3);
  await expect(page.locator('#conversation')).not.toContainText('Explain this decision');

  await page.locator('#question').fill('Can you order a pizza?');
  await page.locator('#question').press('Enter');
  await expect(page.locator('#conversation .assistant-message').last()).toContainText('limited to this illustrated event record');
});

test('replay refreshes answers across events and preserves learning safeguards', async ({ page }) => {
  await correctCleaner(page);
  await tab(page, 'Ask Watchwise').click();
  await page.getByRole('button', { name: 'What did you filter out?' }).click();
  await expect(page.locator('#conversation .assistant-message').last()).not.toContainText('later matching visit');
  await tab(page, 'Decision').click();
  await page.getByRole('button', { name: 'See the next matching visit' }).click();
  await selectEvent(page, 'cleaner');
  await tab(page, 'Ask Watchwise').click();
  await expect(page.locator('#conversation .chat-message')).toHaveCount(1);
  await page.getByRole('button', { name: 'What did you filter out?' }).click();
  await expect(page.locator('#conversation .assistant-message').last()).toContainText('later matching visit was also resolved by memory');
  await page.locator('#question').fill('Does memory recognize the cleaner?');
  await page.locator('#question').press('Enter');
  await expect(page.locator('#conversation .assistant-message').last()).toContainText('another person or behavior can match');
  await expect(page.locator('#conversation .assistant-message').last()).toContainText('vetoes memory hold-backs for that signature');
  await page.locator('#question').fill('Can I use installed chat?');
  await page.locator('#question').press('Enter');
  await expect(page.locator('#conversation .assistant-message').last()).toContainText('Ask Watchwise works in Telegram');
  await expect(page.locator('#conversation .assistant-message').last()).toContainText('duty-roster additions or removals');
  await expect(page.locator('#conversation .assistant-message').last()).toContainText('Only tapping ✅ Confirm applies');
  await expect(page.locator('#conversation .assistant-message').last()).toContainText('typing “confirm”, “yes” or “ok” applies nothing');
  await expect(page.locator('#conversation .assistant-message').last()).toContainText('After 6 quiet hours');
});

test('installed Telegram and camera setup are separate from the simulated story', async ({ page }) => {
  await page.getByRole('link', { name: 'Explore Watchwise on your phone' }).click();
  await expect(page).toHaveURL(/#live-verification$/);
  await expect(page.locator('.verification-record')).toContainText('Add or remove duty-roster rules');
  await expect(page.locator('.verification-record')).toContainText('Each chat proposal needs ✅ Confirm');
  await expect(page.getByRole('link', { name: 'Read the verification records' })).toHaveAttribute('href', 'https://github.com/PAOPAOtomato/watchwise/blob/main/docs/verification.md');
  await page.locator('.verification-limits summary').focus();
  await page.keyboard.press('Enter');
  await expect(page.locator('.verification-limit-grid')).toBeVisible();
  await expect(page.locator('.verification-limits')).toContainText('The screener receives the arming state');
  await expect(page.locator('.verification-limits')).toContainText('A roster rule does not clear the person by itself');
  await expect(page.locator('.verification-limits')).toContainText('Clip opens Frigate’s recording page');
  await expect(page.locator('.verification-limits')).toContainText('Whole-event summaries can raise an earlier decision, never lower it');
  await expect(page.locator('.verification-limits')).toContainText('nvidia-16gb profile uses one qwen3-vl:8b-instruct model');
  await expect(page.locator('.verification-limits')).toContainText('write to Frigate only after ✅ Add');
  await expect(page.locator('.verification-limits')).toContainText('send previews through Telegram when images are enabled');
  await expect(page.locator('.verification-limits summary')).toBeFocused();
});

test('reduced motion and clipboard-denied fallback remain usable', async ({ page }) => {
  expect(await page.evaluate(() => matchMedia('(prefers-reduced-motion: reduce)').matches)).toBe(true);
  expect(await page.locator('html').evaluate((element) => getComputedStyle(element).scrollBehavior)).toBe('auto');
  await tab(page, 'Decision').focus();
  expect(await tab(page, 'Decision').evaluate((element) => getComputedStyle(element).outlineStyle)).toBe('solid');
  await page.evaluate(() => Object.defineProperty(navigator.clipboard, 'writeText', { value: async () => { throw new Error('Clipboard denied'); } }));
  await page.getByRole('button', { name: 'Copy setup commands' }).click();
  await expect(page.locator('#copy-status')).toContainText('Commands selected');
  expect(await page.evaluate(() => getSelection().toString())).toContain('watchwise serve');
});

test('typed markup is displayed as text and cannot inject elements or execute scripts', async ({ page }) => {
  await tab(page, 'Ask Watchwise').click();
  const payload = '<img src=x onerror="window.__injected=true"> What evidence did you use?';
  await page.locator('#question').fill(payload);
  await page.locator('#question').press('Enter');
  await expect(page.locator('#conversation .user-message p')).toHaveText(payload);
  await expect(page.locator('#conversation img')).toHaveCount(0);
  expect(await page.evaluate(() => window.__injected)).toBeUndefined();
  await expect(page.locator('#conversation .assistant-message').last()).toContainText('An unexpected visitor:');
});

test('copy commands and internal navigation work, with repository links correctly targeted', async ({ page, context }) => {
  await context.grantPermissions(['clipboard-read', 'clipboard-write']);
  await page.getByRole('button', { name: 'Copy setup commands' }).click();
  await expect(page.locator('#copy-status')).toHaveText('Setup commands copied to clipboard.');
  const copied = await page.evaluate(() => navigator.clipboard.readText());
  expect(copied.replace(/\r\n/g, '\n')).toBe('watchwise doctor\nwatchwise init\nwatchwise check\nwatchwise serve');
  await page.getByRole('link', { name: 'See it in action' }).click();
  await expect(page).toHaveURL(/#experience$/);
  await page.getByRole('link', { name: 'How it works', exact: true }).click();
  await expect(page).toHaveURL(/#how-it-works$/);
  const brokenAnchors = await page.locator('a[href^="#"]').evaluateAll((links) => links.map((link) => link.getAttribute('href')).filter((href) => href !== '#' && !document.querySelector(href)));
  expect(brokenAnchors).toEqual([]);
  await expect(page.getByRole('link', { name: 'View on GitHub' })).toHaveAttribute('href', 'https://github.com/PAOPAOtomato/watchwise');
  await expect(page.getByRole('link', { name: 'Get started on GitHub' })).toHaveAttribute('href', /^https:\/\/github\.com\/PAOPAOtomato\/watchwise#/);
});

for (const width of [320, 375, 390, 768]) {
  test(`story remains functional without horizontal overflow at ${width}px`, async ({ page }, testInfo) => {
    await page.setViewportSize({ width, height: 844 });
    await expect(page.getByRole('heading', { level: 1 })).toHaveText('An AI guard for the cameras you already have.', { useInnerText: true });
    const expectNoOverflow = async () => {
      expect(await page.evaluate(() => document.documentElement.scrollWidth), 'Page width should fit the viewport').toBeLessThanOrEqual(width);
      await expect(page.locator('#scene')).toBeVisible();
    };
    await expectNoOverflow();
    await page.locator('.verification-limits summary').click();
    await expect(page.locator('.verification-limit-grid')).toBeVisible();
    await expectNoOverflow();
    await page.locator('.verification-limits summary').click();
    await correctCleaner(page);
    await expectNoOverflow();
    await page.getByRole('button', { name: 'See the next matching visit' }).click();
    await expectNoOverflow();
    await tab(page, 'Evidence').click();
    await expect(page.locator('#panel-evidence')).toBeVisible();
    await expectNoOverflow();
    await tab(page, 'Ask Watchwise').click();
    await page.getByRole('button', { name: 'Why did you alert me?' }).click();
    await expect(page.locator('#conversation .assistant-message').last()).toContainText('resolved this later event at the gate');
    await expectNoOverflow();
    await page.getByRole('button', { name: 'Reset story' }).click();
    if (width === 390) {
      await page.evaluate(() => window.scrollTo(0, 0));
      await page.screenshot({ path: testInfo.outputPath('mobile.png'), fullPage: true });
    }
  });
}
