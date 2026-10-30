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

test('switching language preserves story, conversation, draft, and open evidence', async ({ page }) => {
  await page.getByRole('link', { name: 'See it in action' }).click();
  await correctCleaner(page);
  await page.getByRole('button', { name: 'See the next matching visit' }).click();
  await tab(page, 'Ask Watchwise').click();
  await page.getByRole('button', { name: 'Why did you alert me?' }).click();
  await page.locator('#question').fill('还有哪些证据？');
  await page.locator('#language').selectOption('zh-CN');
  await expect(page.locator('html')).toHaveAttribute('lang', 'zh-CN');
  await expect(page).toHaveURL(/\?lang=zh-CN#experience$/);
  await expect(page.locator('#language')).toHaveAccessibleName('语言');
  await expect(page.getByRole('heading', { level: 1 })).toContainText('配备 AI 守卫。');
  await expect(page).toHaveTitle('Watchwise — 为现有摄像头配备 AI 守卫');
  await expect(page.locator('meta[name="description"]')).toHaveAttribute('content', /默认本地运行/);
  await expectCounts(page, 5, 3, 1);
  await expect(page.locator('[data-event="return"]')).toHaveAttribute('aria-pressed', 'true');
  await expect(page.getByRole('tab', { name: '询问 Watchwise' })).toHaveAttribute('aria-selected', 'true');
  await expect(page.locator('#conversation .user-message p')).toHaveText('Why did you alert me?');
  await expect(page.locator('#conversation .assistant-message').last()).toContainText('信任度 0.75，相似度 0.95');
  await expect(page.locator('#question')).toHaveValue('还有哪些证据？');
  await expect(page.locator('.installation-copy .button')).toHaveAttribute('href', /README.zh-CN.md#试用路由模拟/);
  await page.locator('#language').selectOption('en');
  await expectCounts(page, 5, 3, 1);
  await expect(page.locator('#question')).toHaveValue('还有哪些证据？');
  await expect(page.locator('#conversation .assistant-message').last()).toContainText('3 corrections, trust 0.75, and similarity 0.95');
  await expect(page.locator('.installation-copy .button')).toHaveAttribute('href', /watchwise#try-the-routing-simulation/);
  await tab(page, 'Evidence').click();
  await page.locator('.evidence-timeline summary').nth(1).click();
  await page.locator('#language').selectOption('zh-CN');
  await expect(page.getByRole('tab', { name: /^证据/ })).toHaveAttribute('aria-selected', 'true');
  await expect(page.locator('.evidence-timeline details').nth(1)).toHaveAttribute('open', '');
  await expect(page.locator('.evidence-timeline details').nth(1)).toContainText('你的纠正带来了变化');
});

test('Chinese events, evidence, questions, corrections, undo, and reset stay coherent', async ({ page }, testInfo) => {
  await page.locator('#language').selectOption('zh-CN');
  const cases = [
    ['headlights', '车道', '只是路过。', '20 次纠正', 3],
    ['courier', '正门', '有人送来了包裹。', '并未识别快递员身份', 3],
    ['cleaner', '侧门', '你知道是熟人，它还不确定。', '只有两次纠正', 4],
    ['visitor', '侧门', '有人在侧门逗留。', '有人在后门附近逗留', 4],
  ];
  for (const [id, camera, title, answer, count] of cases) {
    await selectEvent(page, id);
    await expect(page.locator('#camera-name')).toHaveText(camera);
    await expect(page.locator('#camera-title')).toHaveText(title);
    await expect(page.locator('#scene')).toHaveAttribute('alt', /^插画：/);
    await page.getByRole('tab', { name: /^证据/ }).click();
    await expect(page.locator('.evidence-timeline li')).toHaveCount(count);
    await page.getByRole('tab', { name: '询问 Watchwise' }).click();
    await page.getByRole('button', { name: '为什么提醒我？' }).click();
    await expect(page.locator('#conversation .assistant-message').last()).toContainText(answer);
  }
  await page.getByRole('button', { name: '你依据了哪些证据？' }).click();
  await expect(page.locator('#conversation .assistant-message').last()).toContainText('置信度为 0.87');
  for (const [question, answer] of [
    ['你过滤了哪些事件？', '保洁员来访仍需要你的说明'],
    ['记忆如何学习？', '低于默认阈值 3'],
    ['如何使用安装后的聊天？', '只有点击 ✅ Confirm'],
    ['我的数据会发到云端吗？', 'Telegram 还会中转'],
    ['帮我点披萨', '仅限于这个插画事件记录'],
  ]) {
    await page.locator('#question').fill(question);
    await page.locator('#question').press('Enter');
    await expect(page.locator('#conversation .assistant-message').last()).toContainText(answer);
  }
  await page.getByRole('button', { name: '教会 Watchwise' }).click();
  await page.getByRole('button', { name: '这是预期到访的保洁员' }).click();
  await expect(page.locator('.correction-box')).toContainText('纠正次数：2 → 3');
  await page.getByRole('button', { name: '查看下次匹配的来访' }).click();
  await expectCounts(page, 5, 3, 1);
  await page.getByRole('tab', { name: '询问 Watchwise' }).click();
  await page.getByRole('button', { name: '你过滤了哪些事件？' }).click();
  await expect(page.locator('#conversation .assistant-message').last()).toContainText('后续匹配的来访也由记忆解决');
  await page.locator('#question').fill('记忆如何学习？');
  await page.locator('#question').press('Enter');
  await expect(page.locator('#conversation .assistant-message').last()).toContainText('默认在 30 天内否决');
  await selectEvent(page, 'cleaner');
  await page.getByRole('tab', { name: '决策', exact: true }).click();
  await page.getByRole('button', { name: '撤销我的纠正' }).click();
  await expectCounts(page, 4, 2, 2);
  await page.getByRole('button', { name: '这是预期到访的保洁员' }).click();
  await page.getByRole('button', { name: '重置场景' }).click();
  await expectCounts(page, 4, 2, 2);
  await expect(page.locator('#camera-title')).toHaveText('有人在侧门逗留。');
  await page.screenshot({ path: testInfo.outputPath('chinese-desktop.png'), fullPage: true });
});

test('language preference persists, URLs take priority, and invalid values fall back', async ({ page, context }) => {
  await page.locator('#language').selectOption('zh-CN');
  await page.goto('/#experience');
  await expect(page.locator('html')).toHaveAttribute('lang', 'zh-CN');
  await context.grantPermissions(['clipboard-read', 'clipboard-write']);
  await page.getByRole('button', { name: '复制设置命令' }).click();
  await expect(page.locator('#copy-status')).toHaveText('设置命令已复制到剪贴板。');
  expect(await page.evaluate(() => navigator.clipboard.readText())).toBe('watchwise doctor\nwatchwise init\nwatchwise check\nwatchwise serve');
  await page.locator('#language').selectOption('en');
  await expect(page.locator('#copy-status')).toHaveText('Setup commands copied to clipboard.');
  await page.goto('/?lang=zh-CN#experience');
  await expect(page.locator('html')).toHaveAttribute('lang', 'zh-CN');
  await page.goto('/?lang=en#experience');
  await expect(page.locator('html')).toHaveAttribute('lang', 'en');
  await page.evaluate(() => localStorage.setItem('watchwise.language', 'unsupported'));
  await page.goto('/?lang=unsupported');
  await expect(page.locator('html')).toHaveAttribute('lang', 'en');
});

test('keyboard language selection works with storage and clipboard unavailable', async ({ page }) => {
  await page.addInitScript(() => {
    Storage.prototype.getItem = () => { throw new Error('Storage disabled'); };
    Storage.prototype.setItem = () => { throw new Error('Storage disabled'); };
  });
  await page.goto('/?lang=zh-CN#experience');
  await expect(page.locator('html')).toHaveAttribute('lang', 'zh-CN');
  await page.locator('#language').focus();
  expect(await page.locator('#language').evaluate(element => getComputedStyle(element).outlineStyle)).toBe('solid');
  await page.keyboard.press('ArrowUp');
  await page.keyboard.press('Enter');
  await expect(page.locator('html')).toHaveAttribute('lang', 'en');
  await expect(page.locator('#language')).toBeFocused();
  await page.keyboard.press('ArrowDown');
  await page.keyboard.press('Enter');
  await expect(page.locator('html')).toHaveAttribute('lang', 'zh-CN');
  await expect(page.locator('#language')).toBeFocused();
  await page.evaluate(() => Object.defineProperty(navigator.clipboard, 'writeText', { value: async () => { throw new Error('Clipboard disabled'); } }));
  await page.getByRole('button', { name: '复制设置命令' }).click();
  await expect(page.locator('#copy-status')).toContainText('已选中命令');
  expect(await page.evaluate(() => getSelection().toString())).toContain('watchwise serve');
  await page.getByRole('tab', { name: '询问 Watchwise' }).click();
  await page.locator('#question').fill('<img src=x onerror="window.__injected=true"> 你依据了哪些证据？');
  await page.locator('#question').press('Enter');
  await expect(page.locator('#conversation img')).toHaveCount(0);
  expect(await page.evaluate(() => window.__injected)).toBeUndefined();
});

test('Chinese page translates prose and accessibility labels, preserving technical names', async ({ page }) => {
  await page.locator('#language').selectOption('zh-CN');
  const untranslated = await page.evaluate(() => {
    const allowed = new Set(['watchwise', 'GitHub', 'GitHub ↗', 'CPU', 'GPU', 'nvidia-16gb', 'qwen3-vl:8b-instruct', 'watchwise doctor', 'watchwise init', 'watchwise check', 'watchwise serve']);
    const result = [];
    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
    while (walker.nextNode()) {
      const node = walker.currentNode;
      if (node.parentElement.closest('script, noscript, option')) continue;
      const value = node.textContent.trim().replace(/\s+/g, ' ');
      if (/[a-z]/i.test(value) && !/\p{Script=Han}/u.test(value) && !allowed.has(value)) result.push(value);
    }
    for (const element of document.querySelectorAll('[aria-label], [placeholder]')) {
      for (const name of ['aria-label', 'placeholder']) {
        const value = element.getAttribute(name);
        if (value && /[a-z]/i.test(value) && !/\p{Script=Han}/u.test(value)) result.push(`${name}: ${value}`);
      }
    }
    return result;
  });
  expect(untranslated).toEqual([]);
});

for (const width of [320, 375, 390, 768]) {
  test(`Chinese layout and language switch fit ${width}px`, async ({ page }, testInfo) => {
    await page.setViewportSize({ width, height: 844 });
    await page.locator('#language').selectOption('zh-CN');
    await expect(page.locator('.scenario-label')).toBeVisible();
    await expect(page.locator('.scenario-label')).toHaveText('插画模拟场景');
    const noOverflow = async () => expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(width);
    await noOverflow();
    const languageBox = await page.locator('#language').boundingBox();
    expect(languageBox.x).toBeGreaterThanOrEqual(0);
    expect(languageBox.x + languageBox.width).toBeLessThanOrEqual(width);
    await page.locator('.verification-limits summary').click();
    await expect(page.locator('.verification-limits')).toContainText('一个人脸库，所有摄像头共用');
    await noOverflow();
    await page.locator('.verification-limits summary').click();
    await page.getByRole('button', { name: '教会 Watchwise' }).click();
    await page.getByRole('button', { name: '这是预期到访的保洁员' }).click();
    await page.getByRole('button', { name: '查看下次匹配的来访' }).click();
    await page.getByRole('tab', { name: /^证据/ }).click();
    await noOverflow();
    await page.getByRole('tab', { name: '询问 Watchwise' }).click();
    await page.getByRole('button', { name: '为什么提醒我？' }).click();
    await expect(page.locator('#conversation .assistant-message').last()).toContainText('我在关卡解决了这次后续事件');
    await noOverflow();
    await page.locator('#language').selectOption('en');
    await expectCounts(page, 5, 3, 1);
    await noOverflow();
    await page.locator('#language').selectOption('zh-CN');
    await page.getByRole('button', { name: '重置场景' }).click();
    if (width === 390) {
      await page.evaluate(() => window.scrollTo(0, 0));
      await page.screenshot({ path: testInfo.outputPath('chinese-mobile.png'), fullPage: true });
    }
  });
}
