# Watchwise product website

A self-contained product website for [Watchwise](https://github.com/PAOPAOtomato/watchwise), built with Vite, plain JavaScript, CSS, and original SVG camera illustrations. Vite produces static files in `dist/`; public hosting needs no Python process, database, server functions, cameras, model inference, or API keys. Fonts and illustrations are served with the site.

## Work locally

Use Node.js 24, as in CI. From the repository root:

```sh
cd website
npm ci
npm run dev
```

Build and inspect the production output:

```sh
npm run build
npm run preview
```

Install Chromium once, then run the interaction checks:

```sh
npx playwright install chromium
npm test
```

Linux CI uses `npx playwright install --with-deps chromium` to install browser system dependencies too. The [website workflow](../.github/workflows/website.yml) runs the production build and Playwright checks when website files or that workflow change; it does not replace the Python workflow.

## Languages

Visitors can select **English** or **简体中文** in the header. English is the default. A selection is remembered in browser storage when available; `?lang=en` or `?lang=zh-CN` takes priority for a shared link, including links such as `/?lang=zh-CN#experience`. Switching still works when storage is unavailable.

Switching language preserves the selected event, tab, corrections, replay state, open evidence, and question draft. Earlier Watchwise answers are rendered in the selected language, while visitors' original messages remain unchanged. Suggested questions and typed questions understand English and Simplified Chinese keywords, with deterministic responses in the selected interface language. Questions remain in browser memory; only the language preference is stored across visits.

The translation covers page copy, event stories, evidence, answers, controls, accessible names, image descriptions, and page metadata. Architecture and installation links point to the corresponding README language. Commands, configuration keys, model names, and numerical scenario data stay the same. Chinese uses system fonts, so language selection adds no external font or translation requests.

English page copy lives in `index.html` and `src/main.js`. [`src/locales/zh-CN.js`](src/locales/zh-CN.js) maps English UI text to Chinese and supplies event-text overlays and answers; [`src/i18n.js`](src/i18n.js) applies the selected language without replacing the static markup. Update the translation whenever English text changes, including accessible labels. Both languages share the event mechanics in `src/scenario.js`. Run the browser checks for both languages, including persistence, switching during a corrected story, typed questions, keyboard controls, and mobile layouts.

## Structure and source of truth

| File | Purpose |
| --- | --- |
| [`index.html`](index.html) | Product story, navigation, installation copy, and accessible interface structure |
| [`src/main.js`](src/main.js) | Selected event, panels, correction/replay/reset, conversation, and clipboard controls |
| [`src/scenario.js`](src/scenario.js) | Authored event records, evidence, routing explanations, and deterministic question responses |
| [`src/i18n.js`](src/i18n.js) | Language selection, browser preference, URL override, static copy, and localized README links |
| [`src/locales/zh-CN.js`](src/locales/zh-CN.js) | Simplified Chinese page text, event descriptions, and question responses |
| [`src/style.css`](src/style.css) | Responsive layout, focus states, typography, and reduced-motion behavior |
| [`public/scenes/`](public/scenes/) | Original SVG scene illustrations |
| [`vercel.json`](vercel.json) | Static build settings and response headers |

Keep product claims aligned with the [repository README](../README.md), [hardware guide](../docs/hardware.md), [verification notes](../docs/verification.md), [hardware profiles](../watchwise/hw/profiles.yml), [gate implementation](../watchwise/tier2/gate.py), [configuration defaults](../watchwise/config.py), [pattern memory](../watchwise/memory/long_term.py), and [feedback loop](../watchwise/memory/feedback.py). The website does not call these Python modules at runtime.

The camera scenes and their evidence are authored illustrations. Timestamps, confidence values, correction counts, and changing totals are simulated values for this particular interactive story, not live cameras, production activity, accuracy guarantees, or measured performance. Do not turn the README's simulated routing results into real-world benchmarks. The courier story describes an initial snapshot assessment, rather than promising a description of the whole event: at the start of a live review, recording frames may not yet be available.

The story uses Home mode and shows the model-free gate (`gate.memory_mode: free`, the configuration default and the CPU and cloud profiles' setting; the GPU and Apple Silicon profiles let the vision model check learned routines instead). The cleaner starts with an existing activity pattern: two corrections and trust `0.67`. A visitor's correction reinforces that pattern and agrees with its prior assessment, giving three corrections and trust `0.75`. The next matching visit has similarity `0.95`, meeting the current default requirements: at least three corrections, similarity at least `0.86`, and gate trust at least `0.70`. Correction count and trust are separate: trust is the smoothed agreement ratio `(agreements + 1) / (agreements + disagreements + 2)`, not a direct conversion from correction count. The later gate confidence is the rounded product of trust and similarity, `0.71`.

This illustrates camera/label/zone pattern matching. The gate does not recognize identity or distinguish behaviors within the same signature: another person or action can also match. A correction does not edit the duty roster, and the original decision remains in the evidence history. Conflicting reviews lower trust, and in the product a 🚨 Alert me ruling also stops memory from holding back that signature for 30 days (`gate.real_veto_days`); the story does not show that. Age decay affects search ranking, not gate eligibility or pattern expiration. Undo and reset restore the story's initial state. Replay clears prior answers because filtered-event totals have changed. All selections, corrections, and conversations live only in browser memory and reset on page reload. Suggested questions and typed questions select authored responses; no model or chat service receives visitor input. In an installation, questions go to Ask Watchwise in Telegram.

The headlights pattern has 20 corrections, including 18 scored agreements and no disagreements, giving trust `19 / 20 = 0.95`. Combined with similarity `0.95`, this produces confidence `0.90` after rounding. The gate uses the event signature, not the illustrated frame.

The later event is downgraded to a detection, rather than fully suppressed: its `0.71` confidence is below the policy's `0.85` suppression threshold. The courier's benign vision result is also downgraded. Both remain available for audit without a request for human review. See the [decision policy](../watchwise/decision.py).

Typography uses the locally bundled Manrope font under the [SIL Open Font License](public/licenses/manrope.txt). Scene illustrations are original project assets.

## Telegram and camera setup panels

The `#live-verification` panel explains installed Ask Watchwise in Telegram: questions from event records, replies to cards, and proposals to add or remove duty-roster rules, switch arming, mute a camera or room, and set “tell me when…” watches. Only ✅ Confirm applies a chat proposal; typed confirmations apply nothing. After 6 quiet hours, a question starts a fresh conversation; `/new` restarts at any time, and long conversations omit older turns from the model prompt. Browser questions still select authored answers in local browser state and cannot apply installed changes.

The expandable camera-setup panel describes arming-aware screening, roster context that permits a benign investigation without automatically clearing a person, armed-card reasons, the unmatched-face exception during rostered visits, the Alert me memory veto, and raise-only whole-event summaries. Clip opens Frigate's recording page and requires the phone's network access and login. The `nvidia-16gb` profile uses one `qwen3-vl:8b-instruct` model with 16k context, not a pair of models.

Faces use one Frigate Face Library for all cameras. `/face` lists its contents; `/face <name> [camera]` enrolls a current-frame photo, and 👤 on an unknown-face card offers an event crop for training. Both paths send previews through Telegram when images are enabled and write to Frigate only after ✅ Add. Installed questions, answers, and enabled photos also pass through Telegram; a cloud `chat` role receives questions and retrieved text/tool results.

Keep dated live-test records only in [verification notes](../docs/verification.md), distinguishing phone observations from automated tests and code inspection. Use those records and the [chat service](../watchwise/chat/service.py), [chat tools](../watchwise/chat/tools.py), [face enrollment](../watchwise/notify/faces.py), and [decision policy](../watchwise/decision.py) to check product copy. Never publish private footage, credentials, addresses, or identifiable people as part of this page.

## Deploy on Vercel

Reuse the existing Vercel project connected to `PAOPAOtomato/watchwise` and serving <https://watchwise-six-ebon.vercel.app/>. Do not create another project or replace the established domain. Inspect the existing project settings before deploying; a GitHub account connection alone does not establish automatic deployments.

Use these project settings:

| Setting | Value |
| --- | --- |
| Framework Preset | `Vite` |
| Root Directory | `website` |
| Node.js Version | `24.x` |
| Install Command | `npm ci` |
| Build Command | `npm run build` |
| Output Directory | `dist` |
| Production Branch | `main` |
| Environment Variables | None |

The root directory is a Vercel project setting; `website/vercel.json` supplies the install, build, and output settings relative to that root. Keep files outside the root directory excluded: this frontend does not need the Python application or its configuration. Never copy the repository's `.env` or camera/provider credentials into this project.

For an update, confirm that the existing project's deployment becomes **Ready** for the pushed commit and open <https://watchwise-six-ebon.vercel.app/>. Check the event selector, evidence panels, correction and later visit, question responses, reset, mobile layout, and GitHub links on that URL. Vercel's [Git deployment guide](https://vercel.com/docs/git) and [build configuration guide](https://vercel.com/docs/builds/configure-a-build) describe project settings. If deployment access is unavailable, record which checks succeeded and the exact action still needed; do not infer deployment from a successful Git push.

## Maintain and publish updates

1. Update copy, scene illustrations, or scenario records together so each event's camera, evidence, explanation, and conversation agree.
2. Run `npm run build` and `npm test`; inspect the production preview at desktop and mobile widths, including keyboard navigation and reduced motion.
3. Commit the website source and lockfile, then push to `main`. Keep `node_modules/`, `dist/`, Playwright results, and local `.vercel/` metadata out of Git.
4. Check the website CI result and the existing Vercel project's deployment for that commit. Once the Git integration is configured, pushes to the production branch trigger production deployments; pull requests can receive preview deployments.

Dependencies are confined to this directory. Review updates periodically, commit the resulting lockfile, and repeat the same checks. Static hosting keeps maintenance simple and avoids inference, database, and server-compute costs for website interactions; hosting usage remains subject to the account's Vercel plan.
