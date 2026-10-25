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

## Structure and source of truth

| File | Purpose |
| --- | --- |
| [`index.html`](index.html) | Product story, navigation, installation copy, and accessible interface structure |
| [`src/main.js`](src/main.js) | Selected event, panels, correction/replay/reset, conversation, and clipboard controls |
| [`src/scenario.js`](src/scenario.js) | Authored event records, evidence, routing explanations, and deterministic question responses |
| [`src/style.css`](src/style.css) | Responsive layout, focus states, typography, and reduced-motion behavior |
| [`public/scenes/`](public/scenes/) | Original SVG scene illustrations |
| [`vercel.json`](vercel.json) | Static build settings and response headers |

This update describes the [published Python baseline](../docs/verification.md#source-and-availability); concurrent local Python changes are kept separate. Keep product claims aligned with the [repository README](../README.md), [hardware guide](../docs/hardware.md), [hardware profiles](../watchwise/hw/profiles.yml), [gate implementation](../watchwise/tier2/gate.py), [configuration defaults](../watchwise/config.py), [pattern memory](../watchwise/memory/long_term.py), and [feedback loop](../watchwise/memory/feedback.py). The website does not call these Python modules at runtime.

The camera scenes and their evidence are authored illustrations. Timestamps, confidence values, correction counts, and changing totals describe this particular interactive story, not live cameras, production activity, accuracy guarantees, or measured performance. Do not turn the README's simulated routing results into real-world benchmarks. The courier story describes an initial snapshot assessment, rather than promising a description of the whole event: at the start of a live review, recording frames may not yet be available.

The cleaner story starts with an existing activity pattern: two corrections and trust `0.67`. A visitor's correction reinforces that pattern and agrees with its prior assessment, giving three corrections and trust `0.75`. The next matching visit has similarity `0.95`, meeting the current default requirements: at least three corrections, similarity at least `0.86`, and gate trust at least `0.70`. Correction count and trust are separate: trust is the smoothed agreement ratio `(agreements + 1) / (agreements + disagreements + 2)`, not a direct conversion from correction count. The later gate confidence is the rounded product of trust and similarity, `0.71`.

This illustrates camera/label/zone pattern matching. It does not recognize identity or distinguish behaviors within the same signature: another person or action can also match. A correction does not edit the duty roster, and the original decision remains in the evidence history. Conflicting reviews can lower trust but do not provide a hard veto against future gate decisions. Age decay affects search ranking, not gate eligibility or pattern expiration. Undo and reset restore the story's initial state. Replay clears prior answers because filtered-event totals have changed. All selections, corrections, and conversations live only in browser memory and reset on page reload. Suggested questions and typed questions select authored responses; no model or chat service receives visitor input. Installed conversational chat is planned; installed Telegram alert actions are separate.

The headlights pattern has 20 corrections, including 18 scored agreements and no disagreements, giving trust `19 / 20 = 0.95`. Combined with similarity `0.95`, this produces confidence `0.90` after rounding. The gate uses the event signature, not the illustrated frame.

The later event is downgraded to a detection, rather than fully suppressed: its `0.71` confidence is below the policy's `0.85` suppression threshold. The courier's benign vision result is also downgraded. Both remain available for audit without a request for human review. See the [decision policy](../watchwise/decision.py).

Typography uses the locally bundled Manrope font under the [SIL Open Font License](public/licenses/manrope.txt). Scene illustrations are original project assets.

## Verified product progress

The live-verification panel describes the October 22–23, 2026 test separately from the illustrated story: an external USB camera through go2rtc and Tailscale to Frigate 0.18.2 and MQTT, Ollama models on a 24 GB NVIDIA L4, and Telegram cards on a phone. Cards updated in place, Expected and Real feedback reached memory, and a later matching event was downgraded at the gate without a model call or phone card after enough corrections and trust.

Keep the qualifications visible: this was one indoor camera setup, not a reliability or accuracy benchmark. The screener lacks arming-state context; memory cannot distinguish behavior within a signature. Clip-link rendering is implemented, but phone playback still needs verification. Frigate summary ingestion and raise-only decisions pass automated tests; the live summary attempt exceeded the model context. Watchwise-generated whole-event summaries and additional arming/memory safeguards are local work in progress and are not included in this publication. The tested model pair exceeded the memory advertised by the `nvidia-16gb` profile name. See the [verification notes](../docs/verification.md#test-scope) for scope and current limitations. Never publish private footage, credentials, addresses, or identifiable people as part of this page.

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
