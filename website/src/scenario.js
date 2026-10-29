// These authored records illustrate the repository's routing behavior, not live
// detections or measured model output. No camera or inference API is called.
export const events = [
  {
    id: 'headlights', title: 'Passing headlights', short: 'Headlights', time: '18:42', stamp: 'THU · 18:42:19', camera: 'Driveway', cameraId: 'CAM 01',
    scene: 'headlights', alt: 'Illustrated dusk scene of a passing car casting headlights across a modern home’s driveway',
    kicker: 'THE FAMILIAR, FILTERED', headline: 'Just passing through.',
    summary: 'Light sweeps across the driveway. A trusted recurring pattern resolves the event.',
    label: 'Resolved quietly', tone: 'green', decision: 'Headlights. Nothing more.',
    reason: 'The camera, car label, and driveway zone match a reviewed pattern associated with passing headlights. The gate resolves that signature without examining the image or calling a model.',
    route: 0, routeLabel: 'Resolved by the gate', confidence: '0.90', disposition: 'Suppressed',
    trace: [
      { title: 'A familiar signature', detail: 'Camera: driveway · label: car · zone: driveway. The signature matches an existing false-alarm pattern.' },
      { title: 'Memory has earned its trust', detail: 'This illustrated pattern has 20 human corrections, including 18 scored agreements and no disagreements: trust 0.95. Similarity is 0.95. It meets the configured correction, trust, and similarity thresholds.' },
      { title: 'No model needed', detail: 'The gate records a benign verdict with confidence 0.90. The event stays in the audit trail. This gate cannot distinguish different behaviors sharing the same camera, labels, and zones.' },
    ],
    answer: 'I did not flag these headlights for review. The driveway + car + driveway-zone signature matched a trusted pattern with 20 corrections. The gate resolved it without looking at the image or calling a model. You can still inspect the record; a different behavior with this same signature could also match.',
  },
  {
    id: 'courier', title: 'A parcel at the door', short: 'Courier', time: '18:48', stamp: 'THU · 18:48:32', camera: 'Front entry', cameraId: 'CAM 03',
    scene: 'courier', alt: 'Illustrated view of a courier carrying a parcel to the warmly lit front door',
    kicker: 'A QUICK LOOK AT THE EVIDENCE', headline: 'A delivery at the door.',
    summary: 'An available snapshot shows a person carrying a parcel at the entry. The screen judges it routine.',
    label: 'Resolved quietly', tone: 'green', decision: 'A parcel at the door.',
    reason: 'The available snapshot shows a person carrying a parcel at the entry. Vision screening interprets this as routine delivery activity. This first assessment does not describe the whole visit.',
    route: 1, routeLabel: 'Resolved by vision screening', confidence: '0.92', disposition: 'Downgraded',
    trace: [
      { title: '18:48:32 — Review received', detail: 'Frigate reports a person at the front entry. No trusted gate pattern settles this event; an initial screen is requested.' },
      { title: 'Available snapshot examined', detail: 'The snapshot shows a person with a box at the door. At a live event’s start, recording frames may not yet exist, so screening can fall back to this one snapshot.' },
      { title: 'An initial decision recorded', detail: 'The authored vision verdict is benign, confidence 0.92. It supports a routine-delivery interpretation, not the courier’s identity or proof of a later departure. No additional investigation is requested in this scenario.' },
    ],
    answer: 'I did not request your attention for the courier. The available snapshot shows a person carrying a parcel at the entry. Vision screening returns a benign verdict in this story; it does not identify the courier or establish how the whole visit ended.',
  },
  {
    id: 'cleaner', title: 'The expected cleaner', short: 'Cleaner', time: '18:53', stamp: 'THU · 18:53:11', camera: 'Side entry', cameraId: 'CAM 02',
    scene: 'cleaner', alt: 'Illustrated view of a cleaner with a cleaning bag arriving at the side gate',
    kicker: 'YOU KNOW THE MISSING CONTEXT', headline: 'Expected by you. Unclear to Watchwise.',
    summary: 'The cleaner arrives outside the configured visit window. A little human context can help.',
    label: 'Needs your context', tone: 'amber', decision: 'A familiar visit, still uncertain.',
    reason: 'There is similar activity in memory, but only two corrections. The 18:53 arrival also falls outside the 18:00–18:30 roster window. Watchwise keeps it for review while investigating.',
    route: 2, routeLabel: 'Human context needed', confidence: '0.58', disposition: 'Human review',
    trace: [
      { title: 'Camera health checked', detail: 'The side-entry camera is reachable. The event has usable frames; a camera fault is not the reason for uncertainty.' },
      { title: 'Similar activity, limited trust', detail: 'The side_entry + person + service_path signature has 2 prior corrections and trust 0.67. Default gate thresholds require at least 3 corrections and trust 0.70.' },
      { title: 'Duty roster checked', detail: 'The configured Thursday cleaning window is 18:00–18:30. Arrival at 18:53 is outside that window; a correction will not silently change the roster.' },
      { title: 'Your review closes the loop', detail: 'The vision verdict remains uncertain at confidence 0.58. Marking this expected reinforces the existing activity pattern and records your judgment.' },
    ],
    answer: 'I requested context because this 18:53 arrival is outside the roster window, and the matching activity has only two corrections. I am not identifying the person. If this is the expected cleaner, your review can reinforce the existing pattern.',
  },
  {
    id: 'visitor', title: 'An unexpected visitor', short: 'Visitor', time: '21:16', stamp: 'THU · 21:16:08', camera: 'Side entry', cameraId: 'CAM 02',
    scene: 'visitor', alt: 'Illustrated nighttime view of a person remaining near a home’s side gate',
    kicker: 'THIS ONE DESERVES YOUR ATTENTION', headline: 'Someone at the side gate.',
    summary: 'A person remains near the gate. No matching visit is on the roster.',
    label: 'Attention needed', tone: 'amber', decision: 'Worth your attention.',
    reason: 'A person remains near the rear-gate zone after the expected visit window. The camera is healthy, and the investigation finds no matching visit. This deserves a human look.',
    route: 2, routeLabel: 'Investigated · flagged for review', confidence: '0.87', disposition: 'Alert',
    trace: [
      { title: 'Camera health checked', detail: 'The camera is reachable and providing frames. The activity is not explained by a device fault.' },
      { title: 'Duty roster checked', detail: 'There is no expected visit at 21:16 in the configured duty roster. Absence from the roster is context, not proof of intent.' },
      { title: 'Related events compared', detail: 'A front-entry event at 21:14 precedes side-entry activity at 21:16. The timeline correlates events; it does not establish that both show the same person.' },
      { title: 'Earlier frames reviewed', detail: 'Frames at 21:16:08 and 21:17:24 show continued presence near the rear-gate zone. The authored investigation verdict is suspicious, confidence 0.87; a person should decide what to do.' },
    ],
    answer: 'I flagged this event because the person remains near the rear gate, no expected visit matches 21:16, and the camera is working. Earlier frames show continued presence. That supports asking you to look; it does not establish anyone’s intent.',
  },
];

export function getEvent(id, corrected = false) {
  if (id === 'return') {
    return {
      ...events[2], id: 'return', title: 'The next matching visit', short: 'Next visit', time: 'NEXT THU', stamp: 'NEXT THU · 18:53:09',
      kicker: 'YOUR CONTEXT, REMEMBERED', headline: 'Same kind of visit. Less interruption.',
      summary: 'The next matching activity meets the learned pattern’s thresholds. No model call needed.',
      label: 'Resolved from memory', tone: 'green', decision: 'A lesson put to work.',
      reason: 'Your third correction helped this pattern earn enough trust. Matching side-entry activity now resolves at the gate. The decision stays recorded, and the pattern can still lose trust if you disagree.',
      route: 0, routeLabel: 'Resolved by learned memory', confidence: '0.71', disposition: 'Downgraded',
      trace: [
        { title: 'The same activity signature', detail: 'Side-entry camera · person label · service_path zone. This does not recognize the cleaner’s identity or behavior; another person or action with the same signature can also match.' },
        { title: 'Your correction made a difference', detail: 'The illustrated pattern now has 3 corrections, trust 0.75, and similarity 0.95. All three configured thresholds are met.' },
        { title: 'Resolved at the gate', detail: 'The pattern produces a benign verdict at confidence 0.71, with no model call. Policy downgrades the event to a detection; full suppression requires confidence 0.85. The record remains auditable.' },
      ],
      answer: 'I resolved this later event at the gate because the activity signature matches the pattern you helped review. It now has 3 corrections, trust 0.75, and similarity 0.95. This does not recognize a face or behavior, or create a roster entry. Another action with the same signature can also match.',
    };
  }
  const event = events.find((item) => item.id === id) || events[3];
  if (event.id !== 'cleaner' || !corrected) return event;
  return {
    ...event, label: 'Reviewed by you', tone: 'green', decision: 'Your context is now part of the record.',
    reason: 'You marked this visit as the expected cleaner. The original uncertain decision stays in the history, alongside your correction. The existing activity pattern now has three corrections and trust 0.75.',
    routeLabel: 'Corrected by you · memory reinforced', disposition: 'Human: expected',
    answer: 'You corrected this event as expected. I kept the original uncertain verdict and recorded your judgment. That reinforced the existing pattern to 3 corrections and trust 0.75. Replay the next matching visit to see the gate use that context.',
    trace: [...event.trace, { title: 'Human correction recorded', detail: 'Expected cleaner. Correction count: 2 → 3. Trust: 0.67 → 0.75 after agreement with the prior pattern. The duty roster remains unchanged.' }],
  };
}

export function answerQuestion(question, event, corrected, replayed) {
  const q = question.toLowerCase();
  if (/filter|quiet|suppress|noise/.test(q)) return `In this story, passing headlights were resolved by a trusted gate pattern, and the courier by vision screening.${replayed ? ' The later matching visit was also resolved by memory, after your correction.' : ''} ${corrected ? 'You reviewed the first cleaner visit yourself.' : 'The cleaner still needs your context.'} The unexpected visitor remains flagged. All of those records are available in the event strip.`;
  if (/evidence|proof|frame|camera|confiden/.test(q)) return `${event.title}: ${event.trace.map((step) => step.title).join('; ')}. The ${event.id === 'cleaner' && corrected ? 'original ' : ''}recorded confidence is ${event.confidence}. These are simulated values for this illustrated story, not measured field performance. Open Evidence for the full record.`;
  if (/learn|correct|memory|recogn|cleaner|next|trust/.test(q)) return corrected
    ? 'Your correction reinforced a pattern that already had two reviews. It now has 3 corrections and trust 0.75; the later event’s similarity 0.95 meets the default 0.86 requirement. The signature uses camera, labels, and zones, so another person or behavior can match. In the installed product, Alert me lowers trust and vetoes memory hold-backs for that signature for 30 days by default (gate.real_veto_days). This story shows Expected feedback. The roster is unchanged.'
    : 'The cleaner’s activity pattern already has 2 corrections, below the default threshold of 3. It also needs trust at least 0.70 and similarity at least 0.86. Select the cleaner and mark the visit as expected. That adds your judgment to memory; it does not change the duty roster or recognize identity or behavior.';
  if (/chat|install|telegram/.test(q)) return 'These answers are authored for the selected browser scenario; your questions stay in this page. Ask Watchwise works in Telegram: ask questions or reply to cards, and propose duty-roster additions or removals, arming changes, muting, or “tell me when…” watches. Only tapping ✅ Confirm applies a chat proposal; typing “confirm”, “yes” or “ok” applies nothing. After 6 quiet hours a question starts afresh; /new starts over any time. Installed messages and enabled photos pass through Telegram; a cloud chat role also receives questions and retrieved text.';
  if (/local|cloud|privacy|data|model/.test(q)) return 'Watchwise runs models locally by default. In a hybrid profile, text can go to your chosen cloud provider while frames stay local; a full cloud profile sends images and text. Telegram also relays messages and enabled photos or face previews. This website uses only simulated, illustrated records in your browser—no cameras, model calls, or API keys.';
  if (/why|alert|happen|visit|decision|explain|expected|summary/.test(q)) return event.answer;
  return `I can explain the selected ${event.short.toLowerCase()} event, its evidence, the filtered activity, or how corrections affect memory. Try “Why did you alert me?” or “What evidence did you use?” Answers here are limited to this illustrated event record.`;
}
