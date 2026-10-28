# Verification notes

Dated records of live tests on the maintainer's setup. Each record names the
commit whose code ran. They show what was exercised end to end on one setup;
they do not claim that every capability or hardware configuration has been
validated, and no real-world accuracy rate, false-alarm reduction percentage, or
latency guarantee is claimed. The automated Python suite covers behavior with
fake Frigate, model and Telegram servers.

## The setup

All records used an external USB camera on a Windows laptop, go2rtc 1.9.14,
RTSP over Tailscale, Frigate 0.18.2 with an OpenVINO CPU detector, and a
Mosquitto broker. Watchwise ran on an Ubuntu CPU VM; Ollama 0.40.8 served the
models on a separate NVIDIA L4 GPU VM with 24 GB VRAM. These were
operator-managed cloud VMs, so the tests were not entirely on-premises
deployments. Camera models were not recorded; the records do not establish
support for a particular camera brand. Cards went to a real phone through
Telegram.

## October 22–23, 2026 (`7e7c4aa`)

Screening used `qwen3-vl:8b-instruct` and investigation `qwen3:14b`.

| Capability | Evidence and scope |
|---|---|
| Live ingestion and screening | Real indoor person events traveled through Frigate MQTT and HTTP into Watchwise. Early screening used a snapshot when recordings were not yet available. |
| Phone alerts | The phone received cards, including an initial “Analyzing…” state followed by a verdict or investigation update. |
| Learning from corrections | Expected and Real feedback (the Real button now reads 🚨 Alert me) affected pattern trust. A later event with the same camera/label/zone signature was downgraded by the gate, with no model call or new phone card. After a memory reset, repeated Expected feedback also taught a new pattern. |
| Deeper investigation | Ambiguous live events invoked the tool-calling investigator. The test does not validate every forensic tool or the correctness of every conclusion. |
| Recording links | Cards displayed a Clip button. Phone playback was not verified in this test: the phone lacked access to the private network hosting Frigate. |
| Frigate review summaries | Parsing, persistence, card text, and raise-only decisions were covered by automated tests. Live generation of longer-event summaries exceeded the model context; Frigate GenAI was disabled in that setup. |
| Other platforms and providers | Profiles and provider adapters exist. Automated stand-ins and hardware fixtures do not establish live compatibility for every listed option. |

The implementation in this test could not tell different behaviors sharing a
signature apart, did not give the screener the arming state, and could not
describe an entire event from an early snapshot. The records below cover the
changes made for these.

## October 24, 2026 (`bca60ed`)

One `qwen3-vl:8b-instruct` played every role.

| Capability | Evidence and scope |
|---|---|
| Arming-aware screening | While armed away, a person the vision model called ordinary was alerted and investigated instead of being dismissed. |
| Model-checked memory | Routine events were checked by the vision model against the owner's earlier rulings, instead of being held back by the gate alone. |
| Teaching in the owner's words | After 🚨 Real (now 🚨 Alert me), the owner's reply to “What was wrong?” became the example; a later matching gesture alerted at the first look. Gestures that the sampled frames did not show were missed. |
| Whole-event summaries | Watchwise's own summaries described events across their length, and one raised a gesture to an alert. |
| Faces | With Frigate's face recognition on, events in which Frigate named a resident were held back with no model call; photos were added from the camera with `/face`. Frigate needed a restart after face recognition was first enabled. The unknown-face rule was covered by automated tests only. |
| Recording links | With the phone on the private network, ▶️ Clip opened Frigate's recording after its certificate warning and login. |

## October 26–27, 2026 (`268b5c2`)

| Capability | Evidence and scope |
|---|---|
| `/setup` | Saved camera notes and an arming default to the configuration file after ✅ Apply, with a backup, and the running service used them. The recognition check reported the resident Frigate recognized. |
| `/arm` | Arming was switched from Telegram. |
| More cameras | Four streams ran at once: the USB camera, the laptop's built-in camera, and two iPhones pushing RTMP from Larix Broadcaster into go2rtc. The built-in camera opened only without explicit size and frame-rate options. |
| Rooms | With two cameras in one room, a walk through it produced one card naming both cameras; events in other rooms got their own cards. The test also showed a card taking new events for as long as someone stayed in the room; the next commit limits a card to five minutes and gives standout events their own card. |
| Faces from new angles | A camera with no photo from its own angle raised unknown-face alerts for a person whom another camera recognized as a resident at the same moment; it recognized the resident later in the test. |

## October 27, 2026 (`f8eb33f`)

| Capability | Evidence and scope |
|---|---|
| Ask Watchwise | Questions in Telegram were answered from Watchwise's records by the local model. An arming switch and a duty-roster rule were proposed with ✅ Confirm and ✖️ Cancel, then cancelled; nothing changed. Confirmed changes and photos in answers were covered by automated tests only. |
| Watches | A watch for someone falling did not run for a recognized resident and, once it did, matched while a person sat at a desk: the model read a lying posture into sampled frames. Fall detection is not supported. The watch check that followed asks a separate yes/no question with evidence once per event, also for recognized residents. |
