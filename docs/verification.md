# Verification notes

This record covers the October 22–23, 2026 test and the published implementation
used by the [product README](../README.md). It is a dated test record, not a claim
that every capability or hardware configuration has been validated.

## Source and availability

The documented Python implementation is
[`7e7c4aa`](https://github.com/PAOPAOtomato/watchwise/commit/7e7c4aa9f0a6c1c8ade631fd06209fafcaaecf3f).
Subsequent documentation and website commits do not change that implementation.
Separate local Python changes are excluded from this publication; their
availability and validation must be assessed when they are published.

## Test scope

The live setup used an external USB camera on a Windows laptop, go2rtc 1.9.14,
RTSP over Tailscale, Frigate 0.18.2 with an OpenVINO CPU detector, and a Mosquitto
broker. Watchwise ran on an Ubuntu CPU VM; Ollama 0.40.8 served
`qwen3-vl:8b-instruct` (screening) and `qwen3:14b` (investigation) on a separate
NVIDIA L4 GPU VM with 24 GB VRAM. These were operator-managed cloud VMs, so this
particular test was not an entirely on-premises deployment. The USB camera model
was not recorded; it does not establish support for a particular camera brand.

| Capability | Evidence and scope |
|---|---|
| Live ingestion and screening | Real indoor person events traveled through Frigate MQTT and HTTP into Watchwise. Early screening used a snapshot when recordings were not yet available. |
| Phone alerts | A real Telegram phone received cards, including an initial “Analyzing…” state followed by a verdict or investigation update. |
| Learning from corrections | Expected and Real feedback affected pattern trust. A later event with the same camera/label/zone signature was downgraded by the gate, with no model call or new phone card. After a memory reset, repeated Expected feedback also taught a new pattern. |
| Deeper investigation | Ambiguous live events invoked the tool-calling investigator. The test does not validate every forensic tool or the correctness of every conclusion. |
| Recording links | Cards displayed a Clip button. Phone playback was not verified during this test: the phone lacked access to the private network hosting Frigate. |
| Frigate review summaries | Parsing, persistence, card text, and raise-only decisions are implemented and covered by automated tests. Live generation of longer-event summaries exceeded the model context; Frigate GenAI was disabled in that setup. |
| Other platforms and providers | Profiles and provider adapters exist. Automated stand-ins and hardware fixtures do not establish live compatibility for every listed option. |

The implementation exercised in this test could not distinguish different
behaviors sharing a signature, did not include arming state in the screener
prompt, and could not describe an entire event from an early snapshot. These are detailed under
[Known limitations](../README.md#known-limitations). No real-world accuracy rate, false-alarm
reduction percentage, or latency guarantee is claimed.
