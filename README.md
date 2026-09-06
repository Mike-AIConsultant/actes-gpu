# actes-gpu

The GPU half of [actes.realintelligence.consulting](https://actes.realintelligence.consulting),
a Catalan meeting-minutes tool.

This image runs as a **RunPod serverless** worker in the European Union. It receives a signed,
short-lived link to one audio file, transcribes it with the Barcelona Supercomputing Center's
Catalan Whisper model, separates the speakers, sends the words back, and deletes everything.

It keeps no data. The website, the database, the audio and the minutes all stay on a server in
Spain.

## The model shelf

The job input carries a `model` key. The shelf lives in `registry.py`; adding a language is
one entry there.

| key | model | who made it | in the image? |
|---|---|---|---|
| `ca` (default) | `BSC-LT/faster-whisper-large-v3-ca-punctuated-3370h` | Barcelona Supercomputing Center | yes |
| `ca-valencia` | `BSC-LT/faster-whisper-3cat-cv21-valencian` | BSC | fetched on first use |
| `ca-balear` | `BSC-LT/faster-whisper-3cat-balearic` | BSC | fetched on first use |
| `ca-4700` | `BSC-LT/faster-whisper-bsc-large-v3-cat` | BSC | fetched on first use |
| `mixed-es` | `BSC-LT/whisper-large-v3-LoS-punctuated` | BSC | yes |
| `es` | the same weights, language forced to Spanish | BSC | yes |
| `gl` | `proxectonos/whisper-large-v3-turbo-gl-v1.0` | Proxecto Nós, Xunta de Galicia + USC | yes |
| `eu` | `HiTZ/whisper-large-v3-eu` | HiTZ, Univ. of the Basque Country | fetched on first use |

All Apache 2.0. A model that has to be converted to CTranslate2 is baked in, because
converting takes minutes; one the publisher already ships in CTranslate2 form is fetched on
demand, because that is a one minute file copy and keeping it out of the image keeps the
cold start down. Basque is the exception: it needs converting and is rare here, so it is on
demand and costs about six minutes the first time a worker sees it.

Speakers, unchanged:

| | |
|---|---|
| Speakers, default | `nvidia/diar_streaming_sortformer_4spk-v2` |
| Speakers, fallback | `pyannote-community/speaker-diarization-community-1` |

Built and published by GitHub Actions to `ghcr.io/<owner>/actes-gpu:latest`.
