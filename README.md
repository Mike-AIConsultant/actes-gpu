# actes-gpu

The GPU half of [actes.realintelligence.consulting](https://actes.realintelligence.consulting),
a Catalan meeting-minutes tool.

This image runs as a **RunPod serverless** worker in the European Union. It receives a signed,
short-lived link to one audio file, transcribes it with the Barcelona Supercomputing Center's
Catalan Whisper model, separates the speakers, sends the words back, and deletes everything.

It keeps no data. The website, the database, the audio and the minutes all stay on a server in
Spain.

Baked in so a cold start never downloads a model:

| | |
|---|---|
| Transcription | `BSC-LT/faster-whisper-large-v3-ca-punctuated-3370h` |
| Speakers, default | `nvidia/diar_streaming_sortformer_4spk-v2` |
| Speakers, fallback | `pyannote-community/speaker-diarization-community-1` |

Built and published by GitHub Actions to `ghcr.io/<owner>/actes-gpu:latest`.
