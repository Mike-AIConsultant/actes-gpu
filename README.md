# actes-gpu

The GPU half of [actes.realintelligence.consulting](https://actes.realintelligence.consulting),
a Catalan meeting-minutes tool.

This image runs as a **RunPod serverless** worker in the European Union. It receives a signed,
short-lived link to one audio file, transcribes it with a speech model chosen per meeting,
separates the speakers, sends the words back, and deletes everything.

It keeps no data. The website, the database, the audio and the minutes all stay on a server in
Spain.

## The model shelf

Every model is free, public and Apache 2.0. All but none of them come from a public body: the
Barcelona Supercomputing Center, Proxecto Nos (Xunta de Galicia and the University of Santiago)
and HiTZ (University of the Basque Country).

| key | what it is | model | in the image? |
|---|---|---|---|
| `ca` | Catalan, the default | `BSC-LT/faster-whisper-large-v3-ca-punctuated-3370h` | baked |
| `ca-valencia` | Valencian | `BSC-LT/faster-whisper-3cat-cv21-valencian` | on demand |
| `ca-balear` | Balearic (Mallorca, Menorca, Eivissa) | `BSC-LT/faster-whisper-3cat-balearic` | on demand |
| `ca-4700` | Catalan, the 4,700 hour model | `BSC-LT/faster-whisper-bsc-large-v3-cat` | on demand |
| `mixt` | Catalan and Spanish in one meeting | `BSC-LT/whisper-large-v3-LoS-punctuated` | baked |
| `es` | Spanish | same weights as `mixt` | baked |
| `gl` | Galician | `proxectonos/whisper-large-v3-turbo-gl-v1.0` | baked |
| `eu` | Basque | `HiTZ/whisper-large-v3-eu` | on demand |

**Baked** means built into the image, so a cold start never downloads it. The rule is that a
model needing CONVERSION to CTranslate2 is baked, because converting costs minutes on every
fresh worker; a model the publisher already ships in CTranslate2 form is fetched on demand,
because that is a plain 3.1 GB file copy, once per worker. Basque is the exception: it needs
converting AND is rare on a Catalan meeting site, so it is left on demand.

## The job contract

```json
{"input": {"audio_url": "https://...", "model": "ca", "diar_engine": "auto"}}
```

`model` is optional and defaults to `ca`. An unknown key falls back to `ca` rather than
failing the meeting, and the key that was actually asked for comes back in
`timings.asr_model_asked`. The shelf entry owns the language, so a caller cannot pick the
Galician model and have it decode Catalan.

Speakers: `nvidia/diar_streaming_sortformer_4spk-v2` by default,
`pyannote-community/speaker-diarization-community-1` when Sortformer hits its four speaker
ceiling. Both are baked in.

Built and published by GitHub Actions. `main` publishes `:latest`, which is what the live
endpoint pulls; any other branch publishes under its own name so a new image can be tested
against a throwaway endpoint first.
