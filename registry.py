"""The model shelf: every speech model this service can run, and where it comes from.

All eight are free, public and Apache 2.0, and all but one are published by a public body:
the Barcelona Supercomputing Center for Catalan, Valencian, Balearic and the "Languages of
Spain" model, Proxecto Nos (Xunta de Galicia + Univ. de Santiago) for Galician, and HiTZ
(Univ. of the Basque Country) for Basque.

`format`
  "ct2"          the publisher already ships a CTranslate2 build, so it is only a download
  "transformers" a plain Whisper checkpoint that has to be converted before faster-whisper
                 can load it (`convert.py`)

`baked`
  a path inside the image. Set for the models that are built in, so a cold start never
  downloads them. The rest land in /models/cache the first time somebody picks them, and
  stay there for the life of that worker.

`language`
  what to hand faster-whisper. None means "work it out per segment", which is what a
  meeting that switches between two languages needs.

Adding a language is one entry here, one line in the website's own copy of this table
(`worker/asr_models.py` on the Arsys box), and one <option> in the page.
"""

CACHE_DIR = "/models/cache"

MODELS = {
    "ca": {
        "repo": "BSC-LT/faster-whisper-large-v3-ca-punctuated-3370h",
        "format": "ct2",
        "language": "ca",
        "baked": "/models/bsc-punct",
        "label": "Catala (general)",
    },
    "ca-valencia": {
        "repo": "BSC-LT/faster-whisper-3cat-cv21-valencian",
        "format": "ct2",
        "language": "ca",
        "label": "Valencia",
    },
    "ca-balear": {
        "repo": "BSC-LT/faster-whisper-3cat-balearic",
        "format": "ct2",
        "language": "ca",
        "label": "Balear",
    },
    "ca-4700": {
        "repo": "BSC-LT/faster-whisper-bsc-large-v3-cat",
        "format": "ct2",
        "language": "ca",
        "label": "Catala (model gran)",
    },
    "mixt": {
        "repo": "BSC-LT/whisper-large-v3-LoS-punctuated",
        "format": "transformers",
        "language": None,          # detect per segment: this is the mixed-meeting model
        "multilingual": True,
        "baked": "/models/los-punct",
        "label": "Catala i castella",
    },
    "es": {
        "repo": "BSC-LT/whisper-large-v3-LoS-punctuated",
        "format": "transformers",
        "language": "es",
        "baked": "/models/los-punct",   # same weights as mixt, different language hint
        "label": "Castella",
    },
    "gl": {
        "repo": "proxectonos/whisper-large-v3-turbo-gl-v1.0",
        "format": "transformers",
        "language": "gl",
        "baked": "/models/gl-turbo",
        "label": "Gallec",
    },
    "eu": {
        "repo": "HiTZ/whisper-large-v3-eu",
        "format": "transformers",
        "language": "eu",
        "label": "Basc",
    },
}

# Old name for "mixt", accepted so a caller written against the first draft keeps working.
ALIASES = {"mixed-es": "mixt"}

DEFAULT_MODEL = "ca"

# The files a CTranslate2 whisper model is made of. Asking for exactly these keeps a download
# from dragging in anything else the repo happens to carry.
CT2_FILES = ["config.json", "model.bin", "tokenizer.json",
             "vocabulary.json", "vocabulary.txt", "preprocessor_config.json"]

# Training leftovers that are not needed to convert a checkpoint and are enormous: the
# Galician repo ships a 6.5 GB optimizer state next to a 3.2 GB model. Everything NOT listed
# here is downloaded, so a repo that needs an unusual file still converts. That is the safe
# way round: a missing file breaks the conversion, an extra one only wastes a moment.
CONVERT_IGNORE = ["optimizer.pt", "scheduler.pt", "scaler.pt", "rng_state*", "*.pth",
                  "training_args.bin", "trainer_state.json", "runs/*", "*.msgpack",
                  "*.h5", "*.onnx", "README.md", ".gitattributes"]


def canonical(key):
    """The shelf key a caller's string means, or None if the shelf has no such model."""
    key = (key or "").strip()
    key = ALIASES.get(key, key)
    return key if key in MODELS else None


def resolve(key):
    """The spec for a key, falling back to the default for anything unknown."""
    return MODELS[canonical(key) or DEFAULT_MODEL]
