"""The model shelf: every speech model this service can run, and where it comes from.

Adding a language is one entry here plus one line in the website's own copy of this table
(`worker/asr_models.py` on the Arsys box) and one <option> in the page.

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
        "label": "Catala (valencia)",
    },
    "ca-balear": {
        "repo": "BSC-LT/faster-whisper-3cat-balearic",
        "format": "ct2",
        "language": "ca",
        "label": "Catala (balear)",
    },
    "ca-4700": {
        "repo": "BSC-LT/faster-whisper-bsc-large-v3-cat",
        "format": "ct2",
        "language": "ca",
        "label": "Catala (model gran)",
    },
    "mixed-es": {
        "repo": "BSC-LT/whisper-large-v3-LoS-punctuated",
        "format": "transformers",
        "language": None,          # detect per segment: this is the mixed-meeting model
        "multilingual": True,
        "baked": "/models/los-punct",
        "label": "Catala i castella barrejats",
    },
    "es": {
        "repo": "BSC-LT/whisper-large-v3-LoS-punctuated",
        "format": "transformers",
        "language": "es",
        "baked": "/models/los-punct",   # same weights as mixed-es, different language hint
        "label": "Castella",
    },
    "gl": {
        "repo": "proxectonos/whisper-large-v3-turbo-gl-v1.0",
        "format": "transformers",
        "language": "gl",
        "baked": "/models/gl-turbo",
        "allow": ["*.json", "*.txt", "model.safetensors", "tokenizer.json"],
        "label": "Gallec",
    },
    "eu": {
        "repo": "HiTZ/whisper-large-v3-eu",
        "format": "transformers",
        "language": "eu",
        "allow": ["*.json", "*.txt", "*.safetensors"],
        "label": "Basc",
    },
}

DEFAULT_MODEL = "ca"

# The six files a CTranslate2 whisper model is made of. Asking for exactly these keeps a
# download from dragging in training leftovers (one repo ships a 6.5 GB optimizer state).
CT2_FILES = ["config.json", "model.bin", "tokenizer.json",
             "vocabulary.json", "vocabulary.txt", "preprocessor_config.json"]


def resolve(key):
    """The spec for a key, falling back to the default for anything unknown."""
    return MODELS.get(key) or MODELS[DEFAULT_MODEL]
