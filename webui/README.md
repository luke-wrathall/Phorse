# Stage 2 test UI (throwaway)

A tiny drag-and-drop page for manually poking at the Stage 2 head: drop an
image, see its horseness score and whether it passes the current gate
threshold. Nothing more.

This folder is deliberately decoupled from the rest of the project — it
only *imports* from `horse/` (`embed.py`, `model.py`, `gate.py`); nothing
in `horse/` knows this exists. Delete `webui/` whenever it's no longer
useful; nothing else depends on it, and its one extra dependency
(`flask`) lives in `webui/requirements.txt`, not the project's main
`requirements.txt`.

## Run

```
pip install -r webui/requirements.txt
python webui/app.py
```

Then open http://127.0.0.1:5000 and drop an image in.

Flags: `--head <path>` (defaults to `horse/data/head.pt`), `--threshold
<float>` (defaults to the same `DEFAULT_THRESHOLD` gate.py uses), `--port
<int>`. Without a trained checkpoint at `--head`, the page shows a banner
and scores are from an untrained (random) head.
