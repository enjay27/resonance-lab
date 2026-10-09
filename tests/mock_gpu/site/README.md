Stub `torch`, `tqdm` and `transformers`, just enough for `scripts/llamafactory/eval.py` to run its own code
(prompt building, the generate loop, scoring, report, predictions) without a GPU. The "model" echoes the
reference translation of the eval line it finds in the prompt. Test-only: `harness.py` puts this folder on
`PYTHONPATH` of the mock run and nowhere else.
