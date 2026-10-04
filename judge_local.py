"""The Gate judge WITHOUT llama-server: the Kev decision model loaded in this Python process (needs `.venv-kev`, requirements-kev.txt).

`LocalKevClient` has the same `choice()` / `check_server()` as `judge_client.SystemOneClient` (it IS one: its `post` runs the
engine in-process instead of sending HTTP), so the parsing, the validation and the errors are the server client's, and
`gate_judge.GateJudge` takes either. Kev (jaredpalmer/kev, Apache-2.0) is the model behind ggml-org/Kev-4B-GGUF: a LoRA adapter and
a pointer head on a Qwen3.5 base, so a run here (bf16) and a llama-server run (Q4_K_M) can be compared on the same sample.

The engine is `answer(request dict) -> the /v1/systemone body` (what `kev.serve.Server.answer` returns). It is injected through
`loader(run, device, dtype)`, so everything above `load_kev` is pure and tested with a stub on any OS; `load_kev` is the only part
that imports torch / kev (lazily) and runs on a CUDA machine only (not verified in CI).
"""

import json

from judge_client import JudgeError, SystemOneClient

DTYPES = ("bf16", "fp32")  # bf16: half the memory, what kev.serve uses on CUDA; fp32: the exact path of every number Kev reports
SMALLER_MODELS = "jaredpalmer/kev-4b or jaredpalmer/kev-0.8b"
SCHEME = "kev-local://"  # the label error messages use where the server client has its URL


def model_name(run, dtype):
    """`jaredpalmer/kev-9b@v1.0+bf16`: the weights AND the precision, because the two answer a little differently."""
    return f"{run}+{dtype}"


class LocalKevClient(SystemOneClient):
    def __init__(self, run, device=None, dtype="bf16", loader=None):
        if dtype not in DTYPES:
            raise ValueError(f"dtype {dtype!r} is not one of {', '.join(DTYPES)}")
        super().__init__(SCHEME + run, post=self._run_engine)
        self.run, self.device, self.dtype = run, device, dtype
        self._loader = loader or load_kev
        self._engine = None

    def _load(self):
        if self._engine is None:
            try:
                self._engine = self._loader(self.run, self.device, self.dtype)
            except JudgeError:
                raise
            except Exception as error:
                if type(error).__name__ == "OutOfMemoryError":
                    raise JudgeError(
                        f"{self.run}: out of memory loading the model: use dtype bf16, a smaller model "
                        f"({SMALLER_MODELS}) or free the GPU (a training run or llama-server holding it)"
                    ) from None
                raise JudgeError(f"{self.run}: could not load the model: {error or type(error).__name__}") from None
        return self._engine

    def _run_engine(self, url, payload, timeout):
        """The injected `post`: (200, the body as JSON text); every failure a JudgeError that names the run."""
        engine = self._load()
        try:
            body = engine.answer(payload)
        except ValueError as error:  # kev's request validation (pydantic's ValidationError is one)
            raise JudgeError(f"{self.run}: invalid request: {error}") from None
        except Exception as error:
            raise JudgeError(f"{self.run}: the model failed: {error or type(error).__name__}") from None
        return 200, json.dumps(body, ensure_ascii=False)

    def check_server(self):
        """Loads the model (the slow part) and answers one trivial question; returns the run and dtype."""
        super().check_server()
        return model_name(self.run, self.dtype)


def load_kev(run, device, dtype):
    """The engine for `run` (a Hub id, `id@revision` or a local directory): a kev.serve.Server behind `answer(dict)`. Model part."""
    try:
        import torch
        from kev.api import SystemOneRequest
        from kev.checkpoint import Checkpoint, LoadOptions
        from kev.serve import Server
    except ImportError as error:
        raise JudgeError(
            f"{run}: the local judge needs kev and torch ({error}): use the `.venv-kev` virtualenv "
            "(requirements-kev.txt; the other pipelines' stacks conflict with it)"
        ) from None
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = Checkpoint(run)
    tok, model = checkpoint.load(device, LoadOptions(dtype=torch.bfloat16 if dtype == "bf16" else None, merge=True))
    server = Server(checkpoint, tok, model, device)

    class Engine:
        @staticmethod
        def answer(request):
            return server.answer(SystemOneRequest.model_validate(request))

    return Engine()
