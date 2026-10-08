<p align="center">
  <img src="https://raw.githubusercontent.com/coccinella-labs/mlapi/main/.github/assets/thumbnail.png" alt="mlapi" width="100%">
</p>

# mlapi

FastAPI service for machine learning inference.

Serves a GPT-2 causal language model over HTTP, with CORS enabled and structured logging.

## Run

```bash
pip install -r requirements.txt
python main.py
```

The server starts with uvicorn. Endpoints:

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | Liveness probe |
| POST | `/chat` | Generate a text completion |
| GET | `/models` | Describe the loaded model |
| GET | `/status` | API and model status |

`POST /chat` takes `{"prompt": "..."}` and returns `{"response": "..."}`. Generation parameters are fixed in `main.py` (`max_length` 256, `top_k` 50, `top_p` 0.95, `temperature` 0.7) and are not configurable at runtime.

## Which model it loads

`main.py` prefers `./fine_tuned_model` and falls back to `gpt2` on the Hub if that directory cannot be loaded.

The committed `fine_tuned_model/` directory contains only `config.json`, `special_tokens_map.json` and `tokenizer_config.json`. There is no weight file, so it is not a usable checkpoint and the server falls back to `gpt2`. This is a configuration and tokenizer scaffold, not a fine-tuned model.

The model is a plain causal LM. It produces text continuations rather than dialogue, and it has no instruction tuning and no chat template. The `/models` endpoint says the same thing.

## Config files

`config.yaml` holds training and dataset parameters for the original fine-tuning run. **The server does not read it.** Model selection, generation parameters and CORS are all hardcoded in `main.py`, so editing `config.yaml` changes nothing at runtime.

## Unused files

Two files exist that are not part of the running service:

- `api_server.py` defines a `/chat` endpoint that loads a different model (`bniladridas/conversational-ai-fine-tuned`) and does not enable CORS or logging. It is not referenced by the Dockerfile, CI or tests.
- `generate_response.py` is a standalone CLI script. Nothing imports it.

The entry point is `main.py`, which is what the Dockerfile runs and what `tests/test_main.py` exercises.

## Deployment

`Dockerfile` and `k8s.yaml` are included. The Docker image runs `python main.py`.

## Tests

```bash
python -m pytest tests/ -v
```

Four tests cover `/health`, `/chat`, `/models` and `/status` against `main.py` using FastAPI's test client. They do not test `api_server.py` or `generate_response.py`.

## Stack

FastAPI, Transformers, PyTorch, uvicorn. Python 3.12 in the Docker image.

## License

MIT. See [LICENSE](LICENSE).
