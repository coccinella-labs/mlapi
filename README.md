<p align="center">
  <img src="https://raw.githubusercontent.com/Coccinella-Labs/mlapi/main/.github/assets/thumbnail.png" alt="mlapi" width="100%">
</p>

# mlapi

FastAPI service for machine learning inference.

Serves a fine-tuned GPT-2 causal model (`./fine_tuned_model`) with CORS enabled, logging, and config in `config.yaml`. Consumed by thread.

## Run

```bash
pip install -r requirements.txt
python main.py
```

Docker (`Dockerfile`) and Kubernetes (`k8s.yaml`) manifests included. Responses generated via `generate_response.py`; server entry in `api_server.py`.

## Stack

FastAPI, Transformers, PyTorch.

## License

MIT.
