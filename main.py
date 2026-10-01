from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import os
import torch
from pydantic import BaseModel

# Initialize logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load fine-tuned model
MODEL_PATH = "./fine_tuned_model"
MODEL_NAME = "gpt2"

try:
    if os.path.exists(MODEL_PATH):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, local_files_only=True)
        MODEL_SOURCE = MODEL_PATH
    else:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        MODEL_SOURCE = MODEL_NAME
except Exception as e:
    logger.warning(f"Failed to load local model: {str(e)}, loading from remote")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    MODEL_SOURCE = MODEL_NAME

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    prompt: str


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        # Log incoming request
        logger.info("Received chat request")

        # Get user prompt
        prompt = request.prompt
        if not prompt:
            raise HTTPException(status_code=400, detail="Prompt is required")

        # Generate response
        inputs = tokenizer.encode(prompt, return_tensors="pt")
        outputs = model.generate(
            inputs,
            max_length=256,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
        )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Log response
        logger.info(f"Generated response: {response}")

        return {"response": response}

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/models")
async def get_models():
    """Get information about available models."""
    return {
        "models": [
            {
                "name": MODEL_SOURCE,
                "type": "causal_lm",
                "description": (
                    f"Autoregressive causal LM loaded from {MODEL_SOURCE}. "
                    "Produces text continuations rather than dialogue; it has no "
                    "instruction tuning and no chat template."
                ),
            }
        ]
    }


@app.get("/status")
async def get_status():
    """Get API and model status."""
    return {
        "status": "running",
        "model_loaded": model is not None,
        "model_name": MODEL_SOURCE,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
