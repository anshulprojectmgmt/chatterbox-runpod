import torch
import torchaudio
import io
import base64
import runpod
import tempfile
import os
from src.chatterbox.tts import ChatterboxTTS
from src.chatterbox.vc import ChatterboxVC

# --- Global State to hold models ---
MODELS = {
    "tts_model": None,
    "vc_model": None,
    "device": "cpu"
}

# --- 1. The init() function (Runs ONCE per worker) ---
def init():
    """
    Load the models into memory once when the worker starts.
    This runs on "warm-up" and "cold start".
    """
    global MODELS
    
    # Set the device, prioritizing cuda
    if torch.cuda.is_available():
        MODELS["device"] = "cuda"
    else:
        MODELS["device"] = "cpu"
        
    print(f"Loading models onto device: {MODELS['device']}")
    
    try:
        MODELS["tts_model"] = ChatterboxTTS.from_pretrained(device=MODELS["device"])
        print("ChatterboxTTS model loaded.")
        MODELS["vc_model"] = ChatterboxVC.from_pretrained(device=MODELS["device"])
        print("ChatterboxVC model loaded.")
    except Exception as e:
        print(f"CRITICAL: Failed to load models: {e}")
        return None # Signal failure

# --- 2. The handler() function (Runs for EVERY job) ---
def handler(job):
    """
    Process one inference job.
    """
    job_input = job['input']
    
    # Check if models are loaded
    if MODELS["tts_model"] is None or MODELS["vc_model"] is None:
        return {"error": "Models are not loaded. Worker may have failed to initialize."}

    # Determine which task to run
    task_type = job_input.get("task", "tts")
    
    try:
        if task_type == "tts":
            return handle_tts(job_input)
        elif task_type == "vc":
            return handle_vc(job_input)
        else:
            return {"error": f"Unknown task type: {task_type}"}
    except Exception as e:
        print(f"Unhandled exception in handler: {e}")
        return {"error": str(e)}

def handle_tts(job_input):
    """Handles the Text-to-Speech task."""
    model = MODELS["tts_model"]
            
    # Get inputs
    text = job_input.get("text")
    ref_audio_b64 = job_input.get("ref_audio_b64")
    if not text or not ref_audio_b64:
        return {"error": "Missing 'text' or 'ref_audio_b64'"}

    # Serverless I/O is in base64. We must decode the string.
    audio_bytes = base64.b64decode(ref_audio_b64)
    
    # We still use a tempfile for the model
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp_audio:
        tmp_audio.write(audio_bytes)
        tmp_audio.flush() # Ensure file is written
        
        # Run generation with optional params
        model.prepare_conditionals(
            tmp_audio.name,
            exaggeration=job_input.get("exaggeration", 0.5)
        )
        wav_tensor = model.generate(
            text,
            temperature=job_input.get("temperature", 0.8),
            cfg_weight=job_input.get("cfg_weight", 0.5)
        )

    # Convert output tensor to in-memory WAV buffer
    buffer = io.BytesIO()
    torchaudio.save(buffer, wav_tensor.cpu(), model.sr, format="wav")
    buffer.seek(0)
    
    # Encode the audio file as base64 to send back
    output_audio_b64 = base64.b64encode(buffer.read()).decode('utf-8')
    
    # Return the output
    return {"audio_b64": output_audio_b64, "content_type": "audio/wav"}

def handle_vc(job_input):
    """Handles the Voice Conversion task."""
    model = MODELS["vc_model"]

    # Get inputs
    source_audio_b64 = job_input.get("source_audio_b64")
    target_voice_b64 = job_input.get("target_voice_b64")
    if not source_audio_b64 or not target_voice_b64:
        return {"error": "Missing 'source_audio_b64' or 'target_voice_b64'"}

    # Decode both files
    source_bytes = base64.b64decode(source_audio_b64)
    target_bytes = base64.b64decode(target_voice_b64)

    # We need two temp files
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp_source, \
         tempfile.NamedTemporaryFile(suffix=".wav") as tmp_target:
        
        tmp_source.write(source_bytes)
        tmp_source.flush()
        tmp_target.write(target_bytes)
        tmp_target.flush()

        # Run generation
        model.set_target_voice(tmp_target.name)
        wav_tensor = model.generate(audio=tmp_source.name)

    # Convert output to base64
    buffer = io.BytesIO()
    torchaudio.save(buffer, wav_tensor.cpu(), model.sr, format="wav")
    buffer.seek(0)
    output_audio_b64 = base64.b64encode(buffer.read()).decode('utf-8')

    return {"audio_b64": output_audio_b64, "content_type": "audio/wav"}


# --- 3. Start the RunPod worker ---
if __name__ == "__main__":
    print("Starting RunPod Serverless Worker...")
    runpod.serverless.start({
        "init": init,
        "handler": handler
    })