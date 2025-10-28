import torch
import torchaudio
import io
import base64
import runpod
import tempfile
import os
import time
import gc # Import garbage collector

# --- Global State ---
# Models start as None and will be loaded on demand
MODELS = {
    "tts_model": None,
    "vc_model": None,
    "device": "cpu" # Will be determined later
}
MODELS_LOADED = {
    "tts": False,
    "vc": False
}

# --- 1. init() function - Now does almost nothing ---
def init():
    """
    Determine device, but DO NOT load models here.
    """
    global MODELS
    print(f"--- [{time.time():.2f}] init() function started (Lazy Loading Mode) ---")
    try:
        print(f"--- [{time.time():.2f}] Determining device...")
        if torch.cuda.is_available():
            MODELS["device"] = "cuda"
            print(f"--- [{time.time():.2f}] CUDA is available. Device set to cuda.")
        else:
            MODELS["device"] = "cpu"
            print(f"--- [{time.time():.2f}] CUDA not available. Device set to cpu.")
        print(f"--- [{time.time():.2f}] init() finished successfully (No models loaded yet). ---")
        return True # Signal init success
    except Exception as e_init:
        print(f"--- [{time.time():.2f}] ❌ CRITICAL: Error during basic init(): {e_init}")
        return None # Signal failure

# --- Helper function to load models on demand ---
def load_model(model_type):
    global MODELS, MODELS_LOADED
    device = MODELS["device"]
    
    if model_type == "tts" and not MODELS_LOADED["tts"]:
        print(f"--- [{time.time():.2f}] LAZY LOAD: Attempting to load ChatterboxTTS model onto device: {device} ---")
        try:
            from src.chatterbox.tts import ChatterboxTTS
            MODELS["tts_model"] = ChatterboxTTS.from_pretrained(device=device)
            MODELS_LOADED["tts"] = True
            print(f"--- [{time.time():.2f}] ✅ LAZY LOAD: ChatterboxTTS loaded successfully.")
            gc.collect() # Try to free up memory
            if device == 'cuda': torch.cuda.empty_cache()
            return True
        except Exception as e:
            print(f"--- [{time.time():.2f}] ❌ LAZY LOAD FAILED: ChatterboxTTS: {e}")
            MODELS_LOADED["tts"] = False # Mark as failed
            return False
            
    elif model_type == "vc" and not MODELS_LOADED["vc"]:
        print(f"--- [{time.time():.2f}] LAZY LOAD: Attempting to load ChatterboxVC model onto device: {device} ---")
        try:
            from src.chatterbox.vc import ChatterboxVC
            MODELS["vc_model"] = ChatterboxVC.from_pretrained(device=device)
            MODELS_LOADED["vc"] = True
            print(f"--- [{time.time():.2f}] ✅ LAZY LOAD: ChatterboxVC loaded successfully.")
            gc.collect() # Try to free up memory
            if device == 'cuda': torch.cuda.empty_cache()
            return True
        except Exception as e:
            print(f"--- [{time.time():.2f}] ❌ LAZY LOAD FAILED: ChatterboxVC: {e}")
            MODELS_LOADED["vc"] = False # Mark as failed
            return False
            
    # Model already loaded or invalid type
    return MODELS_LOADED.get(model_type, False)


# --- 2. handler() function - Now triggers loading ---
def handler(job):
    """
    Process one inference job, loading the required model if necessary.
    """
    job_input = job['input']
    task_type = job_input.get("task", "tts")
    print(f"\n--- [{time.time():.2f}] Handler received job ID: {job.get('id', 'N/A')}, Task: {task_type} ---")

    # Load the required model if it hasn't been loaded for this worker yet
    model_loaded_successfully = load_model(task_type)

    if not model_loaded_successfully:
        # Check if it failed during *this* load attempt or a previous one
        if MODELS_LOADED[task_type] is False: # Explicitly check for False (means load attempt failed)
             error_msg = f"Failed to load required model ({task_type}) during this request. Check logs."
        else: # Model is None, but LOADED is not False (means init might have failed basic checks)
             error_msg = f"Required model ({task_type}) is not available. Worker initialization might have failed."
        print(f"--- [{time.time():.2f}] {error_msg} ---")
        return {"error": error_msg}
        
    # Proceed with the correct handler if model is loaded
    try:
        print(f"--- [{time.time():.2f}] Model for task '{task_type}' ready. Proceeding with handler...")
        if task_type == "tts":
            result = handle_tts(job_input)
        elif task_type == "vc":
            result = handle_vc(job_input)
        else:
            result = {"error": f"Unknown task type: {task_type}"}
        
        print(f"--- [{time.time():.2f}] Handler finished for job ID: {job.get('id', 'N/A')} ---")
        # Optional: Aggressive memory cleanup after job
        # gc.collect()
        # if MODELS['device'] == 'cuda': torch.cuda.empty_cache()
        return result
        
    except Exception as e:
        print(f"--- [{time.time():.2f}] ❌ Unhandled exception in handler: {e}")
        traceback.print_exc() # Print full traceback to logs
        return {"error": f"Handler error: {str(e)}"}

# --- (Keep handle_tts and handle_vc functions - unchanged from previous full version) ---
def handle_tts(job_input):
    """Handles the Text-to-Speech task."""
    model = MODELS["tts_model"]
    # ... (rest of handle_tts function is the same) ...
    text = job_input.get("text")
    ref_audio_b64 = job_input.get("ref_audio_b64")
    if not text or not ref_audio_b64:
        return {"error": "Missing 'text' or 'ref_audio_b64'"}
    audio_bytes = base64.b64decode(ref_audio_b64)
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp_audio:
        tmp_audio.write(audio_bytes)
        tmp_audio.flush()
        model.prepare_conditionals(
            tmp_audio.name,
            exaggeration=job_input.get("exaggeration", 0.5)
        )
        wav_tensor = model.generate(
            text,
            temperature=job_input.get("temperature", 0.8),
            cfg_weight=job_input.get("cfg_weight", 0.5)
        )
    buffer = io.BytesIO()
    torchaudio.save(buffer, wav_tensor.cpu(), model.sr, format="wav")
    buffer.seek(0)
    output_audio_b64 = base64.b64encode(buffer.read()).decode('utf-8')
    return {"audio_b64": output_audio_b64, "content_type": "audio/wav"}

def handle_vc(job_input):
    """Handles the Voice Conversion task."""
    model = MODELS["vc_model"]
    # ... (rest of handle_vc function is the same) ...
    source_audio_b64 = job_input.get("source_audio_b64")
    target_voice_b64 = job_input.get("target_voice_b64")
    if not source_audio_b64 or not target_voice_b64:
        return {"error": "Missing 'source_audio_b64' or 'target_voice_b64'"}
    source_bytes = base64.b64decode(source_audio_b64)
    target_bytes = base64.b64decode(target_voice_b64)
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp_source, \
         tempfile.NamedTemporaryFile(suffix=".wav") as tmp_target:
        tmp_source.write(source_bytes)
        tmp_source.flush()
        tmp_target.write(target_bytes)
        tmp_target.flush()
        model.set_target_voice(tmp_target.name)
        wav_tensor = model.generate(audio=tmp_source.name)
    buffer = io.BytesIO()
    torchaudio.save(buffer, wav_tensor.cpu(), model.sr, format="wav")
    buffer.seek(0)
    output_audio_b64 = base64.b64encode(buffer.read()).decode('utf-8')
    return {"audio_b64": output_audio_b64, "content_type": "audio/wav"}

# --- Start the RunPod worker ---
if __name__ == "__main__":
    print("Starting RunPod Serverless Worker (Lazy Loading Mode)...")
    # Note: For local testing, init() won't run automatically here.
    # The load_model() function will be called by the first request.
    runpod.serverless.start({
        "init": init, # Still register init for basic setup
        "handler": handler
    })

# import runpod
# import time
# import sys

# # NO init function, NO model loading

# def handler(job):
#     """Super simple handler."""
#     print(f"--- [{time.time():.2f}] Handler received a job! ---")
#     job_input = job.get('input', {})
#     print(f"Input received: {job_input}")

#     # Just return a success message immediately
#     return {"message": "Handler ran successfully (simple test)", "received_input": job_input}

# if __name__ == "__main__":
#     print("--- Starting RunPod Serverless Worker (SIMPLE TEST MODE) ---")
#     # Check Python version
#     print(f"Python version: {sys.version}") 

#     # Try importing torch just to see if it works
#     try:
#         import torch
#         print(f"PyTorch version: {torch.__version__}")
#         if torch.cuda.is_available():
#             print(f"CUDA available: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}")
#         else:
#              print("CUDA not available.")
#     except ImportError as e:
#         print(f"❌ FAILED to import torch: {e}")
#     except Exception as e_torch:
#         print(f"❌ Error during torch check: {e_torch}")

#     print("Starting handler...")
#     runpod.serverless.start({
#         "handler": handler 
#         # No 'init' needed for this test
#     })
#     print("--- runpod.serverless.start() finished or failed ---") # Should ideally not be reached if it listens
