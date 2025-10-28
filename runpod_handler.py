# import torch
# import torchaudio
# import io
# import base64
# import runpod
# import tempfile
# import os
# import time # Import time for timestamping if needed

# # --- Global State to hold models ---
# MODELS = {
#     "tts_model": None,
#     "vc_model": None,
#     "device": "cpu"
# }

# # --- 1. The init() function (Runs ONCE per worker) ---
# def init():
#     """
#     Load the models into memory once when the worker starts.
#     This runs on "warm-up" and "cold start".
#     """
#     global MODELS
#     print(f"--- [{time.time():.2f}] init() function started ---")

#     try:
#         print(f"--- [{time.time():.2f}] Importing ChatterboxTTS...")
#         from src.chatterbox.tts import ChatterboxTTS
#         print(f"--- [{time.time():.2f}] ChatterboxTTS imported successfully.")

#         print(f"--- [{time.time():.2f}] Importing ChatterboxVC...")
#         from src.chatterbox.vc import ChatterboxVC
#         print(f"--- [{time.time():.2f}] ChatterboxVC imported successfully.")

#         print(f"--- [{time.time():.2f}] Determining device...")
#         if torch.cuda.is_available():
#             print(f"--- [{time.time():.2f}] CUDA is available.")
#             MODELS["device"] = "cuda"
#         else:
#             print(f"--- [{time.time():.2f}] CUDA not available, falling back to CPU.")
#             MODELS["device"] = "cpu"
#         print(f"--- [{time.time():.2f}] Device set to: {MODELS['device']}")

#         # --- Load TTS Model ---
#         print(f"--- [{time.time():.2f}] Attempting to load ChatterboxTTS model onto device: {MODELS['device']}")
#         try:
#             print(f"--- [{time.time():.2f}] Calling ChatterboxTTS.from_pretrained...")
#             MODELS["tts_model"] = ChatterboxTTS.from_pretrained(device=MODELS["device"])
#             print(f"--- [{time.time():.2f}] ✅ ChatterboxTTS model loaded successfully.")
#         except Exception as e_tts:
#             print(f"--- [{time.time():.2f}] ❌ CRITICAL: Failed during ChatterboxTTS.from_pretrained: {e_tts}")
#             MODELS["tts_model"] = None # Ensure it's None on failure
#             print(f"--- [{time.time():.2f}] init() aborted due to TTS model load failure ---")
#             return None # Exit init early

#         # --- Load VC Model ---
#         # (Only runs if TTS loaded)
#         print(f"--- [{time.time():.2f}] Attempting to load ChatterboxVC model onto device: {MODELS['device']}")
#         try:
#             print(f"--- [{time.time():.2f}] Calling ChatterboxVC.from_pretrained...")
#             MODELS["vc_model"] = ChatterboxVC.from_pretrained(device=MODELS["device"])
#             print(f"--- [{time.time():.2f}] ✅ ChatterboxVC model loaded successfully.")
#         except Exception as e_vc:
#             print(f"--- [{time.time():.2f}] ❌ CRITICAL: Failed during ChatterboxVC.from_pretrained: {e_vc}")
#             MODELS["vc_model"] = None # Ensure it's None on failure
#             print(f"--- [{time.time():.2f}] init() aborted due to VC model load failure ---")
#             return None # Exit init early

#         # Final check
#         if MODELS["tts_model"] is None or MODELS["vc_model"] is None:
#              print(f"--- [{time.time():.2f}] init() finished, but one or more models failed to load. ---")
#              return None # Signal failure

#         print(f"--- [{time.time():.2f}] init() finished successfully. Models ready. ---")
#         return True # Explicitly signal success

#     except ImportError as e_import:
#         print(f"--- [{time.time():.2f}] ❌ CRITICAL: Failed during import: {e_import}")
#         print(f"--- [{time.time():.2f}] init() aborted due to import error ---")
#         return None
#     except Exception as e_init:
#         print(f"--- [{time.time():.2f}] ❌ CRITICAL: Unhandled exception during init(): {e_init}")
#         print(f"--- [{time.time():.2f}] init() aborted due to unhandled exception ---")
#         return None # Signal failure

# # --- 2. The handler() function (Runs for EVERY job) ---
# # (Keep your existing handler function here - no changes needed for logging)
# def handler(job):
#     """
#     Process one inference job.
#     """
#     job_input = job['input']
    
#     # Check if models were loaded successfully during init
#     # We check explicitly for the success signal (True) if you prefer,
#     # or just check if models are not None.
#     if MODELS["tts_model"] is None or MODELS["vc_model"] is None:
#         return {"error": "Models are not loaded. Worker failed during initialization. Check init logs."}

#     # Determine which task to run
#     task_type = job_input.get("task", "tts")
    
#     try:
#         if task_type == "tts":
#             return handle_tts(job_input)
#         elif task_type == "vc":
#             return handle_vc(job_input)
#         else:
#             return {"error": f"Unknown task type: {task_type}"}
#     except Exception as e:
#         print(f"--- [{time.time():.2f}] Unhandled exception in handler: {e}")
#         return {"error": str(e)}

# # (Keep your handle_tts and handle_vc functions here)
# def handle_tts(job_input):
#     """Handles the Text-to-Speech task."""
#     model = MODELS["tts_model"]
            
#     # Get inputs
#     text = job_input.get("text")
#     ref_audio_b64 = job_input.get("ref_audio_b64")
#     if not text or not ref_audio_b64:
#         return {"error": "Missing 'text' or 'ref_audio_b64'"}

#     # Serverless I/O is in base64. We must decode the string.
#     audio_bytes = base64.b64decode(ref_audio_b64)
    
#     # We still use a tempfile for the model
#     with tempfile.NamedTemporaryFile(suffix=".wav") as tmp_audio:
#         tmp_audio.write(audio_bytes)
#         tmp_audio.flush() # Ensure file is written
        
#         # Run generation with optional params
#         model.prepare_conditionals(
#             tmp_audio.name,
#             exaggeration=job_input.get("exaggeration", 0.5)
#         )
#         wav_tensor = model.generate(
#             text,
#             temperature=job_input.get("temperature", 0.8),
#             cfg_weight=job_input.get("cfg_weight", 0.5)
#         )

#     # Convert output tensor to in-memory WAV buffer
#     buffer = io.BytesIO()
#     torchaudio.save(buffer, wav_tensor.cpu(), model.sr, format="wav")
#     buffer.seek(0)
    
#     # Encode the audio file as base64 to send back
#     output_audio_b64 = base64.b64encode(buffer.read()).decode('utf-8')
    
#     # Return the output
#     return {"audio_b64": output_audio_b64, "content_type": "audio/wav"}

# def handle_vc(job_input):
#     """Handles the Voice Conversion task."""
#     model = MODELS["vc_model"]

#     # Get inputs
#     source_audio_b64 = job_input.get("source_audio_b64")
#     target_voice_b64 = job_input.get("target_voice_b64")
#     if not source_audio_b64 or not target_voice_b64:
#         return {"error": "Missing 'source_audio_b64' or 'target_voice_b64'"}

#     # Decode both files
#     source_bytes = base64.b64decode(source_audio_b64)
#     target_bytes = base64.b64decode(target_voice_b64)

#     # We need two temp files
#     with tempfile.NamedTemporaryFile(suffix=".wav") as tmp_source, \
#          tempfile.NamedTemporaryFile(suffix=".wav") as tmp_target:
        
#         tmp_source.write(source_bytes)
#         tmp_source.flush()
#         tmp_target.write(target_bytes)
#         tmp_target.flush()

#         # Run generation
#         model.set_target_voice(tmp_target.name)
#         wav_tensor = model.generate(audio=tmp_source.name)

#     # Convert output to base64
#     buffer = io.BytesIO()
#     torchaudio.save(buffer, wav_tensor.cpu(), model.sr, format="wav")
#     buffer.seek(0)
#     output_audio_b64 = base64.b64encode(buffer.read()).decode('utf-8')

#     return {"audio_b64": output_audio_b64, "content_type": "audio/wav"}


# # --- 3. Start the RunPod worker ---
# if __name__ == "__main__":
#     print("Starting RunPod Serverless Worker...")
#     runpod.serverless.start({
#         "init": init,
#         "handler": handler
#     })


import runpod
import time
import sys

# NO init function, NO model loading

def handler(job):
    """Super simple handler."""
    print(f"--- [{time.time():.2f}] Handler received a job! ---")
    job_input = job.get('input', {})
    print(f"Input received: {job_input}")

    # Just return a success message immediately
    return {"message": "Handler ran successfully (simple test)", "received_input": job_input}

if __name__ == "__main__":
    print("--- Starting RunPod Serverless Worker (SIMPLE TEST MODE) ---")
    # Check Python version
    print(f"Python version: {sys.version}") 

    # Try importing torch just to see if it works
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"CUDA available: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}")
        else:
             print("CUDA not available.")
    except ImportError as e:
        print(f"❌ FAILED to import torch: {e}")
    except Exception as e_torch:
        print(f"❌ Error during torch check: {e_torch}")

    print("Starting handler...")
    runpod.serverless.start({
        "handler": handler 
        # No 'init' needed for this test
    })
    print("--- runpod.serverless.start() finished or failed ---") # Should ideally not be reached if it listens
