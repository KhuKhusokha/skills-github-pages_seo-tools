# app.py
import os
import requests
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from bs4 import BeautifulSoup
# Selenium imports remain if needed for other potential tools
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options as ChromeOptions
# Ollama import
import ollama
# ---- Hugging Face Imports ----
try:
    from transformers import pipeline
    import torch # Assuming PyTorch backend
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
# ----------------------------
import json
import logging
from datetime import datetime
from urllib.parse import quote_plus

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Model configurations
OLLAMA_MODEL = 'gemma3:1b' # Your preferred Ollama model
# ---- Fallback Hugging Face Model ----
# Replace this with your desired <3B uncensored Llama 3 model if you find one
HF_MODEL_FALLBACK = "google/gemma-2b"
# -----------------------------------

# --- Global State Variables ---
ollama_available = False
hf_pipeline = None
GENERATION_METHOD = "None" # Track which method is active

# --- Flask App Initialization ---
# Pass static_url_path='' if CSS/JS are separate and in 'static' folder
# If CSS/JS are embedded in index.html, this isn't strictly needed
app = Flask(__name__, template_folder='templates')
CORS(app)

# --- Helper Functions ---

# (Keep your get_webdriver, fetch_page_source etc. if needed)

def initialize_hf_pipeline():
    """Initializes the Hugging Face pipeline if transformers are available."""
    global hf_pipeline, GENERATION_METHOD
    if not TRANSFORMERS_AVAILABLE:
        logging.error("Transformers library not installed. Cannot initialize Hugging Face pipeline.")
        GENERATION_METHOD = "Error: Transformers Missing"
        return False

    if hf_pipeline is None: # Initialize only once
        try:
            logging.info(f"Initializing Hugging Face fallback pipeline with model: {HF_MODEL_FALLBACK}...")
            # Using device_map="auto" helps automatically use GPU if available
            # You might need to install `accelerate` for device_map: pip install accelerate
            hf_pipeline = pipeline(
                "text-generation",
                model=HF_MODEL_FALLBACK,
                torch_dtype=torch.bfloat16, # Use bfloat16 for efficiency if supported
                device_map="auto" # Automatically uses CUDA if available, else CPU
            )
            logging.info("Hugging Face pipeline initialized successfully.")
            GENERATION_METHOD = "Hugging Face"
            return True
        except Exception as e:
            logging.error(f"Failed to initialize Hugging Face pipeline ({HF_MODEL_FALLBACK}): {e}")
            hf_pipeline = None # Ensure it's None if init fails
            GENERATION_METHOD = "Error: HF Init Failed"
            return False
    return True # Already initialized

def generate_text(prompt, tool_name):
    """Generates text using Ollama if available, otherwise falls back to Hugging Face."""
    global ollama_available, hf_pipeline, GENERATION_METHOD
    result_text = ""
    source = "Unknown"
    error_message = None

    try:
        if ollama_available:
            source = f"Ollama ({OLLAMA_MODEL})"
            logging.info(f"Attempting generation via {source} for {tool_name}...")
            response = ollama.chat(model=OLLAMA_MODEL, messages=[{'role': 'user', 'content': prompt}])
            result_text = response['message']['content']
            logging.info(f"Received response from {source} for {tool_name}.")
        elif TRANSFORMERS_AVAILABLE:
            source = f"Hugging Face ({HF_MODEL_FALLBACK})"
            logging.info(f"Ollama unavailable. Attempting generation via {source} for {tool_name}...")
            # Initialize HF pipeline if it hasn't been already
            if hf_pipeline is None:
                if not initialize_hf_pipeline(): # Try to initialize
                     error_message = f"Error: Failed to initialize Hugging Face fallback model ({HF_MODEL_FALLBACK})."
                     logging.error(error_message)
                     source = "Error" # Update source to reflect error state
                # If initialize_hf_pipeline failed, hf_pipeline is still None

            # Proceed only if pipeline is now available
            if hf_pipeline:
                # Note: Adjust max_length and other generation parameters as needed
                # HF pipeline expects a slightly different call structure
                outputs = hf_pipeline(prompt, max_length=700, num_return_sequences=1, truncation=True)
                result_text = outputs[0]['generated_text']
                 # Optional: remove the prompt from the beginning if the model includes it
                if result_text.startswith(prompt):
                    result_text = result_text[len(prompt):].lstrip()
                logging.info(f"Received response from {source} for {tool_name}.")
            elif not error_message: # If pipeline is None but no init error was logged yet
                error_message = "Error: Hugging Face pipeline is not available after initialization attempt."
                logging.error(error_message)
                source = "Error"

        else: # Ollama unavailable AND Transformers library missing
             error_message = "Error: Ollama is unavailable and the Transformers library is not installed. Cannot generate text."
             logging.error(error_message)
             source = "Error"

    except Exception as e:
        logging.error(f"Error during text generation via {source} for {tool_name}: {e}")
        error_message = f"Error during text generation using {source}: {e}"
        source = "Error" # Mark as error

    # Format the final output or error message
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if source != "Error":
        branded_result = f"--- Wolfgank AI [{tool_name}] Result ({source} @ {timestamp}) ---\n\n{result_text}\n\n--- End of Report ---"
        return branded_result
    else:
        # Provide a more informative error message
        fallback_info = ""
        if GENERATION_METHOD.startswith("Error"):
            fallback_info = f" Fallback Status: {GENERATION_METHOD}."
        elif not ollama_available and not TRANSFORMERS_AVAILABLE:
             fallback_info = " Fallback attempted but Transformers library is missing."
        elif not ollama_available and hf_pipeline is None:
             fallback_info = f" Fallback attempted but HF model ({HF_MODEL_FALLBACK}) failed to initialize."

        final_error_message = error_message or "An unknown error occurred during text generation."
        return f"--- Wolfgank AI Error ({timestamp}) ---\n\n{final_error_message}{fallback_info}\nPlease check logs and model availability.\n\n--- End of Report ---"


# --- NEW ROUTE TO SERVE THE FRONTEND ---
@app.route('/')
def index():
    """Serves the main HTML page."""
    try:
        # Flask looks for 'index.html' in the 'templates' folder
        return render_template('index.html')
    except Exception as e:
        logging.error(f"Error rendering template: {e}")
        return "Error loading the application interface.", 500

# --- API Endpoints ---
# Ensure ALL endpoints now call the new `generate_text` function

@app.route('/api/keyword-hunter', methods=['POST'])
def keyword_hunter():
    try:
        data = request.get_json()
        topic = data.get('topic')
        if not topic:
            return jsonify({"error": "Topik diperlukan."}), 400
        # Keep the prompt specific to the tool's goal
        base_prompt = f"Berikan daftar 15-20 kata kunci SEO long-tail yang relevan untuk topik: '{topic}'. Kategorikan kata kunci ini (misalnya, Informatif, Navigasi, Transaksional, Komersial). Fokus pada Bahasa Indonesia. Format sebagai daftar."
        # Call the unified generation function
        result = generate_text(base_prompt, "Keyword Hunter")
        # Check if the result indicates an error internally (based on formatting)
        if "--- Wolfgank AI Error" in result:
             return jsonify({"error": result}), 500 # Pass the formatted error back
        return jsonify({"result": result})
    except Exception as e:
        logging.error(f"Error in /api/keyword-hunter endpoint: {e}")
        return jsonify({"error": f"Terjadi kesalahan server: {e}"}), 500

# --- Add the rest of your API endpoints here, ensuring they call `generate_text` ---
@app.route('/api/meta-master', methods=['POST'])
def meta_master():
    try:
        data = request.get_json()
        content = data.get('content')
        keywords = data.get('keywords', '')
        if not content:
            return jsonify({"error": "Konten diperlukan."}), 400
        base_prompt = f"Buatlah Judul SEO (Meta Title) yang menarik (maksimal 60 karakter) dan Deskripsi Meta (Meta Description) yang efektif (maksimal 160 karakter) dalam Bahasa Indonesia untuk konten berikut. Jika ada, pertimbangkan kata kunci ini: '{keywords}'.\n\nKonten:\n{content[:1000]}..."
        result = generate_text(base_prompt, "Meta Master")
        if "--- Wolfgank AI Error" in result:
             return jsonify({"error": result}), 500
        return jsonify({"result": result})
    except Exception as e:
         logging.error(f"Error in /api/meta-master endpoint: {e}")
         return jsonify({"error": f"Terjadi kesalahan server: {e}"}), 500

@app.route('/api/article-forge', methods=['POST'])
def article_forge():
    try:
        data = request.get_json()
        topic = data.get('topic')
        keywords = data.get('keywords', '')
        if not topic:
            return jsonify({"error": "Topik diperlukan."}), 400
        base_prompt = f"Tulis draf artikel blog SEO-friendly sekitar 500-700 kata dalam Bahasa Indonesia tentang topik '{topic}'. Masukkan kata kunci berikut secara alami jika memungkinkan: '{keywords}'. Sertakan judul, pendahuluan, beberapa subjudul (H2), dan kesimpulan."
        result = generate_text(base_prompt, "Article Forge")
        if "--- Wolfgank AI Error" in result:
             return jsonify({"error": result}), 500
        return jsonify({"result": result})
    except Exception as e:
        logging.error(f"Error in /api/article-forge endpoint: {e}")
        return jsonify({"error": f"Terjadi kesalahan server: {e}"}), 500


@app.route('/api/seo-analyzer', methods=['POST'])
def seo_analyzer():
    try:
        data = request.get_json()
        url = data.get('url')
        if not url:
            return jsonify({"error": "URL diperlukan."}), 400
        # Placeholder - replace with actual analysis if implemented
        simulated_analysis = f"Analisis SEO Awal untuk {url}:\n- Kecepatan Muat: (Perlu alat eksternal)\n- Responsif Seluler: (Perlu alat eksternal)\n- Tag Judul: (Ambil dari sumber halaman jika memungkinkan)\n- Deskripsi Meta: (Ambil dari sumber halaman jika memungkinkan)\n- Penggunaan HTTPS: Ya\n- Tautan Rusak: (Perlu pemeriksaan tautan)\n\n(Analisis ini disimulasikan.)"
        base_prompt = f"Anda adalah asisten SEO. Berdasarkan URL '{url}' dan analisis awal (simulasi) ini:\n{simulated_analysis}\n\nBerikan ringkasan singkat tentang potensi masalah SEO on-page utama dan saran perbaikan umum dalam Bahasa Indonesia. Fokus pada aspek yang dapat dievaluasi dari data yang diberikan atau pengetahuan SEO umum."
        result = generate_text(base_prompt, "SEO Analyzer")
        if "--- Wolfgank AI Error" in result:
             return jsonify({"error": result}), 500
        return jsonify({"result": result})
    except Exception as e:
        logging.error(f"Error in /api/seo-analyzer endpoint: {e}")
        return jsonify({"error": f"Terjadi kesalahan server: {e}"}), 500


@app.route('/api/news-radar', methods=['POST'])
def news_radar():
    try:
        data = request.get_json()
        search_query = data.get('search_query')
        if not search_query:
            return jsonify({"error": "Query pencarian diperlukan."}), 400
        # Placeholder - replace with actual news fetching if implemented
        simulated_news_snippets = [
            f"Tren pencarian untuk '{search_query}' meningkat di Google Trends minggu ini.",
            f"Sebuah artikel di Kompasiana membahas dampak '{search_query}' pada UMKM lokal.",
            f"Pemerintah mengumumkan regulasi baru yang mungkin mempengaruhi industri terkait '{search_query}'."
        ]
        news_context = "\\n".join(simulated_news_snippets)
        base_prompt = f"Topik/Kata Kunci Pencarian Berita: '{search_query}'\nBerikut adalah beberapa rangkuman berita/informasi terkini yang relevan (simulasi):\n{news_context}\nBerikan analisis singkat mengenai berita ini dalam Bahasa Indonesia. Apa implikasinya dari sudut pandang SEO atau konten? Apa tren utama yang terlihat?"
        result = generate_text(base_prompt, "News Radar")
        if "--- Wolfgank AI Error" in result:
             return jsonify({"error": result}), 500
        return jsonify({"result": result})

    except Exception as e:
        logging.error(f"Error in /api/news-radar endpoint: {e}")
        return jsonify({"error": f"Terjadi kesalahan server: {e}"}), 500


# --- Main Execution ---
if __name__ == '__main__':
    # Check Ollama availability on startup
    try:
        logging.info(f"Checking Ollama availability (Model: {OLLAMA_MODEL})...")
        # A simple check like listing models is enough
        ollama.list()
        ollama_available = True
        GENERATION_METHOD = f"Ollama ({OLLAMA_MODEL})"
        logging.info(f"Ollama service detected. Using primary generation method: {GENERATION_METHOD}")
    except Exception as e:
        ollama_available = False
        logging.warning(f"Could not connect to Ollama. Will attempt Hugging Face fallback. Error: {e}")
        # Attempt to initialize HF pipeline immediately if Ollama fails
        initialize_hf_pipeline() # Result logged within the function

    # Start Flask server
    logging.info(f"Starting Flask server. Text Generation active method: {GENERATION_METHOD}")
    app.run(host='0.0.0.0', port=5000, debug=True) # Remove debug=True for production