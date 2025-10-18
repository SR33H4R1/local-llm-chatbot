# Local Llama 3 8B Chatbot

This project is a fully functional, locally-hosted chatbot powered by the `meta-llama/Meta-Llama-3-8B-Instruct` model.

The primary goal of this project was to meet the job requirement of **"Proven ability to deploy and run open-source models locally."** It demonstrates hands-on experience with Generative AI, LLMs, and the Hugging Face ecosystem on consumer-grade hardware.

## 🚀 Live Demo

This video shows the chatbot running 100% locally on my NVIDIA 4060 8GB laptop. It successfully handles a complex request (`explain chatbot`) after loading, proving the stability of the application.



https://github.com/user-attachments/assets/5e8071ab-5d43-40d1-b69c-5f03c94ac9a6


## 🛠️ Key Features & Technical Challenges Solved

This project successfully overcomes the two main challenges of running large language models on a local machine: VRAM (GPU memory) and system RAM.

* **GPU VRAM Management (The 8GB VRAM Problem):**
    * The 8-billion parameter Llama 3 model is loaded using 4-bit quantization via the `bitsandbytes` library.
    * This "shrinks" the model from its original size (over 15GB) to fit comfortably within the 8GB of VRAM on the NVIDIA 4060.

* **Gated Model Access:**
    * Successfully handled the "gated" (protected) status of the Llama 3 model by authenticating with the Hugging Face Hub using `huggingface-cli login`.

* **Interactive Web UI:**
    * A simple, real-time, streaming chatbot interface was built using the `Gradio` library.

## 💻 Tech Stack

* **Python**
* **PyTorch**
* **Hugging Face `transformers`:** For loading the model and tokenizer.
* **`bitsandbytes`:** For 4-bit quantization.
* **`Gradio`:** For the web UI.
* **`accelerate`:** For optimizing hardware utilization.
* **`hf_xet`:** For high-speed model downloading.

## ⚙️ How to Run

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/SR33H4R1/local-llm-chatbot.git](https://github.com/SR33H4R1/local-llm-chatbot.git)
    cd local-llm-chatbot
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    # Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Log in to Hugging Face:**
    * *You must have already requested and been granted access to the [Llama 3 model on Hugging Face](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct).*
    ```bash
    huggingface-cli login
    ```
5.  **Run the application:**
    ```bash
    python app.py
    ```
6.  **Open the UI:**
    * Open the local URL in your browser (e.g., `http://127.0.0.1:7860`).
