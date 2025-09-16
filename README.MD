An interactive AI-powered automobile manual assistant that helps users query and understand automobile manuals with ease.
This project uses RAG (Retrieval Augmented Generation) with LangChain, HuggingFace embeddings, and ChromaDB to provide precise answers from automobile manuals.
It also supports voice input and text-to-speech responses, making it more user-friendly and accessible.

✨ Features

📘 Automobile Manual Q&A – Ask any question from the uploaded manual and get accurate answers.

🗂️ RAG Pipeline – Uses LangChain with HuggingFace embeddings and ChromaDB for efficient information retrieval.

🎤 Voice Input – Ask questions using speech recognition.

🔊 Voice Output – Get responses in natural speech (Text-to-Speech).

🖥️ Streamlit UI – Simple and interactive frontend interface.

⚡ Lightweight & Fast – Runs locally without requiring API keys (Groq / HuggingFace local models).

🛠️ Tech Stack

Frontend: Streamlit

Backend: Python

NLP & AI: LangChain, HuggingFace Transformers

Vector Database: ChromaDB

Embeddings: HuggingFace sentence-transformers

Voice Support: SpeechRecognition, gTTS / pyttsx3
📂 Project Structure
AI-Automobile-Assistant/
│── app.py                 # Main Streamlit application
│── requirements.txt       # Python dependencies
│── models/                # Pre-installed local LLM models
│── data/                  # Automobile manuals (PDFs)
│── vectorstore/           # ChromaDB storage
│── utils/                 # Helper functions (speech, text, etc.)
│── README.md              # Project documentation

⚙️ Installation & Setup

Clone this repository

git clone https://github.com/your-username/AI-Automobile-Assistant.git
cd AI-Automobile-Assistant


Create a virtual environment & activate it

python -m venv venv
source venv/bin/activate   # (Linux/Mac)
venv\Scripts\activate      # (Windows)


Install dependencies

pip install -r requirements.txt


Run the Streamlit app

streamlit run app.py

🚀 Usage

Upload your automobile manual (PDF).

Ask questions by typing or speaking.

The assistant will search, retrieve, and answer from the manual.

Listen to the response using text-to-speech.

📸 Screenshots

(Add your app UI images here — input box, voice recording, and answer display.)

🔮 Future Enhancements

Integration with multilingual support.

Advanced summarization of manuals.

Mobile app version for on-road accessibility.

Real-time sensor data integration with cars.