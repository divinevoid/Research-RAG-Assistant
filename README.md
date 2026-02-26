# 📚 Research RAG Assistant

A **Research-oriented Retrieval-Augmented Generation (RAG) Assistant** designed to help researchers efficiently discover, evaluate, and ideate on academic literature.

This system retrieves **relevant research papers** for a given query, **reranks** them using a **hybrid scoring approach**, and generates **novel research ideas** grounded in the retrieved papers along with **novelty scores**.

---

## 🚀 Key Features

- 🔍 **Query-based Paper Retrieval**
- 📊 **Hybrid Scoring Mechanism**
  - Semantic similarity
  - Relevance-based reranking
- 🧠 **LLM-powered Novel Research Idea Generation**
- 🧪 **Novelty Scoring** for generated ideas
- 📈 **Evaluation Pipeline** for quantitative assessment
- 🌐 **Interactive Streamlit Interface**

---

## 🏗️ Project Architecture

Research-RAG-Assistant/
│
├── app.py                  # Streamlit application
├── create_embeddings.py    # Embedding creation pipeline
├── run_evaluation.py       # Evaluation script
├── requirements.txt        # Project dependencies
├── .env                    # API keys (not committed)
├── eval/                   # Evaluation logic and metrics
└── a.py                    # Core RAG pipeline logic

## ▶️ How to Run the Project

Follow the steps below to set up and run the **Research RAG Assistant** locally.

---

### 1️⃣ Create a Virtual Environment

**Windows**
```bash
python -m venv .venv
```
**Macos/Linux**
```bash
python3 -m venv .venv
```
### 2️⃣ Activate the Virtual Environment

**Windows**
```bash
.venv\Scripts\activate
```
**Macos/Linux**
```bash
source .venv/bin/activate
```
### 3️⃣ Install Project Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Configure Environment Variables

Create a .env file in the project root and add your Gemini API key:
GEMINI_API_KEY=your_api_key_here

### 5️⃣ Generate Embeddings

Run the embedding creation script:
```bash
python create_embeddings.py
```

### 6️⃣ Run the Application

Start the Streamlit app:
```bash
streamlit run app.py
```

### 7️⃣ Run Evaluation (Optional)

Evaluate retrieval results
```bash
python run_evaluation.py
```
