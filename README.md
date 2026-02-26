# Research-RAG-Assistant
This is research RAG assistant that can be used to retrieve relevant papers based on given query, rernaks them, uses a hybrid scoring method and generates novel research ideas grounded from retrieved papers along with the novelty scores.

**To run the code**
**1. Create a virtual environment**
python -m venv .venv (for windows)
python3 -m venv .venv (for ios)

**2. Run the virtual environment**
.venv\Scripts\activate (for windows)
source .venv/bin/activate (for ios)

**3. Install the requirements**
pip install -r requirements.txt

**4. Create a .env file and put your gemini api key inside it.**

**5. Run create_embeddings.py**
python create_embeddings.py

**6. Run the app.py**
streamlit run app.py

**7. Run the run_evaluation.py to evaluate the results**
python run_evaluation.py

