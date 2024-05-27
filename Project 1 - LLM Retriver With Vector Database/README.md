# Project Setup

## Steps to Set Up the Project

1. **Create a New Python Virtual Environment with `venv`** (install `venv` if not already installed)
   ```powershell
   python -m venv LLM_env
2. activate the virtual environment
   ```powershell
   .\LLM_env\Scripts\Activate

3. Install All Required Packages Specified in requirements.txt
   ```powershell
    pip install -r requirements.txt
4. Change the Name of `.env.shared` to `.env`
5. Enter Your API Keys
    Open the `.env` file and enter your OpenAI API key and Pinecone API key in the appropriate variables.
6. Run the first script - **download_pdf.py** for download all PDF files
   ```powershell
   python download_pdf.py
7. Run the second script - **vectorizing_pdf.py** - vectorize all pdf to pinecone
   ```powershell
   python vectorizing_pdf.py
7. Run the third script - **chat_bot_app.py** - launching the chatbot app
   ```powershell
   streamlit run chat_bot_app.py