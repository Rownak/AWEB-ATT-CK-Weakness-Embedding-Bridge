# AWEB-ATT&CK-Weakness-Embedding-Bridge
AWEB (ATT&amp;CK Weakness Embedding Bridge) is a model designed to connect ATT&amp;CK techniques with Common Weakness Enumerations (CWEs) by leveraging advanced embeddings from LLMs and graph-based networks. This repository provides a simple interface to retrieve ATT&amp;CK and CWE descriptions relevant to a given query using the AWEB model.

# Setup Instructions:
Follow the steps below to set up the environment and run the application:

1. Create a Virtual Environment
   
   -> python3 -m venv aweb_env
   
   -> source aweb_env/bin/activate # For Linux/Mac
   
   -> aweb_env\Scripts\activate   # For Windows
   
3. Install Dependencies
   
   Option A: Use requirements.txt
   
   Install all required libraries with a single command:
   
   -> pip install -r requirements.txt
   
   Or follow Option B to install the required libraries manually.
   
   Option B: Manual Installation
   
   If requirements.txt does not work, manually install the required libraries:
   
   -> pip install torch==1.13.1
   
   -> pip install transformers==4.47.1
   
   -> pip install scikit-learn==1.2.0
   
   -> pip install sentence-transformers==3.2.0
   
   -> pip install gradio==4.44.1
   
   -> pip install numpy==1.24.3
   
5. Navigate to the Application Directory
   
   Change into the application folder where the main script is located:
   
   -> cd application
   
7. Run the Application
   
   Start the application by running the main script:
   
   -> python app.py

# Notes
Ensure that Python 3.8 or higher is installed on your system.
Use the virtual environment created in Step 1 to avoid dependency conflicts.
# Contact
For questions or support, please open an issue in the repository or reach out to the maintainers - (rownak.utep@gmail.com)
