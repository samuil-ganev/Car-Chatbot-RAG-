# Car Manual RAG QA System

## Project Description

This project implements a Retrieval Augmented Generation (RAG) Question Answering system focused on car manuals. Users can ask questions about various car models (e.g., Mustang, Honda, Volkswagen, Daewoo), and the system will retrieve relevant information from a knowledge base of car manuals and generate an answer. The project features an intuitive user interface built with Streamlit.

## Prerequisites

*   Python 3.13.3.
*   Git

## Setup and Running the Project

1.  **Clone the Repository:**
    Open your terminal or command prompt and run:
    ```bash
    git clone https://github.wdf.sap.corp/C5397307/RAG.git
    cd RAG
    ```

2.  **Create and Activate a Virtual Environment:**
    It's highly recommended to use a virtual environment to manage project dependencies.

    *   **On macOS and Linux:**
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```
    *   **On Windows (Command Prompt):**
        ```bash
        python -m venv venv
        venv\Scripts\activate.bat
        ```
    *   **On Windows (PowerShell):**
        ```bash
        python -m venv venv
        .\venv\Scripts\Activate.ps1
        ```
    *   **On Windows (Git Bash):**
        ```bash
        python -m venv venv
        source venv/Scripts/activate
        ```

3.  **Install Dependencies:**
    With the virtual environment activated, install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Up Environment Variables:**
    You need to have credentials.json file in assets/secrets/

5.  **Run the Streamlit Application:**
    Start the Streamlit application:
    ```bash
    streamlit run .\frontend\app.py
    ```

    The application should open in your web browser.
