# Car Manual RAG QA System

## Project Description

This project implements a Retrieval Augmented Generation (RAG) Question Answering system focused on car manuals. Users can ask questions about various car models (e.g., Mustang, Honda, Volkswagen, Daewoo), and the system will retrieve relevant information from a knowledge base of car manuals and generate an answer. The project features an intuitive user interface built with Streamlit.

## Prerequisites

*   Python 3.13.3.
*   Git

## Setup and Running the Project

### Method 1: Running Locally

1.  **Clone the Repository:**
    Open your terminal or command prompt and run:
    ```bash
    git clone https://github.com/samuil-ganev/Car-Chatbot-RAG-.git RAG
    cd RAG
    ```

2.  **Install Git LFS:**
    This project uses Git LFS to manage large model files. If you don't have Git LFS installed, please install it first. You can download it from [https://git-lfs.com](https://git-lfs.com).
    After installation, initialize Git LFS:
    ```bash
    git lfs install
    ```

3.  **Pull LFS Files:**
    After cloning and ensuring Git LFS is installed, pull the LFS files:
    ```bash
    git lfs pull
    ```

4.  **Create and Activate a Virtual Environment:**
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

5.  **Install Dependencies:**
    With the virtual environment activated, install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

6.  **Set Up Environment Variables:**
    You need a `credentials.json` file within the `assets/secrets/` directory.
    The `credentials.json` file should contain necessary credentials.

7.  **Run the Streamlit Application:**
    Start the Streamlit application:
    *   **On macOS and Linux:**
    ```bash
    streamlit run ./frontend/app.py
    ```
    *   **On Windows:**
    ```bash
    streamlit run .\frontend\app.py
    ```

    The application should open in your web browser.

---

### Method 2: Running with Docker

This method containerizes the application, making it portable. It uses the project's own `assets` directory for secrets and the vector store.

1.  **Clone the Repository (if not already done):**
    ```bash
    git clone https://github.com/samuil-ganev/Car-Chatbot-RAG-.git RAG
    cd RAG
    ```

2.  **Install Git LFS & Pull LFS Files:**
    ```bash
    git lfs install
    git lfs pull
    ```

3.  **Set Up Credentials for Docker (Crucial Step - Same as Local Setup):**
    *   You need a `credentials.json` file within the `assets/secrets/` directory of this cloned `RAG` project.
    *   The final path on your host machine should be `RAG/assets/secrets/credentials.json`.

4.  **Build the Docker Image:**
    Ensure you are in the root of the `RAG` project directory (where the `Dockerfile` is located).
    ```bash
    docker build -t car-bot-rag .
    ```
    This will build the Docker image and tag it as `car-bot-rag`.

5.  **Run the Docker Container:**
    The following commands assume you are running them from the root of the `RAG` project directory.

    *   **Command for PowerShell (Windows):**
        ```powershell
        docker run -d -p 8501:8501 --name car-bot-rag-instance `
          -v "${PWD}/assets/secrets/credentials.json:/app/assets/secrets/credentials.json:ro" `
          -v "${PWD}/assets/vector_store:/app/assets/vector_store" `
          car-bot-rag
        ```
    *   **Command for Bash (Linux/macOS/Git Bash on Windows):**
        ```bash
        docker run -d -p 8501:8501 --name car-bot-rag-instance \
          -v "$(pwd)/assets/secrets/credentials.json:/app/assets/secrets/credentials.json:ro" \
          -v "$(pwd)/assets/vector_store:/app/assets/vector_store" \
          car-bot-rag
        ```

6.  **Access the Application:**
    Once the container is running, open your web browser and navigate to `http://localhost:8501`.

### Managing the Docker Container

*   **View logs:**
    ```bash
    docker logs car-bot-rag-instance
    ```
*   **Stop the container:**
    ```bash
    docker stop car-bot-rag-instance
    ```
*   **Start a stopped container:**
    ```bash
    docker start car-bot-rag-instance
    ```
*   **Remove the container (must be stopped first):**
    ```bash
    docker rm car-bot-rag-instance
    ```

---
