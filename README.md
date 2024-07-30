# Chat with the University

Welcome to the "Chat with the University" application! This application allows users to interact with an AI model to get information about various university topics. It uses `Streamlit` for the web interface, `LangChain` for embeddings and conversation management, and HuggingFace models for language generation.

## Features

- **Interactive Chat**: Users can ask questions about various university topics.
- **Topic-Based Retrieval**: The application supports multiple topics by using pre-built vector stores.
- **Advanced AI Model**: Leverages HuggingFace models for natural language understanding and generation.

## Requirements

To run this application, you will need:

- Python 3.8 or higher
- Streamlit
- LangChain
- HuggingFace Transformers
- PyTorch
- Additional libraries (see `requirements.txt`)

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/your-username/chat-with-the-university.git
   cd chat-with-the-university
   ```

2. **Create a Virtual Environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up API Keys**

   Create a `.streamlit/secrets.toml` file in the project directory and add your HuggingFace API key:

   ```toml
   [general]
   HUGGINGFACEHUB_API_KEY = "your_huggingface_api_key_here"
   ```

## Usage

1. **Run the Streamlit App**

   ```bash
   streamlit run app.py
   ```

2. **Interact with the App**

   Open your browser and navigate to `http://localhost:8501`. You will see the chat interface where you can ask questions about university topics.

3. **Select a Topic**

   Use the sidebar to select a topic related to your query. The application will load the appropriate vector store and process your questions.

## Application Structure

- **`app.py`**: Main Streamlit application file that handles user input and interaction.
- **`requirements.txt`**: Lists all Python package dependencies.
- **`htmlTemplates.py`**: Contains HTML templates for styling chat messages.
- **`embedding_script.py`**: Script used to create and save FAISS vector stores from web content.

## Code Overview

- **`app.py`**: Manages the Streamlit app interface and integrates with LangChain for conversational retrieval.
  - Loads API keys.
  - Provides functions to handle user input and manage chat history.
  - Sets up vector stores and conversation chains based on selected topics.

- **`embedding_script.py`**: Handles the creation and storage of FAISS vector stores.
  - Retrieves and processes web content.
  - Splits content into chunks and embeds it.
  - Saves vector stores locally for efficient retrieval.

## Contributing

Feel free to open issues or submit pull requests if you have suggestions or improvements. Please follow the project's code of conduct and contribution guidelines.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any questions or feedback, please contact [rafsunsheikh116@gmail.com](mailto:rafsunsheikh116@gmail.com).
