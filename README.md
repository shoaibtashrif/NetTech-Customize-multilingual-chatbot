# Urdu Bot - BISP Nashonuma Program Assistant

A chat-based AI system for the BISP Nashonuma program that classifies messages as complaints or information requests and provides intelligent responses.

## Features

- **Multi-Model Support**: Switch between OpenAI GPT-4 and Groq API
- **FAISS RAG**: Advanced retrieval-augmented generation using FAISS vector database
- **Session Management**: Track conversation history and session types
- **Complaint Categorization**: Automatically categorize complaints into money, eligibility, district, ingredients, or other
- **Language Detection**: Respond in the same language as user input (English, Urdu, Roman Urdu)
- **Knowledge Base Integration**: Upload and use custom knowledge bases
- **Smart Suggestions**: Context-aware quick questions
- **Docker Support**: Easy deployment with Docker

## Model Configuration

The system supports two AI models:

1. **OpenAI** (`openai`): Uses GPT-4 for chat completions
2. **Groq** (`groq`): Uses Groq's fast inference API with Mixtral model

### Switching Models

Use the `/set_model` endpoint to switch between models:

```bash
curl -X POST http://localhost:5000/set_model \
  -H "Content-Type: application/json" \
  -d '{"model": "groq"}'
```

## FAISS RAG Implementation

The system uses FAISS (Facebook AI Similarity Search) for efficient knowledge retrieval:

- **Automatic Chunking**: Knowledge base is automatically split into overlapping chunks
- **Vector Embeddings**: Uses sentence-transformers for semantic search
- **Relevant Retrieval**: Returns only the most relevant knowledge chunks for each query
- **Persistent Storage**: FAISS index is saved to disk and loaded on startup

## Environment Variables

```bash
OPENAI_API_KEY=your_openai_api_key
GROQ_API_KEY=your_groq_api_key
```

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up environment variables
4. Run the application:
   ```bash
   python app.py
   ```

## Docker Deployment

```bash
docker build -t urdu-bot .
docker run -p 5000:5000 urdu-bot
```

## API Endpoints

- `POST /chat`: Main chat endpoint
- `POST /set_model`: Switch AI model (openai or groq)
- `POST /upload`: Upload knowledge base or system prompt
- `POST /suggestions`: Get smart suggestions
- `POST /start_toggle`: Start/end chat session
- `GET /logs`: Get application logs

## Usage

1. Start a session using `/start_toggle`
2. Choose session type (complaint or info) on first message
3. Chat with the AI assistant
4. End session to get summary and categorization

## Architecture

- **Backend**: Flask with CORS support
- **AI Models**: OpenAI and Groq APIs
- **Vector Database**: FAISS for semantic search
- **Frontend**: HTML/CSS/JavaScript with glass morphism UI
- **Session Management**: In-memory storage with session types
- **Logging**: Comprehensive logging with UI display

## Troubleshooting

- **Model Switching Issues**: Ensure API keys are properly set for the selected model
- **FAISS Errors**: Check if sentence-transformers is properly installed
- **Memory Issues**: FAISS index is loaded into memory, consider using smaller chunk sizes for large knowledge bases
- **OpenAI API Issues**: Make sure you're using the correct API key and have sufficient credits
