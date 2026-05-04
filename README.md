# MailMind - Gmail RAG System

A sophisticated Python-based RAG (Retrieval-Augmented Generation) system that indexes Gmail threads and attachments into a Qdrant vector database for semantic retrieval and intelligent email management.

## 🚀 Features

### Core Architecture
- **Thread-Centric Indexing**: Groups messages by thread_id, sorts chronologically, and concatenates body text for comprehensive context
- **Attachment Pipeline**: Extracts text from PDF, DOCX, and images (PNG/JPG) using OCR
- **Sliding Context**: Uses GPT-4o-mini to generate running summaries for threads exceeding 4000 tokens
- **State Management**: PostgreSQL tracks last_synced_at and message_id to prevent duplicate processing

### Advanced Features
- **Action-Item Extraction**: Detects and stores tasks, commitments, and deadlines during embedding phase
- **Cross-Thread Relationships**: Links threads based on shared project codes, invoice numbers, and URLs
- **Semantic Search**: OpenAI embeddings with Qdrant for intelligent thread retrieval
- **RESTful API**: FastAPI endpoints for sync, search, and management operations

## 🏗️ Architecture

```
MailMind/
├── models.py                 # PostgreSQL database schema
├── main.py                   # FastAPI application
├── service/
│   └── gmail_client.py       # Gmail API integration
├── core/
│   ├── attachment_processor.py    # PDF/DOCX/Image OCR
│   ├── sliding_context.py          # Thread summarization
│   ├── action_extractor.py        # Task extraction
│   ├── relationship_mapper.py     # Cross-thread linking
│   └── vector_store.py           # Qdrant integration
├── config/                   # Configuration files
└── tests/                    # Test suite
```

## 🛠️ Tech Stack

- **Framework**: FastAPI + LangChain
- **Database**: PostgreSQL (SQLAlchemy) + Qdrant (Vector)
- **Auth**: Google OAuth2 (Gmail API)
- **AI**: OpenAI GPT-4o-mini + text-embedding-3-small
- **Processing**: AsyncIO background tasks
- **OCR**: Tesseract/EasyOCR for images

## 📋 Prerequisites

- Python 3.9+
- PostgreSQL 13+
- Qdrant server
- OpenAI API key
- Gmail API credentials
- Tesseract OCR (for image processing)

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone <repository-url>
cd MailMind

# Install dependencies
pip install -r requirements.txt

# Install Tesseract OCR
# Ubuntu/Debian:
sudo apt-get install tesseract-ocr

# macOS:
brew install tesseract

# Windows:
# Download from https://github.com/UB-Mannheim/tesseract/wiki
```

### 2. Environment Setup

Create `.env` file:

```bash
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Gmail API Configuration
GMAIL_CREDENTIALS_PATH=path/to/credentials.json
GMAIL_TOKEN_PATH=path/to/token.json

# Database Configuration
DATABASE_URL=postgresql+asyncpg://user:password@localhost/mailmind
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=your_qdrant_api_key

# Application Settings
LOG_LEVEL=INFO
MAX_WORKERS=10
```

### 3. Gmail API Setup

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing
3. Enable Gmail API
4. Create OAuth2 credentials (Desktop Application)
5. Download credentials JSON and save as `credentials.json`

### 4. Database Setup

```bash
# Create PostgreSQL database
createdb mailmind

# Run migrations (when implemented)
alembic upgrade head
```

### 5. Start Qdrant

```bash
# Using Docker
docker run -p 6333:6333 qdrant/qdrant

# Or download and run locally
# https://qdrant.tech/documentation/install/
```

### 6. Run Application

```bash
# Development server
python main.py

# Or using uvicorn directly
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## 📚 API Documentation

### Authentication

All endpoints require Bearer token authentication:

```bash
curl -H "Authorization: Bearer <user_id>" http://localhost:8000/health
```

### Endpoints

#### Health Check
```bash
GET /health
```

#### Gmail Authentication
```bash
POST /auth/gmail
```

#### Sync Gmail Threads
```bash
POST /sync/gmail
Content-Type: application/json

{
  "user_id": "user123",
  "max_threads": 50
}
```

#### Search Threads
```bash
POST /search
Content-Type: application/json

{
  "user_id": "user123",
  "query": "project planning meeting",
  "limit": 10,
  "filters": {
    "participants": ["alice@example.com"],
    "date_from": "2024-01-01",
    "has_attachments": true
  }
}
```

#### Get Thread Details
```bash
GET /threads/{thread_id}
```

#### List Threads
```bash
GET /threads?limit=20&offset=0&sort_by=last_message_date
```

## 🔧 Configuration

### Thread Processing

- **Token Limit**: 4000 tokens before sliding context activation
- **Summary Model**: GPT-4o-mini
- **Embedding Model**: text-embedding-3-small
- **Max Messages per Thread**: No limit (chronological processing)

### Attachment Processing

Supported formats:
- PDF files (PyPDF2 + pdfplumber)
- DOCX files (python-docx)
- Images: PNG, JPG (Tesseract OCR)

### Relationship Detection

- **Project Codes**: Patterns like `PROJ-123`, `ABC456`
- **Invoice Numbers**: Patterns like `invoice #12345`
- **URLs**: HTTP/HTTPS links
- **Documents**: References to attachments
- **Meetings**: Date-based meeting references

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=mailmind

# Run specific test
pytest tests/test_gmail_client.py
```

## 📊 Monitoring

### Health Checks

```bash
# Application health
curl http://localhost:8000/health

# Component status
curl http://localhost:8000/health | jq '.components'
```

### Sync Status

```bash
# Check sync operation
GET /sync/status/{sync_id}
```

### Vector Store Stats

```bash
# Collection statistics
curl http://localhost:8000/vector/stats
```

## 🔒 Security

- OAuth2 for Gmail authentication
- JWT tokens for API security
- Environment-based configuration
- Input validation and sanitization
- Rate limiting (Gmail API quotas)

## 🚀 Deployment

### Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose

```yaml
version: '3.8'
services:
  mailmind:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql+asyncpg://postgres:password@db:5432/mailmind
      - QDRANT_URL=http://qdrant:6333
    depends_on:
      - db
      - qdrant

  db:
    image: postgres:13
    environment:
      - POSTGRES_DB=mailmind
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data

  qdrant:
    image: qdrant/qdrant
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage

volumes:
  postgres_data:
  qdrant_data:
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Troubleshooting

### Gmail Authentication Issues
- Ensure OAuth2 credentials are correctly configured
- Check that Gmail API is enabled in Google Cloud Console
- Verify redirect URI matches your application

### OCR Issues
- Install Tesseract OCR system package
- Verify Tesseract is in PATH (Windows)
- Check image file permissions

### Database Issues
- Verify PostgreSQL is running
- Check connection string format
- Ensure database exists

### Vector Store Issues
- Verify Qdrant server is accessible
- Check API key configuration
- Monitor collection status

## 📈 Performance

### Optimization Tips
- Use batch processing for large syncs
- Implement proper indexing in PostgreSQL
- Monitor Qdrant memory usage
- Cache frequently accessed threads

### Scaling
- Horizontal scaling with multiple workers
- Database connection pooling
- Vector store sharding for large datasets
- CDN for attachment storage

## 🔄 Roadmap

- [ ] Web dashboard for thread management
- [ ] Advanced filtering and faceted search
- [ ] Email composition assistance
- [ ] Integration with other email providers
- [ ] Mobile API endpoints
- [ ] Advanced analytics and reporting

## 📞 Support

For support and questions:
- Create an issue in the repository
- Check the troubleshooting section
- Review API documentation
- Join our community discussions

---

**MailMind** - Transform your email chaos into intelligent, searchable knowledge. 🧠✉️
