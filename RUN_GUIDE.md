# MailMind - Complete Setup and Run Guide

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Docker & Docker Compose
- Groq API Key (for high-speed LLM)
- Google Cloud Project with Gmail API enabled

### Option 1: Docker Compose (Recommended)

1. **Clone and Setup**
   ```bash
   git clone <repository-url>
   cd MailMind
   ```

2. **Configure Environment**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

3. **Start All Services**
   ```bash
   # Start backend services only
   docker-compose up --build -d
   
   # Or start with frontend
   docker-compose --profile frontend up --build -d
   ```

4. **Access Applications**
   - Backend API: http://localhost:8000
   - Frontend Dashboard: http://localhost:8501
   - API Docs: http://localhost:8000/docs

### Option 2: Local Development

1. **Backend Setup**
   ```bash
   cd MailMind
   
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   
   # Start databases with Docker
   docker-compose up db qdrant -d
   
   # Configure environment
   cp .env.example .env
   # Edit .env with your configuration
   ```

2. **Frontend Setup**
   ```bash
   cd frontend
   
   # Install frontend dependencies
   pip install -r requirements.txt
   
   # Configure frontend environment
   cp .env.example .env
   ```

3. **Run Applications**
   ```bash
   # Terminal 1: Backend
   cd MailMind
   python main.py
   
   # Terminal 2: Frontend
   cd frontend
   streamlit run app.py
   ```

## 🔧 Detailed Configuration

### Environment Variables

Create `.env` file in root directory:

```bash
# Groq Configuration (High-speed LLM)
GROQ_API_KEY=your_groq_api_key_here
LLM_MODEL=llama-3.3-70b-versatile

# Database Configuration
DATABASE_URL=postgresql+asyncpg://mailmind:your_password@localhost:5432/mailmind
POSTGRES_PASSWORD=your_secure_password

# Qdrant Configuration
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=your_qdrant_api_key

# Application Configuration
LOG_LEVEL=INFO
MAX_WORKERS=10
DEBUG_MODE=false
USER_ID=soorma2317@gmail.com

# Frontend Configuration (for frontend/.env)
API_BASE_URL=http://localhost:8000
FRONTEND_PORT=8501
```

### Gmail API Setup

1. **Google Cloud Console**
   - Create new project or use existing
   - Enable Gmail API
   - Create OAuth 2.0 credentials (Web Application)
   - Add redirect URI: `http://localhost:8000/auth/gmail/callback`

2. **Download Credentials**
   - Download JSON credentials file
   - Save as `config/credentials.json`

3. **Test Authentication**
   - Start the backend application
   - Visit frontend dashboard
   - Click "Authenticate Gmail" button
   - Complete OAuth flow in browser

### Database Setup

#### PostgreSQL (Docker)
```bash
# Start PostgreSQL
docker-compose up db -d

# Check connection
docker-compose exec db psql -U mailmind -d mailmind -c "SELECT version();"
```

#### Qdrant (Docker)
```bash
# Start Qdrant
docker-compose up qdrant -d

# Check connection
curl http://localhost:6333/health
```

## 🗂️ Project Structure

```
MailMind/
├── main.py                    # FastAPI backend application
├── requirements.txt           # Backend dependencies
├── .env.example              # Environment template
├── docker-compose.yml        # Docker services configuration
├── Dockerfile               # Backend container
├── models.py                # Database models
├── config/                  # Configuration files
│   ├── credentials.json      # Gmail API credentials
│   └── token.json           # Gmail auth tokens
├── core/                    # Intelligence components
│   ├── embedding_pipeline.py
│   ├── draft_reply_agent.py
│   ├── entity_mapper.py
│   ├── groq_client.py
│   └── ...
├── service/                 # External services
│   └── gmail_client.py
├── frontend/                # Streamlit frontend
│   ├── app.py              # Frontend application
│   ├── requirements.txt    # Frontend dependencies
│   ├── Dockerfile          # Frontend container
│   └── .env.example        # Frontend environment
└── logs/                   # Application logs
```

## 🚦 Running the Services

### Docker Compose Commands

```bash
# Start all core services
docker-compose up --build -d

# Start with frontend
docker-compose --profile frontend up --build -d

# View logs
docker-compose logs -f mailmind
docker-compose logs -f mailmind-frontend

# Stop services
docker-compose down

# Clean up volumes
docker-compose down -v
```

### Local Development Commands

```bash
# Backend
python main.py

# Frontend
streamlit run frontend/app.py

# With custom port
streamlit run frontend/app.py --server.port 8502
```

## 🔍 Testing the Setup

### 1. Health Check
```bash
curl http://localhost:8000/health
```

### 2. API Documentation
Visit: http://localhost:8000/docs

### 3. Frontend Access
Visit: http://localhost:8501

### 4. Test Gmail Integration
1. Open frontend dashboard
2. Click "Authenticate Gmail"
3. Complete OAuth flow
4. Check authentication status

### 5. Test Sync Process
1. After authentication, click "Start Sync"
2. Monitor progress in sidebar
3. Wait for completion notification

### 6. Test Search
1. Enter search query like "What are my tasks?"
2. View results with entity highlighting
3. Explore thread relationships

### 7. Test Draft Reply
1. Find action items in Summary View
2. Click "✉️ Draft" button next to any task
3. Customize reply options
4. Generate professional response

## 🐛 Troubleshooting

### Common Issues

**Port Conflicts**
```bash
# Check what's using ports
netstat -tulpn | grep :8000
netstat -tulpn | grep :8501

# Kill processes if needed
sudo kill -9 <PID>
```

**Database Connection**
```bash
# Check PostgreSQL status
docker-compose ps db

# Reset database
docker-compose down -v
docker-compose up db -d
```

**Gmail Authentication**
```bash
# Check credentials file
ls -la config/credentials.json

# Clear tokens and re-authenticate
rm config/token.json
```

**Groq API Issues**
```bash
# Test API key
curl -H "Authorization: Bearer $GROQ_API_KEY" \
     https://groq.com/api/v1/models

# Check environment variables
echo $GROQ_API_KEY
```

**Frontend Not Loading**
```bash
# Check frontend logs
docker-compose logs mailmind-frontend

# Restart frontend
docker-compose restart mailmind-frontend
```

### Debug Mode

Enable debug logging:
```bash
# Add to .env
LOG_LEVEL=DEBUG
DEBUG_MODE=true

# Restart services
docker-compose restart mailmind
```

### Logs Location

- Backend logs: `logs/mailmind.log`
- Docker logs: `docker-compose logs mailmind`
- Frontend logs: Docker container logs

## 📊 Monitoring

### Health Endpoints

- Backend: http://localhost:8000/health
- Database: Check Docker container status
- Qdrant: http://localhost:6333/health

### Performance Metrics

- API response times in frontend
- Sync progress in dashboard
- Database query logs (debug mode)

## 🔒 Security Considerations

1. **API Keys**: Never commit `.env` files to version control
2. **Gmail Tokens**: Store securely, use HTTPS in production
3. **Database**: Use strong passwords, restrict network access
4. **Docker**: Run containers as non-root users

## 🚀 Production Deployment

### Environment Setup
```bash
# Production environment variables
DEBUG_MODE=false
LOG_LEVEL=WARNING
# Use production database URLs
# Configure SSL certificates
# Set up reverse proxy (nginx)
```

### Docker Production
```bash
# Use production compose file
docker-compose -f docker-compose.prod.yml up -d

# Or use Kubernetes
kubectl apply -f k8s/
```

### Monitoring Setup
- Set up application monitoring (Prometheus/Grafana)
- Configure log aggregation (ELK stack)
- Set up alerting for critical errors

## 🔧 Required Configuration

### 1. Groq API Key (Primary LLM)
- Get API key from https://groq.com/
- Add to `.env`: `GROQ_API_KEY=your_groq_api_key_here`
- Example: `GROQ_API_KEY=gsk_your_groq_api_key_here`

### 2. OpenAI API Key (Optional - for embeddings)
- Get API key from https://platform.openai.com/
- Add to `.env`: `OPENAI_API_KEY=your_key_here`

### 3. Environment Variables
```bash
# Required in .env
GROQ_API_KEY=your_groq_api_key_here
DATABASE_URL=postgresql+asyncpg://mailmind:password@localhost:5432/mailmind
POSTGRES_PASSWORD=secure_password
QDRANT_URL=http://localhost:6333

# Optional (for embeddings)
OPENAI_API_KEY=your_openai_key_here
```

## 📚 Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Groq Documentation](https://groq.com/)
- [Gmail API Documentation](https://developers.google.com/gmail/api)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)

## 🆘 Getting Help

1. Check this guide first
2. Review application logs
3. Check GitHub issues
4. Join community discussions
5. Contact support team

---

**MailMind** - Your intelligent email assistant is ready to help! 🧠✉️
