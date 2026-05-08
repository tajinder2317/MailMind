#!/bin/bash

# MailMind Quick Start Script
# This script helps you get MailMind running quickly

set -e

echo "🧠 MailMind Quick Start Script"
echo "================================"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Check if .env file exists
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    cp .env.example .env
    echo "⚠️  Please edit .env file with your API keys before continuing!"
    echo "   Required: OPENAI_API_KEY, Gmail credentials setup"
    read -p "Press Enter after editing .env file..."
fi

# Check if config directory exists
if [ ! -d config ]; then
    echo "📁 Creating config directory..."
    mkdir -p config
fi

# Check if Gmail credentials exist
if [ ! -f config/credentials.json ]; then
    echo "⚠️  Gmail credentials not found in config/credentials.json"
    echo "   Please set up Gmail API in Google Cloud Console and download credentials"
    echo "   See RUN_GUIDE.md for detailed instructions"
    read -p "Press Enter to continue (you can authenticate later)..."
fi

echo "🚀 Starting MailMind services..."

# Start with frontend by default
if [ "$1" = "--frontend" ] || [ "$1" = "-f" ]; then
    echo "🎨 Starting with frontend..."
    docker-compose --profile frontend up --build -d
else
    echo "🔧 Starting backend services only..."
    docker-compose up --build -d
fi

echo "⏳ Waiting for services to start..."
sleep 10

# Check if services are running
echo "🔍 Checking service health..."

# Check backend
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Backend API is running at http://localhost:8000"
else
    echo "❌ Backend API is not responding"
    echo "📋 Check logs with: docker-compose logs mailmind"
fi

# Check frontend if started
if [ "$1" = "--frontend" ] || [ "$1" = "-f" ]; then
    sleep 5
    if curl -s http://localhost:8501/_stcore/health > /dev/null 2>&1; then
        echo "✅ Frontend is running at http://localhost:8501"
    else
        echo "❌ Frontend is not responding"
        echo "📋 Check logs with: docker-compose logs mailmind-frontend"
    fi
fi

# Check database
if docker-compose exec -T db pg_isready -U mailmind > /dev/null 2>&1; then
    echo "✅ PostgreSQL database is running"
else
    echo "❌ PostgreSQL database is not ready"
fi

# Check Qdrant
if curl -s http://localhost:6333/health > /dev/null 2>&1; then
    echo "✅ Qdrant vector database is running"
else
    echo "❌ Qdrant vector database is not ready"
fi

echo ""
echo "🎉 MailMind setup complete!"
echo ""
echo "📚 Quick Guide:"
echo "   • Backend API: http://localhost:8000"
echo "   • API Docs:    http://localhost:8000/docs"
if [ "$1" = "--frontend" ] || [ "$1" = "-f" ]; then
    echo "   • Frontend:    http://localhost:8501"
fi
echo ""
echo "🔗 Next Steps:"
echo "   1. Open frontend dashboard"
echo "   2. Click 'Authenticate Gmail' to connect your email"
echo "   3. Click 'Start Sync' to index your emails"
echo "   4. Try searching for 'What are my tasks?'"
echo ""
echo "📋 Useful Commands:"
echo "   • View logs:    docker-compose logs -f mailmind"
echo "   • Stop all:     docker-compose down"
echo "   • Restart:      docker-compose restart"
echo ""
echo "📖 For detailed instructions, see RUN_GUIDE.md"
