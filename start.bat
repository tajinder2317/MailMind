@echo off
REM MailMind Quick Start Script for Windows
REM This script helps you get MailMind running quickly

echo 🧠 MailMind Quick Start Script
echo ================================

REM Check if Docker is installed
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker is not installed. Please install Docker Desktop first.
    pause
    exit /b 1
)

REM Check if Docker Compose is installed
docker-compose --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker Compose is not installed. Please install Docker Compose first.
    pause
    exit /b 1
)

REM Check if .env file exists
if not exist .env (
    echo 📝 Creating .env file from template...
    copy .env.example .env
    echo ⚠️  Please edit .env file with your API keys before continuing!
    echo    Required: OPENAI_API_KEY, Gmail credentials setup
    pause
)

REM Check if config directory exists
if not exist config (
    echo 📁 Creating config directory...
    mkdir config
)

REM Check if Gmail credentials exist
if not exist config\credentials.json (
    echo ⚠️  Gmail credentials not found in config\credentials.json
    echo    Please set up Gmail API in Google Cloud Console and download credentials
    echo    See RUN_GUIDE.md for detailed instructions
    pause
)

echo 🚀 Starting MailMind services...

REM Start with frontend if argument provided
if "%1"=="--frontend" (
    echo 🎨 Starting with frontend...
    docker-compose --profile frontend up --build -d
) else if "%1"=="-f" (
    echo 🎨 Starting with frontend...
    docker-compose --profile frontend up --build -d
) else (
    echo 🔧 Starting backend services only...
    docker-compose up --build -d
)

echo ⏳ Waiting for services to start...
timeout /t 10 /nobreak >nul

REM Check if services are running
echo 🔍 Checking service health...

REM Check backend
curl -s http://localhost:8000/health >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Backend API is running at http://localhost:8000
) else (
    echo ❌ Backend API is not responding
    echo 📋 Check logs with: docker-compose logs mailmind
)

REM Check frontend if started
if "%1"=="--frontend" (
    timeout /t 5 /nobreak >nul
    curl -s http://localhost:8501/_stcore/health >nul 2>&1
    if %errorlevel% equ 0 (
        echo ✅ Frontend is running at http://localhost:8501
    ) else (
        echo ❌ Frontend is not responding
        echo 📋 Check logs with: docker-compose logs mailmind-frontend
    )
) else if "%1"=="-f" (
    timeout /t 5 /nobreak >nul
    curl -s http://localhost:8501/_stcore/health >nul 2>&1
    if %errorlevel% equ 0 (
        echo ✅ Frontend is running at http://localhost:8501
    ) else (
        echo ❌ Frontend is not responding
        echo 📋 Check logs with: docker-compose logs mailmind-frontend
    )
)

REM Check database
docker-compose exec -T db pg_isready -U mailmind >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ PostgreSQL database is running
) else (
    echo ❌ PostgreSQL database is not ready
)

REM Check Qdrant
curl -s http://localhost:6333/health >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Qdrant vector database is running
) else (
    echo ❌ Qdrant vector database is not ready
)

echo.
echo 🎉 MailMind setup complete!
echo.
echo 📚 Quick Guide:
echo    • Backend API: http://localhost:8000
echo    • API Docs:    http://localhost:8000/docs
if "%1"=="--frontend" (
    echo    • Frontend:    http://localhost:8501
) else if "%1"=="-f" (
    echo    • Frontend:    http://localhost:8501
)
echo.
echo 🔗 Next Steps:
echo    1. Open frontend dashboard
echo    2. Click 'Authenticate Gmail' to connect your email
echo    3. Click 'Start Sync' to index your emails
echo    4. Try searching for 'What are my tasks?'
echo.
echo 📋 Useful Commands:
echo    • View logs:    docker-compose logs -f mailmind
echo    • Stop all:     docker-compose down
echo    • Restart:      docker-compose restart
echo.
echo 📖 For detailed instructions, see RUN_GUIDE.md
pause
