# MailMind Frontend

A sophisticated Streamlit-based web interface for the MailMind email intelligence system.

## Features

### 📋 Summary View
- **Top 5 Action Items**: Displays the most important tasks across all threads
- **Priority-based Sorting**: Urgent and high-priority tasks appear first
- **Overdue Detection**: Automatically highlights overdue tasks
- **Thread Navigation**: One-click access to source threads

### 🔍 Contextual Search
- **Smart Search Bar**: Natural language queries with intent detection
- **Entity Highlighting**: Visual badges for projects, invoices, JIRA tickets
- **Search Type Indicators**: Shows whether results came from task, entity, or semantic search
- **Rich Result Cards**: Comprehensive thread previews with metadata

### 🕸️ Thread Map
- **Relationship Visualization**: Interactive graph showing thread connections
- **Entity-based Connections**: Lines represent shared entities between threads
- **Color-coded Nodes**: Different colors for threads vs. entities
- **Network Statistics**: Density, connection counts, and relationship metrics

### 🔄 One-Click Sync
- **Gmail Authentication**: OAuth2 integration with Google
- **Progress Tracking**: Real-time sync progress with detailed metrics
- **Background Processing**: Non-blocking sync operations
- **Status Monitoring**: Live updates on threads, messages, and attachments

## Installation

### Local Development

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Run the frontend:**
   ```bash
   streamlit run app.py
   ```

4. **Access the dashboard:**
   Open your browser to `http://localhost:8501`

### Docker Deployment

1. **Build and run with Docker Compose:**
   ```bash
   # Include frontend profile
   docker-compose --profile frontend up --build
   ```

2. **Access the dashboard:**
   Open your browser to `http://localhost:8501`

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `API_BASE_URL` | `http://localhost:8000` | Backend API URL |
| `FRONTEND_PORT` | `8501` | Frontend port |
| `USER_ID` | `demo_user` | Default user ID |
| `CACHE_TTL` | `300` | Cache duration in seconds |
| `MAX_GRAPH_NODES` | `50` | Maximum nodes in thread map |
| `DEFAULT_SEARCH_LIMIT` | `10` | Default search result limit |

### Backend Requirements

The frontend requires the MailMind backend API to be running:
- FastAPI server on port 8000
- Qdrant vector database
- PostgreSQL database
- OpenAI API integration

## Usage

### Getting Started

1. **Authenticate with Gmail:**
   - Click "Authenticate Gmail" in the sidebar
   - Follow the OAuth2 flow
   - Wait for successful connection confirmation

2. **Sync Your Emails:**
   - Click "Start Sync" to begin incremental sync
   - Monitor progress in the sidebar
   - Wait for completion notification

3. **Explore Your Intelligence:**
   - Check the Summary View for top action items
   - Use the Search tab to find specific information
   - View the Thread Map to see relationships

### Search Examples

**Task Queries:**
- "What are my tasks?"
- "Show me overdue tasks"
- "Tasks assigned to John"
- "High priority action items"

**Entity Queries:**
- "Find project PROJ-123"
- "Invoice INV-456"
- "JIRA tickets"
- "Meeting notes"

**Semantic Queries:**
- "Quarterly planning"
- "Budget discussions"
- "Client communications"

### Thread Map Navigation

- **Nodes**: Blue circles are threads, colored circles are entities
- **Connections**: Lines show shared entities between threads
- **Interactions**: Hover for details, click to explore
- **Statistics**: View network metrics below the graph

## Architecture

### Frontend Components

```
frontend/
├── app.py                 # Main Streamlit application
├── requirements.txt        # Python dependencies
├── Dockerfile             # Container configuration
├── .env.example          # Environment template
└── README.md              # This file
```

### API Integration

The frontend communicates with the backend through REST endpoints:

- `GET /health` - System health check
- `POST /auth/gmail` - Gmail authentication
- `POST /search` - Intelligent search
- `GET /tasks` - Top action items
- `POST /sync/gmail` - Start sync
- `GET /sync/status/{id}` - Sync progress

### Data Flow

1. **User Action** → Frontend UI
2. **API Request** → Backend Service
3. **Data Processing** → Intelligence Pipeline
4. **Response** → Frontend Display
5. **Visualization** → Interactive Components

## Development

### Adding New Features

1. **UI Components**: Add new Streamlit elements to `app.py`
2. **API Integration**: Update the `MailMindAPI` class
3. **Visualization**: Extend Plotly charts and graphs
4. **Styling**: Modify CSS in the markdown sections

### Testing

```bash
# Run locally
streamlit run app.py

# Test with Docker
docker build -t mailmind-frontend .
docker run -p 8501:8501 mailmind-frontend
```

### Debugging

- Check browser console for JavaScript errors
- Monitor Streamlit logs for Python errors
- Verify API connectivity with health check
- Test backend endpoints independently

## Performance

### Optimization

- **Caching**: Built-in response caching (5-minute TTL)
- **Lazy Loading**: Components load data on demand
- **Pagination**: Search results limited to 10 items
- **Graph Limits**: Thread map limited to 50 nodes

### Monitoring

- **Health Checks**: Backend API status monitoring
- **Error Handling**: Graceful error display and recovery
- **Performance Metrics**: Search time and result counts
- **User Feedback**: Visual indicators for all actions

## Troubleshooting

### Common Issues

**API Connection Error:**
- Verify backend is running on port 8000
- Check API_BASE_URL environment variable
- Ensure network connectivity between containers

**Authentication Failed:**
- Verify Gmail API credentials
- Check OAuth2 redirect URI configuration
- Ensure proper Google Cloud Console setup

**No Search Results:**
- Verify Gmail sync has completed
- Check vector database connection
- Ensure threads have been processed

**Thread Map Empty:**
- Perform a search first to generate results
- Check for shared entities between threads
- Verify relationship mapping is working

### Support

For issues and questions:
1. Check the application logs
2. Verify backend API health
3. Test individual API endpoints
4. Review configuration settings

## Future Enhancements

### Planned Features

- **Real-time Updates**: WebSocket integration for live sync updates
- **Advanced Filtering**: Multi-criteria search and filtering
- **Export Functionality**: Download tasks and search results
- **Mobile Optimization**: Responsive design for mobile devices
- **User Preferences**: Customizable dashboard settings

### Integration Opportunities

- **Calendar Integration**: Sync tasks with Google Calendar
- **Slack Integration**: Post updates to Slack channels
- **Email Notifications**: Send task reminders via email
- **Third-party Tools**: Integration with project management tools

---

**MailMind Frontend** - The intelligent interface for your email intelligence. 🧠✉️
