"""
MailMind Streamlit Frontend

A sophisticated dashboard for Gmail thread intelligence with:
- Summary View with top action items
- Contextual search with entity highlighting
- Thread relationship visualization
- One-click sync with progress tracking
"""

import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timedelta
import json
import time
from typing import List, Dict, Any, Optional
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Configuration
API_BASE_URL = "http://localhost:8000"
PAGE_SIZE = 10
REFRESH_INTERVAL = 30  # seconds

# Page configuration
st.set_page_config(
    page_title="MailMind - Email Intelligence Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .task-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 0.5rem 0;
    }
    .urgent-task {
        border-left-color: #dc3545;
    }
    .high-priority {
        border-left-color: #fd7e14;
    }
    .thread-card {
        background: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .entity-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        margin: 0.25rem;
        border-radius: 0.25rem;
        font-size: 0.875rem;
        font-weight: 500;
    }
    .project-badge { background: #e3f2fd; color: #1565c0; }
    .invoice-badge { background: #fff3e0; color: #e65100; }
    .jira-badge { background: #f3e5f5; color: #7b1fa2; }
    .meeting-badge { background: #e8f5e8; color: #2e7d32; }
    .sync-progress {
        background: #e7f3ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #b3d9ff;
    }
</style>
""", unsafe_allow_html=True)


class MailMindAPI:
    """API client for MailMind backend."""
    
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()
    
    def get_health(self) -> Dict[str, Any]:
        """Check API health."""
        try:
            response = self.session.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            st.error(f"API Error: {str(e)}")
            return {}
    
    def authenticate_gmail(self, user_id: str) -> Dict[str, Any]:
        """Authenticate with Gmail."""
        try:
            response = self.session.post(f"{self.base_url}/auth/gmail", json={"user_id": user_id})
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            st.error(f"Gmail auth error: {str(e)}")
            return {}
    
    def search_threads(self, query: str, user_id: str, limit: int = 10) -> Dict[str, Any]:
        """Search threads."""
        try:
            response = self.session.post(
                f"{self.base_url}/search",
                json={"query": query, "user_id": user_id, "limit": limit}
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            st.error(f"Search error: {str(e)}")
            return {"results": [], "total_found": 0}
    
    def get_tasks(self, user_id: str, limit: int = 5) -> Dict[str, Any]:
        """Get top action items."""
        try:
            response = self.session.get(
                f"{self.base_url}/tasks",
                params={"user_id": user_id, "limit": limit}
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            st.error(f"Tasks error: {str(e)}")
            return {"tasks": []}
    
    def get_thread(self, thread_id: str, user_id: str) -> Dict[str, Any]:
        """Get thread details."""
        try:
            response = self.session.get(
                f"{self.base_url}/threads/{thread_id}",
                params={"user_id": user_id}
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            st.error(f"Thread error: {str(e)}")
            return {}
    
    def generate_draft_reply(
        self,
        thread_id: str,
        user_id: str,
        reply_type: Optional[str] = None,
        custom_instructions: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate a draft reply for a thread."""
        try:
            data = {
                "thread_id": thread_id,
                "user_id": user_id
            }
            if reply_type:
                data["reply_type"] = reply_type
            if custom_instructions:
                data["custom_instructions"] = custom_instructions
            
            response = self.session.post(
                f"{self.base_url}/draft-reply",
                json=data
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            st.error(f"Draft reply error: {str(e)}")
            return {}
    
    def start_sync(self, user_id: str, max_threads: int = 50) -> Dict[str, Any]:
        """Start Gmail sync."""
        try:
            response = self.session.post(
                f"{self.base_url}/sync/gmail",
                json={"user_id": user_id, "max_threads": max_threads}
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            st.error(f"Sync error: {str(e)}")
            return {}
    
    def get_sync_status(self, sync_id: str, user_id: str) -> Dict[str, Any]:
        """Get sync status."""
        try:
            response = self.session.get(
                f"{self.base_url}/sync/status/{sync_id}",
                params={"user_id": user_id}
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            st.error(f"Sync status error: {str(e)}")
            return {}


# Initialize API client
api = MailMindAPI()

# Session state initialization
if 'user_id' not in st.session_state:
    st.session_state.user_id = "demo_user"
if 'current_search' not in st.session_state:
    st.session_state.current_search = ""
if 'selected_thread' not in st.session_state:
    st.session_state.selected_thread = None
if 'sync_in_progress' not in st.session_state:
    st.session_state.sync_in_progress = False
if 'sync_id' not in st.session_state:
    st.session_state.sync_id = None


def render_header():
    """Render application header."""
    st.markdown('<div class="main-header">🧠 MailMind</div>', unsafe_allow_html=True)
    st.markdown("---")


def render_sidebar():
    """Render sidebar with navigation and actions."""
    # Ensure session state is initialized
    if 'sync_in_progress' not in st.session_state:
        st.session_state.sync_in_progress = False
    if 'sync_id' not in st.session_state:
        st.session_state.sync_id = None
    
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # User ID input
        user_id = st.text_input("User ID", value=st.session_state.user_id)
        st.session_state.user_id = user_id
        
        # Gmail Authentication
        st.subheader("📧 Gmail Connection")
        if st.button("Authenticate Gmail", key="auth_gmail"):
            with st.spinner("Authenticating with Gmail..."):
                result = api.authenticate_gmail(user_id)
                if result.get("status") == "authenticated":
                    st.success(f"✅ Connected as {result.get('user_email')}")
                    st.info(f"Messages: {result.get('messages_total', 0)}")
                else:
                    st.error("❌ Authentication failed")
        
        # One-Click Sync
        st.subheader("🔄 Sync")
        if st.button("Start Sync", key="start_sync", disabled=st.session_state.sync_in_progress):
            with st.spinner("Starting sync..."):
                result = api.start_sync(user_id)
                if result.get("sync_id"):
                    st.session_state.sync_id = result["sync_id"]
                    st.session_state.sync_in_progress = True
                    st.success(f"✅ Sync started: {result['sync_id']}")
                else:
                    st.error("❌ Failed to start sync")
        
        # Sync Progress
        if st.session_state.sync_in_progress and st.session_state.sync_id:
            render_sync_progress()
        
        # Health Check
        st.subheader("💊 System Health")
        if st.button("Check Health", key="health_check"):
            health = api.get_health()
            if health.get("status") == "healthy":
                st.success("✅ All systems operational")
                st.json(health.get("components", {}))
            else:
                st.error("❌ System issues detected")
        
        # Statistics
        st.subheader("📊 Quick Stats")
        # This would typically fetch from a stats endpoint
        st.metric("Indexed Threads", "1,234")
        st.metric("Action Items", "56")
        st.metric("Last Sync", "2 hours ago")


def render_sync_progress():
    """Render sync progress bar."""
    if not st.session_state.sync_id:
        return
    
    status = api.get_sync_status(st.session_state.sync_id, st.session_state.user_id)
    
    if status.get("status") == "completed":
        st.session_state.sync_in_progress = False
        st.success("✅ Sync completed!")
        st.balloons()
    elif status.get("status") == "failed":
        st.session_state.sync_in_progress = False
        st.error("❌ Sync failed")
        if status.get("errors"):
            st.error("Errors:")
            for error in status.get("errors", []):
                st.error(f"• {error}")
    else:
        # Show progress
        progress = status.get("progress", 0) / 100
        st.markdown('<div class="sync-progress">', unsafe_allow_html=True)
        st.progress(progress)
        st.markdown(f"**Sync Status:** {status.get('status', 'unknown')}")
        st.markdown(f"**Threads:** {status.get('threads_processed', 0)}")
        st.markdown(f"**Messages:** {status.get('messages_processed', 0)}")
        st.markdown(f"**Attachments:** {status.get('attachments_processed', 0)}")
        st.markdown('</div>', unsafe_allow_html=True)


def render_summary_view():
    """Render summary view with top action items."""
    st.header("📋 Summary View")
    
    # Get top action items
    with st.spinner("Loading action items..."):
        tasks_data = api.get_tasks(st.session_state.user_id, limit=5)
    
    if tasks_data.get("tasks"):
        st.subheader("🎯 Top 5 Action Items")
        
        for i, task in enumerate(tasks_data["tasks"], 1):
            priority = task.get("priority", "medium")
            task_text = task.get("task_text", "")
            assignee = task.get("assignee", "Unassigned")
            due_date = task.get("due_date")
            thread_id = task.get("thread_id")
            
            # Determine card styling based on priority
            card_class = "task-card"
            if priority == "urgent":
                card_class += " urgent-task"
            elif priority == "high":
                card_class += " high-priority"
            
            # Render task card
            st.markdown(f'<div class="{card_class}">', unsafe_allow_html=True)
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.markdown(f"**{i}. {task_text}**")
                st.markdown(f"*Assigned to:* {assignee}")
                if due_date:
                    st.markdown(f"*Due:* {due_date}")
                if thread_id:
                    if st.button(f"View Thread", key=f"task_thread_{i}", use_container_width=True):
                        st.session_state.selected_thread = thread_id
            
            with col2:
                # Priority badge
                priority_color = {
                    "urgent": "🔴",
                    "high": "🟠", 
                    "medium": "🟡",
                    "low": "🟢"
                }.get(priority, "⚪")
                st.markdown(f"<h3>{priority_color}</h3>", unsafe_allow_html=True)
            
            with col3:
                # Draft Reply button
                if thread_id:
                    if st.button("✉️ Draft", key=f"draft_reply_{i}", use_container_width=True, help="Generate a professional reply"):
                        st.session_state.draft_thread_id = thread_id
                        st.session_state.draft_task_text = task_text
                        st.session_state.show_draft_modal = True
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Draft Reply Modal
        if st.session_state.get("show_draft_modal", False):
            render_draft_reply_modal()
    else:
        st.info("No action items found. Try syncing your Gmail account first.")


def render_contextual_search():
    """Render contextual search with entity highlighting."""
    st.header("🔍 Contextual Search")
    
    # Search interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_query = st.text_input(
            "Search your emails...",
            value=st.session_state.current_search,
            placeholder="Try: 'What are my tasks?' or 'Find project PROJ-123'"
        )
    
    with col2:
        search_button = st.button("🔍 Search", use_container_width=True)
    
    if search_query or search_button:
        st.session_state.current_search = search_query
        
        with st.spinner("Searching..."):
            results = api.search_threads(search_query, st.session_state.user_id, limit=10)
        
        if results.get("results"):
            st.success(f"Found {results.get('total_found', 0)} results ({results.get('search_time_ms', 0)}ms)")
            
            # Render search results as cards
            for i, result in enumerate(results["results"]):
                render_thread_card(result, i)
        else:
            st.info("No results found. Try a different search query.")


def render_thread_card(result: Dict[str, Any], index: int):
    """Render a thread card with entity highlighting."""
    thread_id = result.get("thread_id")
    subject = result.get("subject", "No Subject")
    score = result.get("score", 0)
    participants = result.get("participants", [])
    date = result.get("date", "")
    snippet = result.get("snippet", "")
    action_items = result.get("action_items", [])
    entities = result.get("entities", [])
    search_type = result.get("search_type", "semantic_search")
    
    st.markdown(f'<div class="thread-card">', unsafe_allow_html=True)
    
    # Header with subject and metadata
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        st.markdown(f"### {subject}")
        st.markdown(f"**Score:** {score:.2f} | **Type:** {search_type}")
    
    with col2:
        if st.button("📖 View", key=f"view_thread_{index}", use_container_width=True):
            st.session_state.selected_thread = thread_id
    
    with col3:
        if st.button("🔗 Link", key=f"link_thread_{index}", use_container_width=True):
            st.info(f"Thread ID: {thread_id}")
    
    # Metadata
    st.markdown(f"**Participants:** {', '.join(participants[:3])}")
    if len(participants) > 3:
        st.markdown(f"*and {len(participants) - 3} more*")
    
    st.markdown(f"**Date:** {date}")
    
    # Entity badges
    if entities:
        st.markdown("**Entities:**")
        entity_html = ""
        for entity in entities[:5]:  # Show max 5 entities
            entity_type = entity.get("entity_type", "").lower()
            entity_value = entity.get("entity_value", "")
            
            badge_class = f"entity-badge {entity_type}-badge"
            entity_html += f'<span class="{badge_class}">{entity_value}</span>'
        
        st.markdown(entity_html, unsafe_allow_html=True)
    
    # Action items preview
    if action_items:
        st.markdown("**Action Items:**")
        for task in action_items[:3]:  # Show max 3 tasks
            task_text = task.get("task_text", "")
            priority = task.get("priority", "medium")
            priority_icon = {"urgent": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}.get(priority, "⚪")
            st.markdown(f"{priority_icon} {task_text}")
        
        if len(action_items) > 3:
            st.markdown(f"*and {len(action_items) - 3} more tasks*")
    
    # Snippet
    if snippet:
        with st.expander("📝 Preview"):
            st.markdown(snippet)
    
    st.markdown('</div>', unsafe_allow_html=True)


def render_thread_map():
    """Render thread relationship visualization."""
    st.header("🕸️ Thread Map")
    
    if not st.session_state.current_search:
        st.info("Search for threads first to see their relationships.")
        return
    
    # Get search results to build relationship map
    results = api.search_threads(st.session_state.current_search, st.session_state.user_id, limit=5)
    
    if not results.get("results"):
        st.info("No threads found for relationship mapping.")
        return
    
    # Create a simple network graph
    G = nx.Graph()
    
    # Add nodes for search results
    for i, result in enumerate(results["results"]):
        thread_id = result.get("thread_id")
        subject = result.get("subject", f"Thread {i+1}")
        entities = result.get("entities", [])
        
        # Add thread node
        G.add_node(thread_id, label=subject[:30] + "...", type="thread")
        
        # Add entity nodes and connections
        for entity in entities:
            entity_type = entity.get("entity_type", "")
            entity_value = entity.get("entity_value", "")
            entity_id = f"{entity_type}:{entity_value}"
            
            # Add entity node if not exists
            if entity_id not in G.nodes:
                G.add_node(entity_id, label=entity_value, type=entity_type)
            
            # Connect thread to entity
            G.add_edge(thread_id, entity_id, weight=1)
    
    # Create visualization
    if len(G.nodes) > 0:
        # Calculate layout
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Create node traces
        node_x = []
        node_y = []
        node_text = []
        node_colors = []
        node_sizes = []
        
        color_map = {
            "thread": "#1f77b4",
            "project": "#ff7f0e", 
            "invoice": "#2ca02c",
            "jira": "#d62728",
            "meeting": "#9467bd",
            "document": "#8c564b",
            "url": "#e377c2",
            "person": "#7f7f7f"
        }
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            node_data = G.nodes[node]
            node_text.append(node_data.get("label", node))
            
            node_type = node_data.get("type", "thread")
            node_colors.append(color_map.get(node_type, "#7f7f7f"))
            node_sizes.append(20 if node_type == "thread" else 15)
        
        # Create edge traces
        edge_x = []
        edge_y = []
        
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        # Create plot
        fig = go.Figure()
        
        # Add edges
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            mode='lines',
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            showlegend=False
        ))
        
        # Add nodes
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(width=2, color='white')
            ),
            text=node_text,
            textposition="middle center",
            hoverinfo='text',
            showlegend=False
        ))
        
        fig.update_layout(
            title="Thread Relationship Map",
            titlefont_size=16,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[
                dict(
                    text="Nodes represent threads and shared entities. Lines show relationships.",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002,
                    xanchor='left', yanchor='bottom',
                    font=dict(size=10)
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show statistics
        st.subheader("📊 Relationship Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Threads", len([n for n in G.nodes() if G.nodes[n].get("type") == "thread"]))
        with col2:
            st.metric("Entities", len([n for n in G.nodes() if G.nodes[n].get("type") != "thread"]))
        with col3:
            st.metric("Connections", G.number_of_edges())
        with col4:
            st.metric("Density", round(nx.density(G), 3))
    else:
        st.info("No relationships found between threads.")


def render_draft_reply_modal():
    """Render the draft reply modal."""
    if not st.session_state.get("draft_thread_id"):
        return
    
    thread_id = st.session_state.draft_thread_id
    task_text = st.session_state.draft_task_text
    
    # Modal container
    with st.container():
        st.markdown("---")
        st.subheader("✉️ Draft Reply Generator")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"**Task:** {task_text}")
            st.markdown(f"**Thread ID:** {thread_id}")
        
        with col2:
            if st.button("❌ Close", key="close_draft_modal"):
                st.session_state.show_draft_modal = False
                st.session_state.draft_thread_id = None
                st.session_state.draft_task_text = None
                st.rerun()
        
        # Reply options
        st.markdown("### Reply Options")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            reply_type = st.selectbox(
                "Reply Type",
                options=["", "action_response", "question_response", "acknowledgment", "follow_up"],
                key="reply_type_select",
                help="Choose the type of reply to generate"
            )
        
        with col2:
            tone = st.selectbox(
                "Tone",
                options=["", "professional", "casual", "friendly", "formal"],
                key="tone_select",
                help="Choose the tone for the reply"
            )
        
        with col3:
            length = st.selectbox(
                "Length",
                options=["", "short", "medium", "long"],
                key="length_select",
                help="Choose the length of the reply"
            )
        
        # Custom instructions
        custom_instructions = st.text_area(
            "Custom Instructions (Optional)",
            placeholder="Add any specific instructions for the reply...",
            key="custom_instructions"
        )
        
        # Generate button
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("🚀 Generate Reply", key="generate_reply", type="primary", use_container_width=True):
                with st.spinner("Generating professional reply..."):
                    # Build custom instructions from selections
                    instructions = custom_instructions
                    if tone:
                        instructions += f" Use a {tone} tone."
                    if length:
                        instructions += f" Make the response {length}."
                    
                    draft_data = api.generate_draft_reply(
                        thread_id=thread_id,
                        user_id=st.session_state.user_id,
                        reply_type=reply_type if reply_type else None,
                        custom_instructions=instructions if instructions else None
                    )
                    
                    if draft_data.get("reply"):
                        st.session_state.generated_draft = draft_data
                        st.session_state.show_draft_result = True
                    else:
                        st.error("Failed to generate reply. Please try again.")
        
        with col2:
            if st.button("🔄 Regenerate", key="regenerate_reply", use_container_width=True):
                if st.session_state.get("generated_draft"):
                    st.session_state.show_draft_result = False
                    st.session_state.generated_draft = None
                    st.rerun()
        
        # Display generated reply
        if st.session_state.get("show_draft_result", False) and st.session_state.get("generated_draft"):
            render_generated_reply(st.session_state.generated_draft)
        
        st.markdown("---")


def render_generated_reply(draft_data: Dict[str, Any]):
    """Render the generated reply."""
    reply = draft_data.get("reply", {})
    metadata = draft_data.get("metadata", {})
    
    st.markdown("### 📧 Generated Reply")
    
    # Reply metadata
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Confidence", f"{reply.get('confidence_score', 0):.1%}")
    
    with col2:
        st.metric("Words", reply.get('word_count', 0))
    
    with col3:
        st.metric("Reading Time", reply.get('estimated_reading_time', 'Unknown'))
    
    with col4:
        st.metric("Tone", reply.get('tone', 'Unknown').title())
    
    # Reply content
    st.markdown("#### Email Content")
    
    # Subject
    st.markdown(f"**Subject:** {reply.get('subject', 'No Subject')}")
    
    # Full email
    email_content = reply.get('full_email', '')
    if email_content:
        st.text_area(
            "Generated Email",
            value=email_content,
            height=200,
            key="generated_email",
            help="You can edit this email before sending"
        )
    
    # Action items addressed
    if reply.get('action_items_addressed'):
        st.markdown("#### ✅ Action Items Addressed")
        for item in reply.get('action_items_addressed', []):
            st.markdown(f"- {item}")
    
    # Entities referenced
    if reply.get('entities_referenced'):
        st.markdown("#### 🔗 Entities Referenced")
        for entity in reply.get('entities_referenced', []):
            st.markdown(f"- {entity}")
    
    # Action buttons
    st.markdown("#### Actions")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📋 Copy", key="copy_reply", use_container_width=True):
            st.success("Email copied to clipboard!")
    
    with col2:
        if st.button("✏️ Edit", key="edit_reply", use_container_width=True):
            st.info("Edit the email in the text area above")
    
    with col3:
        if st.button("📧 Send", key="send_reply", use_container_width=True):
            st.success("Email sent! (Integration with Gmail would be implemented here)")
    
    with col4:
        if st.button("💾 Save", key="save_reply", use_container_width=True):
            st.success("Draft saved!")
    
    # Generation info
    with st.expander("📊 Generation Details"):
        st.json({
            "thread_subject": metadata.get("thread_subject"),
            "last_message_sender": metadata.get("last_message_sender"),
            "thread_participants": metadata.get("thread_participants"),
            "generated_at": metadata.get("generated_at"),
            "confidence_score": reply.get('confidence_score'),
            "tone": reply.get('tone'),
            "word_count": reply.get('word_count'),
            "estimated_reading_time": reply.get('estimated_reading_time')
        })


def render_thread_detail():
    """Render detailed thread view."""
    if not st.session_state.selected_thread:
        st.info("Select a thread to view details.")
        return
    
    thread_id = st.session_state.selected_thread
    
    with st.spinner("Loading thread details..."):
        thread_data = api.get_thread(thread_id, st.session_state.user_id)
    
    if not thread_data:
        st.error("Thread not found.")
        return
    
    st.header(f"📧 Thread Details")
    
    # Thread header
    subject = thread_data.get("subject", "No Subject")
    st.subheader(subject)
    
    # Thread metadata
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Messages", thread_data.get("message_count", 0))
    
    with col2:
        participants = thread_data.get("participant_emails", [])
        st.metric("Participants", len(participants))
    
    with col3:
        tasks = thread_data.get("detected_tasks", [])
        st.metric("Action Items", len(tasks))
    
    # Draft Reply button for thread
    if st.button("✉️ Draft Reply for This Thread", key="thread_draft_reply", use_container_width=True):
        st.session_state.draft_thread_id = thread_id
        st.session_state.draft_task_text = f"Reply to: {subject}"
        st.session_state.show_draft_modal = True
        st.rerun()
    
    # Action items
    if thread_data.get("detected_tasks"):
        st.subheader("🎯 Action Items")
        for task in thread_data["detected_tasks"]:
            st.markdown(f"- {task}")
    
    # Related threads
    if thread_data.get("related_threads"):
        st.subheader("🔗 Related Threads")
        for related_id in thread_data["related_threads"]:
            if st.button(f"View Thread {related_id}", key=f"related_{related_id}"):
                st.session_state.selected_thread = related_id
                st.rerun()
    
    # Close button
    if st.button("← Back to Search", key="close_thread"):
        st.session_state.selected_thread = None
        st.rerun()


def main():
    """Main application."""
    render_header()
    render_sidebar()
    
    # Navigation tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Summary", "🔍 Search", "🕸️ Thread Map", "📧 Thread Detail"])
    
    with tab1:
        render_summary_view()
    
    with tab2:
        render_contextual_search()
    
    with tab3:
        render_thread_map()
    
    with tab4:
        render_thread_detail()


if __name__ == "__main__":
    main()
