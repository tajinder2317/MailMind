"""
Gmail API Client for MailMind

This module provides a comprehensive interface for fetching Gmail threads,
messages, and attachments with proper authentication and rate limiting.
"""

import asyncio
import base64
import email
import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional, AsyncGenerator, Tuple, Set, Any
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.message import Message
from pathlib import Path
from urllib.parse import urlencode
import webbrowser

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow, Flow
from googleapiclient.discovery import build, Resource
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload

import aiohttp
import logging
from dataclasses import dataclass
from enum import Enum
import heapq

logger = logging.getLogger(__name__)


class SyncPriority(Enum):
    """Sync priority levels for threads."""
    HIGH = "high"      # Last 24 hours
    NORMAL = "normal"   # Last 7 days
    LOW = "low"        # Historical (older than 7 days)


@dataclass
class GmailMessage:
    """Data class representing a Gmail message."""
    message_id: str
    thread_id: str
    subject: str
    from_email: str
    to_emails: List[str]
    cc_emails: List[str]
    bcc_emails: List[str]
    body_text: str
    body_html: Optional[str]
    gmail_date: datetime
    internal_date: int
    history_id: Optional[str]
    attachments: List[Dict[str, any]]


@dataclass
class GmailThread:
    """Data class representing a Gmail thread with messages."""
    thread_id: str
    subject: str
    messages: List[GmailMessage]
    participant_emails: List[str]
    history_id: Optional[str]
    sync_priority: SyncPriority


class GmailClient:
    """
    Gmail API client with OAuth2 authentication and incremental sync capabilities.
    
    Features:
    - OAuth2 authentication with token refresh
    - Rate limiting and retry logic
    - Thread-centric message fetching
    - Attachment handling
    - Incremental sync using History API
    - Priority-based sync queue
    """
    
    SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
    MAX_RETRIES = 3
    RETRY_DELAY = 1  # seconds
    
    def __init__(self, credentials_path: str, token_path: str):
        """
        Initialize Gmail client.
        
        Args:
            credentials_path: Path to OAuth2 credentials JSON file
            token_path: Path to store/refresh OAuth2 tokens
        """
        self.credentials_path = Path(credentials_path) if credentials_path else Path("config/credentials.json")
        self.token_path = Path(token_path)
        self.service: Optional[Resource] = None
        self._credentials: Optional[Credentials] = None
        self._flow: Optional[Flow] = None
        self._auth_code: Optional[str] = None

    def _get_client_config(self) -> Dict[str, Any]:
        """
        Load OAuth client config from JSON file, or fall back to env vars.

        Supported env vars:
        - GOOGLE_OAUTH_CLIENT_ID
        - GOOGLE_OAUTH_CLIENT_SECRET
        """
        if self.credentials_path.exists():
            with open(self.credentials_path, "r") as f:
                return json.load(f)

        client_id = os.getenv("GOOGLE_OAUTH_CLIENT_ID")
        client_secret = os.getenv("GOOGLE_OAUTH_CLIENT_SECRET")
        if not client_id or not client_secret:
            raise FileNotFoundError(
                f"Missing Gmail OAuth credentials file at {self.credentials_path} and env vars "
                f"GOOGLE_OAUTH_CLIENT_ID/GOOGLE_OAUTH_CLIENT_SECRET are not set."
            )

        # Minimal "installed app" config compatible with google-auth-oauthlib.
        return {
            "installed": {
                "client_id": client_id,
                "client_secret": client_secret,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
            }
        }
        
    async def initiate_oauth_flow(self, redirect_uri: str = "http://localhost:8080/callback") -> str:
        """
        Initiate OAuth2 flow and return authorization URL.
        
        Args:
            redirect_uri: Callback URL for OAuth2
            
        Returns:
            Authorization URL for user to visit
        """
        try:
            client_config = self._get_client_config()
            # Create OAuth2 flow
            self._flow = InstalledAppFlow.from_client_config(
                client_config,
                self.SCOPES,
                redirect_uri=redirect_uri
            )
            
            # Generate authorization URL
            auth_url, state = self._flow.authorization_url(
                access_type='offline',
                include_granted_scopes='true',
                prompt='consent'
            )
            
            logger.info(f"OAuth2 flow initiated, state: {state}")
            return auth_url
            
        except Exception as e:
            logger.error(f"Failed to initiate OAuth2 flow: {str(e)}")
            raise
    
    async def complete_oauth_flow(self, auth_code: str, state: str) -> bool:
        """
        Complete OAuth2 flow with authorization code.
        
        Args:
            auth_code: Authorization code from OAuth2 callback
            state: State parameter from OAuth2 flow
            
        Returns:
            True if authentication successful, False otherwise
        """
        try:
            if not self._flow:
                raise ValueError("OAuth2 flow not initiated")
            
            # Exchange auth code for credentials
            self._flow.fetch_token(code=auth_code)
            
            # Get credentials
            self._credentials = self._flow.credentials
            
            # Save credentials
            await self._save_credentials()
            
            # Build service
            self.service = build('gmail', 'v1', credentials=self._credentials)
            
            logger.info("OAuth2 authentication completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to complete OAuth2 flow: {str(e)}")
            return False
    
    async def authenticate(self) -> bool:
        """
        Authenticate with Gmail API using OAuth2.
        Supports both existing tokens and new OAuth2 flow.
        
        Returns:
            True if authentication successful, False otherwise
        """
        try:
            # Load existing tokens
            if self.token_path.exists():
                creds = Credentials.from_authorized_user_file(str(self.token_path), self.SCOPES)
                
                if creds and creds.valid:
                    self._credentials = creds
                    self.service = build('gmail', 'v1', credentials=creds)
                    logger.info("Gmail authentication successful (existing tokens)")
                    return True
                elif creds and creds.expired and creds.refresh_token:
                    try:
                        creds.refresh(Request())
                        self._credentials = creds
                        await self._save_credentials()
                        self.service = build('gmail', 'v1', credentials=creds)
                        logger.info("Gmail authentication successful (token refreshed)")
                        return True
                    except Exception as e:
                        logger.warning(f"Token refresh failed: {str(e)}")
            
            # No valid credentials, need OAuth2 flow
            logger.info("No valid credentials found, OAuth2 flow required")
            return False
            
        except Exception as e:
            logger.error(f"Gmail authentication failed: {str(e)}")
            return False
    
    async def _save_credentials(self):
        """Save credentials to token file."""
        try:
            # Ensure directory exists
            self.token_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save credentials
            with open(self.token_path, 'w') as token:
                token.write(self._credentials.to_json())
                
        except Exception as e:
            logger.error(f"Failed to save credentials: {str(e)}")
            raise
    
    async def get_user_profile(self) -> Dict[str, any]:
        """
        Get the authenticated user's profile information.
        
        Returns:
            User profile data including email address
        """
        if not self.service:
            raise ValueError("Service not initialized. Call authenticate() first.")
        
        try:
            profile = self.service.users().getProfile(userId='me').execute()
            return {
                'email_address': profile['emailAddress'],
                'messages_total': profile['messagesTotal'],
                'threads_total': profile['threadsTotal'],
                'history_id': profile['historyId']
            }
        except HttpError as e:
            logger.error(f"Failed to get user profile: {str(e)}")
            raise
    
    async def get_threads(
        self, 
        max_results: int = 50,
        page_token: Optional[str] = None,
        query: Optional[str] = None
    ) -> Tuple[List[str], Optional[str]]:
        """
        Get list of thread IDs.
        
        Args:
            max_results: Maximum number of threads to return
            page_token: Token for pagination
            query: Gmail search query string
            
        Returns:
            Tuple of (thread_ids, next_page_token)
        """
        if not self.service:
            raise ValueError("Service not initialized. Call authenticate() first.")
        
        try:
            kwargs = {
                'userId': 'me',
                'maxResults': max_results,
            }
            
            if page_token:
                kwargs['pageToken'] = page_token
            
            if query:
                kwargs['q'] = query
            
            result = self.service.users().threads().list(**kwargs).execute()
            
            threads = result.get('threads', [])
            thread_ids = [thread['id'] for thread in threads]
            next_page_token = result.get('nextPageToken')
            
            return thread_ids, next_page_token
            
        except HttpError as e:
            logger.error(f"Failed to get threads: {str(e)}")
            raise
    
    async def get_thread(self, thread_id: str) -> GmailThread:
        """
        Get full thread details including all messages.
        
        Args:
            thread_id: Gmail thread ID
            
        Returns:
            GmailThread object with all messages
        """
        if not self.service:
            raise ValueError("Service not initialized. Call authenticate() first.")
        
        try:
            result = self.service.users().threads().get(
                userId='me',
                id=thread_id,
                format='full'
            ).execute()
            
            messages = []
            participant_emails = set()
            
            for msg_data in result.get('messages', []):
                message = await self._parse_message(msg_data)
                messages.append(message)
                
                # Collect participant emails
                participant_emails.add(message.from_email)
                participant_emails.update(message.to_emails)
                participant_emails.update(message.cc_emails)
                participant_emails.update(message.bcc_emails)
            
            # Sort messages chronologically
            messages.sort(key=lambda x: x.gmail_date)
            
            # Get thread history ID from the thread data
            thread_history_id = result.get('historyId')
            
            return GmailThread(
                thread_id=thread_id,
                subject=messages[0].subject if messages else "",
                messages=messages,
                participant_emails=list(participant_emails),
                history_id=thread_history_id,
                sync_priority=SyncPriority.NORMAL  # Will be updated by sync queue
            )
            
        except HttpError as e:
            logger.error(f"Failed to get thread {thread_id}: {str(e)}")
            raise
    
    async def _parse_message(self, msg_data: Dict[str, any]) -> GmailMessage:
        """
        Parse Gmail message data into structured format.
        
        Args:
            msg_data: Raw Gmail API message data
            
        Returns:
            Parsed GmailMessage object
        """
        headers = {h['name']: h['value'] for h in msg_data.get('payload', {}).get('headers', [])}
        
        # Extract email addresses
        from_email = headers.get('From', '')
        to_emails = self._parse_email_addresses(headers.get('To', ''))
        cc_emails = self._parse_email_addresses(headers.get('Cc', ''))
        bcc_emails = self._parse_email_addresses(headers.get('Bcc', ''))
        
        # Extract body content
        body_text, body_html = await self._extract_message_body(msg_data.get('payload', {}))
        
        # Extract attachments
        attachments = await self._extract_attachments(msg_data.get('payload', {}))
        
        # Parse dates
        gmail_date = datetime.fromtimestamp(int(msg_data.get('internalDate', 0)) / 1000)
        internal_date = int(msg_data.get('internalDate', 0))
        
        # Extract history ID
        history_id = msg_data.get('historyId')
        
        return GmailMessage(
            message_id=msg_data['id'],
            thread_id=msg_data['threadId'],
            subject=headers.get('Subject', ''),
            from_email=from_email,
            to_emails=to_emails,
            cc_emails=cc_emails,
            bcc_emails=bcc_emails,
            body_text=body_text,
            body_html=body_html,
            gmail_date=gmail_date,
            internal_date=internal_date,
            history_id=history_id,
            attachments=attachments
        )
    
    async def _extract_message_body(self, payload: Dict[str, any]) -> Tuple[str, Optional[str]]:
        """
        Extract text and HTML body from message payload.
        
        Args:
            payload: Gmail message payload
            
        Returns:
            Tuple of (body_text, body_html)
        """
        body_text = ""
        body_html = None
        
        if payload.get('mimeType') == 'text/plain':
            body_text = base64.urlsafe_b64decode(
                payload.get('body', {}).get('data', '')
            ).decode('utf-8', errors='ignore')
        elif payload.get('mimeType') == 'text/html':
            html_data = base64.urlsafe_b64decode(
                payload.get('body', {}).get('data', '')
            ).decode('utf-8', errors='ignore')
            body_html = html_data
            # Extract text from HTML (simplified)
            body_text = html_data.replace('<br>', '\n').replace('</p>', '\n')
        elif payload.get('mimeType', '').startswith('multipart/'):
            # Handle multipart messages
            parts = payload.get('parts', [])
            for part in parts:
                part_text, part_html = await self._extract_message_body(part)
                if part_text:
                    body_text += part_text + '\n'
                if part_html and not body_html:
                    body_html = part_html
        
        return body_text.strip(), body_html
    
    async def _extract_attachments(self, payload: Dict[str, any]) -> List[Dict[str, any]]:
        """
        Extract attachment information from message payload.
        
        Args:
            payload: Gmail message payload
            
        Returns:
            List of attachment metadata dictionaries
        """
        attachments = []
        
        if payload.get('mimeType', '').startswith('multipart/'):
            parts = payload.get('parts', [])
            for part in parts:
                attachments.extend(await self._extract_attachments(part))
        elif payload.get('filename') and payload.get('body', {}).get('attachmentId'):
            # This is an attachment
            attachment = {
                'filename': payload['filename'],
                'mime_type': payload.get('mimeType', ''),
                'size': payload.get('body', {}).get('size', 0),
                'attachment_id': payload['body']['attachmentId']
            }
            attachments.append(attachment)
        
        return attachments
    
    async def download_attachment(self, message_id: str, attachment_id: str) -> bytes:
        """
        Download attachment content.
        
        Args:
            message_id: Gmail message ID
            attachment_id: Attachment ID
            
        Returns:
            Attachment content as bytes
        """
        if not self.service:
            raise ValueError("Service not initialized. Call authenticate() first.")
        
        try:
            attachment = self.service.users().messages().attachments().get(
                userId='me',
                messageId=message_id,
                id=attachment_id
            ).execute()
            
            data = base64.urlsafe_b64decode(attachment.get('data', ''))
            return data
            
        except HttpError as e:
            logger.error(f"Failed to download attachment {attachment_id}: {str(e)}")
            raise
    
    async def get_threads_since(
        self, 
        since_date: datetime,
        max_results: int = 50
    ) -> AsyncGenerator[GmailThread, None]:
        """
        Get threads modified since a specific date.
        
        Args:
            since_date: Only return threads modified after this date
            max_results: Maximum results per page
            
        Yields:
            GmailThread objects
        """
        # Format date for Gmail query
        date_str = since_date.strftime('%Y/%m/%d')
        query = f"after:{date_str}"
        
        page_token = None
        while True:
            thread_ids, page_token = await self.get_threads(
                max_results=max_results,
                page_token=page_token,
                query=query
            )
            
            if not thread_ids:
                break
            
            # Fetch full thread details
            for thread_id in thread_ids:
                try:
                    thread = await self.get_thread(thread_id)
                    yield thread
                except HttpError as e:
                    logger.warning(f"Failed to get thread {thread_id}: {str(e)}")
                    continue
            
            if not page_token:
                break
    
    def _parse_email_addresses(self, address_string: str) -> List[str]:
        """
        Parse email address string into list of addresses.
        
        Args:
            address_string: Comma-separated email addresses
            
        Returns:
            List of email addresses
        """
        if not address_string:
            return []
        
        addresses = []
        for addr in address_string.split(','):
            addr = addr.strip()
            if '<' in addr and '>' in addr:
                # Extract email from "Name <email@domain.com>"
                email_part = addr.split('<')[1].split('>')[0]
                addresses.append(email_part.strip())
            else:
                addresses.append(addr)
        
        return addresses
    
    async def get_thread_updates(
        self,
        user_id: str,
        last_history_id: Optional[str] = None,
        last_sync_time: Optional[datetime] = None,
        max_results: int = 50
    ) -> AsyncGenerator[GmailThread, None]:
        """
        Get thread updates using Gmail History API or timestamp queries.
        
        Args:
            user_id: User ID for filtering
            last_history_id: Last history ID from previous sync
            last_sync_time: Last sync timestamp as fallback
            max_results: Maximum results to return
            
        Yields:
            GmailThread objects with updates
        """
        if not self.service:
            raise ValueError("Service not initialized. Call authenticate() first.")
        
        try:
            # Try History API first if we have a history ID
            if last_history_id:
                logger.info(f"Using History API with last_history_id: {last_history_id}")
                async for thread in self._get_history_updates(last_history_id, max_results):
                    yield thread
            else:
                # Fallback to timestamp-based query
                sync_time = last_sync_time or (datetime.utcnow() - timedelta(days=1))
                logger.info(f"Using timestamp-based query since: {sync_time}")
                async for thread in self.get_threads_since(sync_time, max_results):
                    yield thread
                    
        except Exception as e:
            logger.error(f"Failed to get thread updates: {str(e)}")
            raise
    
    async def _get_history_updates(
        self,
        start_history_id: str,
        max_results: int = 50
    ) -> AsyncGenerator[GmailThread, None]:
        """
        Get thread updates using Gmail History API.
        
        Args:
            start_history_id: Starting history ID
            max_results: Maximum results to return
            
        Yields:
            GmailThread objects with updates
        """
        try:
            history_list = self.service.users().history()
            
            # Get history records
            history_result = history_list.list(
                userId='me',
                startHistoryId=start_history_id,
                historyTypes=['messageAdded', 'labelAdded', 'labelRemoved'],
                maxResults=max_results
            ).execute()
            
            histories = history_result.get('history', [])
            
            if not histories:
                logger.info("No new history records found")
                return
            
            # Extract thread IDs from history
            thread_ids = set()
            for history in histories:
                for record in history.get('messagesAdded', []):
                    message = record.get('message', {})
                    thread_id = message.get('threadId')
                    if thread_id:
                        thread_ids.add(thread_id)
            
            # Fetch full thread details
            for thread_id in thread_ids:
                try:
                    thread = await self.get_thread(thread_id)
                    if thread:
                        yield thread
                except HttpError as e:
                    if e.resp.status == 404:
                        logger.warning(f"Thread {thread_id} not found (may have been deleted)")
                    else:
                        raise
                        
        except HttpError as e:
            if e.resp.status == 404:
                logger.warning(f"History ID {start_history_id} not found, falling back to timestamp query")
                # Fallback to timestamp-based query
                fallback_time = datetime.utcnow() - timedelta(days=1)
                async for thread in self.get_threads_since(fallback_time, max_results):
                    yield thread
            else:
                raise
    
    async def create_sync_queue(
        self,
        user_id: str,
        last_sync_time: Optional[datetime] = None
    ) -> List[Tuple[SyncPriority, GmailThread]]:
        """
        Create priority-based sync queue for threads.
        
        Args:
            user_id: User ID for filtering
            last_sync_time: Last sync time for determining priority
            
        Returns:
            List of (priority, thread) tuples sorted by priority
        """
        if not self.service:
            raise ValueError("Service not initialized. Call authenticate() first.")
        
        try:
            sync_queue = []
            now = datetime.utcnow()
            
            # Define time windows for different priorities
            high_priority_cutoff = now - timedelta(hours=24)
            normal_priority_cutoff = now - timedelta(days=7)
            
            # Get all threads since last sync (or last 30 days if no sync time)
            since_time = last_sync_time or (now - timedelta(days=30))
            
            logger.info(f"Creating sync queue since: {since_time}")
            
            async for thread in self.get_threads_since(since_time):
                # Determine priority based on thread age
                if thread.messages and thread.last_message_date:
                    last_msg_time = thread.last_message_date
                    
                    if last_msg_time >= high_priority_cutoff:
                        priority = SyncPriority.HIGH
                    elif last_msg_time >= normal_priority_cutoff:
                        priority = SyncPriority.NORMAL
                    else:
                        priority = SyncPriority.LOW
                    
                    thread.sync_priority = priority
                    sync_queue.append((priority, thread))
            
            # Sort by priority (HIGH -> NORMAL -> LOW)
            priority_order = {SyncPriority.HIGH: 0, SyncPriority.NORMAL: 1, SyncPriority.LOW: 2}
            sync_queue.sort(key=lambda x: priority_order[x[0]])
            
            logger.info(f"Created sync queue with {len(sync_queue)} threads")
            logger.info(f"High priority: {sum(1 for p, _ in sync_queue if p == SyncPriority.HIGH)}")
            logger.info(f"Normal priority: {sum(1 for p, _ in sync_queue if p == SyncPriority.NORMAL)}")
            logger.info(f"Low priority: {sum(1 for p, _ in sync_queue if p == SyncPriority.LOW)}")
            
            return sync_queue
            
        except Exception as e:
            logger.error(f"Failed to create sync queue: {str(e)}")
            raise
    
    async def get_current_history_id(self) -> Optional[str]:
        """
        Get the current history ID for the user's mailbox.
        
        Returns:
            Current history ID or None if not available
        """
        try:
            if not self.service:
                raise ValueError("Service not initialized. Call authenticate() first.")
            
            profile = self.service.users().getProfile(userId='me').execute()
            return profile.get('historyId')
            
        except Exception as e:
            logger.error(f"Failed to get current history ID: {str(e)}")
            return None
    
    async def test_connection(self) -> bool:
        """
        Test Gmail API connection.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            if not self.service:
                return False
            
            # Try to get user profile as a simple test
            await self.get_user_profile()
            return True
            
        except Exception as e:
            logger.error(f"Gmail connection test failed: {str(e)}")
            return False


class GmailRateLimiter:
    """
    Rate limiter for Gmail API calls to prevent quota exhaustion.
    """
    
    def __init__(self, calls_per_second: int = 10):
        self.calls_per_second = calls_per_second
        self.last_call_time = 0
        self._lock = asyncio.Lock()
    
    async def acquire(self):
        """Acquire permission to make an API call."""
        async with self._lock:
            current_time = asyncio.get_event_loop().time()
            time_since_last = current_time - self.last_call_time
            min_interval = 1.0 / self.calls_per_second
            
            if time_since_last < min_interval:
                sleep_time = min_interval - time_since_last
                await asyncio.sleep(sleep_time)
            
            self.last_call_time = asyncio.get_event_loop().time()
