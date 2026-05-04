"""
Gmail API Client for MailMind

This module provides a comprehensive interface for fetching Gmail threads,
messages, and attachments with proper authentication and rate limiting.
"""

import asyncio
import base64
import email
from datetime import datetime, timedelta
from typing import List, Dict, Optional, AsyncGenerator, Tuple
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.message import Message

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build, Resource
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload

import aiohttp
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


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
    attachments: List[Dict[str, any]]


@dataclass
class GmailThread:
    """Data class representing a Gmail thread with messages."""
    thread_id: str
    subject: str
    messages: List[GmailMessage]
    participant_emails: List[str]


class GmailClient:
    """
    Gmail API client with OAuth2 authentication and thread fetching capabilities.
    
    Features:
    - OAuth2 authentication with token refresh
    - Rate limiting and retry logic
    - Thread-centric message fetching
    - Attachment handling
    - Incremental sync support
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
        self.credentials_path = credentials_path
        self.token_path = token_path
        self.service: Optional[Resource] = None
        self._credentials: Optional[Credentials] = None
        
    async def authenticate(self) -> bool:
        """
        Authenticate with Gmail API using OAuth2.
        
        Returns:
            True if authentication successful, False otherwise
        """
        try:
            creds = None
            
            # Load existing tokens
            if self.token_path.exists():
                creds = Credentials.from_authorized_user_file(str(self.token_path), self.SCOPES)
            
            # If no valid credentials, get new ones
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    flow = InstalledAppFlow.from_client_secrets_file(
                        str(self.credentials_path), self.SCOPES
                    )
                    creds = flow.run_local_server(port=0)
                
                # Save credentials
                with open(self.token_path, 'w') as token:
                    token.write(creds.to_json())
            
            self._credentials = creds
            self.service = build('gmail', 'v1', credentials=creds)
            logger.info("Gmail authentication successful")
            return True
            
        except Exception as e:
            logger.error(f"Gmail authentication failed: {str(e)}")
            return False
    
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
            
            return GmailThread(
                thread_id=thread_id,
                subject=messages[0].subject if messages else "",
                messages=messages,
                participant_emails=list(participant_emails)
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
        elif part.get('filename') and part.get('body', {}).get('attachmentId'):
            # This is an attachment
            attachment = {
                'filename': part['filename'],
                'mime_type': part['mimeType'],
                'size': part['body'].get('size', 0),
                'attachment_id': part['body']['attachmentId']
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
