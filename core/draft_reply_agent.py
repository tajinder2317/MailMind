"""
Draft Reply Agent for MailMind

This module provides intelligent email reply generation that:
- Analyzes full thread context and extracted entities
- Learns and adapts to user's writing style
- Generates professional, contextually relevant responses
- Incorporates action items and entity references
"""

import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import re

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class ReplyTone(Enum):
    """Email reply tone styles."""
    PROFESSIONAL = "professional"
    CASUAL = "casual"
    FRIENDLY = "friendly"
    FORMAL = "formal"
    CONCISE = "concise"
    DETAILED = "detailed"


class ReplyType(Enum):
    """Types of email replies."""
    ACKNOWLEDGMENT = "acknowledgment"
    ACTION_RESPONSE = "action_response"
    QUESTION_RESPONSE = "question_response"
    INFORMATION_REQUEST = "information_request"
    FOLLOW_UP = "follow_up"
    DECLINATION = "declination"
    AGREEMENT = "agreement"


@dataclass
class UserWritingStyle:
    """Represents a user's writing style characteristics."""
    tone: ReplyTone
    formality_level: float  # 0.0 (very casual) to 1.0 (very formal)
    average_sentence_length: float
    greeting_style: str  # "Hi", "Hello", "Dear", etc.
    closing_style: str  # "Best", "Regards", "Sincerely", etc.
    signature_included: bool
    use_emojis: bool
    use_bullets: bool
    response_length_preference: str  # "short", "medium", "long"


@dataclass
class ThreadContext:
    """Complete context for reply generation."""
    thread_id: str
    subject: str
    participants: List[str]
    messages: List[Dict[str, Any]]
    action_items: List[Dict[str, Any]]
    entities: List[Dict[str, Any]]
    last_message_sender: str
    last_message_content: str
    user_writing_style: Optional[UserWritingStyle]


@dataclass
class GeneratedReply:
    """Generated email reply with metadata."""
    subject: str
    greeting: str
    body: str
    closing: str
    signature: Optional[str]
    action_items_addressed: List[str]
    entities_referenced: List[str]
    confidence_score: float
    tone: ReplyTone
    estimated_reading_time: str
    word_count: int


class StyleAnalyzer:
    """Analyzes user's writing style from their email history."""
    
    def __init__(self):
        """Initialize the style analyzer."""
        self.style_patterns = {
            "greetings": {
                "formal": ["Dear", "Hello", "Good morning", "Good afternoon"],
                "professional": ["Hi", "Hello"],
                "casual": ["Hey", "Hi there"],
                "friendly": ["Hi", "Hello", "Hey"]
            },
            "closings": {
                "formal": ["Sincerely", "Yours sincerely", "Respectfully"],
                "professional": ["Best regards", "Regards", "Kind regards"],
                "casual": ["Best", "Cheers", "Thanks"],
                "friendly": ["Best", "Talk soon", "Take care"]
            }
        }
    
    async def analyze_user_style(
        self,
        user_id: str,
        sample_messages: List[Dict[str, Any]]
    ) -> UserWritingStyle:
        """
        Analyze user's writing style from sample messages.
        
        Args:
            user_id: User identifier
            sample_messages: Sample messages from the user
            
        Returns:
            UserWritingStyle object with style characteristics
        """
        try:
            if not sample_messages:
                # Default style if no samples available
                return UserWritingStyle(
                    tone=ReplyTone.PROFESSIONAL,
                    formality_level=0.7,
                    average_sentence_length=15.0,
                    greeting_style="Hi",
                    closing_style="Best regards",
                    signature_included=True,
                    use_emojis=False,
                    use_bullets=True,
                    response_length_preference="medium"
                )
            
            # Analyze patterns in user messages
            all_text = " ".join([msg.get("body_text", "") for msg in sample_messages])
            
            # Detect formality level
            formal_indicators = ["Dear", "Sincerely", "Yours", "Respectfully"]
            casual_indicators = ["Hey", "Cheers", "Thanks", "Talk soon"]
            
            formal_count = sum(1 for word in formal_indicators if word.lower() in all_text.lower())
            casual_count = sum(1 for word in casual_indicators if word.lower() in all_text.lower())
            
            total_indicators = formal_count + casual_count
            formality_level = formal_count / total_indicators if total_indicators > 0 else 0.7
            
            # Determine greeting style
            greeting_style = self._detect_greeting_style(all_text)
            
            # Determine closing style
            closing_style = self._detect_closing_style(all_text)
            
            # Calculate average sentence length
            sentences = re.split(r'[.!?]+', all_text)
            avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 15.0
            
            # Detect other preferences
            use_emojis = bool(re.search(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]', all_text))
            use_bullets = bool(re.search(r'^\s*[-*•]\s', all_text, re.MULTILINE))
            
            # Determine tone from formality level
            if formality_level >= 0.8:
                tone = ReplyTone.FORMAL
            elif formality_level >= 0.6:
                tone = ReplyTone.PROFESSIONAL
            elif formality_level >= 0.4:
                tone = ReplyTone.FRIENDLY
            else:
                tone = ReplyTone.CASUAL
            
            # Determine response length preference
            avg_length = len(all_text) / len(sample_messages) if sample_messages else 500
            if avg_length < 200:
                response_length = "short"
            elif avg_length < 500:
                response_length = "medium"
            else:
                response_length = "long"
            
            return UserWritingStyle(
                tone=tone,
                formality_level=formality_level,
                average_sentence_length=avg_sentence_length,
                greeting_style=greeting_style,
                closing_style=closing_style,
                signature_included=True,
                use_emojis=use_emojis,
                use_bullets=use_bullets,
                response_length_preference=response_length
            )
            
        except Exception as e:
            logger.error(f"Style analysis failed: {str(e)}")
            # Return default style on error
            return UserWritingStyle(
                tone=ReplyTone.PROFESSIONAL,
                formality_level=0.7,
                average_sentence_length=15.0,
                greeting_style="Hi",
                closing_style="Best regards",
                signature_included=True,
                use_emojis=False,
                use_bullets=True,
                response_length_preference="medium"
            )
    
    def _detect_greeting_style(self, text: str) -> str:
        """Detect preferred greeting style."""
        text_lower = text.lower()
        
        for category, greetings in self.style_patterns["greetings"].items():
            for greeting in greetings:
                if greeting.lower() in text_lower:
                    return greetings[0]  # Return first greeting in category
        
        return "Hi"  # Default greeting
    
    def _detect_closing_style(self, text: str) -> str:
        """Detect preferred closing style."""
        text_lower = text.lower()
        
        for category, closings in self.style_patterns["closings"].items():
            for closing in closings:
                if closing.lower() in text_lower:
                    return closings[0]  # Return first closing in category
        
        return "Best regards"  # Default closing


class DraftReplyAgent:
    """
    Intelligent email reply generation agent.
    
    Uses GPT-4o-mini to generate professional responses that:
    - Address action items and questions
    - Reference relevant entities
    - Match user's writing style
    - Provide contextually appropriate responses
    """
    
    def __init__(self, openai_client: AsyncOpenAI):
        """Initialize the draft reply agent."""
        self.client = openai_client
        self.model = "gpt-4o-mini"
        self.style_analyzer = StyleAnalyzer()
    
    async def generate_reply(
        self,
        context: ThreadContext,
        reply_type: Optional[ReplyType] = None,
        custom_instructions: Optional[str] = None
    ) -> GeneratedReply:
        """
        Generate a professional email reply.
        
        Args:
            context: Complete thread context
            reply_type: Type of reply to generate (auto-detected if None)
            custom_instructions: Additional user instructions
            
        Returns:
            Generated reply with metadata
        """
        try:
            logger.info(f"Generating reply for thread {context.thread_id}")
            
            # Auto-detect reply type if not specified
            if not reply_type:
                reply_type = self._detect_reply_type(context)
            
            # Analyze thread content and requirements
            thread_analysis = self._analyze_thread_requirements(context)
            
            # Build the prompt for GPT-4o-mini
            prompt = self._build_reply_prompt(
                context, reply_type, thread_analysis, custom_instructions
            )
            
            # Generate the reply
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert email assistant that generates professional, contextually relevant email responses. Always respond in valid JSON format with the specified structure."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=1500,
                temperature=0.3
            )
            
            # Parse and structure the response
            reply_data = self._parse_reply_response(response.choices[0].message.content)
            
            # Create GeneratedReply object
            generated_reply = GeneratedReply(
                subject=reply_data.get("subject", f"Re: {context.subject}"),
                greeting=reply_data.get("greeting", ""),
                body=reply_data.get("body", ""),
                closing=reply_data.get("closing", ""),
                signature=reply_data.get("signature"),
                action_items_addressed=reply_data.get("action_items_addressed", []),
                entities_referenced=reply_data.get("entities_referenced", []),
                confidence_score=reply_data.get("confidence_score", 0.8),
                tone=reply_type,
                estimated_reading_time=self._estimate_reading_time(reply_data.get("body", "")),
                word_count=len(reply_data.get("body", "").split())
            )
            
            logger.info(f"Successfully generated reply for thread {context.thread_id}")
            return generated_reply
            
        except Exception as e:
            logger.error(f"Reply generation failed: {str(e)}")
            raise
    
    def _detect_reply_type(self, context: ThreadContext) -> ReplyType:
        """Auto-detect the most appropriate reply type."""
        last_message = context.last_message_content.lower()
        
        # Check for questions
        if any(q in last_message for q in ["?", "question", "wondering", "curious"]):
            return ReplyType.QUESTION_RESPONSE
        
        # Check for action items
        if context.action_items:
            return ReplyType.ACTION_RESPONSE
        
        # Check for information requests
        if any(req in last_message for req in ["need", "require", "looking for", "could you"]):
            return ReplyType.INFORMATION_REQUEST
        
        # Check for follow-up indicators
        if any(follow in last_message for follow in ["follow up", "checking in", "update"]):
            return ReplyType.FOLLOW_UP
        
        # Default to acknowledgment
        return ReplyType.ACKNOWLEDGMENT
    
    def _analyze_thread_requirements(self, context: ThreadContext) -> Dict[str, Any]:
        """Analyze thread to identify response requirements."""
        requirements = {
            "needs_action_response": bool(context.action_items),
            "has_questions": bool(re.search(r'\?', context.last_message_content)),
            "urgent_indicators": any(urgent in context.last_message_content.lower() 
                                  for urgent in ["urgent", "asap", "immediately", "as soon as possible"]),
            "formal_required": any(formal in context.last_message_content.lower() 
                                 for formal in ["dear", "sincerely", "formal", "official"]),
            "entities_to_mention": [entity.get("entity_value", "") for entity in context.entities],
            "participants_to_cc": [p for p in context.participants if p != context.last_message_sender],
            "thread_length": len(context.messages),
            "complexity": "high" if len(context.messages) > 5 else "medium" if len(context.messages) > 2 else "low"
        }
        
        return requirements
    
    def _build_reply_prompt(
        self,
        context: ThreadContext,
        reply_type: ReplyType,
        thread_analysis: Dict[str, Any],
        custom_instructions: Optional[str]
    ) -> str:
        """Build the comprehensive prompt for reply generation."""
        
        # Get user style preferences
        style = context.user_writing_style or UserWritingStyle(
            tone=ReplyTone.PROFESSIONAL,
            formality_level=0.7,
            average_sentence_length=15.0,
            greeting_style="Hi",
            closing_style="Best regards",
            signature_included=True,
            use_emojis=False,
            use_bullets=True,
            response_length_preference="medium"
        )
        
        prompt = f"""
Generate a professional email reply based on the following context:

THREAD CONTEXT:
- Subject: {context.subject}
- Last Message From: {context.last_message_sender}
- Thread Length: {len(context.messages)} messages
- Participants: {', '.join(context.participants)}

REPLY TYPE: {reply_type.value}
USER STYLE:
- Tone: {style.tone.value}
- Formality Level: {style.formality_level}
- Average Sentence Length: {style.average_sentence_length}
- Greeting Style: {style.greeting_style}
- Closing Style: {style.closing_style}
- Use Emojis: {style.use_emojis}
- Use Bullets: {style.use_bullets}
- Response Length: {style.response_length_preference}

THREAD ANALYSIS:
- Needs Action Response: {thread_analysis['needs_action_response']}
- Has Questions: {thread_analysis['has_questions']}
- Urgent: {thread_analysis['urgent_indicators']}
- Formal Required: {thread_analysis['formal_required']}
- Complexity: {thread_analysis['complexity']}

ACTION ITEMS TO ADDRESS:
{json.dumps(context.action_items, indent=2) if context.action_items else "None"}

ENTITIES TO REFERENCE:
{json.dumps(context.entities, indent=2) if context.entities else "None"}

LAST MESSAGE TO REPLY TO:
{context.last_message_content}

REQUIREMENTS:
1. Address all action items mentioned in the thread
2. Reference relevant entities naturally in the response
3. Match the user's writing style and tone
4. Respond to any questions asked
5. Keep the response {style.response_length_preference} length
6. Use {style.greeting_style} as greeting and {style.closing_style} as closing
7. {"Use emojis appropriately" if style.use_emojis else "Do not use emojis"}
8. {"Use bullet points for lists" if style.use_bullets else "Use paragraphs for lists"}

{f'ADDITIONAL INSTRUCTIONS: {custom_instructions}' if custom_instructions else ''}

Respond in JSON format:
{{
    "subject": "Reply subject line",
    "greeting": "Greeting text",
    "body": "Main email body",
    "closing": "Closing text",
    "signature": "Optional signature",
    "action_items_addressed": ["List of action items addressed"],
    "entities_referenced": ["List of entities referenced"],
    "confidence_score": 0.9
}}
"""
        
        return prompt
    
    def _parse_reply_response(self, response_content: str) -> Dict[str, Any]:
        """Parse the JSON response from GPT-4o-mini."""
        try:
            # Clean up the response
            content = response_content.strip()
            
            # Extract JSON if it's wrapped in code blocks
            if content.startswith("```json"):
                content = content[7:-3].strip()
            elif content.startswith("```"):
                content = content[3:-3].strip()
            
            # Parse JSON
            reply_data = json.loads(content)
            
            # Ensure required fields are present
            defaults = {
                "subject": "Re: No Subject",
                "greeting": "Hi",
                "body": "",
                "closing": "Best regards",
                "signature": None,
                "action_items_addressed": [],
                "entities_referenced": [],
                "confidence_score": 0.8
            }
            
            for key, default_value in defaults.items():
                if key not in reply_data:
                    reply_data[key] = default_value
            
            return reply_data
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse reply JSON: {str(e)}")
            # Return basic fallback response
            return {
                "subject": "Re: (Auto-generated)",
                "greeting": "Hi",
                "body": "I apologize, but I was unable to generate a proper response. Please review the thread and respond manually.",
                "closing": "Best regards",
                "signature": None,
                "action_items_addressed": [],
                "entities_referenced": [],
                "confidence_score": 0.3
            }
    
    def _estimate_reading_time(self, text: str) -> str:
        """Estimate reading time for the email body."""
        word_count = len(text.split())
        # Average reading speed: 200-250 words per minute
        reading_time_minutes = max(1, round(word_count / 220))
        
        if reading_time_minutes == 1:
            return "~1 minute"
        else:
            return f"~{reading_time_minutes} minutes"
    
    async def learn_user_style(
        self,
        user_id: str,
        user_messages: List[Dict[str, Any]]
    ) -> UserWritingStyle:
        """
        Learn and update user's writing style.
        
        Args:
            user_id: User identifier
            user_messages: Recent messages from the user
            
        Returns:
            Updated user writing style
        """
        try:
            # Analyze the user's writing style
            user_style = await self.style_analyzer.analyze_user_style(user_id, user_messages)
            
            logger.info(f"Updated writing style for user {user_id}")
            return user_style
            
        except Exception as e:
            logger.error(f"Style learning failed: {str(e)}")
            # Return default style
            return UserWritingStyle(
                tone=ReplyTone.PROFESSIONAL,
                formality_level=0.7,
                average_sentence_length=15.0,
                greeting_style="Hi",
                closing_style="Best regards",
                signature_included=True,
                use_emojis=False,
                use_bullets=True,
                response_length_preference="medium"
            )
    
    async def get_reply_suggestions(
        self,
        context: ThreadContext,
        num_suggestions: int = 3
    ) -> List[GeneratedReply]:
        """
        Generate multiple reply suggestions with different approaches.
        
        Args:
            context: Thread context
            num_suggestions: Number of suggestions to generate
            
        Returns:
            List of generated reply suggestions
        """
        suggestions = []
        
        # Generate different types of replies
        reply_types = [
            ReplyType.ACTION_RESPONSE,
            ReplyType.QUESTION_RESPONSE,
            ReplyType.ACKNOWLEDGMENT
        ]
        
        for i, reply_type in enumerate(reply_types[:num_suggestions]):
            try:
                suggestion = await self.generate_reply(context, reply_type)
                suggestions.append(suggestion)
            except Exception as e:
                logger.error(f"Failed to generate suggestion {i+1}: {str(e)}")
                continue
        
        return suggestions
