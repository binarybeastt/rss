import os
import secrets
import uuid
import asyncio
import logging
from typing import List, Optional, Dict

import feedparser
import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException, Query, Depends, Security
from fastapi.security.api_key import APIKeyHeader
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
import pytz
import re
from newspaper import Article

# Database simulation (replace with actual database like PostgreSQL/SQLAlchemy)
class APIKeyManager:
    def __init__(self):
        self._keys = {}  # In-memory storage, replace with persistent storage

    def generate_key(self, user_id: str, expires_in: int = 30) -> str:
        """
        Generate a new API key with expiration
        
        Args:
            user_id (str): User identifier
            expires_in (int): Expiration time in days
        
        Returns:
            str: Generated API key
        """
        key = f"an_{secrets.token_urlsafe(32)}"
        expiration = datetime.now(pytz.UTC) + timedelta(days=expires_in)
        
        self._keys[key] = {
            'user_id': user_id,
            'created_at': datetime.now(pytz.UTC),
            'expires_at': expiration,
            'is_active': True
        }
        return key

    def validate_key(self, key: str) -> bool:
        """
        Validate an API key
        
        Args:
            key (str): API key to validate
        
        Returns:
            bool: Whether the key is valid and active
        """
        if key not in self._keys:
            return False
        
        key_info = self._keys[key]
        if not key_info['is_active']:
            return False
        
        return datetime.now(pytz.UTC) < key_info['expires_at']

    def revoke_key(self, key: str):
        """
        Revoke an API key
        
        Args:
            key (str): API key to revoke
        """
        if key in self._keys:
            self._keys[key]['is_active'] = False

    def refresh_key(self, key: str, expires_in: int = 30) -> Optional[str]:
        """
        Refresh an existing API key
        
        Args:
            key (str): Existing API key
            expires_in (int): New expiration time in days
        
        Returns:
            Optional[str]: New API key or None if refresh fails
        """
        if not self.validate_key(key):
            return None
        
        user_id = self._keys[key]['user_id']
        self.revoke_key(key)
        return self.generate_key(user_id, expires_in)

# Initialize API key manager
api_key_manager = APIKeyManager()

# Async API key dependency
async def get_api_key(api_key_header: str = Security(APIKeyHeader(name='X-API-Key'))):
    """
    API key validation dependency
    
    Args:
        api_key_header (str): API key from request header
    
    Raises:
        HTTPException: If API key is invalid
    """
    if not api_key_manager.validate_key(api_key_header):
        raise HTTPException(status_code=403, detail="Invalid or expired API key")
    return api_key_header

class NewsArticle(BaseModel):
    title: str
    link: str
    description: str
    published_at: str
    source: str = Field(default="Google News")
    content: Optional[str] = None

class GoogleNewsRSSFetcher:
    @staticmethod
    async def fetch_news(
        query: str, 
        limit: int = 10, 
        include_content: bool = False
    ) -> List[NewsArticle]:
        """
        Asynchronous news fetching with improved error handling
        
        Args:
            query (str): Search query or topic
            limit (int): Maximum number of articles to return
            include_content (bool): Fetch full article content
        
        Returns:
            List[NewsArticle]: List of news articles
        """
        try:
            # Use asyncio to potentially parallelize content fetching
            rss_url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
            
            # Use requests-futures or httpx for true async
            loop = asyncio.get_event_loop()
            feed = await loop.run_in_executor(None, feedparser.parse, rss_url)
            
            articles = []
            for entry in feed.entries[:limit]:
                article_data = {
                    'title': entry.get('title', ''),
                    'link': entry.get('link', ''),
                    'description': GoogleNewsRSSFetcher._clean_description(entry.get('summary', '')),
                    'published_at': GoogleNewsRSSFetcher._parse_time(entry.get('published', '')),
                }
                
                if include_content:
                    # Parallel content fetching using newspaper3k
                    content = await loop.run_in_executor(
                        None, 
                        GoogleNewsRSSFetcher._fetch_content_advanced, 
                        entry.get('link', '')
                    )
                    article_data['content'] = content
                
                articles.append(NewsArticle(**article_data))
            
            return articles

        except Exception as e:
            logging.error(f"News fetch error: {e}")
            return []

    @staticmethod
    def _clean_description(description: str) -> str:
        """Clean and truncate description"""
        cleaned = re.sub('<[^<]+?>', '', description)
        return (cleaned[:300] + '...') if len(cleaned) > 300 else cleaned

    @staticmethod
    def _parse_time(time_str: str) -> str:
        """Parse time with robust error handling"""
        try:
            parsed_time = feedparser._parse_date(time_str)
            dt = datetime(*parsed_time[:6], tzinfo=pytz.UTC)
            return dt.isoformat()
        except Exception:
            return datetime.now(pytz.UTC).isoformat()

    @staticmethod
    def _fetch_content(url: str) -> str:
        """Fetch article content with timeout and error handling"""
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            paragraphs = soup.find_all('p')
            content = ' '.join([p.get_text(strip=True) for p in paragraphs])
            
            return (content[:2000] + '...') if len(content) > 2000 else content
        
        except requests.RequestException as e:
            logging.warning(f"Content fetch error for {url}: {e}")
            return "Content unavailable"
        
    @staticmethod
    def _fetch_content_advanced(url: str) -> Optional[str]:
        """
        Advanced content extraction using newspaper3k
        
        Args:
            url (str): URL of the article
        
        Returns:
            Optional[str]: Full article content or None
        """
        try:
            # Download and parse the article
            article = Article(url)
            article.download()
            article.parse()
            print(article.text)
            # Extract main article text
            return article.text if article.text else None
        
        except Exception as e:
            logging.warning(f"Advanced content fetch error for {url}: {e}")
            
            # Fallback to traditional method if newspaper3k fails
            try:
                return GoogleNewsRSSFetcher._fetch_content_fallback(url)
            except Exception:
                return None

    @staticmethod
    def _fetch_content_fallback(url: str) -> Optional[str]:
        """
        Fallback content extraction method
        
        Args:
            url (str): URL of the article
        
        Returns:
            Optional[str]: Full article content or None
        """
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # More sophisticated content extraction
            # Try different common content container selectors
            content_selectors = [
                'div.article-body',
                'div.content',
                'article',
                'div.post-content',
                'main',
                'body'
            ]
            
            for selector in content_selectors:
                content_div = soup.select_one(selector)
                if content_div:
                    paragraphs = content_div.find_all('p')
                    content = ' '.join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])
                    return content if content else None
            
            # If no specific selector works, fall back to all paragraphs
            paragraphs = soup.find_all('p')
            content = ' '.join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])
            return content if content else None
        
        except requests.RequestException as e:
            logging.warning(f"Fallback content fetch error for {url}: {e}")
            return None

# FastAPI App Configuration
app = FastAPI(
    title="Google News Fetcher API",
    description="Efficient News Aggregation API with API Key Management",
    version="1.0.0"
)

@app.get('/news/google', response_model=Dict, dependencies=[Depends(get_api_key)])
async def get_google_news(
    query: str = Query(..., description="Search query for Google News"),
    limit: int = Query(10, ge=1, le=50, description="Maximum articles"),
    include_content: bool = Query(False, description="Include article content")
):
    """
    Fetch Google News headlines with optional full content
    Requires valid API key in X-API-Key header
    """
    try:
        headlines = await GoogleNewsRSSFetcher.fetch_news(query, limit, include_content)
        return {
            'status': 'success',
            'query': query,
            'total_articles': len(headlines),
            'articles': [article.dict() for article in headlines]
        }
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post('/api-keys/generate')
def generate_api_key(user_id: str = Query(..., description="User identifier")):
    """Generate a new API key for a user"""
    key = api_key_manager.generate_key(user_id)
    return {"api_key": key, "user_id": user_id}

@app.post('/api-keys/revoke')
def revoke_api_key(api_key: str = Query(..., description="API key to revoke")):
    """Revoke an existing API key"""
    api_key_manager.revoke_key(api_key)
    return {"status": "success", "message": "API key revoked"}

@app.post('/api-keys/refresh')
def refresh_api_key(
    api_key: str = Query(..., description="Existing API key"),
    expires_in: int = Query(30, description="Expiration in days")
):
    """Refresh an existing API key"""
    new_key = api_key_manager.refresh_key(api_key, expires_in)
    if new_key:
        return {"new_api_key": new_key}
    raise HTTPException(status_code=400, detail="Cannot refresh invalid key")