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

from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError

import pydantic
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
import pytz
import re
from newspaper import Article

# SQLAlchemy Setup
Base = declarative_base()
DATABASE_URL = "sqlite:///./api_keys.db"

class APIKeyModel(Base):
    """SQLAlchemy model for API keys"""
    __tablename__ = "api_keys"

    id = Column(Integer, primary_key=True, index=True)
    key = Column(String, unique=True, index=True)
    user_id = Column(String, index=True)
    created_at = Column(DateTime(timezone=True), default=datetime.now(pytz.UTC))
    is_active = Column(Boolean, default=True)

# Create database and tables
engine = create_engine(DATABASE_URL)
Base.metadata.create_all(bind=engine)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class APIKeyManager:
    @staticmethod
    def get_db():
        """
        Create a database session
        
        Returns:
            Session: SQLAlchemy database session
        """
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()

    @staticmethod
    def generate_key(db: Session, user_id: str) -> str:
        """
        Generate a new API key with persistent storage
        
        Args:
            db (Session): Database session
            user_id (str): User identifier
        
        Returns:
            str: Generated API key
        """
        # Generate a secure, unique API key
        key = f"yn_{secrets.token_urlsafe(32)}"
        
        # Create new API key record
        try:
            api_key_record = APIKeyModel(
                key=key, 
                user_id=user_id, 
                created_at=datetime.now(pytz.UTC),
                is_active=True
            )
            db.add(api_key_record)
            db.commit()
            db.refresh(api_key_record)
            return key
        except SQLAlchemyError as e:
            db.rollback()
            logging.error(f"Error generating API key: {e}")
            raise HTTPException(status_code=500, detail="Could not generate API key")

    @staticmethod
    def validate_key(db: Session, key: str) -> bool:
        """
        Validate an API key
        
        Args:
            db (Session): Database session
            key (str): API key to validate
        
        Returns:
            bool: Whether the key is valid and active
        """
        try:
            api_key = db.query(APIKeyModel).filter(
                APIKeyModel.key == key, 
                APIKeyModel.is_active == True
            ).first()
            return api_key is not None
        except SQLAlchemyError:
            return False

    @staticmethod
    def revoke_key(db: Session, key: str):
        """
        Revoke an API key
        
        Args:
            db (Session): Database session
            key (str): API key to revoke
        """
        try:
            # Find and update the key
            api_key = db.query(APIKeyModel).filter(APIKeyModel.key == key).first()
            if api_key:
                api_key.is_active = False
                db.commit()
        except SQLAlchemyError as e:
            db.rollback()
            logging.error(f"Error revoking API key: {e}")
            raise HTTPException(status_code=500, detail="Could not revoke API key")

    @staticmethod
    def list_user_keys(db: Session, user_id: str) -> List[Dict]:
        """
        List all API keys for a user
        
        Args:
            db (Session): Database session
            user_id (str): User identifier
        
        Returns:
            List[Dict]: List of API keys with their details
        """
        try:
            keys = db.query(APIKeyModel).filter(
                APIKeyModel.user_id == user_id
            ).all()
            
            return [
                {
                    'key': key.key,
                    'created_at': key.created_at.isoformat(),
                    'is_active': key.is_active
                } for key in keys
            ]
        except SQLAlchemyError as e:
            logging.error(f"Error listing user keys: {e}")
            raise HTTPException(status_code=500, detail="Could not list API keys")

# Dependency for getting database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Async API key dependency
async def get_api_key(
    api_key_header: str = Security(APIKeyHeader(name='X-API-Key')),
    db: Session = Depends(get_db)
):
    """
    API key validation dependency
    
    Args:
        api_key_header (str): API key from request header
        db (Session): Database session
    
    Raises:
        HTTPException: If API key is invalid
    """
    if not APIKeyManager.validate_key(db, api_key_header):
        raise HTTPException(status_code=403, detail="Invalid or expired API key")
    return api_key_header

class NewsArticle(BaseModel):
    title: str
    link: str
    description: str
    published_at: str
    source: str = Field(default="Yahoo News")
    content: Optional[str] = None

class YahooNewsRSSFetcher:
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
            # Yahoo News RSS URL with query encoding
            rss_url = f"https://news.yahoo.com/rss?p={query}"
            
            # Use asyncio to potentially parallelize content fetching
            loop = asyncio.get_event_loop()
            feed = await loop.run_in_executor(None, feedparser.parse, rss_url)
            
            articles = []
            for entry in feed.entries[:limit]:
                article_data = {
                    'title': entry.get('title', ''),
                    'link': entry.get('link', ''),
                    'description': YahooNewsRSSFetcher._clean_description(entry.get('description', '')),
                    'published_at': YahooNewsRSSFetcher._parse_time(entry.get('published', '')),
                }
                
                if include_content:
                    # Parallel content fetching using newspaper3k
                    content = await loop.run_in_executor(
                        None, 
                        YahooNewsRSSFetcher._fetch_content_advanced, 
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
            
            # Extract main article text
            return article.text if article.text else None
        
        except Exception as e:
            logging.warning(f"Advanced content fetch error for {url}: {e}")
            
            # Fallback to traditional method if newspaper3k fails
            try:
                return YahooNewsRSSFetcher._fetch_content_fallback(url)
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
            
            # More sophisticated content extraction for Yahoo News and general sites
            content_selectors = [
                'div.article-body',
                'div.content',
                'article',
                'div.post-content',
                'div.caas-body',  # Specific to Yahoo News article structure
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

app = FastAPI(
    title="Yahoo News Fetcher API",
    description="Efficient News Aggregation API with Persistent API Key Management",
    version="1.1.0"
)

@app.post('/api-keys/generate')
def generate_api_key(
    user_id: str = Query(..., description="User identifier"),
    db: Session = Depends(get_db)
):
    """Generate a new API key for a user"""
    key = APIKeyManager.generate_key(db, user_id)
    return {"api_key": key, "user_id": user_id}

@app.post('/api-keys/revoke')
def revoke_api_key(
    api_key: str = Query(..., description="API key to revoke"),
    db: Session = Depends(get_db)
):
    """Revoke an existing API key"""
    APIKeyManager.revoke_key(db, api_key)
    return {"status": "success", "message": "API key revoked"}

@app.get('/api-keys/list')
def list_user_keys(
    user_id: str = Query(..., description="User identifier"),
    db: Session = Depends(get_db)
):
    """List all API keys for a user"""
    keys = APIKeyManager.list_user_keys(db, user_id)
    return {
        "status": "success", 
        "user_id": user_id, 
        "keys": keys
    }

@app.get('/news/yahoo', response_model=Dict, dependencies=[Depends(get_api_key)])
async def get_yahoo_news(
    query: str = Query(..., description="Search query for Yahoo News"),
    limit: int = Query(10, ge=1, le=50, description="Maximum articles"),
    include_content: bool = Query(False, description="Include article content")
):
    """
    Fetch Yahoo News headlines with optional full content
    Requires valid API key in X-API-Key header
    """
    try:
        headlines = await YahooNewsRSSFetcher.fetch_news(query, limit, include_content)
        return {
            'status': 'success',
            'query': query,
            'total_articles': len(headlines),
            'articles': [article.dict() for article in headlines]
        }
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")