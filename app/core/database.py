"""
Database configuration and setup (placeholder for future use).
"""

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import logging
from typing import Generator

from app.core.config import get_settings

logger = logging.getLogger(__name__)

# Database base class
Base = declarative_base()

# Database engine and session (will be initialized when needed)
engine = None
SessionLocal = None


def init_database():
    """
    Initialize database connection and session factory.
    
    This is a placeholder for when database functionality is needed.
    """
    global engine, SessionLocal
    
    settings = get_settings()
    
    # For now, use SQLite for development
    # In production, this would use PostgreSQL or another database
    database_url = "sqlite:///./ml_api_platform.db"
    
    engine = create_engine(
        database_url,
        connect_args={"check_same_thread": False}  # Only needed for SQLite
    )
    
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    logger.info(f"Database initialized with URL: {database_url}")


def create_tables():
    """
    Create all database tables.
    
    This should be called during application startup if database is used.
    """
    if engine is None:
        init_database()
    
    # Import all models to ensure they are registered
    from app.models.user import User
    from app.models.prediction_log import PredictionLog
    
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created")


def get_db() -> Generator:
    """
    Dependency to get database session.
    
    Yields:
        Database session
    """
    if SessionLocal is None:
        init_database()
    
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def close_database():
    """
    Close database connections.
    
    This should be called during application shutdown.
    """
    global engine
    if engine:
        engine.dispose()
        logger.info("Database connections closed")