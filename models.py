
import os
import logging
from sqlalchemy import create_engine, Column, Integer, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger(__name__)

Base = declarative_base()

class Badge(Base):

    __tablename__ = "badges"

    id = Column(Integer, primary_key=True)
    image_hash = Column(Text, unique=True, nullable=False)
    source_work = Column(Text, default="")
    character = Column(Text, default="")
    acquisition_difficulty = Column(Text, default="")
    auction_description = Column(Text, default="")
    color_hist = Column(Text, default="")
    url = Column(Text, default="")

class FailedGrok(Base):
    __tablename__ = "failed_grok"

    id = Column(Integer, primary_key=True)
    image_hash = Column(Text, nullable=False)
    url = Column(Text, default="")
    error_detail = Column(Text, default="")


def get_engine():
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise RuntimeError(
            "DATABASE_URL is not set. PostgreSQL connection string is required."
        )

    logger.debug("Creating SQLAlchemy engine for %s", database_url)

    pool_size = int(os.getenv("SQL_POOL_SIZE", 5))
    max_overflow = int(os.getenv("SQL_POOL_MAX_OVERFLOW", 10))
    pool_recycle = int(os.getenv("SQL_POOL_RECYCLE", 1800))
    pool_pre_ping = True

    return create_engine(
        database_url,
        pool_size=pool_size,
        max_overflow=max_overflow,
        pool_recycle=pool_recycle,
        pool_pre_ping=pool_pre_ping,
    )

def get_session_factory():
    engine = get_engine()
    Base.metadata.create_all(engine)

    return sessionmaker(bind=engine, autoflush=False, autocommit=False)
