from pymongo import MongoClient
import logging
from config import Config

logger = logging.getLogger(__name__)

class DatabaseConnection:
    _instance = None
    _client = None
    _db = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatabaseConnection, cls).__new__(cls)
        return cls._instance
    
    def connect(self):
        """Initialize database connection"""
        if self._client is None:
            try:
                self._client = MongoClient(Config.MONGODB_URI)
                self._db = self._client['legalai']
                
                # Test connection
                self._client.admin.command('ismaster')
                logger.info("✅ Connected to MongoDB successfully")
                
            except Exception as e:
                logger.error(f"❌ Failed to connect to MongoDB: {e}")
                raise
    
    @property
    def db(self):
        """Get database instance"""
        if self._db is None:
            self.connect()
        return self._db
    
    def close(self):
        """Close database connection"""
        if self._client:
            self._client.close()
            self._client = None
            self._db = None

# Global database instance
db_connection = DatabaseConnection()

def get_db():
    """Get database instance"""
    return db_connection.db