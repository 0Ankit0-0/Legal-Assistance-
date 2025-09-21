import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Flask Configuration
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-change-in-production')
    FLASK_ENV = os.getenv('FLASK_ENV', 'development')
    DEBUG = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'
    PORT = int(os.getenv('PORT', 5000))
    
    # JWT Configuration
    JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY', 'jwt-secret-change-in-production')
    JWT_ACCESS_TOKEN_EXPIRES_HOURS = int(os.getenv('JWT_ACCESS_TOKEN_EXPIRES_HOURS', 24))
    
    # Database Configuration
    MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/legalai')
    
    # Google Drive Configuration
    GOOGLE_DRIVE_CREDENTIALS_JSON = os.getenv('GOOGLE_DRIVE_CREDENTIALS_JSON')
    T5_MODEL_FOLDER_ID = os.getenv('T5_MODEL_FOLDER_ID')
    BART_MODEL_FOLDER_ID = os.getenv('BART_MODEL_FOLDER_ID')
    
    # AI Configuration
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    
    # File Upload Configuration
    MAX_CONTENT_LENGTH = int(os.getenv('MAX_CONTENT_LENGTH', 16777216))  # 16MB
    UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'uploads')
    ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx', 'txt'}
    
    # Rate Limiting
    RATE_LIMIT_PER_HOUR = int(os.getenv('RATE_LIMIT_PER_HOUR', 100))
    
    @staticmethod
    def validate():
        """Validate required configuration"""
        required_vars = [
            'MONGODB_URI',
            'JWT_SECRET_KEY',
            'GEMINI_API_KEY'
        ]
        
        missing = [var for var in required_vars if not os.getenv(var)]
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")