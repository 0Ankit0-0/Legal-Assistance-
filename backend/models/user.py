from datetime import datetime, timedelta
from bson import ObjectId
from utils.db import get_db
import bcrypt
import logging

logger = logging.getLogger(__name__)

class User:
    def __init__(self, name=None, email=None, password=None, user_id=None):
        self.db = get_db()
        self.collection = self.db.users
        
        if user_id:
            self.data = self.get_by_id(user_id)
        else:
            self.data = {
                'name': name,
                'email': email.lower() if email else None,
                'password': self._hash_password(password) if password else None,
                'created_at': datetime.utcnow(),
                'subscription': 'free',
                'usage_stats': {
                    'documents_processed': 0,
                    'questions_asked': 0,
                    'api_calls_today': 0,
                    'last_reset': datetime.utcnow()
                }
            }
    
    def _hash_password(self, password):
        """Hash password using bcrypt"""
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    
    def verify_password(self, password):
        """Verify password against hash"""
        if not self.data or not self.data.get('password'):
            return False
        return bcrypt.checkpw(password.encode('utf-8'), self.data['password'])
    
    def save(self):
        """Save user to database"""
        try:
            if '_id' in self.data:
                # Update existing user
                result = self.collection.update_one(
                    {'_id': self.data['_id']},
                    {'$set': {k: v for k, v in self.data.items() if k != '_id'}}
                )
                return result.modified_count > 0
            else:
                # Create new user
                result = self.collection.insert_one(self.data)
                self.data['_id'] = result.inserted_id
                logger.info(f"✅ User created: {self.data['email']}")
                return str(result.inserted_id)
        except Exception as e:
            logger.error(f"❌ Error saving user: {e}")
            return False
    
    def get_by_email(self, email):
        """Get user by email"""
        try:
            user_data = self.collection.find_one({'email': email.lower()})
            if user_data:
                self.data = user_data
            return user_data
        except Exception as e:
            logger.error(f"❌ Error getting user by email: {e}")
            return None
    
    def get_by_id(self, user_id):
        """Get user by ID"""
        try:
            user_data = self.collection.find_one({'_id': ObjectId(user_id)})
            if user_data:
                self.data = user_data
            return user_data
        except Exception as e:
            logger.error(f"❌ Error getting user by ID: {e}")
            return None
    
    def update_usage(self, usage_type):
        """Update user usage statistics"""
        try:
            today = datetime.utcnow().date()
            last_reset = self.data['usage_stats'].get('last_reset', datetime.utcnow()).date()
            
            update_data = {'$inc': {}}
            
            # Reset daily counter if it's a new day
            if today > last_reset:
                update_data['$set'] = {
                    'usage_stats.api_calls_today': 1,
                    'usage_stats.last_reset': datetime.utcnow()
                }
            else:
                update_data['$inc']['usage_stats.api_calls_today'] = 1
            
            # Update specific usage type
            if usage_type == 'document':
                update_data['$inc']['usage_stats.documents_processed'] = 1
            elif usage_type == 'question':
                update_data['$inc']['usage_stats.questions_asked'] = 1
            
            result = self.collection.update_one(
                {'_id': self.data['_id']},
                update_data
            )
            
            # Update local data
            if result.modified_count > 0:
                updated_user = self.get_by_id(str(self.data['_id']))
                self.data = updated_user
                
            return result.modified_count > 0
            
        except Exception as e:
            logger.error(f"❌ Error updating user usage: {e}")
            return False
    
    def can_make_request(self):
        """Check if user can make API request based on rate limits"""
        from config import Config
        
        today = datetime.utcnow().date()
        last_reset = self.data['usage_stats'].get('last_reset', datetime.utcnow()).date()
        
        # Reset counter if new day
        if today > last_reset:
            return True
        
        api_calls_today = self.data['usage_stats'].get('api_calls_today', 0)
        return api_calls_today < Config.RATE_LIMIT_PER_HOUR
    
    def to_dict(self, include_sensitive=False):
        """Convert user data to dictionary"""
        if not self.data:
            return None
        
        result = {
            'id': str(self.data['_id']),
            'name': self.data['name'],
            'email': self.data['email'],
            'subscription': self.data['subscription'],
            'usage_stats': self.data['usage_stats'],
            'created_at': self.data['created_at'].isoformat()
        }
        
        if include_sensitive:
            result['password'] = self.data.get('password')
        
        return result
    
    @staticmethod
    def email_exists(email):
        """Check if email already exists"""
        db = get_db()
        return db.users.find_one({'email': email.lower()}) is not None