from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity
from models.user import User
from datetime import timedelta
from config import Config
import logging

logger = logging.getLogger(__name__)

class AuthService:
    @staticmethod
    def register_user(name, email, password):
        """Register a new user"""
        try:
            # Validate input
            if not name or not email or not password:
                return {'success': False, 'message': 'All fields are required'}
            
            if len(password) < 6:
                return {'success': False, 'message': 'Password must be at least 6 characters'}
            
            # Check if user already exists
            if User.email_exists(email):
                return {'success': False, 'message': 'Email already registered'}
            
            # Create new user
            user = User(name=name, email=email, password=password)
            user_id = user.save()
            
            if user_id:
                # Create JWT token
                access_token = create_access_token(
                    identity=user_id,
                    expires_delta=timedelta(hours=Config.JWT_ACCESS_TOKEN_EXPIRES_HOURS)
                )
                
                return {
                    'success': True,
                    'message': 'User registered successfully',
                    'user': user.to_dict(),
                    'access_token': access_token
                }
            else:
                return {'success': False, 'message': 'Failed to create user'}
                
        except Exception as e:
            logger.error(f"❌ Registration error: {e}")
            return {'success': False, 'message': 'Registration failed'}
    
    @staticmethod
    def login_user(email, password):
        """Login user"""
        try:
            # Validate input
            if not email or not password:
                return {'success': False, 'message': 'Email and password are required'}
            
            # Get user from database
            user = User()
            user_data = user.get_by_email(email)
            
            if not user_data:
                return {'success': False, 'message': 'Invalid email or password'}
            
            # Verify password
            if not user.verify_password(password):
                return {'success': False, 'message': 'Invalid email or password'}
            
            # Create JWT token
            access_token = create_access_token(
                identity=str(user_data['_id']),
                expires_delta=timedelta(hours=Config.JWT_ACCESS_TOKEN_EXPIRES_HOURS)
            )
            
            return {
                'success': True,
                'message': 'Login successful',
                'user': user.to_dict(),
                'access_token': access_token
            }
            
        except Exception as e:
            logger.error(f"❌ Login error: {e}")
            return {'success': False, 'message': 'Login failed'}
    
    @staticmethod
    def get_current_user():
        """Get current authenticated user"""
        try:
            user_id = get_jwt_identity()
            if not user_id:
                return None
            
            user = User()
            user_data = user.get_by_id(user_id)
            
            return user if user_data else None
            
        except Exception as e:
            logger.error(f"❌ Error getting current user: {e}")
            return None
    
    @staticmethod
    def update_user_profile(user_id, name=None, email=None):
        """Update user profile"""
        try:
            user = User()
            user_data = user.get_by_id(user_id)
            
            if not user_data:
                return {'success': False, 'message': 'User not found'}
            
            # Update fields if provided
            if name:
                user.data['name'] = name
            
            if email and email != user.data['email']:
                # Check if new email already exists
                if User.email_exists(email):
                    return {'success': False, 'message': 'Email already in use'}
                user.data['email'] = email.lower()
            
            # Save changes
            if user.save():
                return {
                    'success': True,
                    'message': 'Profile updated successfully',
                    'user': user.to_dict()
                }
            else:
                return {'success': False, 'message': 'Failed to update profile'}
                
        except Exception as e:
            logger.error(f"❌ Profile update error: {e}")
            return {'success': False, 'message': 'Profile update failed'}
    
    @staticmethod
    def change_password(user_id, current_password, new_password):
        """Change user password"""
        try:
            if len(new_password) < 6:
                return {'success': False, 'message': 'New password must be at least 6 characters'}
            
            user = User()
            user_data = user.get_by_id(user_id)
            
            if not user_data:
                return {'success': False, 'message': 'User not found'}
            
            # Verify current password
            if not user.verify_password(current_password):
                return {'success': False, 'message': 'Current password is incorrect'}
            
            # Update password
            user.data['password'] = user._hash_password(new_password)
            
            if user.save():
                return {'success': True, 'message': 'Password changed successfully'}
            else:
                return {'success': False, 'message': 'Failed to change password'}
                
        except Exception as e:
            logger.error(f"❌ Password change error: {e}")
            return {'success': False, 'message': 'Password change failed'}