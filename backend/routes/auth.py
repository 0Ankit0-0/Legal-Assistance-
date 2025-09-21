from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from services.auth_service import AuthService
from utils.validators import Validators, ValidationError
import logging

logger = logging.getLogger(__name__)

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/register', methods=['POST'])
def register():
    """User registration endpoint"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'success': False, 'message': 'No data provided'}), 400
        
        # Extract and validate data
        name = Validators.validate_name(data.get('name', ''))
        email = Validators.validate_email(data.get('email', ''))
        password = Validators.validate_password(data.get('password', ''))
        
        # Register user
        result = AuthService.register_user(name, email, password)
        
        status_code = 201 if result['success'] else 400
        return jsonify(result), status_code
        
    except ValidationError as e:
        logger.warning(f"Registration validation error: {e}")
        return jsonify({'success': False, 'message': str(e)}), 400
    except Exception as e:
        logger.error(f"Registration error: {e}")
        return jsonify({'success': False, 'message': 'Registration failed'}), 500

@auth_bp.route('/login', methods=['POST'])
def login():
    """User login endpoint"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'success': False, 'message': 'No data provided'}), 400
        
        # Extract data
        email = data.get('email', '').strip()
        password = data.get('password', '')
        
        # Login user
        result = AuthService.login_user(email, password)
        
        status_code = 200 if result['success'] else 401
        return jsonify(result), status_code
        
    except Exception as e:
        logger.error(f"Login error: {e}")
        return jsonify({'success': False, 'message': 'Login failed'}), 500

@auth_bp.route('/profile', methods=['GET'])
@jwt_required()
def get_profile():
    """Get current user profile"""
    try:
        user = AuthService.get_current_user()
        
        if not user:
            return jsonify({'success': False, 'message': 'User not found'}), 404
        
        return jsonify({
            'success': True,
            'user': user.to_dict()
        })
        
    except Exception as e:
        logger.error(f"Profile fetch error: {e}")
        return jsonify({'success': False, 'message': 'Failed to fetch profile'}), 500

@auth_bp.route('/profile', methods=['PUT'])
@jwt_required()
def update_profile():
    """Update user profile"""
    try:
        data = request.get_json()
        user_id = get_jwt_identity()
        
        if not data:
            return jsonify({'success': False, 'message': 'No data provided'}), 400
        
        # Extract and validate data
        name = data.get('name')
        email = data.get('email')
        
        if name:
            name = Validators.validate_name(name)
        if email:
            email = Validators.validate_email(email)
        
        # Update profile
        result = AuthService.update_user_profile(user_id, name, email)
        
        status_code = 200 if result['success'] else 400
        return jsonify(result), status_code
        
    except ValidationError as e:
        logger.warning(f"Profile update validation error: {e}")
        return jsonify({'success': False, 'message': str(e)}), 400
    except Exception as e:
        logger.error(f"Profile update error: {e}")
        return jsonify({'success': False, 'message': 'Profile update failed'}), 500

@auth_bp.route('/change-password', methods=['POST'])
@jwt_required()
def change_password():
    """Change user password"""
    try:
        data = request.get_json()
        user_id = get_jwt_identity()
        
        if not data:
            return jsonify({'success': False, 'message': 'No data provided'}), 400
        
        # Extract data
        current_password = data.get('current_password', '')
        new_password = data.get('new_password', '')
        
        if not current_password or not new_password:
            return jsonify({'success': False, 'message': 'Current and new passwords required'}), 400
        
        # Validate new password
        new_password = Validators.validate_password(new_password)
        
        # Change password
        result = AuthService.change_password(user_id, current_password, new_password)
        
        status_code = 200 if result['success'] else 400
        return jsonify(result), status_code
        
    except ValidationError as e:
        logger.warning(f"Password change validation error: {e}")
        return jsonify({'success': False, 'message': str(e)}), 400
    except Exception as e:
        logger.error(f"Password change error: {e}")
        return jsonify({'success': False, 'message': 'Password change failed'}), 500

@auth_bp.route('/logout', methods=['POST'])
@jwt_required()
def logout():
    """User logout endpoint"""
    try:
        # In a more sophisticated implementation, you might want to blacklist the JWT token
        # For now, we'll just return success as the frontend will remove the token
        return jsonify({'success': True, 'message': 'Logged out successfully'})
        
    except Exception as e:
        logger.error(f"Logout error: {e}")
        return jsonify({'success': False, 'message': 'Logout failed'}), 500

@auth_bp.route('/verify', methods=['GET'])
@jwt_required()
def verify_token():
    """Verify JWT token validity"""
    try:
        user = AuthService.get_current_user()
        
        if not user:
            return jsonify({'success': False, 'message': 'Invalid token'}), 401
        
        return jsonify({
            'success': True,
            'user': user.to_dict(),
            'message': 'Token is valid'
        })
        
    except Exception as e:
        logger.error(f"Token verification error: {e}")
        return jsonify({'success': False, 'message': 'Token verification failed'}), 401