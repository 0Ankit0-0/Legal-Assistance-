import re
import mimetypes
from werkzeug.utils import secure_filename
from config import Config

class ValidationError(Exception):
    """Custom validation error"""
    pass

class Validators:
    @staticmethod
    def validate_email(email):
        """Validate email format"""
        if not email:
            raise ValidationError("Email is required")
        
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(pattern, email):
            raise ValidationError("Invalid email format")
        
        return email.lower()
    
    @staticmethod
    def validate_password(password):
        """Validate password strength"""
        if not password:
            raise ValidationError("Password is required")
        
        if len(password) < 6:
            raise ValidationError("Password must be at least 6 characters long")
        
        if len(password) > 128:
            raise ValidationError("Password is too long")
        
        return password
    
    @staticmethod
    def validate_name(name):
        """Validate user name"""
        if not name:
            raise ValidationError("Name is required")
        
        if len(name.strip()) < 2:
            raise ValidationError("Name must be at least 2 characters long")
        
        if len(name) > 50:
            raise ValidationError("Name is too long")
        
        # Remove extra whitespace
        return ' '.join(name.split())
    
    @staticmethod
    def validate_file(file):
        """Validate uploaded file"""
        if not file:
            raise ValidationError("No file provided")
        
        if not file.filename:
            raise ValidationError("No file selected")
        
        # Check file extension
        filename = secure_filename(file.filename)
        if '.' not in filename:
            raise ValidationError("File must have an extension")
        
        extension = filename.rsplit('.', 1)[1].lower()
        if extension not in Config.ALLOWED_EXTENSIONS:
            raise ValidationError(f"File type not allowed. Supported types: {', '.join(Config.ALLOWED_EXTENSIONS)}")
        
        # Check file size (file.content_length might not be available in all cases)
        file.seek(0, 2)  # Seek to end
        file_size = file.tell()
        file.seek(0)  # Reset to beginning
        
        if file_size > Config.MAX_CONTENT_LENGTH:
            max_mb = Config.MAX_CONTENT_LENGTH / (1024 * 1024)
            raise ValidationError(f"File too large. Maximum size: {max_mb:.1f}MB")
        
        if file_size == 0:
            raise ValidationError("File is empty")
        
        return {
            'filename': filename,
            'extension': extension,
            'size': file_size
        }
    
    @staticmethod
    def validate_text_input(text, min_length=10, max_length=50000):
        """Validate text input for processing"""
        if not text:
            raise ValidationError("Text is required")
        
        text = text.strip()
        
        if len(text) < min_length:
            raise ValidationError(f"Text must be at least {min_length} characters long")
        
        if len(text) > max_length:
            raise ValidationError(f"Text is too long. Maximum {max_length} characters allowed")
        
        return text
    
    @staticmethod
    def validate_model_choice(model):
        """Validate AI model choice"""
        valid_models = ['t5', 'bart', 'auto', 'compare']
        
        if not model:
            return 'auto'  # Default
        
        if model.lower() not in valid_models:
            raise ValidationError(f"Invalid model choice. Valid options: {', '.join(valid_models)}")
        
        return model.lower()
    
    @staticmethod
    def validate_pagination(page, limit):
        """Validate pagination parameters"""
        try:
            page = int(page) if page else 1
            limit = int(limit) if limit else 20
        except ValueError:
            raise ValidationError("Page and limit must be numbers")
        
        if page < 1:
            raise ValidationError("Page must be at least 1")
        
        if limit < 1 or limit > 100:
            raise ValidationError("Limit must be between 1 and 100")
        
        return page, limit
    
    @staticmethod
    def validate_max_length(max_length):
        """Validate summary max length parameter"""
        if not max_length:
            return 150  # Default
        
        try:
            max_length = int(max_length)
        except ValueError:
            raise ValidationError("Max length must be a number")
        
        if max_length < 50 or max_length > 500:
            raise ValidationError("Max length must be between 50 and 500")
        
        return max_length
    
    @staticmethod
    def sanitize_filename(filename):
        """Sanitize filename for safe storage"""
        filename = secure_filename(filename)
        
        # Remove any remaining special characters
        filename = re.sub(r'[^\w\-_\.]', '_', filename)
        
        # Ensure it's not too long
        if len(filename) > 100:
            name, ext = filename.rsplit('.', 1)
            filename = name[:95] + '.' + ext
        
        return filename