import os
import tempfile
from werkzeug.utils import secure_filename
from utils.ocr import pdf_processor
from utils.validators import Validators, ValidationError
from config import Config
import logging
import mimetypes

logger = logging.getLogger(__name__)

class FileService:
    def __init__(self):
        self.upload_folder = Config.UPLOAD_FOLDER
        os.makedirs(self.upload_folder, exist_ok=True)
    
    def process_uploaded_file(self, file):
        """Process uploaded file and extract text"""
        temp_path = None
        try:
            # Validate file
            file_info = Validators.validate_file(file)
            
            # Create temporary file
            temp_fd, temp_path = tempfile.mkstemp(
                suffix=f".{file_info['extension']}",
                dir=self.upload_folder
            )
            
            # Save uploaded file to temporary location
            with os.fdopen(temp_fd, 'wb') as temp_file:
                file.save(temp_file)
            
            logger.info(f"✅ File saved temporarily: {file_info['filename']}")
            
            # Process based on file type
            if file_info['extension'] in ['pdf']:
                result = self._process_pdf(temp_path, file_info)
            elif file_info['extension'] in ['txt']:
                result = self._process_text_file(temp_path, file_info)
            elif file_info['extension'] in ['doc', 'docx']:
                result = self._process_doc_file(temp_path, file_info)
            else:
                raise ValidationError(f"Unsupported file type: {file_info['extension']}")
            
            # Add file metadata to result
            result['file_info'] = {
                'original_name': file_info['filename'],
                'size': file_info['size'],
                'type': file_info['extension']
            }
            
            logger.info(f"✅ File processed successfully: {len(result.get('text', ''))} characters extracted")
            
            return result
            
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"❌ File processing error: {e}")
            raise ValidationError(f"Failed to process file: {str(e)}")
        finally:
            # Clean up temporary file
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                    logger.info("🗑️ Temporary file cleaned up")
                except Exception as e:
                    logger.warning(f"Failed to clean up temp file: {e}")
    
    def _process_pdf(self, file_path, file_info):
        """Process PDF file with OCR support"""
        try:
            with open(file_path, 'rb') as f:
                pdf_content = f.read()
            
            # Use PDF processor with OCR
            result = pdf_processor.extract_text_from_pdf(pdf_content)
            
            if 'error' in result:
                raise ValidationError(f"PDF processing failed: {result['error']}")
            
            # Format result
            return {
                'text': result['combined_text'],
                'processing_info': {
                    'pages': result['total_pages'],
                    'images_processed': result['images_processed'],
                    'direct_text_length': len(result['direct_text']),
                    'ocr_text_length': len(result['ocr_text']),
                    'has_images': result['images_processed'] > 0
                },
                'extraction_method': 'pdf_with_ocr'
            }
            
        except Exception as e:
            logger.error(f"❌ PDF processing failed: {e}")
            raise ValidationError(f"Failed to process PDF: {str(e)}")
    
    def _process_text_file(self, file_path, file_info):
        """Process plain text file"""
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252']
            text_content = None
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        text_content = f.read()
                    break
                except UnicodeDecodeError:
                    continue
            
            if text_content is None:
                raise ValidationError("Could not decode text file. Please ensure it's in UTF-8 format.")
            
            return {
                'text': text_content,
                'processing_info': {
                    'encoding_used': encoding,
                    'character_count': len(text_content)
                },
                'extraction_method': 'plain_text'
            }
            
        except Exception as e:
            logger.error(f"❌ Text file processing failed: {e}")
            raise ValidationError(f"Failed to process text file: {str(e)}")
    
    def _process_doc_file(self, file_path, file_info):
        """Process DOC/DOCX file"""
        try:
            # For simplicity, we'll implement basic DOC processing
            # In production, you might want to use python-docx for DOCX files
            
            if file_info['extension'] == 'docx':
                return self._process_docx(file_path)
            else:
                # For .doc files, we might need additional libraries
                # For now, return an error message
                raise ValidationError("DOC files are not supported yet. Please convert to PDF or DOCX.")
                
        except Exception as e:
            logger.error(f"❌ DOC file processing failed: {e}")
            raise ValidationError(f"Failed to process DOC file: {str(e)}")
    
    def _process_docx(self, file_path):
        """Process DOCX file (basic implementation)"""
        try:
            # This is a basic implementation
            # In production, you would use python-docx library
            # For now, we'll suggest converting to PDF
            raise ValidationError("DOCX files are not fully supported yet. Please convert to PDF for best results.")
            
        except Exception as e:
            logger.error(f"❌ DOCX processing failed: {e}")
            raise ValidationError(f"Failed to process DOCX file: {str(e)}")
    
    def validate_text_for_processing(self, text):
        """Validate extracted text before AI processing"""
        try:
            # Use validator
            validated_text = Validators.validate_text_input(text, min_length=50, max_length=50000)
            
            # Additional checks for legal document processing
            word_count = len(validated_text.split())
            
            if word_count < 10:
                raise ValidationError("Text is too short for meaningful processing. Please provide at least 10 words.")
            
            # Check for common legal document indicators
            legal_indicators = [
                'court', 'judge', 'plaintiff', 'defendant', 'case', 'law', 'legal',
                'section', 'act', 'petition', 'order', 'judgment', 'appeal',
                'contract', 'agreement', 'whereas', 'therefore', 'hereby'
            ]
            
            text_lower = validated_text.lower()
            legal_word_count = sum(1 for indicator in legal_indicators if indicator in text_lower)
            
            return {
                'text': validated_text,
                'word_count': word_count,
                'character_count': len(validated_text),
                'legal_indicators_found': legal_word_count,
                'is_likely_legal': legal_word_count >= 2,
                'quality_score': min(1.0, legal_word_count / 10)  # Simple quality score
            }
            
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"❌ Text validation failed: {e}")
            raise ValidationError(f"Text validation failed: {str(e)}")

# Global file service instance
file_service = FileService()