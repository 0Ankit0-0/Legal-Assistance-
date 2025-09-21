import easyocr
import fitz  # PyMuPDF
import cv2
import numpy as np
from PIL import Image
import io
import logging

logger = logging.getLogger(__name__)

class OCRProcessor:
    def __init__(self, languages=['en']):
        self.languages = languages
        self.reader = None
        self._initialize_ocr()
    
    def _initialize_ocr(self):
        """Initialize EasyOCR reader"""
        try:
            self.reader = easyocr.Reader(self.languages, gpu=False)
            logger.info(f"✅ OCR initialized for languages: {self.languages}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize OCR: {e}")
            raise
    
    def preprocess_image(self, image):
        """Preprocess image for better OCR accuracy"""
        try:
            # Convert PIL to OpenCV format
            if isinstance(image, Image.Image):
                image_array = np.array(image)
            else:
                image_array = image
            
            # Convert to grayscale if needed
            if len(image_array.shape) == 3:
                gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = image_array
            
            # Apply denoising
            denoised = cv2.fastNlMeansDenoising(gray)
            
            # Apply adaptive thresholding
            thresh = cv2.adaptiveThreshold(
                denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            
            # Apply morphological operations to clean up
            kernel = np.ones((1, 1), np.uint8)
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            return cleaned
            
        except Exception as e:
            logger.warning(f"Image preprocessing failed: {e}")
            return image
    
    def extract_text_from_image(self, image, min_confidence=0.5):
        """Extract text from image using OCR"""
        try:
            if not self.reader:
                self._initialize_ocr()
            
            # Preprocess image
            processed_image = self.preprocess_image(image)
            
            # Perform OCR
            results = self.reader.readtext(processed_image)
            
            # Filter results by confidence and extract text
            extracted_texts = []
            confidences = []
            
            for (bbox, text, confidence) in results:
                if confidence >= min_confidence and text.strip():
                    extracted_texts.append(text.strip())
                    confidences.append(confidence)
            
            # Combine texts
            combined_text = ' '.join(extracted_texts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            logger.info(f"✅ OCR extracted {len(combined_text)} characters with {avg_confidence:.2f} confidence")
            
            return {
                'text': combined_text,
                'confidence': avg_confidence,
                'word_count': len(extracted_texts)
            }
            
        except Exception as e:
            logger.error(f"❌ OCR extraction failed: {e}")
            return {
                'text': '',
                'confidence': 0,
                'word_count': 0
            }

class PDFProcessor:
    def __init__(self, ocr_processor=None):
        self.ocr_processor = ocr_processor or OCRProcessor()
    
    def extract_text_from_pdf(self, pdf_content):
        """Extract text from PDF including OCR for images"""
        try:
            # Open PDF from bytes
            doc = fitz.open(stream=pdf_content, filetype="pdf")
            
            result = {
                'direct_text': '',
                'ocr_text': '',
                'combined_text': '',
                'total_pages': len(doc),
                'images_processed': 0,
                'processing_info': []
            }
            
            all_direct_text = []
            all_ocr_text = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_info = {'page': page_num + 1, 'direct_text_length': 0, 'images_found': 0, 'ocr_text_length': 0}
                
                # Extract direct text
                direct_text = page.get_text()
                page_info['direct_text_length'] = len(direct_text)
                all_direct_text.append(direct_text)
                
                # Extract images and perform OCR
                image_list = page.get_images()
                page_info['images_found'] = len(image_list)
                page_ocr_text = []
                
                for img_index, img in enumerate(image_list):
                    try:
                        # Extract image
                        xref = img[0]
                        pix = fitz.Pixmap(doc, xref)
                        
                        # Skip very small images (likely decorative)
                        if pix.width < 50 or pix.height < 50:
                            pix = None
                            continue
                        
                        # Convert to PIL Image
                        if pix.n - pix.alpha < 4:  # GRAY or RGB
                            img_data = pix.tobytes("ppm")
                            img_pil = Image.open(io.BytesIO(img_data))
                        else:  # CMYK
                            pix1 = fitz.Pixmap(fitz.csRGB, pix)
                            img_data = pix1.tobytes("ppm")
                            img_pil = Image.open(io.BytesIO(img_data))
                            pix1 = None
                        
                        pix = None  # Free memory
                        
                        # Perform OCR
                        ocr_result = self.ocr_processor.extract_text_from_image(img_pil)
                        
                        if ocr_result['text']:
                            page_ocr_text.append(f"[Image {img_index + 1}]: {ocr_result['text']}")
                            result['images_processed'] += 1
                        
                    except Exception as e:
                        logger.warning(f"Failed to process image {img_index + 1} on page {page_num + 1}: {e}")
                        continue
                
                # Combine OCR text for this page
                page_ocr_combined = ' '.join(page_ocr_text)
                page_info['ocr_text_length'] = len(page_ocr_combined)
                all_ocr_text.append(page_ocr_combined)
                
                result['processing_info'].append(page_info)
            
            doc.close()
            
            # Combine all text
            result['direct_text'] = '\n'.join(all_direct_text)
            result['ocr_text'] = '\n'.join(all_ocr_text)
            result['combined_text'] = self._combine_and_clean_text(result['direct_text'], result['ocr_text'])
            
            logger.info(f"✅ PDF processed: {result['total_pages']} pages, {len(result['combined_text'])} characters, {result['images_processed']} images")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ PDF processing failed: {e}")
            return {
                'direct_text': '',
                'ocr_text': '',
                'combined_text': '',
                'total_pages': 0,
                'images_processed': 0,
                'processing_info': [],
                'error': str(e)
            }
    
    def _combine_and_clean_text(self, direct_text, ocr_text):
        """Intelligently combine direct text and OCR text"""
        try:
            # Start with direct text
            combined = direct_text
            
            # Add OCR text that's not already in direct text
            if ocr_text:
                ocr_lines = ocr_text.split('\n')
                for line in ocr_lines:
                    line = line.strip()
                    
                    # Skip empty lines and image markers
                    if not line or line.startswith('[Image'):
                        continue
                    
                    # Check if this content is already in direct text
                    if len(line) > 20:  # Only check substantial content
                        words = line.split()
                        if len(words) >= 3:
                            # Check if first 3 words appear in direct text
                            key_phrase = ' '.join(words[:3]).lower()
                            if key_phrase not in combined.lower():
                                combined += '\n' + line
            
            # Clean up the combined text
            lines = combined.split('\n')
            cleaned_lines = []
            
            for line in lines:
                line = line.strip()
                # Remove very short lines and common PDF artifacts
                if len(line) > 5 and not line.isdigit():
                    cleaned_lines.append(line)
            
            return '\n'.join(cleaned_lines)
            
        except Exception as e:
            logger.error(f"Text combination failed: {e}")
            return direct_text  # Fallback to direct text only

# Global instances
ocr_processor = OCRProcessor()
pdf_processor = PDFProcessor(ocr_processor)