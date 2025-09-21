import os
import time
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from services.drive_service import drive_service
from config import Config
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import logging

logger = logging.getLogger(__name__)

class AIModelManager:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models_cache = {
            't5': {'model': None, 'tokenizer': None, 'loaded_at': None},
            'bart': {'model': None, 'tokenizer': None, 'loaded_at': None}
        }
        self.model_cache_timeout = 3600  # 1 hour
        
        # Initialize Gemini
        self._initialize_gemini()
        
        # Initialize sentence transformer for RAG
        self.embedding_model = None
        self.rag_index = None
        self.rag_corpus = []
        
    def _initialize_gemini(self):
        """Initialize Gemini API"""
        try:
            if Config.GEMINI_API_KEY:
                genai.configure(api_key=Config.GEMINI_API_KEY)
                self.gemini_model = genai.GenerativeModel('gemini-pro')
                logger.info("✅ Gemini API initialized")
            else:
                logger.warning("⚠️ Gemini API key not found")
                self.gemini_model = None
        except Exception as e:
            logger.error(f"❌ Gemini initialization failed: {e}")
            self.gemini_model = None
    
    def _is_model_cache_valid(self, model_name):
        """Check if cached model is still valid"""
        if self.models_cache[model_name]['model'] is None:
            return False
        
        if self.models_cache[model_name]['loaded_at'] is None:
            return False
        
        time_elapsed = time.time() - self.models_cache[model_name]['loaded_at']
        return time_elapsed < self.model_cache_timeout
    
    def _load_model_from_drive(self, model_name):
        """Load model from Google Drive"""
        try:
            logger.info(f"🔄 Loading {model_name} model from Google Drive...")
            
            # Get folder ID based on model type
            if model_name == 't5':
                folder_id = Config.T5_MODEL_FOLDER_ID
                local_dir = f"./models/t5_temp_{int(time.time())}"
            elif model_name == 'bart':
                folder_id = Config.BART_MODEL_FOLDER_ID
                local_dir = f"./models/bart_temp_{int(time.time())}"
            else:
                raise ValueError(f"Unknown model type: {model_name}")
            
            if not folder_id:
                raise ValueError(f"{model_name.upper()} model folder ID not configured")
            
            # Download model files
            success = drive_service.download_model_files(folder_id, local_dir)
            
            if not success:
                raise Exception(f"Failed to download {model_name} model files")
            
            # Load model and tokenizer
            if model_name == 't5':
                tokenizer = T5Tokenizer.from_pretrained(local_dir)
                model = T5ForConditionalGeneration.from_pretrained(local_dir)
            else:  # bart
                tokenizer = AutoTokenizer.from_pretrained(local_dir)
                model = AutoModelForSeq2SeqLM.from_pretrained(local_dir)
            
            # Move to device
            model = model.to(self.device)
            model.eval()
            
            # Cache the model
            self.models_cache[model_name] = {
                'model': model,
                'tokenizer': tokenizer,
                'loaded_at': time.time()
            }
            
            logger.info(f"✅ {model_name} model loaded successfully")
            
            # Clean up temporary files
            import shutil
            if os.path.exists(local_dir):
                shutil.rmtree(local_dir)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load {model_name} model: {e}")
            return False
    
    def get_model(self, model_name):
        """Get model and tokenizer, loading from Drive if necessary"""
        if not self._is_model_cache_valid(model_name):
            success = self._load_model_from_drive(model_name)
            if not success:
                return None, None
        
        cache = self.models_cache[model_name]
        return cache['model'], cache['tokenizer']
    
    def summarize_with_t5(self, text, max_length=150):
        """Summarize text using T5 model"""
        try:
            start_time = time.time()
            
            model, tokenizer = self.get_model('t5')
            if model is None or tokenizer is None:
                return {'error': 'T5 model not available'}
            
            # Prepare input
            input_text = f"summarize: {text}"
            inputs = tokenizer(
                input_text,
                max_length=512,
                truncation=True,
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate summary
            with torch.no_grad():
                summary_ids = model.generate(
                    inputs["input_ids"],
                    max_length=max_length,
                    min_length=30,
                    length_penalty=2.0,
                    num_beams=4,
                    repetition_penalty=1.2,
                    early_stopping=True
                )
            
            # Decode summary
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            
            processing_time = time.time() - start_time
            
            return {
                'summary': summary,
                'model': 'T5',
                'processing_time': round(processing_time, 2),
                'input_length': len(text.split()),
                'summary_length': len(summary.split()),
                'compression_ratio': round(len(summary.split()) / len(text.split()), 3)
            }
            
        except Exception as e:
            logger.error(f"❌ T5 summarization error: {e}")
            return {'error': f'T5 summarization failed: {str(e)}'}
    
    def summarize_with_bart(self, text, max_length=150):
        """Summarize text using DistilBART model"""
        try:
            start_time = time.time()
            
            model, tokenizer = self.get_model('bart')
            if model is None or tokenizer is None:
                return {'error': 'DistilBART model not available'}
            
            # Prepare input
            inputs = tokenizer(
                text,
                max_length=1024,
                truncation=True,
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate summary
            with torch.no_grad():
                summary_ids = model.generate(
                    inputs["input_ids"],
                    max_length=max_length,
                    min_length=30,
                    length_penalty=1.0,
                    num_beams=4,
                    early_stopping=True
                )
            
            # Decode summary
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            
            processing_time = time.time() - start_time
            
            return {
                'summary': summary,
                'model': 'DistilBART',
                'processing_time': round(processing_time, 2),
                'input_length': len(text.split()),
                'summary_length': len(summary.split()),
                'compression_ratio': round(len(summary.split()) / len(text.split()), 3)
            }
            
        except Exception as e:
            logger.error(f"❌ DistilBART summarization error: {e}")
            return {'error': f'DistilBART summarization failed: {str(e)}'}
    
    def compare_models(self, text, max_length=150):
        """Compare both T5 and DistilBART models"""
        try:
            start_time = time.time()
            
            # Get summaries from both models
            t5_result = self.summarize_with_t5(text, max_length)
            bart_result = self.summarize_with_bart(text, max_length)
            
            # Determine recommendations
            recommendation = None
            winner = {}
            
            if 'error' not in t5_result and 'error' not in bart_result:
                # Speed comparison
                winner['speed'] = 'T5' if t5_result['processing_time'] < bart_result['processing_time'] else 'DistilBART'
                
                # Length comparison
                winner['brevity'] = 'T5' if t5_result['summary_length'] < bart_result['summary_length'] else 'DistilBART'
                
                # Simple quality heuristic (prefer model with better compression)
                t5_compression = t5_result['compression_ratio']
                bart_compression = bart_result['compression_ratio']
                
                # Ideal compression is around 0.1-0.2 (10-20% of original)
                t5_score = abs(0.15 - t5_compression)
                bart_score = abs(0.15 - bart_compression)
                
                winner['quality'] = 'T5' if t5_score < bart_score else 'DistilBART'
                
                # Overall recommendation
                t5_wins = sum(1 for v in winner.values() if v == 'T5')
                recommendation = 'T5' if t5_wins >= 2 else 'DistilBART'
            
            total_time = time.time() - start_time
            
            return {
                'comparison': {
                    't5': t5_result,
                    'distilbart': bart_result
                },
                'recommendation': recommendation,
                'winner': winner,
                'total_processing_time': round(total_time, 2),
                'input_preview': text[:200] + "..." if len(text) > 200 else text
            }
            
        except Exception as e:
            logger.error(f"❌ Model comparison error: {e}")
            return {'error': f'Model comparison failed: {str(e)}'}

class RAGService:
    def __init__(self):
        self.embedding_model = None
        self.vector_index = None
        self.corpus = []
        self.gemini_model = None
        self._initialize_rag()
    
    def _initialize_rag(self):
        """Initialize RAG components"""
        try:
            # Initialize embedding model
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Initialize Gemini
            if Config.GEMINI_API_KEY:
                genai.configure(api_key=Config.GEMINI_API_KEY)
                self.gemini_model = genai.GenerativeModel('gemini-pro')
            
            # Load legal corpus (simplified version)
            self._load_legal_corpus()
            
            logger.info("✅ RAG service initialized")
            
        except Exception as e:
            logger.error(f"❌ RAG initialization failed: {e}")
    
    def _load_legal_corpus(self):
        """Load and index legal corpus"""
        try:
            # Simple legal knowledge base for demo
            # In production, this would load from your processed PDFs
            legal_docs = [
                {
                    "text": "The Consumer Protection Act 2019 provides consumers with six fundamental rights including right to safety, information, choice, and redressal.",
                    "source": "CPA2019",
                    "section": "Consumer Rights"
                },
                {
                    "text": "Under RTI Act 2005, citizens can file applications to access government information within 30 days response time.",
                    "source": "RTI",
                    "section": "Application Process"
                },
                {
                    "text": "Motor Vehicle Act prescribes penalties for traffic violations including drunk driving, overspeeding, and driving without license.",
                    "source": "MVA",
                    "section": "Penalties"
                }
                # Add more legal documents here
            ]
            
            # Create embeddings
            texts = [doc["text"] for doc in legal_docs]
            embeddings = self.embedding_model.encode(texts)
            
            # Create FAISS index
            dimension = embeddings.shape[1]
            self.vector_index = faiss.IndexFlatIP(dimension)
            
            # Normalize embeddings
            faiss.normalize_L2(embeddings.astype('float32'))
            self.vector_index.add(embeddings.astype('float32'))
            
            self.corpus = legal_docs
            
            logger.info(f"✅ Legal corpus loaded: {len(legal_docs)} documents")
            
        except Exception as e:
            logger.error(f"❌ Corpus loading failed: {e}")
    
    def search_relevant_docs(self, query, top_k=3):
        """Search for relevant documents"""
        try:
            if not self.vector_index or not self.embedding_model:
                return []
            
            # Create query embedding
            query_embedding = self.embedding_model.encode([query])
            faiss.normalize_L2(query_embedding.astype('float32'))
            
            # Search
            scores, indices = self.vector_index.search(query_embedding.astype('float32'), top_k)
            
            # Format results
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.corpus):
                    doc = self.corpus[idx].copy()
                    doc['relevance_score'] = float(score)
                    results.append(doc)
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Document search failed: {e}")
            return []
    
    def answer_question(self, question):
        """Answer legal question using RAG"""
        try:
            if not self.gemini_model:
                return {'error': 'Gemini API not available'}
            
            start_time = time.time()
            
            # Search for relevant documents
            relevant_docs = self.search_relevant_docs(question, top_k=3)
            
            if not relevant_docs:
                return {'error': 'No relevant legal information found'}
            
            # Build context
            context = "\n\n".join([
                f"Source: {doc['source']} - {doc['section']}\nContent: {doc['text']}"
                for doc in relevant_docs
            ])
            
            # Create prompt
            prompt = f"""You are a legal AI assistant. Answer the following legal question based on the provided legal documents.

LEGAL CONTEXT:
{context}

USER QUESTION: {question}

INSTRUCTIONS:
1. Provide a clear, accurate answer based on the legal context provided
2. Cite specific sources when possible (CPA2019, RTI, MVA, etc.)
3. If the context doesn't fully answer the question, acknowledge this
4. Use professional legal language but keep it understandable
5. Structure your response clearly with key points

ANSWER:"""

            # Generate response
            response = self.gemini_model.generate_content(prompt)
            
            processing_time = time.time() - start_time
            
            return {
                'answer': response.text,
                'sources': relevant_docs,
                'processing_time': round(processing_time, 2),
                'question': question
            }
            
        except Exception as e:
            logger.error(f"❌ Q&A generation failed: {e}")
            return {'error': f'Question answering failed: {str(e)}'}

# Global service instances
ai_manager = AIModelManager()
rag_service = RAGService()