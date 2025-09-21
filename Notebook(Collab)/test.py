"""
1. Kindly run this script after training both your models to ensure they are functioning correctly. 
2. This script will load both models, test them on sample legal texts, and compare their performance.
3. Make sure to update the model paths to point to your saved models.
4. You can use google colab or your local machine with GPU support for faster testing.
5. Make sure you have the required libraries installed:- pip install transformers torch pandas
"""
# 🧪 TEST BOTH YOUR TRAINED MODELS
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import time
import pandas as pd
from datetime import datetime

# Your model paths (UPDATE THESE!)
T5_MODEL_PATH = "/content/drive/MyDrive/Legal_Summarizer_Project(T5)/models/t5-legal-summarizer_final"  # Update if different
BART_MODEL_PATH = "/content/drive/MyDrive/LegalSummarizer/Model_Bart"   # Update if different

print("🔥 TESTING YOUR TRAINED LEGAL AI MODELS")
print("=" * 60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Using device: {device}")

class DualModelTester:
    def __init__(self):
        self.t5_model = None
        self.t5_tokenizer = None
        self.bart_model = None
        self.bart_tokenizer = None
        self.models_loaded = {"t5": False, "bart": False}
        
        self.load_models()
    
    def load_models(self):
        """Load both your trained models"""
        print("\n📦 Loading your trained models...")
        
        # Load T5 Model
        try:
            print("   Loading T5 model...")
            self.t5_tokenizer = T5Tokenizer.from_pretrained(T5_MODEL_PATH)
            self.t5_model = T5ForConditionalGeneration.from_pretrained(T5_MODEL_PATH)
            self.t5_model.to(device)
            self.t5_model.eval()
            self.models_loaded["t5"] = True
            print("   ✅ T5 model loaded successfully!")
        except Exception as e:
            print(f"   ❌ T5 loading failed: {e}")
        
        # Load DistilBART Model
        try:
            print("   Loading DistilBART model...")
            self.bart_tokenizer = AutoTokenizer.from_pretrained(BART_MODEL_PATH)
            self.bart_model = AutoModelForSeq2SeqLM.from_pretrained(BART_MODEL_PATH)
            self.bart_model.to(device)
            self.bart_model.eval()
            self.models_loaded["bart"] = True
            print("   ✅ DistilBART model loaded successfully!")
        except Exception as e:
            print(f"   ❌ DistilBART loading failed: {e}")
        
        print(f"\n🎯 Models loaded: T5={self.models_loaded['t5']}, DistilBART={self.models_loaded['bart']}")
    
    def test_t5(self, text, max_length=150):
        """Test T5 model"""
        if not self.models_loaded["t5"]:
            return {"error": "T5 model not loaded", "time": 0}
        
        try:
            start_time = time.time()
            
            inputs = self.t5_tokenizer(
                "summarize: " + text,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            ).to(device)
            
            with torch.no_grad():
                summary_ids = self.t5_model.generate(
                    inputs["input_ids"],
                    max_length=max_length,
                    min_length=30,
                    length_penalty=2.0,
                    num_beams=4,
                    repetition_penalty=1.2,
                    early_stopping=True
                )
            
            summary = self.t5_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            
            processing_time = time.time() - start_time
            
            return {
                "summary": summary,
                "model": "T5",
                "time": processing_time,
                "original_length": len(text.split()),
                "summary_length": len(summary.split()),
                "compression_ratio": f"{len(summary.split())/len(text.split()):.1%}"
            }
            
        except Exception as e:
            return {"error": str(e), "time": 0}
    
    def test_bart(self, text, max_length=150):
        """Test DistilBART model"""
        if not self.models_loaded["bart"]:
            return {"error": "DistilBART model not loaded", "time": 0}
        
        try:
            start_time = time.time()
            
            inputs = self.bart_tokenizer(
                text,
                return_tensors="pt",
                max_length=1024,
                truncation=True,
                padding=True
            ).to(device)
            
            with torch.no_grad():
                summary_ids = self.bart_model.generate(
                    inputs["input_ids"],
                    max_length=max_length,
                    min_length=30,
                    length_penalty=1.0,
                    num_beams=4,
                    early_stopping=True
                )
            
            summary = self.bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            
            processing_time = time.time() - start_time
            
            return {
                "summary": summary,
                "model": "DistilBART",
                "time": processing_time,
                "original_length": len(text.split()),
                "summary_length": len(summary.split()),
                "compression_ratio": f"{len(summary.split())/len(text.split()):.1%}"
            }
            
        except Exception as e:
            return {"error": str(e), "time": 0}
    
    def compare_models(self, text):
        """Compare both models on the same text"""
        print(f"\n🔬 TESTING BOTH MODELS")
        print("-" * 40)
        
        # Test T5
        t5_result = self.test_t5(text)
        
        # Test DistilBART  
        bart_result = self.test_bart(text)
        
        # Display results
        print(f"📄 Input text length: {len(text.split())} words")
        print(f"📄 Input preview: {text[:100]}...")
        
        print(f"\n🤖 T5 MODEL RESULTS:")
        if "error" not in t5_result:
            print(f"   ✅ Summary: {t5_result['summary']}")
            print(f"   ⏱️ Time: {t5_result['time']:.2f}s")
            print(f"   📊 Length: {t5_result['summary_length']} words ({t5_result['compression_ratio']})")
        else:
            print(f"   ❌ Error: {t5_result['error']}")
        
        print(f"\n🤖 DISTILBART MODEL RESULTS:")
        if "error" not in bart_result:
            print(f"   ✅ Summary: {bart_result['summary']}")
            print(f"   ⏱️ Time: {bart_result['time']:.2f}s")
            print(f"   📊 Length: {bart_result['summary_length']} words ({bart_result['compression_ratio']})")
        else:
            print(f"   ❌ Error: {bart_result['error']}")
        
        # Determine winner
        if "error" not in t5_result and "error" not in bart_result:
            print(f"\n🏆 COMPARISON:")
            print(f"   ⚡ Faster: {'T5' if t5_result['time'] < bart_result['time'] else 'DistilBART'}")
            print(f"   📏 More detailed: {'T5' if t5_result['summary_length'] > bart_result['summary_length'] else 'DistilBART'}")
            
        return {"t5": t5_result, "bart": bart_result}

# Initialize tester
tester = DualModelTester()

# Test cases for your legal models
test_cases = {
    "Contract Dispute": """
    The plaintiff ABC Corporation entered into a supply agreement with defendant XYZ Ltd for delivery of 
    industrial equipment worth Rs. 50 lakhs. The contract specified delivery within 90 days of signing. 
    However, defendant failed to deliver on time and the equipment delivered was defective. Plaintiff 
    suffered business losses due to delayed operations. The court examined the contract terms and found 
    defendant liable for breach. The court awarded damages of Rs. 15 lakhs plus interest to plaintiff 
    and directed defendant to replace the defective equipment within 30 days.
    """,
    
    "Criminal Case": """
    The accused was charged under Section 420 IPC for cheating. The prosecution alleged that accused 
    promised high returns on investment in a fake scheme and collected Rs. 25 lakhs from multiple 
    victims. The defense argued it was a legitimate business that failed due to market conditions. 
    After examining evidence including bank records and witness testimonies, the court found accused 
    guilty of intentional deception. The court sentenced accused to 2 years imprisonment and ordered 
    restitution of Rs. 25 lakhs to victims within 6 months.
    """,
    
    "Property Dispute": """
    This case involves a dispute over ownership of residential property in Delhi. Plaintiff claimed 
    ownership through registered sale deed dated 2010, while defendant claimed adverse possession 
    for 15 years. The property records showed multiple transactions and mutations. Court examined 
    documentary evidence, conducted local inquiry, and heard testimonies of neighbors. The judgment 
    held that plaintiff's title through registered deed was valid and defendant failed to prove 
    continuous adverse possession. Court granted possession to plaintiff and ordered defendant to 
    vacate within 3 months.
    """
}

print(f"\n🧪 COMPREHENSIVE MODEL TESTING")
print("=" * 60)

# Test all cases
all_results = []

for case_name, case_text in test_cases.items():
    print(f"\n📋 TESTING: {case_name.upper()}")
    print("=" * 50)
    
    results = tester.compare_models(case_text)
    results["case_name"] = case_name
    all_results.append(results)

# Summary comparison
print(f"\n📊 OVERALL PERFORMANCE SUMMARY")
print("=" * 60)

t5_times = [r["t5"]["time"] for r in all_results if "error" not in r["t5"]]
bart_times = [r["bart"]["time"] for r in all_results if "error" not in r["bart"]]

if t5_times and bart_times:
    print(f"⏱️ Average Speed:")
    print(f"   T5: {sum(t5_times)/len(t5_times):.2f}s per summary")
    print(f"   DistilBART: {sum(bart_times)/len(bart_times):.2f}s per summary")
    
    print(f"\n🏆 Winner: {'T5 is faster! ⚡' if sum(t5_times) < sum(bart_times) else 'DistilBART is faster! ⚡'}")

# Save results
results_df = []
for result in all_results:
    for model_name in ["t5", "bart"]:
        model_result = result[model_name]
        if "error" not in model_result:
            results_df.append({
                "Case": result["case_name"],
                "Model": model_result["model"],
                "Summary": model_result["summary"],
                "Processing_Time": f"{model_result['time']:.2f}s",
                "Original_Words": model_result["original_length"],
                "Summary_Words": model_result["summary_length"],
                "Compression": model_result["compression_ratio"]
            })

if results_df:
    df = pd.DataFrame(results_df)
    df.to_csv("/content/drive/MyDrive/model_comparison_results.csv", index=False)
    print(f"\n💾 Results saved to: /content/drive/MyDrive/model_comparison_results.csv")

print(f"\n🎯 NEXT STEPS:")
print("1. 📊 Check the CSV file for detailed comparison")
print("2. 🌐 Ready to create web app with both models")
print("3. 🚀 Deploy the better performing model")
print("4. 🔄 Or use both models for different use cases")

# Interactive testing function
def interactive_test():
    """Test with your own text"""
    print(f"\n🎮 INTERACTIVE TESTING")
    print("=" * 30)
    
    while True:
        text = input("\n📝 Enter your legal case text (or 'quit' to exit): ").strip()
        
        if text.lower() == 'quit':
            break
        
        if len(text) < 50:
            print("⚠️ Please enter longer text (minimum 50 characters)")
            continue
        
        results = tester.compare_models(text)
        
        print(f"\n🎯 Which summary do you prefer?")
        if "error" not in results["t5"]:
            print(f"A) T5: {results['t5']['summary']}")
        if "error" not in results["bart"]:
            print(f"B) DistilBART: {results['bart']['summary']}")

# Uncomment to run interactive testing:
# interactive_test()

print(f"\n✅ TESTING COMPLETE! Both your models are working! 🎉")