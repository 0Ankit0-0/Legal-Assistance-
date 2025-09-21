# 📂 Legal AI Models – README  

This folder contains **three main AI workflows** for legal text processing and question answering.  
It is **recommended to use Google Colab** for running these notebooks/scripts for smooth execution.  

👉 For downloading the fine-tuned models, use the links below:  
- **T5 Fine-tuned Model:** [Download here](https://drive.google.com/drive/folders/17y3umzNhUIXfWu1yEdRSQOc1bRNTWGmy?usp=drive_link)  
- **DistilBART Fine-tuned Model:** [Download here](https://drive.google.com/drive/folders/1SF3LDySBDGAh92S_upl9zO-gvmvexY4Y?usp=sharing)  

⚡ **Note:** If you are training or fine-tuning these models yourself, make sure to use **Colab GPU** (recommended) or a **local GPU setup** for best performance.  

---

## 📑 Files Overview  

1. **`Fine_tuning_Pipeline.ipynb`**  
   - Implements **DistilBART** fine-tuning.  
   - Used for summarizing **long legal documents** into concise case summaries.  
   - Trained on Indian Legal Corpus (ILC) dataset.  

2. **`Gemini_Rag.ipynb`**  
   - Retrieval-Augmented Generation (**RAG**) system using **Google Gemini API**.  
   - Requires **Gemini API key** → can be placed in a `now.py` or config file or see `file rag_gemini.py`.  
   - Best for **question answering** with sources and citations.  

3. **`T5_Fine_tune.ipynb`**  
   - Fine-tunes **T5 model** for legal summarization.  
   - Focuses on **document-level summarization** of legal cases.  

---

## ⚖️ Quick Comparison Table  

| Aspect               | T5 / DistilBART Models | RAG + Gemini System |
|-----------------------|-------------------------|----------------------|
| **Primary Function**  | 📄 Summarizes legal documents | ❓ Answers legal questions |
| **Input Type**        | Long legal case text    | Natural language questions |
| **Output Type**       | Short summary paragraphs | Detailed answers with sources |
| **User Interaction**  | "Here's a case → Summarize it" | "What are my rights?" → Detailed answer |
| **Knowledge Base**    | Trained on ILC dataset | PDF docs + Q&A database + Web knowledge |
| **Response Style**    | Brief, factual summary | Conversational, explanatory |
| **Sources**           | No source citations    | ✅ Shows sources & citations |
| **Use Case**          | Document processing    | Legal consultation & research |

---

✅ **Recommendation:**  
- Use **T5/DistilBART** for **summarization tasks** (case briefs, judgments).  
- Use **Gemini RAG** for **interactive Q&A** (rights, precedents, legal advice with citations).  
