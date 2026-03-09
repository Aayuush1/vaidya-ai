# 🌿 VaidyaAI — वैद्य AI

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![ML](https://img.shields.io/badge/ML-Scikit--learn-orange)
![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow)
![Gradio](https://img.shields.io/badge/UI-Gradio-red)
![License](https://img.shields.io/badge/License-MIT-green)

> *"Where 5,000 years of Ayurvedic wisdom meets Modern AI"*

## 🤔 What is VaidyaAI?
An AI-powered Ayurvedic health assistant that combines ancient Indian medicine with cutting-edge Machine Learning and Large Language Models.

## ⚡ 6 Powerful Features

| Feature | Description | Tech |
|---------|-------------|------|
| 🏥 Symptom Analyzer | Predicts top 5 diseases from symptoms | Random Forest + TF-IDF |
| 🧘 Dosha Profiler | Identifies Vata/Pitta/Kapha body type | ML Classification |
| 🌿 Herb Recommender | Personalized Ayurvedic herb suggestions | Knowledge Base AI |
| 📋 Note Summarizer | Clinical notes → Plain English | BART LLM |
| 🥗 Food Guide | Dosha-specific diet recommendations | Rule-based AI |
| 🤖 Vaidya Chatbot | Ask anything about Ayurveda | NLP + Knowledge Base |

## 📊 Model Performance
- ✅ Symptom Classifier Accuracy: **100%** on test set
- ✅ Trained on **4,800+ medical records**
- ✅ Powered by **Facebook BART Large CNN**

## 🏗️ Architecture
```
User Input (Text/Voice)
        ↓
┌─────────────────────────────────────┐
│         VaidyaAI Engine             │
│                                     │
│  🤖 Random Forest → Disease Pred   │
│  🧘 Rule-based ML → Dosha Type     │
│  🤗 BART LLM     → Summarizer      │
│  📚 Knowledge DB → Herb/Food/Yoga  │
│  🤖 OPT-125M     → Chatbot         │
└─────────────────────────────────────┘
        ↓
   Gradio Web App
```

## 🚀 Run Locally
```bash
git clone https://github.com/Aayuush1/vaidya-ai.git
cd vaidya-ai
pip install gradio transformers torch scikit-learn pandas wordcloud
cd app && python app.py
```
Open `http://localhost:7860` 🎉

## 🔮 Roadmap
- [ ] Hindi voice input/output 🇮🇳
- [ ] Fine-tune LLM on Ayurvedic texts
- [ ] Mobile app
- [ ] REST API with FastAPI + Docker
- [ ] Multilingual support (22 Indian languages)

## 👨‍💻 Author
Built by Aayuush with ❤️ and chai ☕

*"Ancient wisdom never gets outdated — it just needs a modern interface"*

---
⚠️ *Educational purposes only. Not a substitute for medical advice.*
```

Click `Commit new file` ✅

**4. Create `requirements.txt`** — `Add file` → `Create new file` → name `requirements.txt`:
```
gradio
transformers
torch
scikit-learn
pandas
numpy
wordcloud
matplotlib
seaborn
