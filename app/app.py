import gradio as gr
import pickle
import os
from transformers import pipeline

# ══════════════════════════════════════
# LOAD MODELS
# ══════════════════════════════════════
with open('../models/symptom_classifier.pkl', 'rb') as f:
    model = pickle.load(f)
with open('../models/symptom_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
with open('../models/disease_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

print("Loading summarizer...")
from transformers import BartForConditionalGeneration, BartTokenizer

tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
print("✅ Summarizer loaded!")

# ══════════════════════════════════════
# KNOWLEDGE BASE
# ══════════════════════════════════════
HERBS = {
    "Vata": [
        {"name": "Ashwagandha (अश्वगंधा)", "benefit": "Reduces anxiety, boosts energy, strengthens nervous system", "dose": "1 tsp in warm milk at night"},
        {"name": "Haritaki (हरीतकी)", "benefit": "Improves digestion, relieves constipation, rejuvenates tissues", "dose": "1 tsp with warm water before bed"},
        {"name": "Shatavari (शतावरी)", "benefit": "Calms nervous system, boosts immunity, nourishes tissues", "dose": "1 tsp in warm milk twice daily"},
        {"name": "Bala (बला)", "benefit": "Strengthens muscles, relieves joint pain, builds stamina", "dose": "1 tsp powder with warm water"},
    ],
    "Pitta": [
        {"name": "Brahmi (ब्राह्मी)", "benefit": "Cools mind, improves memory, reduces inflammation", "dose": "1 tsp with coconut oil or ghee"},
        {"name": "Amalaki (आमलकी)", "benefit": "Rich in Vitamin C, cools body heat, improves liver", "dose": "2 fresh fruits or 1 tsp powder daily"},
        {"name": "Neem (नीम)", "benefit": "Purifies blood, reduces inflammation, antimicrobial", "dose": "Neem tea or 2 capsules daily"},
        {"name": "Shatavari (शतावरी)", "benefit": "Cools body, balances hormones, reduces acidity", "dose": "1 tsp in cool milk daily"},
    ],
    "Kapha": [
        {"name": "Trikatu (त्रिकटु)", "benefit": "Boosts metabolism, clears mucus, stimulates digestion", "dose": "1/4 tsp with honey before meals"},
        {"name": "Guggul (गुग्गुल)", "benefit": "Reduces cholesterol, aids weight loss, anti-inflammatory", "dose": "250mg twice daily with warm water"},
        {"name": "Tulsi (तुलसी)", "benefit": "Clears respiratory system, boosts immunity, reduces stress", "dose": "Tulsi tea 2-3 times daily"},
        {"name": "Ginger (अदरक)", "benefit": "Improves digestion, clears congestion, anti-nausea", "dose": "Fresh ginger tea 2x daily"},
    ]
}

UNIVERSAL_HERBS = [
    {"name": "Triphala (त्रिफला)", "benefit": "Cleanses gut, improves digestion, rejuvenates all tissues", "dose": "1 tsp in warm water before bed"},
    {"name": "Giloy (गिलोय)", "benefit": "Boosts immunity, reduces fever, powerful adaptogen", "dose": "1 tsp powder or 2 capsules daily"},
    {"name": "Turmeric (हल्दी)", "benefit": "Anti-inflammatory, antioxidant, wound healing", "dose": "Golden milk: 1 tsp in warm milk daily"},
    {"name": "Amla (आंवला)", "benefit": "Vitamin C powerhouse, anti-aging, hair and skin health", "dose": "1-2 fresh amla or juice daily"},
]

FOODS = {
    "Vata 💨": {
        "eat": "✅ Warm cooked foods, ghee, sesame oil, nuts, sweet fruits, warm milk, rice, wheat, root vegetables, soups",
        "avoid": "❌ Cold/raw foods, carbonated drinks, dry snacks, beans, caffeine, frozen food",
        "routine": "🕐 Eat at fixed times daily, always eat warm food, never skip meals, eat slowly"
    },
    "Pitta 🔥": {
        "eat": "✅ Cooling foods, coconut water, cucumber, mint, sweet fruits, leafy greens, milk, ghee, rice, oats",
        "avoid": "❌ Spicy food, alcohol, fermented foods, red meat, coffee, vinegar, citrus in excess",
        "routine": "🕐 Largest meal at noon, avoid eating when angry, stay hydrated with cool water"
    },
    "Kapha 🌊": {
        "eat": "✅ Light dry foods, honey, legumes, vegetables, spices, bitter greens, barley, millet, apples",
        "avoid": "❌ Heavy oily foods, dairy excess, sweets, cold drinks, red meat, fried food, bananas",
        "routine": "🕐 Eat light meals, skip breakfast if not hungry, exercise before eating, no late night eating"
    }
}

YOGA = {
    "Vata": {
        "poses": ["Child's Pose (Balasana)", "Cat-Cow Stretch", "Mountain Pose (Tadasana)", "Corpse Pose (Savasana)", "Seated Forward Bend"],
        "pranayama": "Nadi Shodhana (Alternate Nostril Breathing) — 10 mins daily",
        "tips": "Practice slowly and gently. Focus on grounding. Avoid fast-paced yoga."
    },
    "Pitta": {
        "poses": ["Moon Salutation", "Fish Pose (Matsyasana)", "Boat Pose (Navasana)", "Bridge Pose", "Cooling Forward Bends"],
        "pranayama": "Sheetali Pranayama (Cooling Breath) — 5-10 mins daily",
        "tips": "Practice in cool environment. Avoid hot yoga. Focus on surrender and releasing control."
    },
    "Kapha": {
        "poses": ["Sun Salutation (fast)", "Warrior Poses", "Camel Pose (Ustrasana)", "Bow Pose (Dhanurasana)", "Headstand"],
        "pranayama": "Kapalabhati (Skull Shining Breath) — 5 mins daily",
        "tips": "Practice vigorously and energetically. Morning practice is best. Challenge yourself."
    }
}

CHATBOT_KNOWLEDGE = """
You are VaidyaAI, an expert Ayurvedic health assistant. You have deep knowledge of:
- Ayurvedic medicine, Doshas (Vata, Pitta, Kapha), herbs, and treatments
- Modern medical symptoms and diseases
- Yoga, pranayama, and wellness practices
- Indian traditional remedies and their scientific backing

Rules:
1. Always recommend consulting a real doctor for serious conditions
2. Give practical, actionable Ayurvedic advice
3. Explain Sanskrit terms in simple English
4. Be warm, caring and supportive like a traditional Indian Vaidya
5. Keep responses concise but complete
6. Always mention both Ayurvedic AND modern perspective
"""

# ══════════════════════════════════════
# FUNCTIONS
# ══════════════════════════════════════

def analyze_symptoms(symptoms_text):
    if not symptoms_text.strip():
        return "Please enter your symptoms"
    vec = vectorizer.transform([symptoms_text])
    probs = model.predict_proba(vec)[0]
    top5_idx = probs.argsort()[-5:][::-1]
    result = "🏥 AI Disease Prediction\n"
    result += "─" * 40 + "\n\n"
    for i, idx in enumerate(top5_idx):
        disease = le.inverse_transform([idx])[0]
        confidence = probs[idx] * 100
        bar = "█" * int(confidence/5) + "░" * (20-int(confidence/5))
        emoji = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"][i]
        result += f"{emoji} {disease}\n"
        result += f"   {bar} {confidence:.1f}%\n\n"
    result += "─" * 40 + "\n"
    result += "⚠️ Educational only. Always consult a doctor."
    return result

def profile_dosha(body_frame, skin_type, appetite, sleep, energy, stress, digestion):
    if not all([body_frame, skin_type, appetite, sleep, energy, stress, digestion]):
        return "Please answer all questions", ""
    
    scores = {"Vata": 0, "Pitta": 0, "Kapha": 0}
    mapping = {
        "body_frame":  {"thin": "Vata", "medium": "Pitta", "heavy": "Kapha"},
        "skin_type":   {"dry": "Vata", "oily": "Pitta", "smooth": "Kapha"},
        "appetite":    {"irregular": "Vata", "strong": "Pitta", "steady": "Kapha"},
        "sleep":       {"light": "Vata", "moderate": "Pitta", "deep": "Kapha"},
        "energy":      {"variable": "Vata", "intense": "Pitta", "stable": "Kapha"},
        "stress":      {"anxious": "Vata", "irritable": "Pitta", "calm": "Kapha"},
        "digestion":   {"irregular": "Vata", "fast": "Pitta", "slow": "Kapha"},
    }
    answers = {"body_frame": body_frame, "skin_type": skin_type,
               "appetite": appetite, "sleep": sleep, "energy": energy,
               "stress": stress, "digestion": digestion}
    
    for key, val in answers.items():
        if val in mapping[key]:
            scores[mapping[key][val]] += 1
    
    dosha = max(scores, key=scores.get)
    total = sum(scores.values())
    
    descriptions = {
        "Vata": "💨 VATA DOMINANT\n\nYou are creative, quick-thinking, and full of ideas. Natural artist and visionary.\n\nStrengths: Creative, adaptable, energetic\nChallenges: Anxiety, irregular routine, dry skin\nBest season: Summer (avoid cold/dry weather)",
        "Pitta": "🔥 PITTA DOMINANT\n\nYou are sharp, ambitious, and a natural leader. Driven and focused.\n\nStrengths: Intelligence, leadership, passion\nChallenges: Anger, inflammation, perfectionism\nBest season: Winter (avoid heat)",
        "Kapha": "🌊 KAPHA DOMINANT\n\nYou are calm, loving, and incredibly strong. Natural caregiver and anchor.\n\nStrengths: Strength, loyalty, patience\nChallenges: Weight gain, lethargy, attachment\nBest season: Spring (avoid cold/damp)"
    }
    
    result = descriptions[dosha] + "\n\n"
    result += "📊 Your Dosha Breakdown:\n"
    for d, s in scores.items():
        pct = s/total*100
        bar = "█" * int(pct/10) + "░" * (10-int(pct/10))
        emoji = "💨" if d=="Vata" else "🔥" if d=="Pitta" else "🌊"
        result += f"{emoji} {d}: {bar} {pct:.0f}%\n"
    
    herb_text = f"🌿 HERBS FOR {dosha.upper()}\n"
    herb_text += "─" * 40 + "\n\n"
    for herb in HERBS[dosha]:
        herb_text += f"🌱 {herb['name']}\n"
        herb_text += f"   ✦ {herb['benefit']}\n"
        herb_text += f"   💊 {herb['dose']}\n\n"
    herb_text += "🌍 UNIVERSAL HERBS\n"
    herb_text += "─" * 40 + "\n"
    for herb in UNIVERSAL_HERBS:
        herb_text += f"🌱 {herb['name']}\n"
        herb_text += f"   ✦ {herb['benefit']}\n"
        herb_text += f"   💊 {herb['dose']}\n\n"
    herb_text += "⚠️ Consult a qualified Vaidya before use."
    
    return result, herb_text

def summarize_note(text):
    if len(text.strip()) < 50:
        return "Please enter a longer clinical note (minimum 50 words)"
    try:
        inputs = tokenizer(text[:1024], return_tensors="pt", 
                          max_length=1024, truncation=True)
        summary_ids = bart_model.generate(
            inputs["input_ids"], 
            max_length=150, 
            min_length=40,
            length_penalty=2.0,
            num_beams=4
        )
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        orig = len(text.split())
        summ = len(summary.split())
        return f"📋 PLAIN ENGLISH SUMMARY\n{'─'*40}\n\n{summary}\n\n{'─'*40}\n📊 {orig} words → {summ} words ({(1-summ/orig):.0%} shorter)"
    except Exception as e:
        return f"Error: {str(e)}"

def food_guide(dosha):
    if not dosha:
        return "Please select your Dosha"
    info = FOODS[dosha]
    result = f"🥗 AYURVEDIC FOOD GUIDE — {dosha}\n"
    result += "─" * 40 + "\n\n"
    result += f"{info['eat']}\n\n"
    result += f"{info['avoid']}\n\n"
    result += f"{info['routine']}\n\n"
    result += "─" * 40 + "\n"
    result += "💡 Tip: Eat your largest meal at noon when digestion is strongest."
    return result

def yoga_guide(dosha):
    if not dosha:
        return "Please select your Dosha"
    dosha_key = dosha.split()[0]
    info = YOGA[dosha_key]
    result = f"🧘 YOGA GUIDE — {dosha}\n"
    result += "─" * 40 + "\n\n"
    result += "🏃 RECOMMENDED POSES:\n"
    for pose in info['poses']:
        result += f"  • {pose}\n"
    result += f"\n🌬️ PRANAYAMA:\n  {info['pranayama']}\n"
    result += f"\n💡 TIPS:\n  {info['tips']}"
    return result

def chat_with_vaidya(message, history):
    if not message.strip():
        return ""
    
    from transformers import pipeline as hf_pipeline
    
    # Build conversation context
    context = CHATBOT_KNOWLEDGE + "\n\nConversation:\n"
    for human, assistant in history[-3:]:
        context += f"User: {human}\nVaidyaAI: {assistant}\n"
    context += f"User: {message}\nVaidyaAI:"
    
    try:
        generator = hf_pipeline(
            "text-generation",
            model="facebook/opt-125m",
            device=-1,
            max_new_tokens=200
        )
        response = generator(context, do_sample=True, temperature=0.7)[0]['generated_text']
        # Extract only the new response
        response = response.split("VaidyaAI:")[-1].strip()
        if not response:
            response = get_rule_based_response(message)
    except:
        response = get_rule_based_response(message)
    
    return response

def get_rule_based_response(message):
    message_lower = message.lower()
    if any(w in message_lower for w in ['vata', 'pitta', 'kapha', 'dosha']):
        return "🌿 Doshas are the three fundamental energies in Ayurveda. Vata (air/space) governs movement, Pitta (fire/water) governs transformation, and Kapha (earth/water) governs structure. Use the Dosha Profiler tab to discover yours!"
    elif any(w in message_lower for w in ['ashwagandha', 'herb', 'herbs']):
        return "🌱 Ayurvedic herbs are powerful natural medicines. Ashwagandha reduces stress, Brahmi improves memory, Triphala cleanses digestion, and Giloy boosts immunity. Check the Herb Recommender for personalized suggestions!"
    elif any(w in message_lower for w in ['fever', 'cold', 'cough', 'headache']):
        return "🏥 For fever: Giloy + Tulsi tea works wonderfully. For cold/cough: Trikatu with honey. For headache: Brahmi oil massage. Always consult a doctor for persistent symptoms!"
    elif any(w in message_lower for w in ['diet', 'food', 'eat']):
        return "🥗 Ayurvedic diet is personalized to your Dosha. Check the Food Guide tab for your specific recommendations. Generally: eat warm cooked food, avoid processed food, and eat mindfully!"
    elif any(w in message_lower for w in ['yoga', 'exercise', 'meditation']):
        return "🧘 Yoga and pranayama are core to Ayurvedic wellness. Check the Yoga Guide tab for Dosha-specific practices. Daily practice of even 20 minutes transforms health!"
    elif any(w in message_lower for w in ['stress', 'anxiety', 'sleep']):
        return "😌 Ayurveda says stress is primarily a Vata imbalance. Try: Ashwagandha before bed, Abhyanga (oil massage), Nadi Shodhana breathing, and a consistent sleep schedule. Your mind and body are connected!"
    else:
        return "🌿 Namaste! I'm VaidyaAI, your Ayurvedic health companion. I can help you understand your Dosha, recommend herbs, suggest yoga practices, and answer questions about Ayurvedic wellness. What would you like to know?"

# ══════════════════════════════════════
# BUILD UI
# ══════════════════════════════════════
css = """
.gradio-container {
    font-family: 'Arial', sans-serif;
    max-width: 1200px !important;
}
.tab-nav button {
    font-size: 15px !important;
    font-weight: 600 !important;
}
footer { display: none !important; }
"""

with gr.Blocks(css=css, title="🌿 VaidyaAI") as demo:
    
    gr.Markdown("""
    # 🌿 VaidyaAI — वैद्य AI
    ### *Ancient Ayurvedic Wisdom × Modern Artificial Intelligence*
    > Bridging 5,000 years of Indian medical knowledge with cutting-edge AI
    ---
    """)
    
    with gr.Tab("🏥 Symptom Analyzer"):
        gr.Markdown("### Describe symptoms in plain English — AI predicts top 5 conditions")
        with gr.Row():
            with gr.Column(scale=1):
                s_input = gr.Textbox(
                    label="Your Symptoms",
                    placeholder="e.g. fever headache body ache fatigue nausea since 3 days",
                    lines=4
                )
                s_btn = gr.Button("🔍 Analyze Symptoms", variant="primary", size="lg")
            with gr.Column(scale=1):
                s_output = gr.Textbox(label="AI Prediction", lines=12)
        s_btn.click(fn=analyze_symptoms, inputs=s_input, outputs=s_output)

    with gr.Tab("🧘 Dosha Profiler"):
        gr.Markdown("### Discover Your Ayurvedic Body Constitution (Prakriti)")
        with gr.Row():
            bf = gr.Dropdown(["thin", "medium", "heavy"], label="🏃 Body Frame")
            sk = gr.Dropdown(["dry", "oily", "smooth"], label="✋ Skin Type")
            ap = gr.Dropdown(["irregular", "strong", "steady"], label="🍽️ Appetite")
        with gr.Row():
            sl = gr.Dropdown(["light", "moderate", "deep"], label="😴 Sleep Quality")
            en = gr.Dropdown(["variable", "intense", "stable"], label="⚡ Energy Level")
            st = gr.Dropdown(["anxious", "irritable", "calm"], label="🧠 Stress Response")
        dg = gr.Dropdown(["irregular", "fast", "slow"], label="🔄 Digestion Speed")
        d_btn = gr.Button("🌿 Discover My Dosha", variant="primary", size="lg")
        with gr.Row():
            d_result = gr.Textbox(label="Your Dosha Profile", lines=14)
            h_result = gr.Textbox(label="🌱 Herb Recommendations", lines=14)
        d_btn.click(fn=profile_dosha,
                    inputs=[bf, sk, ap, sl, en, st, dg],
                    outputs=[d_result, h_result])

    with gr.Tab("📋 Note Summarizer"):
        gr.Markdown("### Paste any clinical note — get instant plain English summary")
        with gr.Row():
            with gr.Column(scale=1):
                n_input = gr.Textbox(
                    label="Clinical Note",
                    placeholder="Paste doctor's notes, medical reports, discharge summaries here...",
                    lines=10
                )
                n_btn = gr.Button("✨ Summarize in Plain English", variant="primary", size="lg")
            with gr.Column(scale=1):
                n_output = gr.Textbox(label="Plain English Summary", lines=10)
        n_btn.click(fn=summarize_note, inputs=n_input, outputs=n_output)

    with gr.Tab("🥗 Food Guide"):
        gr.Markdown("### Personalized Ayurvedic diet based on your Dosha")
        with gr.Row():
            with gr.Column(scale=1):
                f_input = gr.Dropdown(
                    ["Vata 💨", "Pitta 🔥", "Kapha 🌊"],
                    label="Select Your Dosha"
                )
                f_btn = gr.Button("🥗 Get My Food Guide", variant="primary", size="lg")
            with gr.Column(scale=1):
                f_output = gr.Textbox(label="Your Personalized Food Guide", lines=10)
        f_btn.click(fn=food_guide, inputs=f_input, outputs=f_output)

    with gr.Tab("🧘 Yoga Guide"):
        gr.Markdown("### Dosha-specific yoga and pranayama recommendations")
        with gr.Row():
            with gr.Column(scale=1):
                y_input = gr.Dropdown(
                    ["Vata 💨", "Pitta 🔥", "Kapha 🌊"],
                    label="Select Your Dosha"
                )
                y_btn = gr.Button("🧘 Get My Yoga Guide", variant="primary", size="lg")
            with gr.Column(scale=1):
                y_output = gr.Textbox(label="Your Yoga & Pranayama Guide", lines=10)
        y_btn.click(fn=yoga_guide, inputs=y_input, outputs=y_output)

    with gr.Tab("🤖 Chat with Vaidya"):
        gr.Markdown("### Ask anything about Ayurveda, herbs, symptoms, wellness")
        chatbot = gr.ChatInterface(
            fn=chat_with_vaidya,
            examples=[
                "What is my Dosha and how do I balance it?",
                "Which herbs are good for stress and anxiety?",
                "I have acidity and inflammation, what should I do?",
                "What is Ashwagandha and how should I take it?",
                "How can Ayurveda help with my sleep problems?",
            ],
            title=""
        )

    gr.Markdown("""
    ---
    🌿 **VaidyaAI** | Built with Hugging Face Transformers, Scikit-learn & Gradio
    ⚠️ *For educational purposes only. Not a substitute for professional medical advice.*
    """)

demo.launch()