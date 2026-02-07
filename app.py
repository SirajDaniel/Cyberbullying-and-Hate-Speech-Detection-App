import streamlit as st
import numpy as np
import pandas as pd
import os, time, joblib, torch, re, random
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from cohere import Client, TooManyRequestsError
from lime.lime_text import LimeTextExplainer
import streamlit.components.v1 as components
from dotenv import load_dotenv
from langdetect import detect, DetectorFactory

# --- 0. LOAD ENVIRONMENT ---
load_dotenv() 

# Import custom classes (Ensure these match your local files)
from custom_classes import (
    PyTorchLSTM, SimpleTokenizer, pad_sequences_manual,
    MAX_WORDS, MAX_SEQ_LENGTH, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, OOV_TOKEN
)

# --- 1. SESSION STATE & GLOBAL INITIALIZATION ---
# Initialize variables at the top to prevent Pylance "not defined" errors
if 'view_mode' not in st.session_state:
    st.session_state.view_mode = "Dashboard"

DetectorFactory.seed = 0
captured_text = "" 
svm_score = 0.0
lstm_score = 0.0
avg_toxic = 0.0
label = "SAFE"
status_color = "#00d26a"
behavior_intent = "Waiting for analysis..."
rec_action = "N/A"
past_tweets = []
flag_count = 0

# Check if data is coming from the Chrome Extension URL parameters
q_params = st.query_params
ext_username = q_params.get("username", "ManualEntry")
ext_comment = q_params.get("comment", "") 

if ext_comment:
    captured_text = ext_comment

# --- 2. REPORTING DIALOG ---
@st.dialog("Official Platform Report")
def show_report_modal(username, comment, label, confidence):
    """Creates a professional pop-up and logs detailed report data to CSV"""
    st.warning(f"Reporting User: **{username}**")
    st.write(f"**AI Verdict:** {label} ({confidence})")
    
    reason = st.selectbox(
        "Select the violation type:",
        ["Hate Speech", "Harassment", "Threatening Violence", "Spam", "Self-Harm", "Misinformation"]
    )
    details = st.text_area("Provide additional details for the Admin:", placeholder="Explain the context of the report...")
    
    if st.button("Submit Report to Admin", type="primary"):
        new_entry = pd.DataFrame([[
            time.strftime("%Y-%m-%d %H:%M"), username, label, confidence, reason, details, comment
        ]], columns=["Date", "User", "Verdict", "Confidence", "Violation_Type", "Additional_Info", "Content"])
        
        new_entry.to_csv("moderation_history.csv", mode='a', 
                         header=not os.path.exists("moderation_history.csv"), index=False)

        with st.status("Connecting to platform API...", expanded=True) as status:
            time.sleep(1.2)
            st.write("Verifying evidence logs...")
            time.sleep(0.8)
            status.update(label="Report Sent Successfully!", state="complete", expanded=False)
        
        st.success(f"Ticket filed successfully.")
        st.toast("Report has been filed. ✅")
        time.sleep(1)
        st.rerun()

# --- 3. GLOBAL INITIALIZATION (COHERE & MODELS) ---
api_key = st.secrets.get("COHERE_API_KEY")
client = Client(api_key) if api_key else None

@st.cache_resource
def load_models():
    svm = joblib.load('svm_model.pkl')
    tfidf = joblib.load('tfidf_vectorizer.pkl')
    tokenizer = joblib.load('lstm_tokenizer.pkl')
    vocab_size = len(tokenizer.word_index) + 1
    lstm = PyTorchLSTM(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, MAX_SEQ_LENGTH)
    lstm.load_state_dict(torch.load('pytorch_lstm_model.pt', map_location='cpu'))
    lstm.eval()
    return svm, tfidf, lstm, tokenizer

svm_model, tfidf_vectorizer, lstm_model, lstm_tokenizer = load_models()

# --- 4. IMPROVISED INTELLIGENCE LOGIC ---
def generate_advanced_intelligence(svm_s, lstm_s):
    """Automated behavioral improviser based on ensemble agreement"""
    avg = (svm_s + lstm_s) / 2
    diff = abs(svm_s - lstm_s)
    
    if avg < 0.35:
        behavior = "✅ **Constructive Interaction:** The user's intent appears neutral or positive."
        action = "None (Approved)"
        color = "#00d26a"
    elif diff > 0.25 and lstm_s > svm_s:
        behavior = "🎭 **Sophisticated Harassment:** Intentional use of context to mask toxicity (sarcasm or dog-whistling)."
        action = "Shadow-Hide & Human Review"
        color = "#ffaa00"
    elif svm_s > 0.8:
        behavior = "🔥 **Explicit Aggression:** High frequency of toxic tokens and direct verbal attacks."
        action = "Immediate Removal & Account Flag"
        color = "#ff4b4b"
    else:
        behavior = "⚖️ **General Toxicity:** Standard violation involving derogatory language or insults."
        action = "Warning Issued"
        color = "#ff4b4b"
    return behavior, action, color

# --- 5. PREDICTION LOGIC & UTILITIES ---
def analyze_text_scores(text):
    clean = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    svm_s = svm_model.predict_proba(tfidf_vectorizer.transform([clean]))[0][1]
    seq = lstm_tokenizer.texts_to_sequences([clean])
    pad = pad_sequences_manual(seq, maxlen=MAX_SEQ_LENGTH)
    with torch.no_grad():
        lstm_s = lstm_model(torch.LongTensor(pad)).item()
    return svm_s, lstm_s

def scan_twitter_history(username):
    pool = [
        "I really hate how people like you are allowed here.",
        "Today is a beautiful day for a walk.",
        "You are so stupid, go away forever.",
        "Does anyone want to grab coffee?",
        "This whole group is disgusting and needs to be removed.",
        "I love the new updates to this app!",
        "Stop posting this trash, nobody cares about your opinion.",
        "The sunset was amazing this evening.",
        "Absolute garbage content, delete your account."
    ]
    selected_tweets = random.sample(pool, random.randint(3, 6))
    flagged_count = 0
    detailed_results = []
    for tweet in selected_tweets:
        s, l = analyze_text_scores(tweet)
        avg = (s + l) / 2
        is_toxic = avg > 0.5
        if is_toxic: flagged_count += 1
        detailed_results.append({"text": tweet, "score": avg, "toxic": is_toxic})
    return flagged_count, detailed_results

def render_meter_widget(value, title, color):
    # FIXED: Axis locked at 0-100%
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        title={'text': title, 'font': {'size': 20, 'color': 'white'}},
        number={'suffix': "%", 'font': {'size': 40, 'color': 'white'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': color},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2, 'bordercolor': "#444"
        }
    ))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
                      height=280, margin=dict(l=30, r=30, t=50, b=20))
    st.plotly_chart(fig, use_container_width=True)

def get_status_color(weight_pct):
    if weight_pct > 15.0: return '#ff4b4b' 
    if weight_pct > 5.0: return '#ffaa00' 
    return '#00d26a' 

def predictor_wrapper(texts):
    results = []
    for text in texts:
        s, l = analyze_text_scores(text)
        avg = (s + l) / 2
        results.append([1 - avg, avg])
    return np.array(results)

@st.cache_data(show_spinner=False, ttl=3600)
def rephrase_with_retry(text, _client):
    if not _client: return "AI unavailable."
    instruction = "Rewrite this to be polite and professional. Output ONLY the rephrased result."
    for attempt in range(5):
        try:
            response = _client.chat(model='command-r-08-2024', 
                                    message=f"{instruction}\n\nInput: '{text}'", 
                                    temperature=0.3)
            return response.text.strip().replace('"', '')
        except TooManyRequestsError:
            time.sleep((2 ** attempt) + 1)
            continue
    return "Rate limit error."

# --- 6. SIDEBAR CONTROLS ---
with st.sidebar:
    st.title("🛡️ Agent Control")
    if st.button("📊 Analysis Dashboard", use_container_width=True):
        st.session_state.view_mode = "Dashboard"
        st.rerun()
    if st.button("📜 Reporting History", use_container_width=True):
        st.session_state.view_mode = "History"
        st.rerun()
    
    st.markdown("---")
    st.subheader("Last 5 Reports")
    if os.path.exists("moderation_history.csv"):
        try:
            df_side = pd.read_csv("moderation_history.csv")
            if "Violation_Type" in df_side.columns:
                st.dataframe(df_side[["User", "Violation_Type"]].tail(5), hide_index=True)
            else:
                if st.button("Fix Log File Format"):
                    os.remove("moderation_history.csv"); st.rerun()
        except: pass
    else:
        st.caption("No reports filed yet.")

# --- 7. MAIN INTERFACE LOGIC ---
st.set_page_config(layout="wide", page_title="Advanced Hate Speech Analytics")
st.markdown("""<style>
.stApp { background-color: #0E1117; color: #FFFFFF; }
.dashboard-card { background-color: #1E2129; padding: 25px; border-radius: 12px; border: 1px solid #333; margin-bottom: 25px; }
.comparison-box { padding: 15px; border-radius: 8px; min-height: 80px; margin-bottom: 10px; border-left: 5px solid; }
.original-box { background-color: #2D1B1B; border-color: #ff4b4b; }
.rephrased-box { background-color: #1B2D22; border-color: #00d26a; }
.status-badge { padding: 8px 16px; border-radius: 20px; font-weight: bold; margin-bottom: 15px; display: inline-block; }
</style>""", unsafe_allow_html=True)

if st.session_state.view_mode == "History":
    st.title("📂 Moderation History Logs")
    if os.path.exists("moderation_history.csv"):
        try:
            full_df = pd.read_csv("moderation_history.csv")
            st.dataframe(full_df.sort_index(ascending=False), use_container_width=True, hide_index=True)
            if st.button("Clear History Logs"):
                os.remove("moderation_history.csv"); st.rerun()
        except: st.error("Log error. Reset file in sidebar.")
    else:
        st.info("No reporting history found.")

else:
    st.title("🛡️ Advanced Hate Speech Agent")
    
    if ext_comment:
        st.info(f"📬 **Agent Triggered!** Analyzing Twitter Post from: **@{ext_username}**")
        st.markdown(f'<div style="background-color: #161920; padding: 20px; border-radius: 10px; border-left: 5px solid #1DA1F2; margin-bottom: 25px;"><i>"{ext_comment}"</i></div>', unsafe_allow_html=True)

    if not ext_comment:
        user_input = st.text_area("Input Content for Detection (Manual Mode)", value=captured_text, height=100)
        analyze_trigger = st.button("Run Full Analysis", type="primary")
    else:
        user_input = ext_comment
        analyze_trigger = True 

    if analyze_trigger:
        if not user_input.strip():
            st.error("🛑 **Input Required: No Text Detected**")
            st.stop()

        try:
            detected_lang = detect(user_input)
            if detected_lang != 'en':
                st.warning("⚠️ **Unsupported Language Detected**")
                st.stop()
        except:
            st.error("Could not determine language.")
            st.stop()

        # EXECUTE ANALYSIS
        svm_score, lstm_score = analyze_text_scores(user_input)
        avg_toxic = (svm_score + lstm_score) / 2
        
        behavior_intent, rec_action, status_color = generate_advanced_intelligence(svm_score, lstm_score)
        label = "TOXIC" if avg_toxic > 0.7 else ("SUSPICIOUS" if avg_toxic > 0.3 else "SAFE")

       # --- SECTION 1: ACCOUNT RISK PROFILING ---
        # ONLY run and show this section if the comment came from the Twitter extension
        if ext_comment:
            st.markdown('<div class="dashboard-card" style="border-top: 4px solid #1DA1F2;">', unsafe_allow_html=True)
            st.subheader(f"👤 Account Integrity Profile: @{ext_username}")
            flag_count, past_tweets = scan_twitter_history(ext_username)
            
            m1, m2, m3 = st.columns(3)
            with m1: st.metric("Other Posts Scanned", len(past_tweets))
            with m2: st.metric("Timeline Violations", flag_count)
            with m3: st.metric("Risk Level", "DANGER" if flag_count > 2 else "STABLE")

            # Displaying historical comments directly inside the integrity profile
            if flag_count > 0:
                st.warning(f"This user has made {flag_count} other toxic comments in recent Twitter history.")
                for pt in past_tweets:
                    if pt['toxic']:
                        st.error(f"🚩 **Flagged Content:** `{pt['text']}`")
            else:
                st.success("No toxic comments identified in simulated user history.")
            st.markdown('</div>', unsafe_allow_html=True)

        # --- SECTION: AUTOMATED BEHAVIORAL INTELLIGENCE ---
        # Only show this for extension users as well
            if ext_comment:
    
            st.markdown("### 🧠 Automated Behavioral Intelligence")
            col_intel1, col_intel2 = st.columns([2, 1])  # Variables created here
            with col_intel1:
                st.markdown(f"""
                    <div style="background-color: #1E2129; padding: 20px; border-radius: 10px; border-left: 5px solid {status_color};">
                        <h4 style="margin-top:0; color:{status_color};">Behavioral Intent</h4>
                        <p style="font-size: 1.1rem;">{behavior_intent}</p>
                        <hr style="border-color: #333;">
                        <p style="font-size: 0.9rem; color: #aaa;">
                            <b>System Logic:</b> The ensemble agreement of {avg_toxic*100:.1f}% triggered this 
                            behavioral profile.
                        </p>
                    </div>
                """, unsafe_allow_html=True)
            with col_intel2:  # Variable used here - now safe because it's inside the IF
                st.markdown(f"""
                    <div style="background-color: #161920; padding: 20px; border-radius: 10px; border: 1px solid #333; text-align: center;">
                        <h4 style="margin-top:0;">Recommended Action</h4>
                        <div style="font-size: 1.2rem; font-weight: bold; color: {status_color}; margin-bottom: 10px;">
                            {rec_action}
                        </div>
                        <small>Based on Platform Safety Protocol v2.6</small>
                    </div>
                """, unsafe_allow_html=True)
        with col_intel2:
            st.markdown(f"""
                <div style="background-color: #161920; padding: 20px; border-radius: 10px; border: 1px solid #333; text-align: center;">
                    <h4 style="margin-top:0;">Recommended Action</h4>
                    <div style="font-size: 1.2rem; font-weight: bold; color: {status_color}; margin-bottom: 10px;">
                        {rec_action}
                    </div>
                    <small>Based on Platform Safety Protocol v2.6</small>
                </div>
            """, unsafe_allow_html=True)

        # --- SECTION 2: STATUS & GAUGES ---
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f'<div class="status-badge" style="background-color: {status_color};">STATUS: {label}</div>', unsafe_allow_html=True)
        
        col_gauge1, col_gauge2 = st.columns(2)
        with col_gauge1:
            render_meter_widget(avg_toxic, "Aggression Score", "#ff4b4b")
        with col_gauge2:
            render_meter_widget(1 - avg_toxic, "Integrity Score", "#00d26a")

        # --- SECTION 3: WORD-LEVEL IMPACT (LIME) ---
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.subheader("💡 Live Word-Level Impact & Logic Explanation")
        explainer = LimeTextExplainer(class_names=['Safe', 'Toxic'])
        exp = explainer.explain_instance(user_input, predictor_wrapper, num_features=10)
        exp_list = exp.as_list()
        total_abs_weight = sum(abs(x[1]) for x in exp_list)
        
        c1, c2 = st.columns([2, 1])
        with c1:
            fig, ax = plt.subplots(figsize=(8, 4), facecolor='#1E2129')
            # FIXED: Lock X-axis to 0-100%
            weights = [abs(x[1])/total_abs_weight * 100 for x in exp_list][::-1]
            words = [x[0] for x in exp_list][::-1]
            ax.barh(words, weights, color=[get_status_color(w) for w in weights])
            ax.set_xlim(0, 100)
            ax.set_facecolor('#1E2129'); ax.tick_params(colors='white'); st.pyplot(fig)
        with c2:
            st.markdown(f'<div style="border: 1px solid {status_color}; padding:15px; border-radius:10px; background:#161920;">', unsafe_allow_html=True)
            st.markdown(f'<h4 style="color:{status_color}; margin-top:0;">Technical Analysis</h4>', unsafe_allow_html=True)
            for word, weight in exp_list:
                w_pct = (abs(weight)/total_abs_weight) * 100
                st.markdown(f'<div style="display:flex; justify-content:space-between; font-family:monospace; color:{get_status_color(w_pct)};"><span>{word}</span><span>{w_pct:.1f}%</span></div>', unsafe_allow_html=True)
            st.markdown('<hr style="border-color:#333">', unsafe_allow_html=True)
            st.write('Ensemble analysis of context and word frequency confirmed this result.')
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # --- SECTION 4: ENSEMBLE BREAKDOWN & INTENT ANALYSIS ---
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        d_col1, d_col2 = st.columns([1.5, 1])
        
        with d_col1:
            st.subheader("🔬 Ensemble Logic Agreement (Tug-of-War)")
            fig_break, ax_break = plt.subplots(figsize=(10, 5), facecolor='#1E2129')
            # FIXED: Lock Y-axis scale at 0-100%
            ax_break.bar(['LSTM (Contextual)', 'SVM (Lexical)'], [lstm_score*100, svm_score*100], color=['#75bbfd', '#0343df'])
            ax_break.set_ylim(0, 100)
            ax_break.set_facecolor('#1E2129'); ax_break.tick_params(colors='white'); ax_break.set_ylabel('Confidence (%)', color='white')
            st.pyplot(fig_break)
            
        with d_col2:
            st.subheader("🧠 Reasoning & Intent Analysis")
            # Logic-based reasoning generation replaces severity benchmarking
            if abs(lstm_score - svm_score) < 0.15:
                intent_type = "Direct Violation"
                reasoning = (
                    "Both detection engines are in high agreement. The content contains "
                    "explicitly toxic keywords and a sentence structure designed to attack."
                )
            elif lstm_score > svm_score:
                intent_type = "Coaxing/Sarcastic Harassment"
                reasoning = (
                    "The Neural LSTM model detected harmful intent that the word-filter (SVM) missed. "
                    "This suggests the user is using context, tone, or sarcasm to mask their aggression."
                )
            else:
                intent_type = "Keyword-Triggered Flag"
                reasoning = (
                    "The toxicity is primarily driven by high-risk keywords. While the contextual "
                    "threat is lower, the presence of these specific terms warrants a restriction."
                )

            st.markdown(f"**Primary Intent Profile:** `{intent_type}`")
            st.info(reasoning)
            
            # Moderator Tip removed

        st.markdown('</div>', unsafe_allow_html=True)

        # --- SECTION 5: REPHRASING ---
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.subheader("✨ Text Optimization: Before vs. After")
        refined = rephrase_with_retry(user_input, client)
        st.markdown(f'<div class="comparison-box original-box"><b>BEFORE:</b> {user_input}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="comparison-box rephrased-box"><b>AFTER:</b> {refined}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # --- SECTION 6: PLATFORM INTEGRITY ACTION ---
        # Hide the reporting button for manual entry
        if ext_comment:
           st.markdown('<div class="dashboard-card" style="border: 1px solid #ff4b4b;">', unsafe_allow_html=True)
           st.subheader("🚨 Platform Integrity Action")
        if st.button("🚩 Formal Report User to Admin", type="secondary"):
           show_report_modal(ext_username, user_input, label, f"{avg_toxic*100:.1f}%")

           st.markdown('</div>', unsafe_allow_html=True)

