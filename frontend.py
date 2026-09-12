import streamlit as st
import requests
import base64
import pandas as pd

# Configure the UI layout
st.set_page_config(page_title="Altur Voice Security", layout="centered")
st.title("🛡️ Altur Deepfake Detection Dashboard")
st.write("Upload a live stereo intercept to verify human biological acoustic markers.")

uploaded_file = st.file_uploader("Upload Stereo Audio (WAV)", type=["wav"])

if uploaded_file is not None:
    # Playable audio widget in the UI
    st.audio(uploaded_file, format="audio/wav")

    if st.button("Run Real-Time Analysis"):
        with st.spinner("Extracting 40-feature array and checking phase alignment..."):
            # Encode and send the payload to your FastAPI server
            encoded_audio = base64.b64encode(uploaded_file.read()).decode("utf-8")
            response = requests.post(
                "http://127.0.0.1:8000/detect", 
                json={"audio_base64": encoded_audio}
            )
            
            if response.status_code == 200:
                data = response.json()
                confidence = data["confidence"]
                
                # Dynamic UI State based on your API's recommended_action
                if data["recommended_action"] == "trigger_whatsapp_2fa":
                    st.error(f"🚨 SYNTHETIC VOICE DETECTED (Confidence: {confidence:.4f})")
                    st.warning("WhatsApp 2FA verification payload dispatched to user device.")
                else:
                    st.success(f"✅ Authentic Human Voice (Confidence: {confidence:.4f})")
                
                # Render the 40-Feature Breakdown as a chart
                st.subheader("Decision Matrix Breakdown")
                breakdown = data["breakdown"]
                
                # Clean the percentage strings for graphing
                chart_data = pd.DataFrame({
                    "Weight (%)": [float(v.strip('%')) for v in breakdown.values()]
                }, index=[k.replace("_", " ").title() for k in breakdown.keys()])
                
                st.bar_chart(chart_data)
            else:
                st.error(f"API Error: {response.status_code}")