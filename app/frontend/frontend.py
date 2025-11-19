import streamlit as st
import requests
from PIL import Image
from dotenv import load_dotenv
import os

load_dotenv()

st.set_page_config(page_title="Saudi Date Classifier", layout="centered")

st.title("🌴🇸🇦 Saudi Date Classifier")
st.subheader("صنّف تمرك بـذكاء اصطناعي سعودي!")

# Information section (UI text left as-is)
st.markdown("""
> 📸 **ارفع صورة للتمرة اللي عندك**  
> 🧠 والنموذج بيعرف نوعها تلقائيًا!

---

🌟 **الأنواع المدعومة:**
- 🟤 **Sokari** - سكري  
- 🟠 **Sagai** - صقعي  
- ⚫ **Ajwa** - عجوة  
- 🟤 **Medjool** - مجدول

🛑 **لأفضل دقة ممكنة:**
- 📷 ارفع صورة فيها **تمرة وحدة فقط**
- ☁️ خل الخلفية نظيفة
- 🚫 تجنب خلط التمر مع فنجان قهوة أو أشياء ثانية

---
Done By: Abdulrahman Almejna
Linkedin: https://www.linkedin.com/in/abdulrahman-almejna-1786b21b4/
""", unsafe_allow_html=True)

API_INTERNAL = os.getenv("API_INTERNAL")
API_EXTERNAL = os.getenv("API_EXTERNAL")

# Session state for prediction and description
if "prediction_result" not in st.session_state:
    st.session_state["prediction_result"] = None

if "description_data" not in st.session_state:
    st.session_state["description_data"] = None

# File upload
uploaded_file = st.file_uploader("📤 ارفع صورة التمرة", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="📷 الصورة اللي رفعتها")

    if st.button("🔎 صنّف التمرة"):
        try:
            files = {"file": uploaded_file.getvalue()}
            response = requests.post(f"{API_INTERNAL}/predict", files=files)

            if response.status_code == 200:
                result = response.json()
                st.session_state["prediction_result"] = result
                st.session_state["description_data"] = None  # reset description on new prediction
            else:
                st.error("❌ Error during prediction")
        except Exception as e:
            st.error(f"⚠️ Unexpected error: {e}")

# After prediction, always show latest result from session_state
result = st.session_state.get("prediction_result")

if result:
    # English-to-Arabic label translation
    translations = {
        "Sokari": "سكري",
        "Sugaey": "صقعي",
        "Ajwa": "عجوة",
        "Medjool": "مجدول"
    }

    predicted_en = result["class"]
    predicted_ar = translations.get(predicted_en, "غير معروف")

    # Show predicted class
    st.markdown(f"""
        <div style='font-size:30px; font-weight:bold; color:#008000;'>
            🧠 النوع المتوقع: {predicted_ar}
        </div>
    """, unsafe_allow_html=True)

    # Display processed YOLO image
    full_image_url = f"{API_EXTERNAL}{result['image_url']}"
    st.image(full_image_url, caption="🔍 الصورة مع التنبؤ")

    # Only allow description if class is known
    if predicted_en != "Unknown":
        if st.button("🎧 اسمع وصف التمرة"):
            try:
                with st.spinner("🧠 جالس أوصف التمرة لك..."):
                    desc_response = requests.post(
                        f"{API_INTERNAL}/describe",
                        params={"date_type": predicted_en}
                    )

                if desc_response.status_code == 200:
                    st.session_state["description_data"] = desc_response.json()
                else:
                    st.warning("ما قدرت أطلع وصف صوتي للتمرة حالياً.")
            except Exception as e:
                st.error(f"⚠️ Unexpected error: {e}")

# Show description + audio if available
desc_data = st.session_state.get("description_data")
if desc_data:
    description = desc_data.get("description")
    if description:
        st.markdown("---")
        st.markdown("### 📝 وصف التمرة باللهجة السعودية")
        st.write(description)

    audio_url = desc_data.get("audio_url")
    if audio_url:
        full_audio_url = f"{API_EXTERNAL}{audio_url}"
        st.markdown("### 🔊 اسمع الوصف")
        st.audio(full_audio_url, format="audio/mp3")