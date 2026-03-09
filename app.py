import streamlit as st
import tempfile
from solution import DocFusionSolution
import joblib
from PIL import Image

st.set_page_config(page_title="Receipt Forgery Detector", layout="centered")
solution = DocFusionSolution()
@st.cache_resource
def load_model():
    return joblib.load("model/model.pkl")

model = load_model()

st.title("Receipt Forgery Detection")
uploaded_file = st.file_uploader("Upload the receipt image", type=["png","jpg","jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Receipt", width=400)

    # Save uploaded file temporarily
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    temp_file.write(uploaded_file.getbuffer())
    temp_file.close()

    image_path = temp_file.name
    try:
        vendor, date, total, numbers_of_lines, text_length, num_prices, avg_price, max_price, total_diff = solution.extract_features(image_path)

        total_val = total if total else 0

        features = [[
            total_val,
            numbers_of_lines,
            text_length,
            num_prices,
            avg_price,
            max_price,
            total_diff
        ]]

        prediction = model.predict(features)[0]

        st.write("Receipt Information")
        st.write("-------------------")
        st.write("Vendor :", vendor)
        st.write("Date   :", date)
        st.write("Total  :", total)

        st.write("------------------")
        st.write("Forgery Prediction")
        st.write("------------------")
        if prediction == 1:
            st.error("This is likely a forged receipt")
        else:
            st.success("This is likely an authentic receipt")
    except:
        st.error("Error occurred while processing the receipt.")