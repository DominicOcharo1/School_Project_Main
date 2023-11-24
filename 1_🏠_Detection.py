# Python In-built packages
from pathlib import Path
import PIL
import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.image as mpimg

# Local Modules
import settings
import helper

# Setting page layout
st.set_page_config(
    page_title="Plant Pests and Disease Detection",
    page_icon="☢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
with st.container():
    st.markdown(
        "<h1 style='text-align: center; color: blue; background-color: lightblue; padding: 20px;'>DL POWERED PESTS AND DISEASE DETECTION IN MAIZE</h1>",
        unsafe_allow_html=True)

    st.markdown("<h2 style='text-align: center; color: black;'>This detection model detects the following types of diseases in maize:</h2>",
                unsafe_allow_html=True)

    st.markdown("<h3 style='text-align: center; color: black;'>Leaf Blight, Leaf Spot, Fall Armyworm, and Grasshopper</h3>",
                unsafe_allow_html=True)

    st.markdown("<h4 style='text-align: center; color: gray;'>By Dominic Ocharo and Shadrack Onjiri</h4>", unsafe_allow_html=True)

# Sidebar
# Model Options
st.sidebar.header("ML Model Config")
model_task = st.sidebar.radio(
    "Select Task", ['Detection', 'Classification'])

confidence = float(st.sidebar.slider(
    "Select Model Confidence", 20, 10, 1)) / 20

# Detection and Classification Code
with st.container():
    st.write("---")

    if model_task == 'Detection':
        st.header("Image/Video Config")
        source_radio = st.radio(
            "Select Source", settings.SOURCES_LIST)

        source_img = None

        if source_radio == settings.IMAGE:
            st.write("---")
            st.markdown('Updload a photo of an affected maize leaf, then click the Detect Objects button and check the result.')

            source_img = st.file_uploader(
                "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

            col1, col2 = st.columns(2)

            with col1:
                try:
                    if source_img:
                        uploaded_image = PIL.Image.open(source_img)
                        st.image(source_img, caption="Uploaded Image",
                                 use_column_width=True)
                except Exception as ex:
                    st.error("Error occurred while opening the image.")
                    st.error(ex)

            with col2:
                if st.button('Detect Objects'):
                    if model_task == 'Detection':
                        model_path = Path(settings.DETECTION_MODEL)
                        model = helper.load_model(model_path)
                        res = model.predict(uploaded_image, conf=confidence)
                        boxes = res[0].boxes
                        res_plotted = res[0].plot()[:, :, ::-1]
                        st.image(res_plotted, caption="Detected Image",
                                 use_column_width=True)

                        try:
                            with st.expander("Detection Results"):
                                for box in boxes:
                                    st.write(box.data)
                        except Exception as ex:
                            st.write("No image is uploaded yet!")

        # Add other source types (video, webcam, youtube) and their corresponding functions here

    elif model_task == 'Classification':
        st.title("Image Classification App")
        uploaded_file = st.file_uploader("Choose an image...", type="jpg")

        col1, col2 = st.columns(2)

        with col1:
            if uploaded_file:
                uploaded_image = PIL.Image.open(uploaded_file)
                st.image(uploaded_file,
                         caption="Uploaded Image",
                         use_column_width=True)

        if st.button("Predict"):
            if uploaded_file is not None:
                with open("temp_image.jpg", "wb") as f:
                    f.write(uploaded_file.read())
                
                # saved_model_path = Path(settings.CLASSIFICATION_MODEL)
               # model = tf.saved_model.load(saved_model_path)
                model = tf.load(settings.CLASSIFICATION_MODEL)
                class_names = ['fall_armyworm', 'grasshopper', 'leaf_blight', 'leaf_spot']

                idx, predicted_class, confidence = helper.predict_class("temp_image.jpg", model, class_names)

                with col2:
                    st.write(f"Predicted Class Index: {idx}")
                    st.write(f"Predicted Class Name: {predicted_class}")
                    st.write(f"Confidence: {confidence:.2f}")

                    if idx == 0:
                        st.write("Display content for dom1")
                    elif idx == 1:
                        st.write("Display content for dom2")
                    elif idx == 2:
                        st.write("Display content for dom3")
                    elif idx == 3:
                        st.write("Display content for dom4")
            else:
                st.write("Please upload an image before pressing the Predict button.")
