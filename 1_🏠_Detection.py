# Python In-built packages
from pathlib import Path
import PIL
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.image as mpimg

import os
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.layers import Dense, Conv2D, GlobalAvgPool2D, Input
from tensorflow.keras import callbacks, optimizers
from tensorflow.keras.preprocessing.image import load_img, ImageDataGenerator
import numpy as np
import cv2

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

    st.markdown("<h2 style='text-align: center; color: black;'>This model detects/classifies the following types of diseases in maize:</h2>",
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
        st.markdown("<h2 style='text-align: center; color: black;'>Classification Website</h2>", unsafe_allow_html=True)
        
        model = tf.keras.models.load_model("classification_model/mdl_wt .hdf5")
        ### load file
        uploaded_file = st.file_uploader("Choose a image file", type=("jpg", "jpeg", "png", 'bmp', 'webp'))
        
        map_dict = {0: 'armyworm',
                    1: 'blight',
                    2: 'grasshopper',
                    3: 'leaf_spot'}
        
        
        if uploaded_file is not None:   
            Convert the file to an opencv image.
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            opencv_image = cv2.imdecode(file_bytes, 1)
            opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(opencv_image,(224,224))
            Now do something with the image! For example, let's display it:
            st.image(opencv_image, channels="RGB")
        
            resized = mobilenet_v2_preprocess_input(resized)
            img_reshape = resized[np.newaxis,...]
        
            Genrate_pred = st.button("Generate Prediction")    
            if Genrate_pred:
                prediction = model.predict(img_reshape).argmax()
                st.title("Predicted Pests/Diseases: {}".format(map_dict [prediction]))
                st.markdown("<h4 style='text-align: center; color: gray;'>REMEDIES</h4>", unsafe_allow_html=True)
                
                if format(map_dict [prediction]) == "armyworm":
                                st.write(
                                    """
                                    
                                    - Use biopesticides based on naturally occurring organisms, to target FAW larvae.
                                    - Apply insecticidal sprays containing active ingredients effective against FAW larvae, such as pyrethroids, spinosyns, or diamides.
                                    - Use pheromone traps to attract and monitor FAW adults.
                                
                                    """)
                elif format(map_dict [prediction]) == "grasshopper":
                                st.write(
                                    """
        
                                    - Use insecticidal baits containing insecticides to lure and kill grasshoppers. 
                                    - Apply insecticidal sprays to directly kill grasshoppers.
                                    - Create physical barriers, such as sticky traps around fields, to prevent grasshoppers from entering cultivated areas.
                                    
                                    """)
                elif format(map_dict [prediction]) == "blight":
                                st.write(
                                    """
                                    - Apply fungicides to all plant surfaces early in the morning or late in the evening.
                                    - Practice crop rotation to break the disease cycle. Avoid planting maize in the same field consecutively.
                                    - Avoid over-irrigation, as excessive moisture creates favorable conditions for fungal growth.
                                    - Seek advice from local plant pathologists for specific recommendations tailored to your region.
                                    
                                    """)
                elif format(map_dict [prediction]) == "leaf_spot":
                                st.write(
                                    """
        
                                    - Control weeds in and around maize fields to eliminate potential hosts for leaf spot pathogens. 
                                    - Apply fungicides containing active ingredients effective against the specific leaf spot pathogen affecting your crops.
                                    - Avoid overhead irrigation, which can create a favorable environment for leaf spot pathogens to thrive.
                                    
                                    """)
                    
