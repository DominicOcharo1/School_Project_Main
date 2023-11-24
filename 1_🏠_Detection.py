# Python In-built packages
from pathlib import Path
import PIL

# External packages
import streamlit as st

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
with st.container(): # st.title("DL POWERED PESTS AND DISEASE DETECTION IN MAIZE")
    st.markdown(
        "<h1 style='text-align: center; color: blue; background-color: lightblue; padding: 20px;'>DL POWERED PESTS AND DISEASE DETECTION IN MAIZE</h1>",
        unsafe_allow_html=True)
    
    # st.markdown('<h2 style="color:gray;">This detection model detects the following types of diseases in maize:</h2>', unsafe_allow_html=True)

    st.markdown("<h2 style='text-align: center; color: black;'>This detection model detects the following types of diseases in maize:</h2>", unsafe_allow_html=True)
    
    st.markdown("<h3 style='text-align: center; color: black;'>Leaf Blight, Leaf Spot, Fall Armyworm, and Grasshopper</h3>", unsafe_allow_html=True)
    
    st.markdown("<h4 style='text-align: center; color: gray;'>By Dominic Ocharo and Shadrack Onjiri</h4>", unsafe_allow_html=True)

# Sidebar
# Model Options
st.sidebar.header("ML Model Config")
model_type = st.sidebar.radio(
    "Select Task", ['Detection', 'Detection1'])

confidence = float(st.sidebar.slider(
    "Select Model Confidence", 20, 10, 1)) / 20


with st.container():
    st.write("---")
    
    # Selecting Detection1 Or Detection2
    if model_type == 'Detection':
        model_path = Path(settings.DETECTION_MODEL)
    elif model_type == 'Detection1':
        model_path = Path(settings.SEGMENTATION_MODEL)
    
    # Load Pre-trained ML Model
    try:
        model = helper.load_model(model_path)
    except Exception as ex:
        st.error(f"Unable to load model. Check the specified path: {model_path}")
        st.error(ex)
    
    st.header("Image/Video Config")
    source_radio = st.radio(
        "Select Source", settings.SOURCES_LIST)
    
    source_img = None
    # If image is selected
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
                    res = model.predict(uploaded_image,
                                        conf=confidence
                                        )
                    boxes = res[0].boxes
                    res_plotted = res[0].plot()[:, :, ::-1]
                    st.image(res_plotted, caption="Detected Image",
                             use_column_width=True)
                    try:
                        with st.expander("Detection Results"):
                            for box in boxes:
                                st.write(box.data)
                    except Exception as ex:
                        # st.write(ex)
                        st.write("No image is uploaded yet!")
    
    elif source_radio == settings.VIDEO:
        helper.play_stored_video(confidence, model)
        # helper.play_uploaded_video1(confidence, model)

    elif source_radio == settings.WEBCAM:
        helper.play_webcam(confidence, model)
    
    elif source_radio == settings.YOUTUBE:
        helper.play_youtube_video(confidence, model)
    
    else:
        st.error("Please select a valid source type!")
