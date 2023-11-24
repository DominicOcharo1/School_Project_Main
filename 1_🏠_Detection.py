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
        def predict_class(image_path, model):
            # Load and preprocess the image
            img = Image.open(image_path)
            img = img.resize((256, 256))  # Resize the image
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)
            img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        
            # Make a prediction
            predictions = model(tf.constant(img_array, dtype=tf.float32))
        
            # Convert predictions to numpy array
            predictions = np.array(predictions)
        
            # Assuming the model returns class indices and confidence scores
            class_indices = np.argmax(predictions, axis=1)
            confidence_scores = tf.reduce_max(tf.nn.softmax(predictions), axis=1).numpy()
        
            # Get the predicted class name
            predicted_class = class_names[class_indices[0]]
        
            return class_indices[0], predicted_class, confidence_scores[0]
        
        # Path to the directory containing the SavedModel (.pb model)
        saved_model_path = Path("classification_model")
        
        # Load the SavedModel
        model = tf.saved_model.load(saved_model_path)
        
        # List of class names corresponding to the indices
        class_names = ['fall_armyworm', 'grasshopper', 'leaf_blight', 'leaf_spot']
        
        # Streamlit app
        st.title("Image Classification")
        
        # File uploader
        uploaded_file = st.file_uploader("Choose an image...", type="jpg")
        
        col1, col2 = st.columns(2)
        
        # Adding image to the first column if image is uploaded
        with col1:
            if uploaded_file:
                # Opening the uploaded image
                uploaded_image = PIL.Image.open(uploaded_file)
                # Adding the uploaded image to the page with a caption
                st.image(uploaded_file,
                         caption="Uploaded Image",
                         use_column_width=True
                         )
        
        # Button to make predictions
        if st.button("Predict"):
            if uploaded_file is not None:
                # Save the uploaded file
                with open("temp_image.jpg", "wb") as f:
                    f.write(uploaded_file.read())
        
                # Get predictions using the function
                idx, predicted_class, confidence = predict_class("temp_image.jpg", model)
                
                with col2:
                    # Display the uploaded image
                    # st.image(mpimg.imread("temp_image.jpg"), caption="Uploaded Image.", use_column_width=True)
        
                    # Print the predicted class name and confidence score
                    st.write(f"Predicted Class Index: {idx}")
                    st.write(f"Predicted Class Name: {predicted_class}")
                    # st.write(f"Confidence: {confidence:.2f}")
                    st.markdown("<h3 style='text-align: center; color: black;'>REMEDIES</h3>",
                unsafe_allow_html=True)

        
                            # Display different content based on the predicted class
                    if idx == 0:
                        st.write(
                            """
                            
                            - Use biopesticides based on naturally occurring organisms, to target FAW larvae.
                            - Apply insecticidal sprays containing active ingredients effective against FAW larvae, such as pyrethroids, spinosyns, or diamides.
                            - Use pheromone traps to attract and monitor FAW adults.
                        
                            """)
                    elif idx == 1:
                        st.write(
                            """

                            - Use insecticidal baits containing insecticides to lure and kill grasshoppers. 
                            - Apply insecticidal sprays to directly kill grasshoppers.
                            - Create physical barriers, such as sticky traps around fields, to prevent grasshoppers from entering cultivated areas.
                            
                            """)
                    elif idx == 2:
                        st.write(
                            """
                            - Apply fungicides to all plant surfaces early in the morning or late in the evening.
                            - Practice crop rotation to break the disease cycle. Avoid planting maize in the same field consecutively.
                            - Avoid over-irrigation, as excessive moisture creates favorable conditions for fungal growth.
                            - Seek advice from local plant pathologists for specific recommendations tailored to your region.
                            
                            """)
                    elif idx == 3:
                        st.write(
                            """

                            - Control weeds in and around maize fields to eliminate potential hosts for leaf spot pathogens. 
                            - Apply fungicides containing active ingredients effective against the specific leaf spot pathogen affecting your crops.
                            - Avoid overhead irrigation, which can create a favorable environment for leaf spot pathogens to thrive.
                            
                            """)
            else:
                st.write("Please upload an image before pressing the Predict button.")
