import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import PIL

# Replace the relative path to your weight file
model_path =  Path(settings.CLASSIFICATION_MODEL)

# Setting page layout
st.set_page_config(
    page_title="Plant Pests and Disease Detection",  # Setting page title
    page_icon="🤖",  # Setting page icon
    layout="wide",  # Setting layout to wide
    initial_sidebar_state="expanded",  # Expanding sidebar by default
)



# Creating main page heading
st.title("DL POWERED PESTS AND DISEASE DETECTION IN MAIZE")
st.caption("Updload a photo of an affected maize leaf.")
st.caption("Then click the :blue[Detect Objects] button and check the result.")


import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

# Function to predict class
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
saved_model_path = Path(settings.CLASSIFICATION_MODEL)

# Load the SavedModel
model = tf.saved_model.load(saved_model_path)

# List of class names corresponding to the indices
class_names = ['fall_armyworm', 'grasshopper', 'leaf_blight', 'leaf_spot']

# Streamlit app
st.title("Image Classification App")

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
            st.write(f"Confidence: {confidence:.2f}")

                    # Display different content based on the predicted class
            if idx == 0:
                st.write("Display content for dom1")
            elif idx == 1:
                st.write(
                    """
                    - spray with pestiscide
                    - spray twice
                    - weed
                    
                    """)
            elif idx == 2:
                st.write("Display content for dom3")
            elif idx == 3:
                st.write("Display content for dom4")
            
    else:
        st.write("Please upload an image before pressing the Predict button.")
