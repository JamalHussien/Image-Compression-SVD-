from scipy.linalg import svd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
from io import BytesIO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av

# Define the VideoProcessor class
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame = None

    def recv(self, frame):
        self.frame = frame.to_ndarray(format="bgr24")
        return frame

# Function to capture an image from the camera
def capture_image():
    ctx = webrtc_streamer(
        key="example",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=VideoProcessor,
        async_processing=True,
    )
    if ctx.video_processor:
        st.write("Camera is active. Click the button below to capture an image.")
        if st.button("Capture Now"):
            captured_image = ctx.video_processor.frame
            if captured_image is not None:
                return captured_image
            else:
                st.warning("Failed to capture image. Please try again.")
    return None

# Function to compress an image using SVD
def compress_img(pic, k):
    if isinstance(pic, np.ndarray):
        img = pic
    else:
        img = plt.imread(pic)

    # Separate RGB channels
    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    # Apply SVD to each channel
    Ur, Sr, Vtr = svd(R)
    Ug, Sg, Vtg = svd(G)
    Ub, Sb, Vtb = svd(B)

    # Reconstruct each channel with k singular values
    R_compressed = np.dot(Ur[:, :k], np.dot(np.diag(Sr[:k]), Vtr[:k, :]))
    G_compressed = np.dot(Ug[:, :k], np.dot(np.diag(Sg[:k]), Vtg[:k, :]))
    B_compressed = np.dot(Ub[:, :k], np.dot(np.diag(Sb[:k]), Vtb[:k, :]))

    # Stack compressed channels
    compressed_img = np.stack((R_compressed, G_compressed, B_compressed), axis=2)
    # Normalize the reconstructed image to [0, 255]
    compressed_img -= np.min(compressed_img)
    compressed_img /= np.max(compressed_img)
    compressed_img *= 255
    compressed_img = np.clip(compressed_img, 0, 255).astype(np.uint8)

    return compressed_img

# Function to save the compressed image
def save_compressed_img(compressed_img):
    img = Image.fromarray((compressed_img))
    buf = BytesIO()
    img.save(buf, format="JPEG")
    byte_im = buf.getvalue()
    return byte_im

# Streamlit app
st.title("Image Compression App")

# Option to upload or capture image
option = st.radio("Choose an option:", ("Upload Image", "Capture from Camera"))

if option == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.session_state.image = np.array(image)  # Store the image in session state
else:
    st.write("Click the button below to capture an image from your camera.")
    captured_image = capture_image()
    if captured_image is not None:
        st.session_state.image = captured_image  # Store the image in session state
    else:
        st.warning("No image captured yet.")

# Check if an image is available in session state
if 'image' in st.session_state:
    image = st.session_state.image

    # Resize the image to reduce its size
    max_width, max_height = 500, 500  # target max size
    if len(image.shape) == 3:  # Check if the image is in RGB format
        image_pil = Image.fromarray(image)
        image_pil.thumbnail((max_width, max_height))
        image = np.array(image_pil)
    else:
        st.warning("Invalid image format. Please upload or capture a valid image.")

    st.image(image, caption='Uploaded/Captured Image', use_column_width=False)

    # Get the dimensions of the image (height and width)
    m, n = image.shape[:2]

    k = st.slider("Number of singular values to keep", min_value=1, max_value=min(m, n), value=20)

    # Option to show images after compression
    showpic = st.checkbox("Show compressed image", value=True)

    # Compress the image when the button is clicked
    if st.button("Compress Image"):
        compressed_img = compress_img(image, k)

        if showpic:
            st.image(compressed_img, caption='Compressed Image', use_column_width="auto")

        # Convert compressed image to download format
        img_bytes = save_compressed_img(compressed_img)

        # Provide download button
        st.download_button(
            label="Download Compressed Image",
            data=img_bytes,
            file_name="compressed_image.jpeg",
            mime="image/jpeg")