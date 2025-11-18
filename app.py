import streamlit as st
import numpy as np
from PIL import Image
from skimage import color, filters, feature

st.set_page_config(layout="wide")
st.title("Canny Edge Detection - Step-by-Step (Horizontal Visual)")

uploaded_file = st.file_uploader("Upload Gambar", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)

    # 1. Grayscale
    gray = color.rgb2gray(img_np)

    # 2. Gaussian Blur
    blurred = filters.gaussian(gray, sigma=1.4)

    # 3. Sobel X and Y
    sobel_x = filters.sobel_h(blurred)
    sobel_y = filters.sobel_v(blurred)

    # Gabungkan menjadi Sobel XY
    sobel_xy = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel_xy = (sobel_xy / sobel_xy.max()) * 255
    sobel_xy = sobel_xy.astype(np.uint8)

    # 4. Gradient Angle (untuk NMS)
    angle = np.arctan2(sobel_y, sobel_x) * 180 / np.pi
    angle[angle < 0] += 180

    # Non-Max Suppression
    def non_max_suppression(mag, ang):
        H, W = mag.shape
        result = np.zeros((H, W), dtype=np.uint8)

        angle_q = ang.copy()
        angle_q[(angle_q >= 0) & (angle_q < 22.5)] = 0
        angle_q[(angle_q >= 22.5) & (angle_q < 67.5)] = 45
        angle_q[(angle_q >= 67.5) & (angle_q < 112.5)] = 90
        angle_q[(angle_q >= 112.5) & (angle_q < 157.5)] = 135
        angle_q[(angle_q >= 157.5)] = 0

        for i in range(1, H - 1):
            for j in range(1, W - 1):
                q = 255
                r = 255

                if angle_q[i, j] == 0:
                    q = mag[i, j + 1]
                    r = mag[i, j - 1]
                elif angle_q[i, j] == 45:
                    q = mag[i + 1, j - 1]
                    r = mag[i - 1, j + 1]
                elif angle_q[i, j] == 90:
                    q = mag[i + 1, j]
                    r = mag[i - 1, j]
                elif angle_q[i, j] == 135:
                    q = mag[i - 1, j - 1]
                    r = mag[i + 1, j + 1]

                if mag[i, j] >= q and mag[i, j] >= r:
                    result[i, j] = mag[i, j]

        return result

    nms = non_max_suppression(sobel_xy, angle)

    # 5. Double Threshold
    high_threshold = 70
    low_threshold = 30

    strong = 255
    weak = 75

    strong_edges = (nms >= high_threshold).astype(np.uint8) * strong
    weak_edges = ((nms >= low_threshold) & (nms < high_threshold)).astype(np.uint8) * weak

    dt = strong_edges + weak_edges

    # 6. Hysteresis
    def hysteresis(img):
        H, W = img.shape
        for i in range(1, H - 1):
            for j in range(1, W - 1):
                if img[i, j] == weak:
                    if 255 in [
                        img[i+1, j], img[i-1, j], img[i, j+1], img[i, j-1],
                        img[i+1, j+1], img[i-1, j-1], img[i+1, j-1], img[i-1, j+1]
                    ]:
                        img[i, j] = 255
                    else:
                        img[i, j] = 0
        return img

    hysteresis_result = hysteresis(dt.copy())

    #===========================#
    #   DISPLAY HORIZONTAL     #
    #===========================#

    st.subheader("📌 Pipeline Result (Horizontal Layout)")

    cols = st.columns(6)

    with cols[0]:
        st.write("**Original**")
        st.image(img_np)

    with cols[1]:
        st.write("**Gaussian Blur**")
        st.image((blurred*255).astype(np.uint8), clamp=True)

    with cols[2]:
        st.write("**Sobel XY (Gradient Magnitude)**")
        st.image(sobel_xy, clamp=True)

    with cols[3]:
        st.write("**Non-Max Suppression**")
        st.image(nms, clamp=True)

    with cols[4]:
        st.write("**Double Threshold**")
        st.image(dt, clamp=True)

    with cols[5]:
        st.write("**Hysteresis (Final Edge)**")
        st.image(hysteresis_result, clamp=True)
