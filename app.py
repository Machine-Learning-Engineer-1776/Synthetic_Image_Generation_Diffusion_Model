import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import os
import random
import cv2
import pandas as pd
from io import BytesIO

# ============================================================================
# MAIN HEADER
# ============================================================================
st.title("Brain Suite")
st.markdown("**AI-Powered Brain Tumor Detection & Synthetic Image Generation**")
st.markdown("---")

# ============================================================================
# SYNTHETIC MRI SECTION
# ============================================================================
st.markdown("""
<div style='text-align: center; padding: 2rem 0;'>
    <h3 style='color: #1f77b4;'>🎯 Synthetic MRI</h3>
    <p style='color: #666; font-size: 1.1em;'>AI-generated brain MRI scans that closely mimic real clinical imaging for research and training</p>
</div>
""", unsafe_allow_html=True)

# Load model
try:
    model = tf.keras.models.load_model('../Models/model.h5', compile=False)
except Exception as e:
    st.error(f"Model loading failed: {str(e)}")
    st.stop()

CATEGORIES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

# Synthetic image display - FIXED CYCLING LOGIC
SYNTHETIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "synthetic_images")
synthetic_images = [f for f in os.listdir(SYNTHETIC_DIR) if f.endswith('.png')] if os.path.exists(SYNTHETIC_DIR) else []

if synthetic_images:
    # Sort to ensure consistent order and filter for our 3 specific images
    synthetic_images.sort()
    valid_images = [f for f in synthetic_images if f.endswith(('1.png', '2.png', '3.png'))]
    
    if len(valid_images) >= 3:
        # Use session state to track which image to show next (cycling 1->2->3->1...)
        if 'synthetic_counter' not in st.session_state:
            st.session_state.synthetic_counter = 0
        
        if st.button("🎯 Generate Synthetic Image", type="secondary"):
            # Cycle through the 3 images
            st.session_state.synthetic_counter = (st.session_state.synthetic_counter + 1) % 3
            selected_index = st.session_state.synthetic_counter
            selected_image = valid_images[selected_index]
            synth_path = os.path.join(SYNTHETIC_DIR, selected_image)
            synth_img = Image.open(synth_path).convert('RGB').resize((224, 224), Image.Resampling.LANCZOS)
            # Center the image using columns
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(synth_img, caption=f"Generated: {selected_image}", width=196)
    else:
        st.warning(f"Expected 3 synthetic images (ending in 1.png, 2.png, 3.png) but found {len(valid_images)} in the synthetic_images folder.")
else:
    st.warning("No synthetic images found in the synthetic_images folder.")

st.markdown("---")

# ============================================================================
# TUMOR CLASSIFIER SECTION
# ============================================================================
st.markdown("""
<div style='text-align: center; padding: 2rem 0;'>
    <h3 style='color: #d62728;'>🔬 Tumor Classifier</h3>
    <p style='color: #666; font-size: 1.1em;'>Advanced AI analysis of brain MRI scans with comprehensive radiology reporting</p>
</div>
""", unsafe_allow_html=True)

# TEST IMAGE SELECTION - FIXED VERSION
st.subheader("📁 Pre-loaded Test Images")

TEST_IMAGES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test-images")
test_images = {}
if os.path.exists(TEST_IMAGES_DIR):
    # List all files and try to identify images even without extensions
    all_files = os.listdir(TEST_IMAGES_DIR)
    
    for filename in all_files:
        filepath = os.path.join(TEST_IMAGES_DIR, filename)
        # Try to open as image to verify it's actually an image
        try:
            with Image.open(filepath) as img:
                # It's an image! Add it to our list
                test_images[filename] = filepath
        except (IOError, OSError):
            # Not an image, skip it
            continue
    
    if test_images:
        # Default to empty selection
        test_image_options = ["Select an image..."] + list(test_images.keys())
        selected_test_image = st.selectbox(
            "Choose a test image:",
            options=test_image_options,
            index=0,
            help="Select an image from the test images folder"
        )
        
        # Create columns for dropdown and button
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("🔍 Classify Image", type="primary", disabled=(selected_test_image == "Select an image...")):
                # Load the selected test image as PIL Image
                test_image_path = test_images[selected_test_image]
                test_img_pil = Image.open(test_image_path).convert('RGB')
                
                # Store the PIL image directly in session state
                st.session_state.selected_test_image = test_img_pil
                st.session_state.selected_test_filename = selected_test_image
                st.session_state.show_results = True
                st.rerun()
    else:
        st.warning("No valid image files found in the test-images folder.")
else:
    st.warning(f"test-images folder not found at {TEST_IMAGES_DIR}")

# Single image upload section
st.subheader("📤 Upload Your Own Image")
upload_placeholder = st.empty()
uploaded_file = upload_placeholder.file_uploader("Upload MRI Image", type=["jpg", "npy"], accept_multiple_files=False)

# ============================================================================
# RESULTS DISPLAY SECTION
# ============================================================================
if 'show_results' in st.session_state and st.session_state.show_results:
    # Handle test image classification
    test_img_pil = st.session_state.selected_test_image
    test_filename = st.session_state.selected_test_filename
    
    # Clear previous content
    st.empty()
    st.empty()
    st.empty()

    # Process the test image
    img = np.array(test_img_pil)
    img_pil = test_img_pil
    img_resized = img_pil.resize((224, 224), Image.Resampling.LANCZOS)
    img_array = np.array(img_resized).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    try:
        pred = model.predict(img_array)
        probs = tf.nn.softmax(pred[0]).numpy()
        max_prob = float(np.max(probs))
        label_idx = np.argmax(probs)
        label = CATEGORIES[label_idx]
        if label == "No Tumor":
            col1, col2, col3 = st.columns([1, 2, 1])  # CENTERED
            with col2:
                st.image(img_resized, caption="Test Image", width=400)
            st.write("**No Tumor Detected**")
            st.markdown("---")
            st.success("Clear scan - no abnormalities detected.")
            st.stop()
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        st.stop()
    
    # Generate tumor likelihood heatmap (approximation)
    gray_img = cv2.cvtColor(np.array(img_resized), cv2.COLOR_RGB2GRAY)
    heatmap = cv2.normalize(gray_img, None, 0, 255, cv2.NORM_MINMAX)
    heatmap = cv2.GaussianBlur(heatmap, (5, 5), 0)
    coords = []
    for _ in range(3):
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(heatmap)
        coords.append((max_loc[0], max_loc[1], max_val / 255.0))  # X, Y, normalized probability
        cv2.circle(heatmap, max_loc, 5, 0, -1)  # Mask the found point

    # Merge close coordinates
    merged_coords = []
    threshold = 10
    while coords:
        x, y, prob = coords.pop(0)
        group = [(x, y, prob)]
        remaining = []
        for cx, cy, cp in coords:
            if abs(cx - x) < threshold and abs(cy - y) < threshold:
                group.append((cx, cy, cp))
            else:
                remaining.append((cx, cy, cp))
        coords = remaining
        avg_x = int(sum(c[0] for c in group) / len(group))
        avg_y = int(sum(c[1] for c in group) / len(group))
        avg_prob = max(c[2] for c in group)
        merged_coords.append((avg_x, avg_y, avg_prob))

    # Calculate average regional probability
    avg_prob = np.mean([coord[2] for coord in merged_coords]) if merged_coords else 0.0

    # Display resized original image with report - CENTERED
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(img_resized, caption=f"Test Image: {test_filename} - Prediction: {label} ({max_prob:.2f})", width=400)
    
    st.markdown("---")
    st.subheader("📋 Comprehensive Radiology Report")
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown("### Assessment")
    with col2:
        st.markdown("**Tumor Classification:** " + ("✅ " if label == "No Tumor" else "⚠️ ") + f"{label}")
        st.markdown(f"**Confidence:** {max_prob * 100:.1f}%")
    
    st.markdown("### Tumor Likelihood Assessment")
    st.markdown(f"**Regional Intensity Analysis:** {avg_prob * 100:.0f}%")
    
    if merged_coords:
        st.markdown("### Top 3 Suspected Tumor Regions")
        report_data = [
            [f"Region {i+1}", f"X: {coord[0]}, Y: {coord[1]}", f"{coord[2] * 100:.1f}%", "Core" if coord[2] > 0.7 else "Periphery"]
            for i, coord in enumerate(merged_coords)
        ]
        df = pd.DataFrame(report_data, columns=["Region", "Coordinates", "Likelihood", "Region Type"])
        st.table(df)
    
    st.markdown("### Clinical Notes")
    st.markdown("""
    - Coordinates indicate pixel locations with highest intensity suggesting potential tumor presence
    - Core regions (>70% likelihood) indicate high-confidence tumor areas
    - Periphery regions suggest possible tumor margins or edema
    - **Further evaluation with biopsy and clinical correlation recommended**
    """)

    # Display original image with circles and colored pixels - CENTERED
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        marked_img = np.array(img_resized)
        for x, y, prob in merged_coords:
            radius = 10 if len([c for c in merged_coords if abs(c[0] - x) < threshold and abs(c[1] - y) < threshold]) > 1 else 5
            cv2.circle(marked_img, (x, y), radius, (0, 0, 255), 2)  # Red circle
            # Color pixels within circle
            for i in range(max(0, x - radius), min(224, x + radius + 1)):
                for j in range(max(0, y - radius), min(224, y + radius + 1)):
                    if (i - x) ** 2 + (j - y) ** 2 <= radius ** 2:
                        marked_img[j, i] = [0, 0, 255] if prob > 0.5 else [255, 0, 0]  # Red for high prob, blue for lower
        st.image(marked_img, caption="🔴 Tumor Highlighted Regions (Red=High Probability, Blue=Moderate)", width=400)

    # Reset session state after showing results
    st.markdown("---")
    if st.button("🔄 Classify Another Image", type="secondary"):
        for key in ['selected_test_image', 'selected_test_filename', 'show_results']:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

elif uploaded_file:
    # Handle uploaded file (original logic)
    # Clear previous content
    st.empty()
    st.empty()
    st.empty()

    # Load image
    if uploaded_file.name.endswith('.npy'):
        img = np.load(uploaded_file)
        if img.ndim == 2:  # Grayscale to RGB
            img = np.stack([img] * 3, axis=-1)
    else:
        img = np.array(Image.open(uploaded_file).convert('RGB'))
    
    # Resize to model's expected size (224x224)
    img_pil = Image.fromarray(img)
    img_resized = img_pil.resize((224, 224), Image.Resampling.LANCZOS)
    img_array = np.array(img_resized).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    try:
        pred = model.predict(img_array)
        probs = tf.nn.softmax(pred[0]).numpy()
        max_prob = float(np.max(probs))
        label_idx = np.argmax(probs)
        label = CATEGORIES[label_idx]
        if label == "No Tumor":
            col1, col2, col3 = st.columns([1, 2, 1])  # CENTERED
            with col2:
                st.image(img_resized, caption="Uploaded Image", width=400)
            st.write("**No Tumor Has Been Detected**")
            st.stop()
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        st.stop()
    
    # Generate tumor likelihood heatmap (approximation)
    gray_img = cv2.cvtColor(np.array(img_resized), cv2.COLOR_RGB2GRAY)
    heatmap = cv2.normalize(gray_img, None, 0, 255, cv2.NORM_MINMAX)
    heatmap = cv2.GaussianBlur(heatmap, (5, 5), 0)
    coords = []
    for _ in range(3):
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(heatmap)
        coords.append((max_loc[0], max_loc[1], max_val / 255.0))  # X, Y, normalized probability
        cv2.circle(heatmap, max_loc, 5, 0, -1)  # Mask the found point

    # Merge close coordinates
    merged_coords = []
    threshold = 10
    while coords:
        x, y, prob = coords.pop(0)
        group = [(x, y, prob)]
        remaining = []
        for cx, cy, cp in coords:
            if abs(cx - x) < threshold and abs(cy - y) < threshold:
                group.append((cx, cy, cp))
            else:
                remaining.append((cx, cy, cp))
        coords = remaining
        avg_x = int(sum(c[0] for c in group) / len(group))
        avg_y = int(sum(c[1] for c in group) / len(group))
        avg_prob = max(c[2] for c in group)
        merged_coords.append((avg_x, avg_y, avg_prob))

    # Calculate average regional probability
    avg_prob = np.mean([coord[2] for coord in merged_coords]) if merged_coords else 0.0

    # Display resized original image with report - CENTERED
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(img_resized, caption=f"Uploaded Image: {uploaded_file.name} - Prediction: {label} ({max_prob:.2f})", width=400)
    
    st.subheader("Comprehensive Radiology Report")
    st.write("### Tumor Likelihood Assessment")
    st.write(f"**Tumor Probability (Average Regional Intensity)**: {avg_prob * 100:.0f}%")
    st.write("**Top 3 Suspected Tumor Regions (Pixel Coordinates and Likelihood)**")
    report_data = [
        [f"Region {i+1}", f"X: {coord[0]}, Y: {coord[1]}", f"Probability: {coord[2] * 100:.1f}%", "Core" if coord[2] > 0.7 else "Periphery"]
        for i, coord in enumerate(merged_coords)
    ]
    st.table(pd.DataFrame(report_data, columns=["Region", "Coordinates", "Likelihood", "Region Type"]))
    st.write("**Notes**: Coordinates indicate pixel locations with highest intensity, suggesting tumor presence. Further biopsy recommended for confirmation.")

    # Display original image with circles and colored pixels - CENTERED
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        marked_img = np.array(img_resized)
        for x, y, prob in merged_coords:
            radius = 10 if len([c for c in merged_coords if abs(c[0] - x) < threshold and abs(c[1] - y) < threshold]) > 1 else 5
            cv2.circle(marked_img, (x, y), radius, (0, 0, 255), 2)  # Red circle
            # Color pixels within circle
            for i in range(max(0, x - radius), min(224, x + radius + 1)):
                for j in range(max(0, y - radius), min(224, y + radius + 1)):
                    if (i - x) ** 2 + (j - y) ** 2 <= radius ** 2:
                        marked_img[j, i] = [0, 0, 255] if prob > 0.5 else [255, 0, 0]  # Red for high prob, blue for lower
        st.image(marked_img, caption="Tumor Highlighted Regions", width=400)