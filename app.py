import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import requests

# Page setup
st.set_page_config(page_title="AI Pet Detector", page_icon="🐾")
st.title("🐱 Cat vs 🐶 Dog Predictor")

# Load MobileNetV2 (This is the stable, lightweight model)
@st.cache_resource
def load_model():
    model = models.mobilenet_v2(weights='DEFAULT')
    model.eval()
    # Get the official labels
    labels_url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    labels = requests.get(labels_url).text.splitlines()
    return model, labels

# Load the brain
model, labels = load_model()

# User upload
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Your Image", use_container_width=True)
    
    # Process image
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Run AI
    input_tensor = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
    
    # Get prediction
    _, index = torch.max(output, 1)
    result = labels[index[0]]
    
    st.success(f"### I think this is a: **{result.upper()}**")
    st.balloons()
