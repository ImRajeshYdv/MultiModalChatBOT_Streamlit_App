import os
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from PIL import Image
import base64
import io
from datetime import datetime
import time
import mimetypes

load_dotenv()

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

# Initialize session state for conversation history
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

if 'messages' not in st.session_state:
    st.session_state.messages = []

# Initialize session state for model parameters
if 'temperature' not in st.session_state:
    st.session_state.temperature = 0.7

if 'top_k' not in st.session_state:
    st.session_state.top_k = 40

# File size limit constants
MAX_FILE_SIZE_MB = 5
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

def validate_file_size(uploaded_file):
    """Validate if uploaded file size is within limits"""
    if uploaded_file.size > MAX_FILE_SIZE_BYTES:
        file_size_mb = uploaded_file.size / (1024 * 1024)
        return False, f"File size ({file_size_mb:.1f}MB) exceeds the maximum allowed size of {MAX_FILE_SIZE_MB}MB"
    return True, "File size is acceptable"

def encode_image_to_base64(image):
    """Convert PIL image to base64 string for API consumption"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def chatbot_with_image(user_prompt, image=None, temperature=0.7, top_k=40):
    """Enhanced chatbot function that can handle both text and images"""
    messages = []
    
    # Add conversation history for context
    for msg in st.session_state.messages:
        messages.append(msg)
    
    # Prepare the current message
    if image is not None:
        # Convert image to base64 and attach as data URL to mimic multi-modal payload
        img_base64 = encode_image_to_base64(image)
        message_content = [
            {"type": "text", "text": user_prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}},
        ]
        messages.append({"role": "user", "content": message_content})
    else:
        messages.append({"role": "user", "content": user_prompt})
    
    # Map UI top_k (1..100) to top_p (0.01..1.0) to avoid unsupported top_k param
    top_p = max(0.01, min(1.0, top_k / 100.0))

    try:
        model = "meta-llama/llama-4-maverick-17b-128e-instruct"

        t0 = time.perf_counter()
        chat_completion = client.chat.completions.create(
            messages=messages,
            model=model,
            max_tokens=1000,
            temperature=temperature,
            top_p=top_p
        )
        t1 = time.perf_counter()

        response = chat_completion.choices[0].message.content
        latency_ms = int((t1 - t0) * 1000)
        return response, latency_ms
        
    except Exception as e:
        # If vision-like payload fails, fall back to text-only description flow
        if image is not None:
            try:
                st.warning("Vision analysis not available with current model. Using text-only approach...")
                fallback_prompt = (
                    f"User uploaded an image and asked: {user_prompt}. "
                    "Please provide a helpful response based on the question, noting that I cannot see the image content."
                )
                messages_text_only = []
                for msg in st.session_state.messages:
                    if isinstance(msg.get('content'), str):
                        messages_text_only.append(msg)
                messages_text_only.append({"role": "user", "content": fallback_prompt})

                top_p = max(0.01, min(1.0, top_k / 100.0))
                t0 = time.perf_counter()
                chat_completion = client.chat.completions.create(
                    messages=messages_text_only,
                    model=model,
                    max_tokens=1000,
                    temperature=temperature,
                    top_p=top_p
                )
                t1 = time.perf_counter()
                response = chat_completion.choices[0].message.content
                latency_ms = int((t1 - t0) * 1000)
                return response, latency_ms
            except Exception as fallback_error:
                return f"Error: {str(fallback_error)}", None
        else:
            return f"Error: {str(e)}", None

# Streamlit UI
st.set_page_config(page_title="PIC Insights ChatBot", layout="wide")
st.title("🤖 PIC Insights ChatBot")
st.markdown("*Ask questions, upload images, or both! Your conversation history is automatically saved.*")

# Sidebar for conversation management and model parameters
with st.sidebar:
    st.header("⚙️ Model Configuration")

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=st.session_state.temperature,
        step=0.1,
        help="Controls randomness in responses."
    )

    top_k = st.slider(
        "Top-K",
        min_value=1,
        max_value=100,
        value=st.session_state.top_k,
        step=1,
        help="Higher = more diverse choices; internally mapped to top_p."
    )

    st.session_state.temperature = temperature
    st.session_state.top_k = top_k

    st.markdown("---")
    st.header("🧹 Session")

    if st.button("Clear Conversation"):
        st.session_state.conversation_history = []
        st.session_state.messages = []
        st.success("Conversation cleared!")
        st.rerun()

    if st.session_state.conversation_history:
        st.info(f"Messages in history: {len(st.session_state.conversation_history)}")

# Main chat interface
col1, col2 = st.columns([3, 1])

with col1:
    # Display conversation history
    st.subheader("💬 Conversation")
    
    # Create a container for the chat history
    chat_container = st.container()
    
    with chat_container:
        for entry in st.session_state.conversation_history:
            if entry['type'] == 'user':
                with st.chat_message("user"):
                    st.write(entry['content'])
                    if 'image_data' in entry and entry['image_data'] is not None:
                        try:
                            data = entry['image_data']
                            # Show directly if it's a PIL image
                            if isinstance(data, Image.Image):
                                st.image(data, width=20)
                            else:
                                # Legacy/back-compat: support dict with base64, or raw/base64 strings/bytes
                                img_obj = None
                                if isinstance(data, dict) and 'b64' in data:
                                    raw = base64.b64decode(data['b64'])
                                    img_obj = Image.open(io.BytesIO(raw))
                                elif isinstance(data, (bytes, bytearray)):
                                    img_obj = Image.open(io.BytesIO(data))
                                elif isinstance(data, str):
                                    try:
                                        raw = base64.b64decode(data)
                                        img_obj = Image.open(io.BytesIO(raw))
                                    except Exception:
                                        img_obj = None
                                if img_obj is not None:
                                    st.image(img_obj, width=20)
                        except Exception as e:
                            st.error(f"Error displaying image: {str(e)}")
            elif entry['type'] == 'assistant':
                with st.chat_message("assistant"):
                    st.write(entry['content'])
                    if 'latency_ms' in entry and entry['latency_ms'] is not None:
                        st.caption(f"⏱ Response time: {entry['latency_ms']} ms")

with col2:
    # st.subheader(" Input")
    
    
    # Image upload
    uploaded_image = st.file_uploader(
        "Upload an image (optional)", 
        type=['png', 'jpg', 'jpeg'], 
        key="image_uploader"
    )
     
   
    
    if uploaded_image:
        # Validate file size
        is_valid_size, size_message = validate_file_size(uploaded_image)
        
        if is_valid_size:
            try:
                image = Image.open(uploaded_image)
                # st.image(image, caption="Uploaded Image", width=30)
                st.image(image, width=20)
                st.success("✅ Image uploaded successfully!")
                
                # Show file size info
                file_size_mb = uploaded_image.size / (1024 * 1024)
                st.caption(f"📊 File size: {file_size_mb:.2f}MB")
                
            except Exception as e:
                st.error(f"❌ Error loading image: {str(e)}")
                uploaded_image = None
        else:
            st.error(f"❌ {size_message}")
            uploaded_image = None  # Reset to prevent processing oversized file

# Chat input
user_input = st.chat_input("Type your message here...")

if user_input:
    # Process the input
    image_for_analysis = None
    if uploaded_image:
        # Double-check file size before processing
        is_valid_size, _ = validate_file_size(uploaded_image)
        if is_valid_size:
            try:
                image_for_analysis = Image.open(uploaded_image)
            except Exception as e:
                st.error(f"❌ Error processing image: {str(e)}")
                image_for_analysis = None
        else:
            st.error("❌ File size validation failed")
            image_for_analysis = None
    
    # Add user message to history
    user_entry = {
        'type': 'user',
        'content': user_input,
        'timestamp': datetime.now().isoformat(),
    }
    if image_for_analysis is not None:
        user_entry['image_data'] = image_for_analysis
    st.session_state.conversation_history.append(user_entry)
    
    # Get bot response with current temperature and top_k values
    with st.spinner("🤔 Thinking..."):
        bot_response, latency_ms = chatbot_with_image(
            user_input, 
            image_for_analysis, 
            temperature=st.session_state.temperature,
            top_k=st.session_state.top_k
        )
    
    # Add bot response to history
    bot_entry = {
        'type': 'assistant',
        'content': bot_response,
        'timestamp': datetime.now().isoformat(),
        'latency_ms': latency_ms
    }
    st.session_state.conversation_history.append(bot_entry)
    
    # Update messages for API context
    if image_for_analysis:
        # For image + text queries, include both in the message
        st.session_state.messages.append({
            "role": "user", 
            "content": f"[Image uploaded] {user_input}"
        })
    else:
        # For text-only queries, just include the text
        st.session_state.messages.append({
            "role": "user", 
            "content": user_input
        })
    
    st.session_state.messages.append({
        "role": "assistant", 
        "content": bot_response
    })
    
    # Rerun to update the display
    st.rerun()

# Footer with dynamic tips
st.markdown("---")

# Show different tips based on current state
if uploaded_image:
    st.markdown("*💡 Tip: You can ask specific questions about the uploaded image!*")
else:
    st.markdown("*💡 Tip: Ask any question or upload an image for analysis!*")