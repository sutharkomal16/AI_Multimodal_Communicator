import os

import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image

from gemini_utility import (load_gemini_pro_model,
                            gemini_pro_vision_response,
                            gemini_pro_response,
                            get_pdf_text,
                            get_text_chunks,
                            get_vector_store,
                            get_conversational_chain,
                            user_input,
                            about)




working_dir = os.path.dirname(os.path.abspath(__file__))
# print(working_dir)

# setting up page configeration
st.set_page_config(
    page_title="Generative AI",
    page_icon="🧠",
    layout="centered"
)

# setting up sidebar option menu
with st.sidebar:
    selected = option_menu(
                            "Generative AI",
                            ["ChatBot","Image Captioning","Ask me anything","Chat with PDF","About"],
                            menu_icon="robot",
                            icons=['chat-dots-fill', 'image-fill', 'patch-question-fill','file-pdf-fill','file-person-fill'],
                            default_index=0
                        )
    
def translate_role_for_streamlit(user_role):
    if user_role == "model":
        return "assistant"
    else:
        return user_role

# chatbot page
if selected == 'ChatBot':
    model = load_gemini_pro_model()

    # Initialize chat session in Streamlit if not already present
    if "chat_session" not in st.session_state:  # Renamed for clarity
        st.session_state.chat_session = model.start_chat(history=[])

    # Display the chatbot's title on the page
    st.title("🤖 ChatBot")

    # Display the chat history
    for message in st.session_state.chat_session.history:
        with st.chat_message(translate_role_for_streamlit(message.role)):
            st.markdown(message.parts[0].text)

    # Input field for user's message
    user_prompt = st.chat_input("Ask Gemini-Pro...")  # Renamed for clarity
    if user_prompt:
        # Add user's message to chat and display it
        st.chat_message("user").markdown(user_prompt)

        # Send user's message to Gemini-Pro and get the response
        gemini_response = st.session_state.chat_session.send_message(user_prompt)  # Renamed for clarity

        # Display Gemini-Pro's response
        with st.chat_message("assistant"):
            st.markdown(gemini_response.text)


# image captioning page
if selected == "Image Captioning":

    st.title("📷 Image captioning")
    upload_image = st.file_uploader("upload an image...",type=["jpg,jpeg","png"])

    if st.button("generate caption"):
        image = Image.open(upload_image)

        col1,col2 = st.columns(2)

        with col1:
            resized_image = image.resize((800,500))
            st.image(resized_image)

        default_prompt = "write a short caption for this image"

        caption = gemini_pro_vision_response(default_prompt,image)

        with col2:
            st.info(caption)  

# Ask me anything page
if selected == "Ask me anything":
    st.title("❓Ask me a Question")

    user_prompt = st.text_area(label="",placeholder="Ask me a Question...")

    if st.button("Get an Answer"):
        response = gemini_pro_response(user_prompt)
        st.markdown(response)

if selected == 'Chat with PDF':

    st.header("📁 Chat with PDF 💁")

    pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)

    if st.button("Submit & Process"):
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Done")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

if selected == "About":
    about()