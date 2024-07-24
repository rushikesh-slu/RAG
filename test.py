import os
import gradio as gr
import PyPDF2
import docx

file_content = ''

def extract_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def extract_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text.encode('utf-8').decode('utf-8', 'ignore')

def extract_doc(file_path):
    doc = docx.Document(file_path)
    full_text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
    return full_text.encode('utf-8').decode('utf-8', 'ignore')

def extract_file_content(file_path):
    _, file_extension = os.path.splitext(file_path)
    print("----------------")
    if file_extension.lower() == '.txt':
        return extract_txt(file_path)
    elif file_extension.lower() == '.pdf':
        return extract_pdf(file_path)
    elif file_extension.lower() in ['.doc', '.docx']:
        return extract_doc(file_path)
    else:
        return "error"


import vertexai

from vertexai.generative_models import GenerativeModel, ChatSession



vertexai.init(project="myprojectrag")

model = GenerativeModel("gemini-1.5-flash-001")

chat = model.start_chat()
def get_chat_response(chat: ChatSession, prompt: str) -> str:
    text_response = []
    responses = chat.send_message(prompt, stream=True)
    for chunk in responses:
        text_response.append(chunk.text)
    return "".join(text_response)


def process_message(prompt):
    # print(prompt)
    print(prompt)
    return get_chat_response(chat, prompt)


messages = [("system", "You are a helpful assistant")]

def respond(history, message):
    """Handles the chat interaction with the language model."""
    global messages
    global file_content
    print("Message received:", message)

    # Handle file uploads
    if message["files"]:
        try:
            filepath = message["files"][0]
            file_content = extract_file_content(filepath)
            if file_content == 'error':
                history.append((message["text"], 'Error occur while parsing the file, only pdf,doc and text files allowed.'))
                file_content = ''
                return history, gr.MultimodalTextbox(value=None, interactive=True)
            else:
                if message['text']:
                    prompt = f"Based on the following content from the file, {file_content},\n answer the below question,\n {message['text']}"
                    process_message(prompt)
                    file_content = ''
                else:
                    history.append(("File uploaded: {}".format(os.path.basename(filepath)), "How can I assist with the file?"))
                    return history, gr.MultimodalTextbox(value=None, interactive=True)
                    
                    
        except Exception as e:
            print(f"Error handling file : {e}")
        
        

    # Handle text input
    if message["text"]:
        try:
            # Append user input to messages
            messages.append(("human", message["text"]))
            prompt = message['text']
            if file_content:
                prompt = f"Based on the following content, {file_content},\n answer {message['text']}"
                file_content = ''
            response_content = process_message(prompt)
            messages.append(("assistant", response_content))
        
            # Append to chat history for display in the UI
            history.append((message["text"], response_content))
        except Exception as e:
            print(f"Error handling text input: {e}")

    return history, gr.MultimodalTextbox(value=None, interactive=True)



with gr.Blocks(fill_height=True) as demo:
    chatbot = gr.Chatbot(
        elem_id="chatbot",
        bubble_full_width=False,
        scale=1,
    )

    # chat_input = gr.MultimodalTextbox(interactive=True,
    #                                   file_count="multiple",
    #                                   placeholder="Enter message or upload file...", show_label=False)

    chat_input = gr.MultimodalTextbox(
        interactive=True,
        file_types=[".pdf", ".doc", ".docx", ".txt"],
        file_count="single",
        placeholder="Enter message or upload file...",
        show_label=False
        
    )

    chat_msg = chat_input.submit(respond, [chatbot, chat_input], [chatbot, chat_input])
    chat_msg.then(lambda: gr.MultimodalTextbox(interactive=True), None, [chat_input])


demo.queue()
demo.launch(share=True)




