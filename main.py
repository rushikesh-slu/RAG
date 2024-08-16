import os
import gradio as gr
import PyPDF2
import docx
# from embeddings import load_and_split_document,query_rag
from embeddings import query_bq
import threading

multipleFilecontent = ''

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
    
    if file_extension.lower() == '.txt':
        return extract_txt(file_path)
    elif file_extension.lower() == '.pdf':
        return extract_pdf(file_path)
    # elif file_extension.lower() in ['.doc', '.docx']:
    #     return extract_doc(file_path)
    else:
        return "error"


import vertexai

from vertexai.generative_models import GenerativeModel, ChatSession



vertexai.init(project="durable-return-430917-b5")

model = GenerativeModel("gemini-1.5-flash-001")

chat = model.start_chat()
def get_chat_response(chat: ChatSession, prompt: str) -> str:
    text_response = []
    responses = chat.send_message(prompt, stream=True)
    for chunk in responses:
        text_response.append(chunk.text)
    return "".join(text_response)

# def generateEmbeddings(file_path):
#     load_and_split_document(file_path)


def process_message(prompt):
    # print(prompt)
    global multipleFilecontent
    multipleFilecontent = ''
    print(prompt)
    return get_chat_response(chat, prompt)
    # return "Output from Gemini API."


messages = [("system", "You are a helpful assistant")]

def respond(history, text_input,files,rag_check):
    """Handles the chat interaction with the language model."""
    global messages
    global multipleFilecontent
    
    file_names = []
    if files:
        if isinstance(files, list):  # If multiple files are uploaded
            file_names = [file.name for file in files]
        else:  # If a single file is uploaded
            file_names = [files.name]
        
    message = {'text':text_input,'files':file_names}
    print("Message received:", message)

    # Handle file uploads
    if message["files"]:
        try:
            for filepath in message["files"]:
            # filepath = message["files"][0]
                file_content = extract_file_content(filepath)
                if file_content == 'error':
                    history.append((message["text"], 'Error occur while parsing the file, only pdf,doc and text files allowed.'))
                    file_content = ''
                    # return history, gr.MultimodalTextbox(value=None, interactive=True)
                    return history, gr.update(value=""), gr.update(value=None)
                else:
                #    background_thread = threading.Thread(
                #     target=generateEmbeddings,
                #     args=(filepath,),  # Pass arguments as a tuple
                #     daemon=True)
                #    background_thread.start()
                   multipleFilecontent += f"Filename : {filepath}, File content: {file_content}\n\n"
                    
                    
        except Exception as e:
            print(f"Error handling file : {e}")
        
        

    # Handle text input
    if message["text"]:
        try:
            # Append user input to messages
            messages.append(("human", message["text"]))
            prompt = message['text']
            if multipleFilecontent:
                
                prompt = f"Based on the following content, {multipleFilecontent},\n answer {message['text']}"
                file_content = ''
            if not rag_check:
                context = query_bq(message['text'])
                if multipleFilecontent:
                    prompt = f"Based on the following content, {multipleFilecontent} and {context},\n answer {message['text']}"
                else:
                    prompt = f"Based on the following content, {context},\n answer {message['text']}"

                

            response_content = process_message(prompt)
            messages.append(("assistant", response_content))
        
            # Append to chat history for display in the UI
            history.append((message["text"], response_content))
        except Exception as e:
            print(f"Error handling text input: {e}")
    
    else:
        if file_names:
            files_str = ",".join(os.path.basename(file_names))
            history.append(("Files uploaded: {}".format(files_str), "How can I assist with the files?"))
        # return history, gr.MultimodalTextbox(value=None, interactive=True)
        return history, gr.update(value=""), gr.update(value=None)

    # return history, gr.MultimodalTextbox(value=None, interactive=True)
    return history, gr.update(value=""), gr.update(value=None)


css = """
.small-file-upload > .file-upload {
    height: 30px !important;
    line-height: 30px !important;
    border-radius: 4px !important;
}
.small-file-upload > .file-upload::before {
    height: 28px !important;
    line-height: 28px !important;
    font-size: 12px !important;
}
.small-file-upload > .file-upload input[type="file"] {
    height: 30px !important;
}
"""


with gr.Blocks(css = css,fill_height=True) as demo:
    chatbot = gr.Chatbot(
        elem_id="chatbot",
        bubble_full_width=False,
        scale=1,
    )

    # chat_input = gr.MultimodalTextbox(interactive=True,
    #                                   file_count="multiple",
    #                                   placeholder="Enter message or upload file...", show_label=False)

    # chat_input = gr.MultimodalTextbox(
    #     interactive=True,
    #     file_types=[".pdf", ".doc", ".docx", ".txt"],
    #     file_count="single",
    #     placeholder="Enter message or upload file...",
    #     show_label=False
        
    # )
    # print(type(chat_input))
    # chat_msg = chat_input.submit(respond, [chatbot, chat_input], [chatbot, chat_input])
    # chat_msg.then(lambda: gr.MultimodalTextbox(interactive=True), None, [chat_input])


    with gr.Row():
        with gr.Column(scale=5):
            checkbox_input = gr.Checkbox(label="Not related to the domain", scale=1)
            text_input = gr.Textbox(placeholder="Enter message...", show_label=False)
        with gr.Column(scale = 1):
            file_input = gr.File(
                file_count="multiple", 
                file_types=[".pdf", ".txt"], 
                label="Upload files", 
                scale=1,
                elem_classes="small-file-upload"
            )

    chat_msg = gr.Button("Submit")
    chat_msg.click(respond, [chatbot, text_input,file_input,checkbox_input], [chatbot, text_input,file_input])#{'text':text_input,'files':file_input}
    


demo.queue()
demo.launch(share=True)






