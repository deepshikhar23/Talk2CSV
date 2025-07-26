# app.py (Corrected Gradio Version)

import gradio as gr
import gradio.themes as themes # CORRECT way to import themes
import pandas as pd
from agent.agent_core import create_agent
import uuid

# --- State Management ---
# We use a simple dictionary to hold the state for each user session.
sessions = {}

def get_session_data(session_id: str):
    """Gets or creates session data for a user."""
    if session_id not in sessions:
        sessions[session_id] = {
            "agent_executor": None,
            "chat_history": [] # This will now be a list of dictionaries
        }
    return sessions[session_id]

# --- Handler Functions ---

def process_file(file, session_id: str):
    """
    This function is triggered when a user uploads a CSV file.
    It reads the CSV into a pandas DataFrame and initializes the agent for the session.
    """
    if file is None:
        return None, "Please upload a valid CSV file."

    session_data = get_session_data(session_id)
    try:
        df = pd.read_csv(file.name)
        session_data["agent_executor"] = create_agent(df)
        session_data["chat_history"] = [] # Reset history for the new file
        
        initial_message = f"File '{file.name.split('/')[-1]}' uploaded successfully with {df.shape[0]} rows. What would you like to know?"
        
        # The new format for the chatbot requires a list of dictionaries
        return [{"role": "assistant", "content": initial_message}], gr.update(placeholder="Ask a question about the data...", interactive=True)
    except Exception as e:
        error_message = f"Error processing file: {e}"
        return [{"role": "assistant", "content": error_message}], gr.update(interactive=False)


def chat_with_agent(user_message, history, session_id: str):
    """
    The main chat function. It takes the user's message, invokes the agent,
    and returns the agent's response.
    """
    session_data = get_session_data(session_id)
    agent_executor = session_data["agent_executor"]

    # Append the user's message to the history immediately
    history.append({"role": "user", "content": user_message})

    if agent_executor is None:
        history.append({"role": "assistant", "content": "Error: Please upload a CSV file first."})
        return history, ""

    try:
        # The agent expects a list of HumanMessage/AIMessage objects, not dictionaries.
        # We need to format the history for the agent, excluding the last user message which is the current input.
        chat_history_for_agent = []
        for msg in history[:-1]:
            if msg["role"] == "user":
                chat_history_for_agent.append({"role": "human", "content": msg["content"]})
            elif msg["role"] == "assistant":
                 chat_history_for_agent.append({"role": "ai", "content": msg["content"]})

        # Invoke the agent
        response = agent_executor.invoke({
            "input": user_message,
            "chat_history": chat_history_for_agent
        })
        
        bot_message = response.get("output", "Sorry, I encountered an error.")
        history.append({"role": "assistant", "content": bot_message})

    except Exception as e:
        history.append({"role": "assistant", "content": f"An error occurred: {e}"})

    # Clear the input box after responding
    return history, ""


def clear_all(session_id: str):
    """Clears the session data and resets the UI."""
    if session_id in sessions:
        del sessions[session_id]
    return [], gr.update(interactive=False, placeholder="Upload a CSV to begin..."), None


# --- Gradio UI Layout ---
with gr.Blocks(theme=themes.Soft(), title="Talk to CSV") as demo:
    # Hidden state to store a unique session ID for each user
    session_id_state = gr.State(lambda: str(uuid.uuid4()))

    gr.Markdown("# 📊 Talk to CSV")
    gr.Markdown("Upload a CSV file and ask questions about your data in plain English.")

    with gr.Row():
        with gr.Column(scale=1):
            file_uploader = gr.File(label="Upload your CSV file", file_types=[".csv"])
            clear_button = gr.Button("🗑️ Clear and Start Over")

        with gr.Column(scale=2):
            # Updated gr.Chatbot with type="messages" and removed deprecated parameter
            chatbot = gr.Chatbot(
                label="Conversation",
                type="messages",
                avatar_images=(None, "https://i.imgur.com/9kQ1Abw.png"),
            )
            chat_input = gr.Textbox(
                lines=1,
                placeholder="Upload a CSV to begin...",
                label="Ask your question",
                interactive=False
            )

    # --- Event Handling ---
    
    # When a file is uploaded, process it
    file_uploader.upload(
        process_file,
        inputs=[file_uploader, session_id_state],
        outputs=[chatbot, chat_input]
    )

    # When a user submits a message, call the chat function
    chat_input.submit(
        chat_with_agent,
        inputs=[chat_input, chatbot, session_id_state],
        outputs=[chatbot, chat_input] # chatbot is updated, chat_input is cleared
    )
    
    # When the clear button is clicked
    clear_button.click(
        clear_all,
        inputs=[session_id_state],
        outputs=[chatbot, chat_input, file_uploader]
    )


if __name__ == "__main__":
    demo.launch(debug=True, share=True)