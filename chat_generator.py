import os
import requests
from datetime import datetime

# =====================================================================
# CONFIGURATION
# =====================================================================
OLLAMA_BASE_URL = "http://localhost:11434/api"
OLLAMA_MODEL_NAME = "gemma4:12b"  # Replace with your active Ollama model if different

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA4_DIR = os.path.join(BASE_DIR, "data4")
os.makedirs(DATA4_DIR, exist_ok=True)

# Dataset file where live conversation will be saved
# timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_FILE = os.path.join(DATA4_DIR, f"live_chat_dataset.txt")

# Special Tokens
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
THOUGHT_START = "[THOUGHT]"
THOUGHT_END = "[/THOUGHT]"

SYSTEM_INSTRUCTION = "You are Luna, an advanced AI reasoning assistant. Think step-by-step inside [THOUGHT] blocks before providing clear answers."


# Terminal Colors for Live Chat
class TerminalColors:
    GREY = '\033[90m'      # Dim Grey for Thought
    GREEN = '\033[92m'     # Green for Assistant Response
    BLUE = '\033[94m'      # Blue for User Prompt
    RESET = '\033[0m'      # Reset formatting


def check_ollama_status():
    """Verify if Ollama service is active."""
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/tags")
        if r.status_code == 200:
            models = [m.get("name") for m in r.json().get("models", [])]
            return any(OLLAMA_MODEL_NAME in m for m in models)
        return False
    except Exception:
        return False


def query_ollama_live(user_input):
    """Sends user query to Ollama enforcing step-by-step reasoning with unlimited token output."""
    prompt_instruction = (
        f"Question: {user_input}\n\n"
        "Instructions:\n"
        "1. First, write step-by-step reasoning inside [THOUGHT] ... [/THOUGHT] blocks.\n"
        "2. Then, provide the final answer right after [/THOUGHT].\n"
    )

    try:
        payload = {
            "model": OLLAMA_MODEL_NAME,
            "prompt": prompt_instruction,
            "stream": False,
            "options": {
                "num_predict": -1,  # -1 disables output token limit for full response generation
                "temperature": 0.4,
            }
        }
        r = requests.post(f"{OLLAMA_BASE_URL}/generate", json=payload)
        if r.status_code == 200:
            return r.json().get("response", "").strip()
        return None
    except Exception as e:
        print(f"\n❌ Ollama Request Error: {e}")
        return None


def format_and_save_turn(user_input, raw_response, file_handle):
    """Formats conversation turn with special ChatML tokens and appends to .txt dataset."""
    # Ensure [THOUGHT] tags are properly structured
    if THOUGHT_START not in raw_response:
        raw_response = f"{THOUGHT_START}\nStep 1: Process query '{user_input}'.\nStep 2: Formulate response.\n{THOUGHT_END}\n" + raw_response

    if THOUGHT_END not in raw_response and THOUGHT_START in raw_response:
        parts = raw_response.split("\n\n", 1)
        if len(parts) > 1:
            raw_response = f"{parts[0]}\n{THOUGHT_END}\n\n{parts[1]}"
        else:
            raw_response = f"{raw_response}\n{THOUGHT_END}"

    # Build ChatML text structure
    chatml_entry = (
        f"{IM_START}system\n{SYSTEM_INSTRUCTION}\n{IM_END}\n"
        f"{IM_START}user\n{user_input.strip()}\n{IM_END}\n"
        f"{IM_START}assistant\n{raw_response.strip()}\n{IM_END}\n\n"
    )

    # Save immediately to text file
    file_handle.write(chatml_entry)
    file_handle.flush()


def display_formatted_response(raw_response):
    """Prints response in terminal with Grey Thoughts and Green Answer."""
    if THOUGHT_START in raw_response and THOUGHT_END in raw_response:
        parts = raw_response.split(THOUGHT_END, 1)
        thought_text = parts[0].replace(THOUGHT_START, "").strip()
        answer_text = parts[1].strip()

        print(f"{TerminalColors.GREY}[THOUGHT]\n{thought_text}\n[/THOUGHT]{TerminalColors.RESET}\n")
        print(f"{TerminalColors.GREEN}{answer_text}{TerminalColors.RESET}\n")
    else:
        # Fallback if tags are missing
        print(f"{TerminalColors.GREEN}{raw_response}{TerminalColors.RESET}\n")


def run_interactive_session():
    print("=" * 65)
    print("🤖 Live Interactive Ollama Chat & Dataset Recorder")
    print(f"📁 Saving ChatML Dataset to: '{OUTPUT_FILE}'")
    print("💡 Commands: Type 'exit' or 'quit' to end session.")
    print("=" * 65 + "\n")

    if not check_ollama_status():
        print(f"❌ Error: Could not connect to Ollama server or model '{OLLAMA_MODEL_NAME}' is missing.")
        print("   Make sure Ollama is active (`ollama serve`).")
        return

    print(f"✅ Active Ollama Model: '{OLLAMA_MODEL_NAME}'\n")

    # Open file for real-time appending
    with open(OUTPUT_FILE, "a", encoding="utf-8") as dataset_file:
        while True:
            try:
                user_input = input(f"{TerminalColors.BLUE}YOU: {TerminalColors.RESET}").strip()
            except (KeyboardInterrupt, EOFError):
                print("\n\nExiting session. Goodbye!")
                break

            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye! Your chat session has been saved.")
                break

            print(f"{TerminalColors.GREEN}OLLAMA:{TerminalColors.RESET} ", end="", flush=True)
            raw_response = query_ollama_live(user_input)

            if raw_response:
                print()
                display_formatted_response(raw_response)
                format_and_save_turn(user_input, raw_response, dataset_file)
            else:
                print("❌ Failed to get response from Ollama.\n")


if __name__ == "__main__":
    run_interactive_session()