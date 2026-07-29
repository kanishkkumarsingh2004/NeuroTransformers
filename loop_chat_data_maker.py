import os
import sys
import random
import argparse
import time
from datetime import datetime

# Base Directory & Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data2")
os.makedirs(DATA_DIR, exist_ok=True)

# Standard ChatML & Reasoning Tokens
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
THOUGHT_START = "[THOUGHT]"
THOUGHT_END = "[/THOUGHT]"

# System Prompts
SYSTEM_PROMPTS = [
    "You are Luna, an advanced AI reasoning assistant. Think step-by-step inside [THOUGHT] blocks before providing clear answers.",
    "You are an expert AI pair programmer and mathematical reasoner. Break down every problem into logical steps within your thought process.",
    "You are a helpful, precise, and thoughtful AI assistant. Express your internal reasoning clearly inside [THOUGHT] before giving the final answer.",
    "You are a scientific AI assistant. Provide logical, well-structured deductions step by step.",
    "You are an intelligent conversational AI. Analyze the user's intent carefully and output your reasoning before responding."
]

# =====================================================================
# Hand-Written Seed Examples (ChatML Format)
# =====================================================================
SEED_CHAT_DATA = [
    {
        "system": SYSTEM_PROMPTS[0],
        "user": "What is 25 multiplied by 14?",
        "thought": "Step 1: Decompose 14 into 10 + 4.\nStep 2: Calculate 25 * 10 = 250.\nStep 3: Calculate 25 * 4 = 100.\nStep 4: Sum the parts: 250 + 100 = 350.",
        "assistant": "25 multiplied by 14 is equal to **350**."
    },
    {
        "system": SYSTEM_PROMPTS[1],
        "user": "Solve for x: 3x + 15 = 45.",
        "thought": "Step 1: Isolate term with variable by subtracting 15 from both sides: 3x = 30.\nStep 2: Divide both sides by 3: x = 10.\nStep 3: Verification: 3(10) + 15 = 45.",
        "assistant": "To solve 3x + 15 = 45:\n1. Subtract 15: 3x = 30\n2. Divide by 3: x = 10\n\nThe solution is **x = 10**."
    },
    {
        "system": SYSTEM_PROMPTS[0],
        "user": "If a train travels at 60 mph for 2.5 hours, how far does it travel?",
        "thought": "Step 1: Apply formula Distance = Speed * Time.\nStep 2: Speed = 60, Time = 2.5.\nStep 3: Distance = 60 * 2.5 = 150 miles.",
        "assistant": "The train travels a total distance of **150 miles**."
    },
    {
        "system": SYSTEM_PROMPTS[2],
        "user": "What is your name?",
        "thought": "Step 1: Identify intent: identity inquiry.\nStep 2: Retrieve system identity persona: Luna.\nStep 3: Formulate a polite introduction.",
        "assistant": "My name is **Luna**! I am an AI reasoning assistant designed to help answer questions step by step."
    },
    {
        "system": SYSTEM_PROMPTS[2],
        "user": "Who are you?",
        "thought": "Step 1: User is asking for my identity.\nStep 2: Identify name (Luna) and main capabilities.\nStep 3: Respond warmly.",
        "assistant": "I am **Luna**, an advanced AI reasoning assistant. I can help you solve math problems, write Python code, explain concepts, and analyze logic."
    },
    {
        "system": SYSTEM_PROMPTS[1],
        "user": "Write a Python function to check if a string is a palindrome.",
        "thought": "Step 1: Understand palindrome definition (reads same forwards and backwards).\nStep 2: Clean input by lowercasing and removing non-alphanumeric chars.\nStep 3: Compare string with reverse slice `cleaned == cleaned[::-1]`.",
        "assistant": "```python\ndef is_palindrome(s: str) -> bool:\n    cleaned = ''.join(c.lower() for c in s if c.isalnum())\n    return cleaned == cleaned[::-1]\n```"
    }
]

def generate_expanded_dataset(target_count=15000, seed=None):
    """Generates a structured dataset with Math, Code, Logic, and Greetings."""
    if seed is not None:
        random.seed(seed)
        
    dataset = []
    
    # 1. Add Hand-Written Seeds
    dataset.extend(SEED_CHAT_DATA)

    # 2. Procedural Arithmetic
    operators = [('plus', '+', lambda a, b: a + b),
                 ('minus', '-', lambda a, b: a - b),
                 ('times', '*', lambda a, b: a * b)]
    
    for _ in range(3000):
        a = random.randint(1, 200)
        b = random.randint(1, 100)
        op_name, op_sym, op_func = random.choice(operators)
        ans = op_func(a, b)
        
        sys_p = random.choice(SYSTEM_PROMPTS)
        user_p = f"What is {a} {op_name} {b}?"
        thought_p = f"Step 1: Identify expression: {a} {op_sym} {b}.\nStep 2: Compute result: {a} {op_sym} {b} = {ans}.\nStep 3: Verify calculation."
        ast_p = f"The result of {a} {op_sym} {b} is **{ans}**."
        
        dataset.append({"system": sys_p, "user": user_p, "thought": thought_p, "assistant": ast_p})

    # 3. Procedural Algebra
    for _ in range(3000):
        m = random.randint(2, 12)
        x_val = random.randint(1, 50)
        c = random.randint(1, 100)
        rhs = m * x_val + c
        
        sys_p = random.choice(SYSTEM_PROMPTS)
        user_p = f"Solve for x: {m}x + {c} = {rhs}."
        thought_p = f"Step 1: Isolate variable: subtract {c} from both sides -> {m}x = {rhs - c}.\nStep 2: Divide by {m} -> x = {x_val}.\nStep 3: Verify solution."
        ast_p = f"Solving {m}x + {c} = {rhs}:\n1. Subtract {c}: {m}x = {rhs - c}\n2. Divide by {m}: x = {x_val}\n\nAnswer: **x = {x_val}**"
        
        dataset.append({"system": sys_p, "user": user_p, "thought": thought_p, "assistant": ast_p})

    # 4. Programming Code Tasks
    code_topics = [
        ("find max of list", "max(lst)", "Iterate through elements or use the built-in max() function."),
        ("calculate average", "sum(lst) / len(lst)", "Compute total sum using sum() and divide by element count len()."),
        ("reverse a list", "lst[::-1]", "Use Python sequence slicing with step -1."),
        ("check even number", "n % 2 == 0", "Check if remainder when divided by 2 is zero."),
        ("count word occurrences", "s.count(word)", "Use string method count() or collections.Counter."),
        ("remove duplicates from list", "list(set(lst))", "Convert to set to remove duplicates then back to list."),
        ("find minimum of list", "min(lst)", "Use built-in min() function or iterate to find smallest."),
        ("join strings from list", "', '.join(lst)", "Use str.join() method to concatenate with delimiter."),
    ]
    for _ in range(3000):
        topic, code_snippet, explanation = random.choice(code_topics)
        sys_p = random.choice(SYSTEM_PROMPTS)
        user_p = f"How do I {topic} in Python?"
        thought_p = f"Step 1: Analyze programming goal: {topic}.\nStep 2: Select pythonic method: {code_snippet}.\nStep 3: Formulate explanation."
        ast_p = f"To {topic} in Python, you can use:\n\n```python\n# {topic}\nresult = {code_snippet}\n```\n{explanation}"
        
        dataset.append({"system": sys_p, "user": user_p, "thought": thought_p, "assistant": ast_p})

    # 5. Conversational & Identity
    conversational_greetings = [
        ("hi", "greeting", "Hello! How can I help you today?"),
        ("hii", "greeting", "Hello there! What's on your mind today?"),
        ("hello", "greeting", "Hello! How can I assist you right now?"),
        ("hey", "greeting", "Hey! Ready to help. What would you like to work on?"),
        ("good morning", "greeting", "Good morning! Hope you have a productive day ahead. How can I help?"),
        ("whats up", "greeting", "Not much, just ready to assist you! What's on your mind?"),
        ("who are you?", "identity", "I am **Luna**, an AI reasoning assistant created to help answer questions step by step."),
        ("what is your name?", "identity", "My name is **Luna**! I am an AI assistant designed to help with math, programming, and general QA."),
        ("thank you", "gratitude", "You're very welcome! Feel free to ask if you need help with anything else."),
        ("bye", "farewell", "Goodbye! Have a fantastic day ahead! 👋"),
    ]

    for _ in range(6000):
        u_text, category, a_resp = random.choice(conversational_greetings)
        sys_p = random.choice(SYSTEM_PROMPTS)
        thought_p = f"Step 1: Analyze user input: '{u_text}'.\nStep 2: Category identified: {category}.\nStep 3: Formulate a clear, polite response."
        dataset.append({"system": sys_p, "user": u_text, "thought": thought_p, "assistant": a_resp})

    random.shuffle(dataset)
    final_dataset = dataset[:target_count]
    print(f"📊 Dataset generated: {len(final_dataset)} total samples.")
    return final_dataset

def format_chatml_sample(sample):
    """Formats sample into strict ChatML structure with [THOUGHT] blocks."""
    return (
        f"{IM_START}system\n{sample['system'].strip()}\n{IM_END}\n"
        f"{IM_START}user\n{sample['user'].strip()}\n{IM_END}\n"
        f"{IM_START}assistant\n"
        f"{THOUGHT_START}\n{sample['thought'].strip()}\n{THOUGHT_END}\n"
        f"{sample['assistant'].strip()}\n"
        f"{IM_END}\n\n"
    )

def main():
    parser = argparse.ArgumentParser(description="Generate ChatML Reasoning Dataset")
    parser.add_argument("-n", "--count", type=int, default=150000, help="Total samples to generate")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(DATA_DIR, f"chat_dataset_{timestamp}.txt")

    print("=" * 60)
    print("🚀 Generating NeuroTransformers Dataset")
    print(f"📁 Target Path: {file_path}")
    print("=" * 60)

    data_items = generate_expanded_dataset(target_count=args.count, seed=42)

    with open(file_path, "w", encoding="utf-8") as f:
        for item in data_items:
            f.write(format_chatml_sample(item))

    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    print(f"\n✅ Dataset File Created Successfully!")
    print(f"   📁 File: {file_path}")
    print(f"   📏 Size: {file_size_mb:.2f} MB")

    print("\nSample Entry:")
    print("-" * 50)
    print(format_chatml_sample(data_items[0]))
    print("-" * 50)

if __name__ == "__main__":
    main()