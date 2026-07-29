import os
import random

# =====================================================================
# CONFIGURATION
# =====================================================================
NUMBER_OF_QUESTIONS = 100  # Adjust this parameter to change the total question count

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA4_DIR = os.path.join(BASE_DIR, "data4")
os.makedirs(DATA4_DIR, exist_ok=True)
FILE_PATH = os.path.join(DATA4_DIR, "questions.txt")

# =====================================================================
# Seed Question Templates
# =====================================================================
MATH_TEMPLATES = [
    "What is {a} plus {b}?",
    "What is {a} minus {b}?",
    "What is {a} multiplied by {b}?",
    "Solve for x: {a}x + {b} = {c}.",
    "What is {perc}% of {val}?",
    "Calculate the square root of {sq_val}.",
    "What is {a} divided by {b}?",
    "If a car travels at {speed} mph for {time} hours, how far does it go?",
    "What is the perimeter of a rectangle with length {a} and width {b}?",
    "What is the area of a circle with radius {r}?"
]

CODING_TEMPLATES = [
    "How do I {task} in Python?",
    "Write a Python function to {task}.",
    "What is the best way to {task} in JavaScript?",
    "Explain the difference between {concept1} and {concept2} in programming.",
    "How do I handle errors when trying to {task} in Python?",
    "What is a {concept1} in computer science?",
    "How can I optimize a function that performs {task}?"
]

CODING_TASKS = [
    "reverse a string", "check if a number is prime", "sort a list of integers",
    "read a file line by line", "flatten a nested list", "merge two dictionaries",
    "calculate the factorial of a number", "find the maximum value in an array",
    "count the frequency of words in a text", "check for palindromes",
    "remove duplicates from a list", "swap two variables without a temp variable",
    "parse a JSON string", "make an HTTP GET request"
]

CONCEPTS = [
    ("a list", "a tuple"), ("== (equality)", "is (identity)"), 
    ("deep copy", "shallow copy"), ("stack", "queue"), 
    ("process", "thread"), ("BFS", "DFS"), 
    ("recursion", "iteration"), ("mutable", "immutable")
]

SCIENCE_TECH_TEMPLATES = [
    "What is {tech_topic}?",
    "How does {tech_topic} work in practice?",
    "Why is {tech_topic} important in modern technology?",
    "Explain the concept of {tech_topic} simply.",
    "What are the main applications of {tech_topic}?"
]

TECH_TOPICS = [
    "Machine Learning", "Deep Learning", "Artificial Neural Networks",
    "Transformer Models", "Self-Attention Mechanisms", "Tokenization in NLP",
    "Byte-Pair Encoding (BPE)", "Model Fine-Tuning", "Gradient Descent",
    "Overfitting", "Knowledge Distillation", "Large Language Models (LLMs)",
    "Prompt Engineering", "Cloud Computing", "APIs", "Docker Containerization",
    "GPU Acceleration", "Quantum Computing", "Reinforcement Learning"
]

CONVERSATIONAL_TEMPLATES = [
    "Hi, how are you today?",
    "Who are you and what can you do?",
    "What is your name?",
    "Can you help me solve a problem step by step?",
    "Good morning! Are you ready to help me?",
    "Tell me an interesting fun fact.",
    "Thank you for your help!",
    "What programming languages do you understand?",
    "How do you process reasoning tasks?"
]

# =====================================================================
# Question Generator Loop
# =====================================================================
def generate_questions(total_count):
    questions = set()

    # Calculate target category distributions dynamically
    target_math = int(total_count * 0.35)
    target_coding = int(total_count * 0.35)
    target_tech = int(total_count * 0.20)

    # 1. Procedural Math Questions
    while len(questions) < target_math:
        a = random.randint(5, 500)
        b = random.randint(2, 200)
        c = random.randint(10, 1000)
        r = random.randint(1, 50)
        perc = random.choice([5, 10, 15, 20, 25, 50, 75])
        val = random.randint(10, 1000)
        sq_val = random.choice([4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 225, 256, 400, 625, 900, 10000])
        speed = random.choice([30, 40, 50, 60, 70, 80])
        time_h = random.choice([1, 2, 3, 4, 5, 2.5, 1.5])

        tmpl = random.choice(MATH_TEMPLATES)
        q = tmpl.format(a=a, b=b, c=c, r=r, perc=perc, val=val, sq_val=sq_val, speed=speed, time=time_h)
        questions.add(q)

    # 2. Coding Questions
    while len(questions) < (target_math + target_coding):
        task = random.choice(CODING_TASKS)
        c1, c2 = random.choice(CONCEPTS)
        tmpl = random.choice(CODING_TEMPLATES)
        q = tmpl.format(task=task, concept1=c1, concept2=c2)
        questions.add(q)

    # 3. Science & Tech Questions
    while len(questions) < (target_math + target_coding + target_tech):
        topic = random.choice(TECH_TOPICS)
        tmpl = random.choice(SCIENCE_TECH_TEMPLATES)
        q = tmpl.format(tech_topic=topic)
        questions.add(q)

    # 4. Conversational / Fill Remaining up to NUMBER_OF_QUESTIONS
    var_counter = 1
    while len(questions) < total_count:
        q = random.choice(CONVERSATIONAL_TEMPLATES)
        if q in questions:
            q = f"{q} (Variation {var_counter})"
            var_counter += 1
        questions.add(q)

    return list(questions)

# =====================================================================
# Main Script Execution
# =====================================================================
def main():
    print(f"🚀 Generating {NUMBER_OF_QUESTIONS} synthetic questions...")
    question_list = generate_questions(NUMBER_OF_QUESTIONS)

    print(f"✍️ Writing questions to: {FILE_PATH}")
    with open(FILE_PATH, "w", encoding="utf-8") as f:
        for q in question_list:
            f.write(q + "\n")

    print(f"✅ Success! Generated {len(question_list)} unique questions in 'data4/questions.txt'.")

if __name__ == "__main__":
    main()