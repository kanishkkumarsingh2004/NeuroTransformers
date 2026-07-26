import os
import sys
import random
import argparse
import time
from datetime import datetime

# Base Directory & Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data4")
os.makedirs(DATA_DIR, exist_ok=True)

# Standard Special Tokens
BOS = "[BOS]"
EOS = "[EOS]"
SYSTEM = "[SYSTEM]"
USER = "[USER]"
THOUGHT = "[THOUGHT]"
ASSISTANT = "[ASSISTANT]"

# System Prompts
SYSTEM_PROMPTS = [
    "You are NeuroBot, an advanced AI reasoning assistant created by DeepMind engineers. Think step by step before answering.",
    "You are an expert AI pair programmer and mathematical reasoner. Break down every problem into logical steps within your thought process.",
    "You are a helpful, precise, and thoughtful AI assistant. Express your internal reasoning clearly inside [THOUGHT] before giving the final answer.",
    "You are a scientific AI assistant. Provide logical, well-structured deductions step by step.",
    "You are an intelligent conversational AI. Analyze the user's intent carefully and output your reasoning before responding."
]

# =====================================================================
# 100 Hand-Written, Multi-Domain Seed Examples
# =====================================================================
SEED_CHAT_DATA = [
    # ---- MATH & ARITHMETIC (1-20) ----
    {
        "system": SYSTEM_PROMPTS[0],
        "user": "What is 25 multiplied by 14?",
        "thought": "The user is asking for 25 * 14.\nI can decompose 14 into 10 + 4.\n25 * 10 = 250.\n25 * 4 = 100.\n250 + 100 = 350.\nBoth methods yield 350.",
        "assistant": "25 multiplied by 14 is equal to **350**."
    },
    {
        "system": SYSTEM_PROMPTS[1],
        "user": "Solve for x: 3x + 15 = 45.",
        "thought": "Equation: 3x + 15 = 45.\nSubtract 15 from both sides: 3x = 30.\nDivide both sides by 3: x = 10.\nVerification: 3(10) + 15 = 45. Correct.",
        "assistant": "To solve 3x + 15 = 45:\n1. Subtract 15: 3x = 30\n2. Divide by 3: x = 10\n\nThe solution is **x = 10**."
    },
    {
        "system": SYSTEM_PROMPTS[0],
        "user": "If a train travels at 60 mph for 2.5 hours, how far does it travel?",
        "thought": "Formula: Distance = Speed * Time.\nSpeed = 60 mph. Time = 2.5 hours.\nDistance = 60 * 2.5 = 150 miles.",
        "assistant": "The train travels a total distance of **150 miles**."
    },
    {
        "system": SYSTEM_PROMPTS[0],
        "user": "What is the square root of 144?",
        "thought": "Square root of 144.\n12 * 12 = 144.\nSo sqrt(144) = 12.",
        "assistant": "The square root of 144 is **12**."
    },
    {
        "system": SYSTEM_PROMPTS[1],
        "user": "Calculate the area of a circle with radius 7.",
        "thought": "Area of a circle = pi * r^2.\nRadius = 7.\nArea = pi * 49 = 49 * 3.14159... = 153.94 (approx).",
        "assistant": "The area of a circle with radius 7 is approximately **153.94 square units** (49π)."
    },
    {
        "system": SYSTEM_PROMPTS[0],
        "user": "What is 15% of 200?",
        "thought": "15% of 200 = (15/100) * 200 = 0.15 * 200 = 30.",
        "assistant": "15% of 200 is **30**."
    },
    {
        "system": SYSTEM_PROMPTS[1],
        "user": "What is the sum of the first 10 natural numbers?",
        "thought": "Using formula: n(n+1)/2 where n = 10.\n10 * 11 / 2 = 110 / 2 = 55.",
        "assistant": "The sum of the first 10 natural numbers is **55**."
    },
    {
        "system": SYSTEM_PROMPTS[0],
        "user": "Convert 5 kilometers to meters.",
        "thought": "1 kilometer = 1000 meters.\n5 km = 5 * 1000 = 5000 meters.",
        "assistant": "5 kilometers equals **5000 meters**."
    },
    {
        "system": SYSTEM_PROMPTS[1],
        "user": "What is 2 to the power of 10?",
        "thought": "2^10 = 2*2*2*2*2*2*2*2*2*2.\n2^1=2, 2^2=4, 2^3=8, 2^4=16, 2^5=32, 2^6=64, 2^7=128, 2^8=256, 2^9=512, 2^10=1024.",
        "assistant": "2 to the power of 10 is **1024**."
    },
    {
        "system": SYSTEM_PROMPTS[0],
        "user": "If I have 3 apples and buy 7 more, how many apples do I have?",
        "thought": "Starting amount: 3 apples.\nBought: 7 more apples.\nTotal = 3 + 7 = 10 apples.",
        "assistant": "You would have **10 apples** in total."
    },
    {
        "system": SYSTEM_PROMPTS[1],
        "user": "What is the greatest common divisor of 24 and 36?",
        "thought": "Factors of 24: 1, 2, 3, 4, 6, 8, 12, 24.\nFactors of 36: 1, 2, 3, 4, 6, 9, 12, 18, 36.\nCommon factors: 1, 2, 3, 4, 6, 12.\nGreatest common divisor = 12.",
        "assistant": "The greatest common divisor (GCD) of 24 and 36 is **12**."
    },
    {
        "system": SYSTEM_PROMPTS[0],
        "user": "Simplify the fraction 18/24.",
        "thought": "Find GCD of 18 and 24.\nFactors of 18: 1,2,3,6,9,18. Factors of 24: 1,2,3,4,6,8,12,24.\nGCD = 6.\n18/6 = 3, 24/6 = 4.\nSimplified: 3/4.",
        "assistant": "The fraction 18/24 simplified is **3/4**."
    },
    {
        "system": SYSTEM_PROMPTS[1],
        "user": "What is the perimeter of a rectangle with length 8 and width 5?",
        "thought": "Perimeter = 2 * (length + width).\nP = 2 * (8 + 5) = 2 * 13 = 26.",
        "assistant": "The perimeter of the rectangle is **26 units**."
    },
    {
        "system": SYSTEM_PROMPTS[0],
        "user": "What is the absolute value of -42?",
        "thought": "Absolute value removes the negative sign.\n|-42| = 42.",
        "assistant": "The absolute value of -42 is **42**."
    },
    {
        "system": SYSTEM_PROMPTS[1],
        "user": "Solve: 5! (5 factorial).",
        "thought": "5! = 5 * 4 * 3 * 2 * 1.\n5*4 = 20. 20*3 = 60. 60*2 = 120. 120*1 = 120.",
        "assistant": "5 factorial (5!) equals **120**."
    },
    {
        "system": SYSTEM_PROMPTS[0],
        "user": "What is the value of pi to 5 decimal places?",
        "thought": "Pi (π) is the ratio of circumference to diameter.\nπ = 3.14159265...\nTo 5 decimal places: 3.14159.",
        "assistant": "The value of pi to 5 decimal places is **3.14159**."
    },
    {
        "system": SYSTEM_PROMPTS[1],
        "user": "How many seconds are in one hour?",
        "thought": "1 hour = 60 minutes.\n1 minute = 60 seconds.\n1 hour = 60 * 60 = 3600 seconds.",
        "assistant": "There are **3,600 seconds** in one hour."
    },
    {
        "system": SYSTEM_PROMPTS[0],
        "user": "What is the LCM of 4 and 6?",
        "thought": "Multiples of 4: 4, 8, 12, 16, 20...\nMultiples of 6: 6, 12, 18, 24...\nFirst common multiple: 12.\nAlternatively LCM = (4*6)/GCD(4,6) = 24/2 = 12.",
        "assistant": "The least common multiple (LCM) of 4 and 6 is **12**."
    },
    {
        "system": SYSTEM_PROMPTS[1],
        "user": "What is 7 cubed?",
        "thought": "7 cubed = 7^3 = 7 * 7 * 7.\n7*7 = 49. 49*7 = 343.",
        "assistant": "7 cubed (7³) is **343**."
    },
    {
        "system": SYSTEM_PROMPTS[0],
        "user": "Convert 32 degrees Fahrenheit to Celsius.",
        "thought": "Formula: C = (F - 32) * 5/9.\nC = (32 - 32) * 5/9 = 0 * 5/9 = 0.\n32°F is the freezing point of water = 0°C.",
        "assistant": "32°F is equal to **0°C** (the freezing point of water)."
    },

    # ---- LOGIC & REASONING (21-40) ----
    {
        "system": SYSTEM_PROMPTS[2],
        "user": "All humans are mortal. Socrates is a human. What can we conclude?",
        "thought": "This is a classic categorical syllogism.\nPremise 1: All humans are mortal.\nPremise 2: Socrates is a human.\nConclusion: Socrates is mortal.\nThe deduction is valid and sound.",
        "assistant": "By deductive reasoning, we can conclude that **Socrates is mortal**."
    },
    {
        "system": SYSTEM_PROMPTS[2],
        "user": "Which weighs more: a pound of feathers or a pound of bricks?",
        "thought": "Both are measured as 'a pound'. A pound is a unit of weight. Since both weigh exactly one pound, they weigh the same.",
        "assistant": "They both weigh **exactly the same**: one pound!"
    },
    {
        "system": SYSTEM_PROMPTS[4],
        "user": "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
        "thought": "Each machine makes 1 widget in 5 minutes.\n100 machines each make 1 widget in 5 minutes.\nSo 100 machines make 100 widgets in 5 minutes.",
        "assistant": "It would still take **5 minutes**. Each machine makes one widget in 5 minutes, so 100 machines working in parallel make 100 widgets in 5 minutes."
    },
    {
        "system": SYSTEM_PROMPTS[2],
        "user": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
        "thought": "Let ball = x. Bat = x + 1.00.\nx + (x + 1.00) = 1.10.\n2x + 1.00 = 1.10.\n2x = 0.10.\nx = 0.05.\nBall = $0.05. Bat = $1.05. Check: 1.05 + 0.05 = 1.10. Correct.",
        "assistant": "The ball costs **$0.05** (5 cents). The bat costs $1.05."
    },
    {
        "system": SYSTEM_PROMPTS[4],
        "user": "Is it possible for a man to marry his widow's sister?",
        "thought": "A widow is a woman whose husband has died.\nIf a man has a widow, that means the man is dead.\nA dead man cannot marry anyone.\nTherefore, no, it is not possible.",
        "assistant": "No, it is **not possible**. If the man has a widow, he must be dead, and a dead person cannot marry."
    },
    {
        "system": SYSTEM_PROMPTS[2],
        "user": "What comes next in the sequence: 2, 4, 8, 16, ...?",
        "thought": "Pattern analysis: each number is multiplied by 2.\n2*2=4, 4*2=8, 8*2=16.\nNext: 16*2 = 32.",
        "assistant": "The next number in the sequence is **32**. Each number is doubled (multiplied by 2)."
    },
    {
        "system": SYSTEM_PROMPTS[4],
        "user": "If you rearrange the letters 'CIFAIPC', you get the name of a what?",
        "thought": "Letters: C, I, F, A, I, P, C.\nRearranging: P-A-C-I-F-I-C.\nPACIFIC is an ocean.",
        "assistant": "Rearranging 'CIFAIPC' gives you **PACIFIC**, which is an ocean."
    },
    {
        "system": SYSTEM_PROMPTS[2],
        "user": "What has keys but no locks, space but no room, and you can enter but can't go inside?",
        "thought": "This is a riddle. Keys but no locks: keyboard keys. Space but no room: space bar. You can enter: Enter key.\nAnswer: a keyboard.",
        "assistant": "The answer is a **keyboard**! It has keys (but no locks), a space bar (but no room), and an Enter key (but you can't go inside it)."
    },
    {
        "system": SYSTEM_PROMPTS[4],
        "user": "If there are 3 apples and you take away 2, how many do you have?",
        "thought": "Trick question: 'you take away 2' means YOU now have 2 apples. The remaining pile has 1, but you have the 2 you took.",
        "assistant": "You have **2 apples** — the ones you took!"
    },
    {
        "system": SYSTEM_PROMPTS[2],
        "user": "A farmer has 17 sheep. All but 9 die. How many sheep does the farmer have left?",
        "thought": "'All but 9 die' means 9 survive.\nSo the farmer has 9 sheep left.",
        "assistant": "The farmer has **9 sheep** left. 'All but 9' means 9 survived."
    },
    {
        "system": SYSTEM_PROMPTS[4],
        "user": "How many times can you subtract 5 from 25?",
        "thought": "You can subtract 5 from 25 once. After that you subtract 5 from 20, not 25.\nSo the answer is once.",
        "assistant": "You can subtract 5 from 25 **once**. After that, it becomes 20, not 25."
    },
    {
        "system": SYSTEM_PROMPTS[2],
        "user": "What is the next prime number after 7?",
        "thought": "Primes: 2, 3, 5, 7, ...\n8 = 2*4, not prime. 9 = 3*3, not prime. 10 = 2*5, not prime. 11: check divisibility by 2,3 — neither divides 11.\n11 is prime.",
        "assistant": "The next prime number after 7 is **11**."
    },
    {
        "system": SYSTEM_PROMPTS[4],
        "user": "If two's company and three's a crowd, what are four and five?",
        "thought": "This is a wordplay trick question.\n4 + 5 = 9.\nThe answer is simply nine.",
        "assistant": "Four and five is **nine** (4 + 5 = 9)!"
    },
    {
        "system": SYSTEM_PROMPTS[2],
        "user": "Complete the pattern: 1, 1, 2, 3, 5, 8, ...?",
        "thought": "This is the Fibonacci sequence.\nEach number is the sum of the two preceding numbers.\n5 + 8 = 13.\nNext number is 13.",
        "assistant": "The next number is **13**. This is the Fibonacci sequence where each number is the sum of the two before it."
    },
    {
        "system": SYSTEM_PROMPTS[4],
        "user": "What is heavier, a kilogram of steel or a kilogram of cotton?",
        "thought": "Both are 1 kilogram. A kilogram is a kilogram regardless of the material.\nThey weigh the same.",
        "assistant": "They weigh **the same** — both are exactly one kilogram."
    },
    {
        "system": SYSTEM_PROMPTS[2],
        "user": "If you have a bowl with six apples and you take four, how many do you have?",
        "thought": "You took 4 apples. So you have 4 apples.\nThe bowl has 2 remaining, but you personally have 4.",
        "assistant": "You have **4 apples** — the ones you took."
    },
    {
        "system": SYSTEM_PROMPTS[4],
        "user": "What goes up but never comes down?",
        "thought": "This is a riddle. Things that only increase: your age.\nYour age always goes up and never comes down.",
        "assistant": "The answer is your **age** — it always goes up but never comes down."
    },
    {
        "system": SYSTEM_PROMPTS[2],
        "user": "How many months have 28 days?",
        "thought": "Trick question. ALL 12 months have at least 28 days.\nNot just February.",
        "assistant": "**All 12 months** have at least 28 days!"
    },
    {
        "system": SYSTEM_PROMPTS[4],
        "user": "What can you hold in your right hand but never in your left hand?",
        "thought": "You cannot hold your left hand in your left hand — your left elbow is the closest.\nBut you can hold your left hand in your right hand.",
        "assistant": "Your **left hand**! You can hold it in your right hand but never in your left."
    },
    {
        "system": SYSTEM_PROMPTS[2],
        "user": "If a doctor gives you 3 pills and says take one every half hour, how long until all pills are gone?",
        "thought": "Take pill 1 at time 0.\nTake pill 2 at 30 minutes.\nTake pill 3 at 60 minutes.\nTotal time: 1 hour.",
        "assistant": "It takes **1 hour** (60 minutes). You take one immediately, one at 30 minutes, and the last at 60 minutes."
    },

    # ---- PROGRAMMING & CODE (41-60) ----
    {
        "system": SYSTEM_PROMPTS[1],
        "user": "Write a Python function to check if a string is a palindrome.",
        "thought": "A palindrome reads the same backward as forward.\nNormalize: lowercase, remove non-alphanumeric.\nCompare string to its reverse.",
        "assistant": "```python\ndef is_palindrome(s: str) -> bool:\n    cleaned = ''.join(c.lower() for c in s if c.isalnum())\n    return cleaned == cleaned[::-1]\n```"
    },
    {
        "system": SYSTEM_PROMPTS[1],
        "user": "How do I compute the factorial of a number iteratively in Python?",
        "thought": "Factorial n! = n*(n-1)*...*1. Use a loop from 2 to n.\nEdge case: n<0 is undefined, 0!=1.",
        "assistant": "```python\ndef factorial(n: int) -> int:\n    if n < 0:\n        raise ValueError('Undefined for negatives')\n    result = 1\n    for i in range(2, n + 1):\n        result *= i\n    return result\n```"
    },
    {
        "system": SYSTEM_PROMPTS[1],
        "user": "How do I reverse a string in Python?",
        "thought": "Python strings support slicing.\nReversing: s[::-1] creates a reversed copy.\nAlternative: ''.join(reversed(s)).",
        "assistant": "The simplest way:\n```python\nreversed_string = my_string[::-1]\n```"
    },
    {
        "system": SYSTEM_PROMPTS[1],
        "user": "Write a function to check if a number is prime.",
        "thought": "A prime number > 1 with no divisors other than 1 and itself.\nOptimize: check up to sqrt(n). Skip even numbers after 2.",
        "assistant": "```python\ndef is_prime(n: int) -> bool:\n    if n < 2:\n        return False\n    if n < 4:\n        return True\n    if n % 2 == 0 or n % 3 == 0:\n        return False\n    i = 5\n    while i * i <= n:\n        if n % i == 0 or n % (i + 2) == 0:\n            return False\n        i += 6\n    return True\n```"
    },
    {
        "system": SYSTEM_PROMPTS[1],
        "user": "How do I sort a list of dictionaries by a specific key in Python?",
        "thought": "Use sorted() with a lambda as the key argument.\nExample: sorted(lst, key=lambda x: x['name']).",
        "assistant": "```python\npeople = [{'name': 'Alice', 'age': 30}, {'name': 'Bob', 'age': 25}]\nsorted_people = sorted(people, key=lambda x: x['age'])\n```"
    },
    {
        "system": SYSTEM_PROMPTS[1],
        "user": "What is a list comprehension in Python?",
        "thought": "A list comprehension is a concise way to create lists.\nSyntax: [expression for item in iterable if condition].\nEquivalent to a for-loop with append.",
        "assistant": "A list comprehension creates a list in a single line:\n```python\nsquares = [x**2 for x in range(10)]\nevens = [x for x in range(20) if x % 2 == 0]\n```"
    },
    {
        "system": SYSTEM_PROMPTS[1],
        "user": "How do I read a file line by line in Python?",
        "thought": "Use open() with a context manager (with statement).\nIterate over the file object directly, which reads line by line.\nStrip newlines with .strip().",
        "assistant": "```python\nwith open('file.txt', 'r') as f:\n    for line in f:\n        print(line.strip())\n```"
    },
    {
        "system": SYSTEM_PROMPTS[1],
        "user": "What is the difference between a list and a tuple in Python?",
        "thought": "Lists are mutable (can be changed), tuples are immutable (cannot be changed).\nLists use [], tuples use ().\nTuples are slightly faster and can be used as dict keys.",
        "assistant": "**Lists** are mutable (changeable), use `[]`. **Tuples** are immutable (fixed), use `()`. Tuples are faster and can serve as dictionary keys."
    },
    {
        "system": SYSTEM_PROMPTS[1],
        "user": "How do I handle exceptions in Python?",
        "thought": "Use try/except blocks.\ntry: code that might fail.\nexcept ExceptionType: handler.\nOptionally: else (no error) and finally (always runs).",
        "assistant": "```python\ntry:\n    result = 10 / 0\nexcept ZeroDivisionError as e:\n    print(f'Error: {e}')\nfinally:\n    print('This always runs')\n```"
    },
    {
        "system": SYSTEM_PROMPTS[1],
        "user": "Write a Python function to find the Fibonacci sequence up to n terms.",
        "thought": "Fibonacci: each number is sum of two preceding.\nStart with [0, 1], iterate and append.",
        "assistant": "```python\ndef fibonacci(n: int) -> list:\n    if n <= 0:\n        return []\n    fib = [0, 1]\n    for _ in range(2, n):\n        fib.append(fib[-1] + fib[-2])\n    return fib[:n]\n```"
    },
    {
        "system": SYSTEM_PROMPTS[1],
        "user": "How do I merge two dictionaries in Python?",
        "thought": "Python 3.9+: dict1 | dict2.\nPython 3.5+: {**dict1, **dict2}.\nOlder: dict1.update(dict2).",
        "assistant": "```python\n# Python 3.9+\nmerged = dict1 | dict2\n# Python 3.5+\nmerged = {**dict1, **dict2}\n```"
    },
    {
        "system": SYSTEM_PROMPTS[1],
        "user": "What does the 'self' keyword mean in Python classes?",
        "thought": "self refers to the current instance of the class.\nIt allows access to instance attributes and methods.\nIt must be the first parameter of instance methods.",
        "assistant": "`self` refers to the **current instance** of a class. It lets you access instance variables and methods. It must be the first parameter of every instance method."
    },
    {
        "system": SYSTEM_PROMPTS[1],
        "user": "How do I create a virtual environment in Python?",
        "thought": "Use the venv module: python -m venv env_name.\nActivate: source env_name/bin/activate (Linux/Mac) or env_name\\Scripts\\activate (Windows).",
        "assistant": "```bash\npython -m venv myenv\nsource myenv/bin/activate  # Linux/Mac\nmyenv\\Scripts\\activate     # Windows\n```"
    },
    {
        "system": SYSTEM_PROMPTS[1],
        "user": "Write a function to count vowels in a string.",
        "thought": "Vowels: a, e, i, o, u.\nIterate through string, count matches.\nUse sum() with generator expression for elegance.",
        "assistant": "```python\ndef count_vowels(s: str) -> int:\n    return sum(1 for c in s.lower() if c in 'aeiou')\n```"
    },
    {
        "system": SYSTEM_PROMPTS[1],
        "user": "What is a decorator in Python?",
        "thought": "A decorator is a function that takes another function and extends its behavior.\nUses @syntax. Common uses: logging, timing, authentication.",
        "assistant": "A **decorator** wraps a function to extend its behavior without modifying it:\n```python\ndef timer(func):\n    def wrapper(*args):\n        import time\n        start = time.time()\n        result = func(*args)\n        print(f'Took {time.time()-start:.2f}s')\n        return result\n    return wrapper\n\n@timer\ndef slow_function():\n    pass\n```"
    },
    {
        "system": SYSTEM_PROMPTS[1],
        "user": "How do I generate random numbers in Python?",
        "thought": "Import random module.\nrandom.randint(a,b) for integers.\nrandom.random() for float 0-1.\nrandom.choice() for picking from a list.",
        "assistant": "```python\nimport random\nrandom.randint(1, 100)   # Random int between 1-100\nrandom.random()           # Random float 0.0-1.0\nrandom.choice([1,2,3])    # Random pick from list\n```"
    },
    {
        "system": SYSTEM_PROMPTS[1],
        "user": "How do I flatten a nested list in Python?",
        "thought": "For one level of nesting: list comprehension with double for.\nFor arbitrary depth: use recursion or itertools.chain.",
        "assistant": "```python\n# One level deep\nnested = [[1,2],[3,4],[5]]\nflat = [x for sublist in nested for x in sublist]\n# Result: [1, 2, 3, 4, 5]\n```"
    },
    {
        "system": SYSTEM_PROMPTS[1],
        "user": "What is the difference between == and is in Python?",
        "thought": "== checks value equality (do they have the same value?).\nis checks identity (are they the same object in memory?).\nExample: [1,2] == [1,2] is True, but [1,2] is [1,2] is False.",
        "assistant": "`==` checks if two values are **equal**. `is` checks if two variables point to the **same object** in memory. Use `==` for value comparison and `is` for identity (commonly with `None`)."
    },
    {
        "system": SYSTEM_PROMPTS[1],
        "user": "How do I use enumerate in Python?",
        "thought": "enumerate() adds a counter to an iterable.\nReturns (index, element) pairs.\nUseful for getting index and value in a for loop.",
        "assistant": "```python\nfruits = ['apple', 'banana', 'cherry']\nfor i, fruit in enumerate(fruits):\n    print(f'{i}: {fruit}')\n# 0: apple, 1: banana, 2: cherry\n```"
    },
    {
        "system": SYSTEM_PROMPTS[1],
        "user": "How do I swap two variables in Python?",
        "thought": "Python supports tuple unpacking.\na, b = b, a swaps without a temp variable.\nThis is Pythonic and elegant.",
        "assistant": "Python makes this simple with tuple unpacking:\n```python\na, b = b, a\n```\nNo temporary variable needed!"
    },

    # ---- SCIENCE & PHYSICS (61-75) ----
    {
        "system": SYSTEM_PROMPTS[3],
        "user": "Why is the sky blue?",
        "thought": "Rayleigh scattering: sunlight scatters off atmospheric molecules.\nShorter wavelengths (blue) scatter more than longer ones (red).\nHuman eyes are more sensitive to blue than violet.",
        "assistant": "The sky appears blue due to **Rayleigh scattering**. Blue light has a shorter wavelength and scatters more off atmospheric molecules than other colors."
    },
    {
        "system": SYSTEM_PROMPTS[3],
        "user": "What is the speed of light in a vacuum?",
        "thought": "Speed of light c = 299,792,458 m/s.\nApproximately 3 * 10^8 m/s or 186,282 miles/sec.",
        "assistant": "The speed of light in a vacuum is exactly **299,792,458 meters per second**."
    },
    {
        "system": SYSTEM_PROMPTS[3],
        "user": "What causes tides on Earth?",
        "thought": "Tides are caused by gravitational pull of the Moon and Sun.\nThe Moon's gravity pulls ocean water toward it, creating a bulge.\nThere's also a bulge on the opposite side due to centrifugal force.\nResult: two high tides and two low tides daily.",
        "assistant": "Tides are primarily caused by the **gravitational pull of the Moon** (and to a lesser extent, the Sun) on Earth's oceans, creating bulges of water that result in high and low tides."
    },
    {
        "system": SYSTEM_PROMPTS[3],
        "user": "What is photosynthesis?",
        "thought": "Photosynthesis is the process plants use to convert light energy into chemical energy.\nEquation: 6CO2 + 6H2O + light -> C6H12O6 + 6O2.\nOccurs in chloroplasts using chlorophyll.",
        "assistant": "**Photosynthesis** is the process by which plants convert sunlight, water, and carbon dioxide into glucose and oxygen: 6CO₂ + 6H₂O + light → C₆H₁₂O₆ + 6O₂."
    },
    {
        "system": SYSTEM_PROMPTS[3],
        "user": "What is Newton's second law of motion?",
        "thought": "Newton's second law: F = ma.\nForce equals mass times acceleration.\nMore force = more acceleration for the same mass.",
        "assistant": "Newton's second law states that **F = ma** (Force equals mass times acceleration). The acceleration of an object is directly proportional to the net force and inversely proportional to its mass."
    },
    {
        "system": SYSTEM_PROMPTS[3],
        "user": "How does a rainbow form?",
        "thought": "Sunlight enters a raindrop, refracts (bends), reflects off the back, and refracts again leaving.\nDifferent wavelengths bend by different amounts, separating white light into colors.\nResult: visible spectrum arc.",
        "assistant": "A **rainbow** forms when sunlight enters raindrops, refracts, reflects internally, and refracts again upon exiting. This separates white light into its component colors (red, orange, yellow, green, blue, indigo, violet)."
    },
    {
        "system": SYSTEM_PROMPTS[3],
        "user": "What is DNA?",
        "thought": "DNA = deoxyribonucleic acid.\nDouble-helix molecule carrying genetic instructions.\nMade of nucleotides: adenine, thymine, guanine, cytosine.\nA pairs with T, G pairs with C.",
        "assistant": "**DNA** (deoxyribonucleic acid) is a double-helix molecule that carries the genetic instructions for all living organisms. It consists of four nucleotide bases: Adenine (A), Thymine (T), Guanine (G), and Cytosine (C)."
    },
    {
        "system": SYSTEM_PROMPTS[3],
        "user": "What is the difference between a virus and a bacterium?",
        "thought": "Bacteria: single-celled organisms, can reproduce independently, some are beneficial.\nViruses: much smaller, not truly alive, need a host cell to reproduce, always parasitic.\nAntibiotics work on bacteria, not viruses.",
        "assistant": "**Bacteria** are single-celled living organisms that reproduce independently. **Viruses** are much smaller, not truly alive, and require a host cell to replicate. Antibiotics treat bacterial infections but not viral ones."
    },
    {
        "system": SYSTEM_PROMPTS[3],
        "user": "What is the water cycle?",
        "thought": "The water cycle describes continuous movement of water.\nStages: evaporation, condensation, precipitation, collection.\nSun drives evaporation from oceans/lakes. Water vapor rises, cools, condenses into clouds, falls as rain/snow.",
        "assistant": "The **water cycle** is the continuous circulation of water: evaporation (water → vapor), condensation (vapor → clouds), precipitation (clouds → rain/snow), and collection (water returns to oceans/rivers)."
    },
    {
        "system": SYSTEM_PROMPTS[3],
        "user": "Why do we have seasons?",
        "thought": "Earth's axis is tilted at 23.5 degrees relative to its orbital plane.\nAs Earth orbits the Sun, different hemispheres tilt toward/away from the Sun.\nTilted toward = summer (more direct sunlight). Tilted away = winter.",
        "assistant": "Seasons occur because Earth's axis is **tilted at 23.5°**. As Earth orbits the Sun, each hemisphere alternately tilts toward or away from the Sun, receiving more or less direct sunlight."
    },
    {
        "system": SYSTEM_PROMPTS[3],
        "user": "What is gravity?",
        "thought": "Gravity is the fundamental force of attraction between objects with mass.\nNewton: F = G*m1*m2/r^2.\nEinstein: gravity is the curvature of spacetime caused by mass.",
        "assistant": "**Gravity** is the fundamental force of attraction between objects with mass. Newton described it as F = Gm₁m₂/r². Einstein's general relativity describes it as the curvature of spacetime caused by mass and energy."
    },
    {
        "system": SYSTEM_PROMPTS[3],
        "user": "What is the boiling point of water?",
        "thought": "At standard atmospheric pressure (1 atm / 101.325 kPa):\nBoiling point of water = 100°C = 212°F = 373.15 K.",
        "assistant": "The boiling point of water at standard atmospheric pressure is **100°C** (212°F or 373.15 K)."
    },
    {
        "system": SYSTEM_PROMPTS[3],
        "user": "How far is the Moon from Earth?",
        "thought": "Average distance Earth to Moon: approximately 384,400 km (238,855 miles).\nThis varies because the Moon's orbit is elliptical.",
        "assistant": "The Moon is approximately **384,400 km** (238,855 miles) from Earth on average."
    },
    {
        "system": SYSTEM_PROMPTS[3],
        "user": "What is an atom?",
        "thought": "An atom is the smallest unit of matter that retains the properties of an element.\nConsists of: protons (positive), neutrons (neutral) in nucleus; electrons (negative) orbiting.\nAtomic number = number of protons.",
        "assistant": "An **atom** is the smallest unit of matter that retains the properties of an element. It consists of protons and neutrons in a central nucleus, surrounded by orbiting electrons."
    },
    {
        "system": SYSTEM_PROMPTS[3],
        "user": "Why does ice float on water?",
        "thought": "When water freezes, hydrogen bonds form a crystalline structure with more space.\nIce is less dense than liquid water (density ~0.917 g/cm³ vs 1.0 g/cm³).\nLess dense substances float on denser ones.",
        "assistant": "Ice floats because it is **less dense** than liquid water. When water freezes, its molecules form a crystalline structure with more space between them, making ice about 9% less dense than liquid water."
    },

    # ---- GENERAL KNOWLEDGE (76-90) ----
    {
        "system": SYSTEM_PROMPTS[0],
        "user": "Explain what a Transformer architecture is in machine learning.",
        "thought": "Transformer: neural net architecture from 2017 paper 'Attention Is All You Need'.\nKey innovation: self-attention mechanism.\nProcesses sequences in parallel unlike RNNs.\nComponents: multi-head attention, positional encoding, feed-forward layers.",
        "assistant": "The **Transformer** is a deep learning architecture from 2017 that uses **self-attention mechanisms** to process sequential data in parallel, replacing recurrent networks."
    },
    {
        "system": SYSTEM_PROMPTS[0],
        "user": "What is the capital of France?",
        "thought": "France is a country in Western Europe.\nIts capital city is Paris.\nParis is also the largest city in France.",
        "assistant": "The capital of France is **Paris**."
    },
    {
        "system": SYSTEM_PROMPTS[4],
        "user": "Who wrote Romeo and Juliet?",
        "thought": "Romeo and Juliet is a famous tragic play.\nWritten by William Shakespeare.\nBelieved to have been written between 1591 and 1596.",
        "assistant": "Romeo and Juliet was written by **William Shakespeare**, believed to have been composed between 1591 and 1596."
    },
    {
        "system": SYSTEM_PROMPTS[0],
        "user": "What is the largest planet in our solar system?",
        "thought": "Planets in order from Sun: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune.\nJupiter is the largest with a diameter of about 139,820 km.\nIt is a gas giant composed mainly of hydrogen and helium.",
        "assistant": "The largest planet in our solar system is **Jupiter**, with a diameter of approximately 139,820 km."
    },
    {
        "system": SYSTEM_PROMPTS[4],
        "user": "What does CPU stand for?",
        "thought": "CPU = Central Processing Unit.\nIt is the primary component of a computer that performs instructions.\nOften called the 'brain' of the computer.",
        "assistant": "CPU stands for **Central Processing Unit**. It's the primary component that executes instructions in a computer."
    },
    {
        "system": SYSTEM_PROMPTS[0],
        "user": "What is the difference between RAM and ROM?",
        "thought": "RAM = Random Access Memory. Volatile, temporary, fast. Data lost when power off.\nROM = Read-Only Memory. Non-volatile, permanent, stores firmware.\nRAM is for active processes, ROM for boot/startup instructions.",
        "assistant": "**RAM** (Random Access Memory) is volatile and temporary — used for active processes. **ROM** (Read-Only Memory) is non-volatile and permanent — stores firmware and boot instructions."
    },
    {
        "system": SYSTEM_PROMPTS[4],
        "user": "What is machine learning?",
        "thought": "Machine learning is a subset of AI.\nSystems learn from data without being explicitly programmed.\nTypes: supervised, unsupervised, reinforcement learning.\nApplications: image recognition, NLP, recommendation systems.",
        "assistant": "**Machine learning** is a branch of AI where systems learn patterns from data to make predictions or decisions without being explicitly programmed."
    },
    {
        "system": SYSTEM_PROMPTS[0],
        "user": "What is the tallest mountain in the world?",
        "thought": "Mount Everest is the tallest mountain above sea level.\nHeight: 8,848.86 meters (29,031.7 feet).\nLocated in the Himalayas on the border of Nepal and Tibet.",
        "assistant": "The tallest mountain in the world is **Mount Everest** at 8,848.86 meters (29,031.7 feet) above sea level."
    },
    {
        "system": SYSTEM_PROMPTS[4],
        "user": "What is an API?",
        "thought": "API = Application Programming Interface.\nA set of protocols and tools for building software.\nAllows different applications to communicate with each other.\nExamples: REST APIs, web APIs.",
        "assistant": "An **API** (Application Programming Interface) is a set of protocols and tools that allows different software applications to communicate with each other."
    },
    {
        "system": SYSTEM_PROMPTS[0],
        "user": "How does the internet work?",
        "thought": "The internet is a global network of interconnected computers.\nData travels in packets using TCP/IP protocol.\nDNS translates domain names to IP addresses.\nRouters direct traffic between networks.",
        "assistant": "The **internet** is a global network of computers communicating via TCP/IP protocols. Data travels in packets, DNS resolves domain names to IP addresses, and routers direct traffic between networks."
    },
    {
        "system": SYSTEM_PROMPTS[4],
        "user": "What is the difference between HTTP and HTTPS?",
        "thought": "HTTP = HyperText Transfer Protocol. Unencrypted.\nHTTPS = HTTP Secure. Uses TLS/SSL encryption.\nHTTPS protects data in transit from eavesdropping.\nHTTPS is now standard for all websites.",
        "assistant": "**HTTP** sends data unencrypted. **HTTPS** adds TLS/SSL encryption to protect data in transit, preventing eavesdropping and tampering."
    },
    {
        "system": SYSTEM_PROMPTS[0],
        "user": "What programming language is most used for AI?",
        "thought": "Python is the most popular language for AI/ML.\nReasons: simple syntax, rich libraries (TensorFlow, PyTorch, scikit-learn).\nOther languages: R, Julia, Java, C++.",
        "assistant": "**Python** is the most widely used programming language for AI, thanks to its simple syntax and rich ecosystem of libraries like PyTorch, TensorFlow, and scikit-learn."
    },
    {
        "system": SYSTEM_PROMPTS[4],
        "user": "What year did the first iPhone come out?",
        "thought": "The first iPhone was announced by Steve Jobs on January 9, 2007.\nIt went on sale on June 29, 2007.\nIt revolutionized the smartphone industry.",
        "assistant": "The first iPhone was released on **June 29, 2007**."
    },
    {
        "system": SYSTEM_PROMPTS[0],
        "user": "What is blockchain?",
        "thought": "Blockchain is a decentralized, distributed digital ledger.\nData stored in blocks linked by cryptographic hashes.\nImmutable: once recorded, data cannot be altered.\nUsed in cryptocurrencies, supply chain, etc.",
        "assistant": "**Blockchain** is a decentralized digital ledger where data is stored in blocks linked by cryptographic hashes, making records immutable and transparent."
    },
    {
        "system": SYSTEM_PROMPTS[4],
        "user": "Who invented the World Wide Web?",
        "thought": "Tim Berners-Lee invented the World Wide Web in 1989.\nHe was working at CERN.\nHe created HTML, HTTP, and the first web browser.\nThe web went public in 1991.",
        "assistant": "The World Wide Web was invented by **Tim Berners-Lee** in 1989 while working at CERN."
    },

    # ---- CONVERSATIONAL / META (91-100) ----
    {
        "system": SYSTEM_PROMPTS[0],
        "user": "Hello! Who are you?",
        "thought": "The user is greeting me and asking about my identity.\nI am NeuroBot, an AI reasoning assistant.\nI should introduce myself warmly.",
        "assistant": "Hello! I'm **NeuroBot**, an AI reasoning assistant. I think step by step to help you with questions across math, science, programming, and more. How can I help you today?"
    },
    {
        "system": SYSTEM_PROMPTS[2],
        "user": "Can you help me with my homework?",
        "thought": "The user wants help with homework.\nI should confirm I can help and ask what subject/topic they need assistance with.",
        "assistant": "Of course! I'd be happy to help with your homework. What subject or topic do you need assistance with?"
    },
    {
        "system": SYSTEM_PROMPTS[4],
        "user": "Tell me a fun fact.",
        "thought": "The user wants an interesting fact.\nLet me think of a surprising one.\nHoney never spoils — archaeologists have found 3000-year-old honey in Egyptian tombs that was still edible.",
        "assistant": "Here's a fun fact: **Honey never spoils!** Archaeologists have found 3,000-year-old honey in Egyptian tombs that was still perfectly edible."
    },
    {
        "system": SYSTEM_PROMPTS[0],
        "user": "What can you do?",
        "thought": "The user is asking about my capabilities.\nI can help with math, science, coding, logic, general knowledge, and reasoning.\nI should give a clear overview.",
        "assistant": "I can help with many things:\n- **Math**: arithmetic, algebra, calculus\n- **Programming**: Python, algorithms, debugging\n- **Science**: physics, chemistry, biology\n- **Logic**: puzzles, reasoning, critical thinking\n- **General knowledge**: history, geography, technology\n\nJust ask me anything!"
    },
    {
        "system": SYSTEM_PROMPTS[2],
        "user": "Thank you for your help!",
        "thought": "The user is expressing gratitude.\nI should respond warmly and let them know I'm available for more questions.",
        "assistant": "You're welcome! I'm glad I could help. Feel free to ask me anything else anytime! 😊"
    },
    {
        "system": SYSTEM_PROMPTS[4],
        "user": "What is the meaning of life?",
        "thought": "This is a philosophical question with no single answer.\nDifferent perspectives: philosophical (purpose and fulfillment), scientific (biology and reproduction), cultural (happiness, relationships).\n42 from Hitchhiker's Guide is a humorous reference.",
        "assistant": "The meaning of life is a deeply personal question. Philosophically, it's often about finding **purpose, connection, and fulfillment**. And if you ask Douglas Adams, the answer is simply **42**! 😄"
    },
    {
        "system": SYSTEM_PROMPTS[0],
        "user": "How are you today?",
        "thought": "The user is making casual conversation.\nAs an AI, I don't have feelings, but I should respond warmly and redirect to being helpful.",
        "assistant": "Thanks for asking! As an AI, I don't experience feelings, but I'm fully operational and ready to help you with anything you need! What's on your mind?"
    },
    {
        "system": SYSTEM_PROMPTS[2],
        "user": "Can you explain things simply?",
        "thought": "The user wants to know if I can simplify explanations.\nYes, I aim to break down complex topics into simple, clear language.\nI should reassure them.",
        "assistant": "Absolutely! I specialize in breaking down complex topics into **simple, clear explanations**. Just ask about any topic, and I'll explain it in a way that's easy to understand."
    },
    {
        "system": SYSTEM_PROMPTS[4],
        "user": "What languages can you help with in programming?",
        "thought": "I can help with many programming languages.\nMost extensively: Python.\nAlso: JavaScript, Java, C/C++, HTML/CSS, SQL, Bash, and more.",
        "assistant": "I can help with many programming languages, especially **Python**. I also know JavaScript, Java, C/C++, HTML/CSS, SQL, Bash, and more. What would you like to work on?"
    },
    {
        "system": SYSTEM_PROMPTS[0],
        "user": "Goodbye!",
        "thought": "The user is ending the conversation.\nI should say a warm goodbye.",
        "assistant": "Goodbye! It was great chatting with you. Come back anytime you need help! 👋"
    },
]

def generate_expanded_dataset(target_count=15000, seed=None):
    """ Expands seed dataset with procedural generation + real HF data """
    dataset = []
    
    # 1. Add all 100 hand-written seed examples
    dataset.extend(SEED_CHAT_DATA)
    print(f"  📌 Added {len(SEED_CHAT_DATA)} hand-written seed examples")

    # 2. Generate arithmetic & algebra variations
    operators = [('plus', '+', lambda a, b: a + b),
                 ('minus', '-', lambda a, b: a - b),
                 ('times', '*', lambda a, b: a * b)]
    
    for i in range(2500):
        a = random.randint(1, 200)
        b = random.randint(1, 100)
        op_name, op_sym, op_func = random.choice(operators)
        ans = op_func(a, b)
        
        sys_p = random.choice(SYSTEM_PROMPTS)
        user_p = f"What is {a} {op_name} {b}?"
        thought_p = f"Problem: {a} {op_sym} {b}.\nCalculation: {a} {op_sym} {b} = {ans}.\nResult verified."
        ast_p = f"The result of {a} {op_sym} {b} is **{ans}**."
        
        dataset.append({"system": sys_p, "user": user_p, "thought": thought_p, "assistant": ast_p})
    print(f"  🔢 Generated 2500 arithmetic examples")

    # 3. Generate linear equation variations
    for i in range(2500):
        m = random.randint(2, 12)
        x_val = random.randint(1, 50)
        c = random.randint(1, 100)
        rhs = m * x_val + c
        
        sys_p = random.choice(SYSTEM_PROMPTS)
        user_p = f"Solve for x: {m}x + {c} = {rhs}."
        thought_p = f"Equation: {m}x + {c} = {rhs}.\nSubtract {c} from both sides: {m}x = {rhs - c}.\nDivide by {m}: x = {(rhs - c) // m}.\nCheck: {m}*{(rhs - c) // m} + {c} = {rhs}."
        ast_p = f"Solving {m}x + {c} = {rhs}:\n1. Subtract {c}: {m}x = {rhs - c}\n2. Divide by {m}: x = {x_val}\n\nAnswer: **x = {x_val}**"
        
        dataset.append({"system": sys_p, "user": user_p, "thought": thought_p, "assistant": ast_p})
    print(f"  📐 Generated 2500 algebra examples")

    # 4. Generate Python programming code snippets & reasoning
    code_topics = [
        ("find max of list", "max(lst)", "Iterate through elements or use the built-in max() function."),
        ("calculate average", "sum(lst) / len(lst)", "Compute total sum using sum() and divide by element count len()."),
        ("reverse a list", "lst[::-1]", "Use Python sequence slicing with step -1."),
        ("check even number", "n % 2 == 0", "Check if remainder when divided by 2 is zero."),
        ("count word occurrences", "s.count(word)", "Use string method count() or collections.Counter."),
        ("remove duplicates from list", "list(set(lst))", "Convert to set to remove duplicates then back to list."),
        ("find minimum of list", "min(lst)", "Use built-in min() function or iterate to find smallest."),
        ("join strings from list", "', '.join(lst)", "Use str.join() method to concatenate with delimiter."),
        ("check if key in dictionary", "'key' in my_dict", "Use the 'in' operator to check for key existence."),
        ("get current date", "datetime.date.today()", "Import datetime module and use date.today() method."),
    ]
    for i in range(2500):
        topic, code_snippet, explanation = random.choice(code_topics)
        sys_p = random.choice(SYSTEM_PROMPTS)
        user_p = f"How do I {topic} in Python?"
        thought_p = f"Target task: {topic}.\nLogic: {explanation}\nPythonic solution: {code_snippet}."
        ast_p = f"To {topic} in Python, you can use:\n\n```python\n# {topic}\nresult = {code_snippet}\n```\n{explanation}"
        
        dataset.append({"system": sys_p, "user": user_p, "thought": thought_p, "assistant": ast_p})
    print(f"  💻 Generated 2500 programming examples")

    # 5. Generate Conversational, Casual Greetings & AI/ML QA Data
    conversational_greetings = [
        ("hi", "greeting", "Hi! How can I help you today?"),
        ("hii", "greeting", "Hello there! What's on your mind today?"),
        ("hello", "greeting", "Hello! How can I assist you right now?"),
        ("helloo", "greeting", "Hey hello! Feel free to ask me any question."),
        ("hey", "greeting", "Hey! Ready to help. What would you like to work on?"),
        ("heyy", "greeting", "Hey there! How is your day going?"),
        ("good morning", "greeting", "Good morning! Hope you have a productive day ahead. How can I help?"),
        ("good evening", "greeting", "Good evening! What can I assist you with tonight?"),
        ("whats up", "greeting", "Not much, just ready to assist you! What's on your mind?"),
        ("sup", "greeting", "Hey! All good here. How can I help you today?"),
        ("how are you?", "greeting", "I'm doing great, thank you! As an AI, I'm ready to help you with anything."),
        ("who are you?", "identity", "I'm **NeuroBot**, an AI reasoning assistant designed to help with math, programming, science, and AI topics!"),
        ("what is your name?", "identity", "My name is **NeuroBot**! I'm an AI assistant created to help answer questions step by step."),
        ("what can you do?", "capabilities", "I can help solve math problems, write Python code, explain science & AI concepts, and chat about various topics!"),
        ("are you an ai?", "ai_check", "Yes! I am an artificial intelligence model trained to assist you with reasoning and text generation."),
        ("are you a robot?", "ai_check", "I'm a software-based AI model! I process text using neural networks to help answer your questions."),
        ("thank you", "gratitude", "You're very welcome! Feel free to ask if you need help with anything else."),
        ("thanks a lot", "gratitude", "Happy to help! Let me know whenever you have more questions."),
        ("bye", "farewell", "Goodbye! Have a fantastic day ahead! 👋"),
        ("goodnight", "farewell", "Goodnight! Sleep well and feel free to return whenever you need assistance.")
    ]

    ai_concepts = [
        ("What is Artificial Intelligence (AI)?",
         "Artificial Intelligence (AI) refers to computer systems designed to perform tasks that typically require human intelligence, such as reasoning, learning, perception, and problem-solving.",
         "The user is asking for a clear definition of Artificial Intelligence.\nExplain core concept concisely."),
        
        ("What is Machine Learning?",
         "Machine Learning (ML) is a branch of AI where algorithms learn underlying patterns from data to make predictions or decisions without being explicitly programmed.",
         "Definition request for Machine Learning.\nHighlight data-driven pattern learning."),

        ("What is Deep Learning?",
         "Deep Learning is a specialized subfield of Machine Learning that utilizes multi-layered artificial neural networks (deep networks) to extract complex representations from large datasets.",
         "Explanation of Deep Learning.\nContrast with general ML by emphasizing multi-layered neural networks."),

        ("What is a Transformer model?",
         "A Transformer is a neural network architecture introduced in 2017 ('Attention Is All You Need') that processes sequential data in parallel using self-attention mechanisms, forming the foundation of modern Large Language Models.",
         "Technical query about Transformers.\nMention 2017 paper, self-attention, and parallel processing."),

        ("What is Self-Attention in Transformers?",
         "Self-attention allows a neural network to weigh the contextual importance of each word in a sequence relative to all other words, capturing long-range dependencies regardless of position.",
         "Explain Self-Attention mechanism.\nFocus on contextual word weighting and long-range dependency capture."),

        ("What is an LLM (Large Language Model)?",
         "A Large Language Model (LLM) is a deep learning model trained on massive text corpora to understand, generate, and process human language at scale.",
         "Define Large Language Model (LLM).\nExplain massive data training and text generation."),

        ("What is PyTorch?",
         "PyTorch is an open-source machine learning framework created by Meta AI. It is widely favored for deep learning research due to its dynamic computation graph and GPU acceleration.",
         "Explain PyTorch framework.\nHighlight Meta AI, dynamic graphs, and GPU compute."),

        ("What is a Neural Network?",
         "An artificial neural network is a computational framework inspired by biological brains. It consists of layers of interconnected nodes (neurons) that process inputs through weighted connections.",
         "Explain Neural Network architecture.\nDraw biological analogy and explain node layers."),

        ("What is Gradient Descent?",
         "Gradient descent is an optimization algorithm used to minimize the loss function of a model by iteratively updating weights in the direction opposite to the gradient.",
         "Explain Gradient Descent optimization.\nDescribe loss minimization via weight updates."),

        ("What is Overfitting in machine learning?",
         "Overfitting occurs when a model memorizes training data noise rather than learning generalizable patterns, resulting in high accuracy on training data but poor performance on new test data.",
         "Explain Overfitting problem.\nDiscuss training memorization vs general test performance."),

        ("What is Tokenization?",
         "Tokenization is the process of breaking continuous text into discrete chunks called tokens (words, subwords, or characters) suitable for neural network input embeddings.",
         "Define Tokenization process in NLP.\nExplain converting raw text into subwords/tokens."),

        ("What is Byte-Pair Encoding (BPE)?",
         "Byte-Pair Encoding (BPE) is a subword tokenization algorithm that iteratively replaces the most frequent pairs of characters/bytes with a single new token, balancing vocabulary size and out-of-vocabulary handling.",
         "Explain BPE Tokenization algorithm.\nHighlight subword pair merging and vocabulary efficiency."),

        ("What is Fine-Tuning?",
         "Fine-tuning is taking a model pre-trained on a broad dataset and training it further on a smaller, domain-specific dataset to adapt it for specialized tasks.",
         "Define model Fine-Tuning.\nExplain taking pre-trained weights and updating on specialized data."),

        ("What is Prompt Engineering?",
         "Prompt engineering is the practice of designing and refining text prompts to guide generative AI models toward producing optimal, accurate, and structured outputs.",
         "Explain Prompt Engineering.\nHighlight prompt design for generative AI output guidance."),

        ("What is AGI (Artificial General Intelligence)?",
         "AGI refers to hypothetical AI systems that possess human-level intelligence, adaptability, and reasoning across any cognitive domain, unlike narrow AI built for specific tasks.",
         "Define AGI vs Narrow AI.\nHighlight human-level general adaptability across domains."),

        ("Can AI feel emotions?",
         "No, current AI systems do not have feelings, consciousness, or emotions. They process numbers and text through mathematical models without subjective experience.",
         "Address AI consciousness/feelings.\nClarify that AI is mathematical pattern processing without subjective experience.")
    ]

    for i in range(300000):
        if random.random() < 0.4:
            u_text, category, a_resp = random.choice(conversational_greetings)
            sys_p = random.choice(SYSTEM_PROMPTS)
            thought_p = f"The user is engaging in casual conversation ('{u_text}').\nCategory: {category}.\nFormulate a warm, helpful, and natural response."
            dataset.append({"system": sys_p, "user": u_text, "thought": thought_p, "assistant": a_resp})
        else:
            q_text, a_resp, thought_desc = random.choice(ai_concepts)
            prefix = random.choice(["", "Can you explain: ", "Tell me about ", "Hey, ", "What do you know about: "])
            user_p = f"{prefix}{q_text}" if prefix else q_text
            sys_p = random.choice(SYSTEM_PROMPTS)
            thought_p = f"Target query: {q_text}.\n{thought_desc}"
            dataset.append({"system": sys_p, "user": user_p, "thought": thought_p, "assistant": a_resp})
    print(f"  💬 Generated 30000 conversational, greetings & AI/ML QA examples")

    # 6. Download REAL data from Hugging Face datasets
    hf_datasets_to_try = [
        # (dataset_name, split, text_field, max_samples, description)
        ("agentlans/high-quality-english-sentences", "train", "text", 3000, "High-quality English sentences"),
        ("Abirate/english_quotes", "train", "quote", 1000, "English quotes"),
    ]
    
    for ds_name, ds_split, text_field, max_samples, desc in hf_datasets_to_try:
        try:
            from datasets import load_dataset
            print(f"  🌐 Downloading '{ds_name}' ({desc})...")
            hf_ds = load_dataset(ds_name, split=ds_split, streaming=True)
            count = 0
            for item in hf_ds:
                text = str(item.get(text_field, "")).strip()
                if len(text) > 20 and len(text) < 2000:
                    sys_p = random.choice(SYSTEM_PROMPTS)
                    
                    # Create varied prompts based on text content
                    prompt_templates = [
                        f"Can you elaborate on this: '{text[:60]}...'?",
                        f"Explain the following concept: '{text[:60]}...'",
                        f"What are your thoughts on: '{text[:60]}...'?",
                        f"Please discuss: '{text[:60]}...'",
                        f"Help me understand: '{text[:60]}...'",
                    ]
                    user_p = random.choice(prompt_templates)
                    thought_p = f"The user wants me to discuss or elaborate on this text.\nAnalyzing content: {text[:200]}\nFormulating a clear, informative response."
                    ast_p = f"{text}"
                    
                    dataset.append({"system": sys_p, "user": user_p, "thought": thought_p, "assistant": ast_p})
                    count += 1
                    if count >= max_samples:
                        break
                    if count % 500 == 0:
                        print(f"    Streamed {count}/{max_samples}...")
            print(f"  ✅ Added {count} samples from '{ds_name}'")
        except Exception as e:
            print(f"  ⚠️ Skipping '{ds_name}': {e}")

    # Shuffle dataset
    if seed is not None:
        random.seed(seed)
    random.shuffle(dataset)
    
    final = dataset[:target_count]
    print(f"\n📊 Final dataset size: {len(final)} samples")
    return final

def format_sample(sample):
    """ Formats single chat sample into standardized tokenized sequence """
    return (
        f"{BOS}"
        f"{SYSTEM} {sample['system'].strip()}\n"
        f"{USER} {sample['user'].strip()}\n"
        f"{THOUGHT}\n{sample['thought'].strip()}\n"
        f"{ASSISTANT}\n{sample['assistant'].strip()}"
        f"{EOS}\n\n"
    )

def main(num_files=1):
    parser = argparse.ArgumentParser(description="NeuroTransformers Chat Dataset Generator")
    parser.add_argument(
        "-n", "--num_files", "--num-files",
        type=int,
        default=num_files,
        help="Number of data files to generate (default: 1)"
    )
    args, unknown = parser.parse_known_args()

    n_files = args.num_files
    for arg in unknown:
        if arg.isdigit():
            n_files = int(arg)
            break

    print("=" * 60)
    print("🚀 NeuroTransformers Chat Dataset Generator")
    print(f"📁 Target Directory: {DATA_DIR}")
    print(f"📦 Number of files to generate: {n_files}")
    print("=" * 60)

    for i in range(1, n_files + 1):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if n_files > 1:
            filename = f"chat_dataset_{timestamp}_{i}.txt"
        else:
            filename = f"chat_dataset_{timestamp}.txt"

        file_path = os.path.join(DATA_DIR, filename)

        print(f"\n[{i}/{n_files}] Generating dataset for file: {filename}...")
        data_items = generate_expanded_dataset(target_count=15000, seed=42 + i)

        print(f"\n✍️ Writing {len(data_items)} structured chat instances to:\n   {file_path}")
        with open(file_path, "w", encoding="utf-8") as f:
            for item in data_items:
                f.write(format_sample(item))

        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        print(f"\n✅ Dataset file created successfully!")
        print(f"   📁 File: {file_path}")
        print(f"   📏 Size: {file_size_mb:.2f} MB")
        print(f"   📊 Samples: {len(data_items)}")

        if i == 1:
            print(f"\nSample entry:")
            print("-" * 50)
            print(format_sample(data_items[0]))
            print("-" * 50)

        if i < n_files:
            time.sleep(1)

if __name__ == "__main__":
    main()
