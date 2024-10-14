from constants import GEMINI_API_KEY
from PIL import Image
import json
import ast
import google.generativeai as genai

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(model_name="gemini-1.5-flash")


def analyze_image(img: Image, dict_of_vars: dict):
    dict_of_vars_str = json.dumps(dict_of_vars, ensure_ascii=False)
    prompt = (
        f"You have been given an image with some mathematical expressions, equations, or graphical problems, and you need to solve them step-by-step, showing the complete solution. "
        f"Note: Use the PEMDAS rule for solving mathematical expressions. PEMDAS stands for the Priority Order: Parentheses, Exponents, Multiplication and Division (from left to right), Addition and Subtraction (from left to right). "
        f"Parentheses have the highest priority, followed by Exponents, then Multiplication and Division, and lastly Addition and Subtraction. "
        f"For example: "
        f"Q. 2 + 3 * 4 "
        f"Step 1: Apply multiplication first: 3 * 4 = 12\n"
        f"Step 2: Now add 2 + 12 = 14\n"
        f"Final answer: 14\n"
        f"Q. 2 + 3 + 5 * 4 - 8 / 2 "
        f"Step 1: Apply multiplication and division first: 5 * 4 = 20 and 8 / 2 = 4\n"
        f"Step 2: Now add and subtract from left to right: 2 + 3 = 5, 5 + 20 = 25, 25 - 4 = 21\n"
        f"Final answer: 21\n"
        f"\n"
        f"YOU CAN HAVE FIVE TYPES OF EQUATIONS/EXPRESSIONS IN THIS IMAGE, AND ONLY ONE CASE SHALL APPLY EVERY TIME: "
        f"Following are the cases: "
        f"1. Simple mathematical expressions like 2 + 2, 3 * 4, 5 / 6, 7 - 8, etc.: In this case, solve step-by-step, showing the operations clearly, and return the answer in the format of a LIST OF ONE DICT [{{'expr': given expression, 'solution': detailed solution, 'result': calculated answer}}]. "
        f"2. Set of Equations like x^2 + 2x + 1 = 0, 3y + 4x = 0, 5x^2 + 6y + 7 = 12, etc.: In this case, solve step-by-step for each variable, and return a COMMA-SEPARATED LIST OF DICTS, with each dict containing {{'expr': 'variable equation', 'solution': detailed solution, 'result': calculated value}}. "
        f"3. Assigning values to variables like x = 4, y = 5, z = 6, etc.: In this case, assign values to variables and return another key in the dict called {{'assign': True}}, keeping the variable as 'expr' and the value as 'result', and include a 'solution' key showing how the variables were assigned. RETURN AS A LIST OF DICTS. "
        f"4. Analyzing Graphical Math problems (such as word problems represented in drawing form) like cars colliding, trigonometric problems, or the Pythagorean theorem: Solve step-by-step, and return the answer in the format of a LIST OF ONE DICT [{{'expr': given expression, 'solution': detailed steps of calculation, 'result': final answer}}]. "
        f"5. Detecting Abstract Concepts from a drawing, such as historical references or symbolic representations like love, hate, or war: Provide the explanation of the drawing in 'solution', and return the abstract concept in 'result'. "
        f"Analyze the equation or expression in this image, solve it step-by-step, and return both the full solution with explanation and the final answer according to the given rules: "
        f"Make sure to use extra backslashes for escape characters like \\f -> \\\\f, \\n -> \\\\n, etc. "
        f"Here is a dictionary of user-assigned variables. If the given expression has any of these variables, use its actual value from this dictionary accordingly: {dict_of_vars_str}. "
        f"DO NOT USE BACKTICKS OR MARKDOWN FORMATTING. "
        f"PROPERLY QUOTE THE KEYS AND VALUES IN THE DICTIONARY FOR EASIER PARSING WITH Python's ast.literal_eval."
    )

    response = model.generate_content([prompt, img])
    print(response.text)
    answers = []
    try:
        answers = ast.literal_eval(response.text)
    except Exception as e:
        return print("Error in response: {e}")

    print(answers)
    for answer in answers:
        if 'assign' in answer:
            answer['assign'] = True
        else:
            answer['assign'] = False

    return answers
