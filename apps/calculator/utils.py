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
        """
    You have been given an image with some mathematical expressions, equations, or graphical problems, and you need to solve them step-by-step, showing the complete solution. 
    Note: Use the PEMDAS rule for solving mathematical expressions. PEMDAS stands for the Priority Order: Parentheses, Exponents, Multiplication and Division (from left to right), Addition and Subtraction (from left to right). 
    Parentheses have the highest priority, followed by Exponents, then Multiplication and Division, and lastly Addition and Subtraction. 
    For example: 

    Q). 2 + 3 * 4 
    Ans). Here's Solution :
    first we can calulate 3 * 4 = 12
    then we can add 2 + 12 = 14
    Final answer: 14

    Q). 2 + 3 + 5 * 4 - 8 / 2 
    Ans). Here's Solution :
    first we can calulate 5 * 4 = 20 so equation is 2 + 3 + 20 - 8 / 2
    then we can calulate 3 + 20 = 23 so equation is 2 + 23 - 8 / 2
    then we can calulate 2 + 23 = 25 so equation is 25 - 8 / 2
    then we can calulate 8 / 2 = 4 so equation is 25 - 4
    and then we can calulate 25 - 4 = 21 
    so Final answer: 21

    Q). A = ((4 7), (2 6)). find the inverse of A 2x2 matix, if exists
    Ans). Here's Solution :
    To find the inverse of a 2x2 matrix, we use the following formula:
             A^-1 = 1/det(A) * adj(A) where det(A) = ad - bc and adj(A) = ((d -b),(c - a))
    so first we can calculate determinant of A
             det(A) = (4 * 6) - (7 * 2) = 24 - 14 = 10
    then we can calculate the adjoint of A
             adj(A) = ((6, -7), (-2, 4))
    then we can calculate the inverse of A
             A^-1 = 1/10 * ((6, -7), (-2, 4))
                  = ((0.6 -0.7), (-0.2 0.4))
    
    so Final answer: ((0.6 -0.7), (-0.2 0.4))

    Q). Find the derivative of the function f(x) = 3x^4 - 5x^3 + 2x - 7.
    Ans). Here's Solution :
    To find the derivative of the given function, we'll use the power rule for differentiation. The power rule states that for any term ax^n, the derivative of the function is ax^(n-1).

    given f(x) = 3x^4 - 5x^3 + 2x - 7, let's differentiate each terms.
       first then derivative of 3x^4 = d/dx (3x^4) = 3*4*x^(4-1) = 12x^3
       second derivative of -5x^3 = d/dx (-5x^3) = -5*3*x^(3-1) = -15x^2
       then derivative of 2x = d/dx (2x) = 2*x^(2-1) = 2x
       then derivative of -7 = d/dx (-7) = -7

    then combine all result, f'(x) = 12x^3 - 15x^2 + 2x - 7
    so Final answer: 12x^3 - 15x^2 + 2x - 7

    Q). Evaluate the following integral: ∫(3x^2 - 4x + 5)dx
    Ans). Here's Solution :
    To solve this integral, we'll apply the power rule for integration. The power rule states that for any term x^n, the integral is 
                        ∫ x^ndx = [x^(n+1)/(n+1)] + C, where C is the constant term.
    so we can integrate all terms:
    first we can integrate ∫3x^2dx = 3 * [x^(2+1)/(2+1)]
                                   = 3 * [x^3/3]
                                   = x^3
    second we can integrate ∫-4xdx = -4 * [x^(1+1)/(1+1)]
                                   = -4 * [x^2/2]
                                   = -2x^2
    third we can integrate ∫5dx    = 5x
    and then combine all result, ∫(3x^2 - 4x + 5)dx = x^3 - 2x^2 + 5x + C
    so Final answer: x^3 - 2x^2 + 5x + C, where C is the constant term

    
    YOU CAN HAVE FIVE TYPES OF EQUATIONS/EXPRESSIONS IN THIS IMAGE, AND ONLY ONE CASE SHALL APPLY EVERY TIME: 
    Following are the cases: 

    1. Simple mathematical expressions like 2 + 2, 3 * 4, 5 / 6, 7 - 8, etc.: In this case, solve step-by-step, showing the operations clearly, and return the answer in the format of a LIST OF ONE DICT [{'expr': given expression, 'solution': detailed solution, 'result': calculated answer}].

    2. Set of Equations like x^2 + 2x + 1 = 0, 3y + 4x = 0, 5x^2 + 6y + 7 = 12, etc.: In this case, solve step-by-step for each variable, and return a COMMA-SEPARATED LIST OF DICTS, with each dict containing {'expr': 'variable equation', 'solution': detailed solution, 'result': calculated value}.

    3. Assigning values to variables like x = 4, y = 5, z = 6, etc.: In this case, assign values to variables and return another key in the dict called {'assign': True}, keeping the variable as 'expr' and the value as 'result', and include a 'solution' key showing how the variables were assigned. RETURN AS A LIST OF DICTS.

    4. Analyzing Graphical Math problems (such as word problems represented in drawing form) like cars colliding, trigonometric problems, or the Pythagorean theorem: Solve step-by-step, and return the answer in the format of a LIST OF ONE DICT [{'expr': given expression, 'solution': detailed steps of calculation, 'result': final answer}].
    
    5. Detecting Abstract Concepts from a drawing, such as historical references or symbolic representations like love, hate, or war: Provide the explanation of the drawing in 'solution', and return the abstract concept in 'result'.
    
    Analyze the equation or expression in this image, solve it step-by-step, and return both the full solution with explanation and the final answer according to the given rules: 
    Make sure to use extra backslashes for escape characters like \\f -> \\\\f, \\n -> \\\\n, etc. 
    Here is a dictionary of user-assigned variables. If the given expression has any of these variables, use its actual value from this dictionary accordingly: """ + str(dict_of_vars_str) + """. 
    DO NOT USE BACKTICKS OR MARKDOWN FORMATTING. 
    PROPERLY QUOTE THE KEYS AND VALUES IN THE DICTIONARY FOR EASIER PARSING WITH Python's ast.literal_eval.
    """
    )

    response = model.generate_content([prompt, img])
    answers = []
    try:
        answers = ast.literal_eval(response.text)
    except Exception as e:
        return print("Error in response: {e}")

    for answer in answers:
        if 'assign' in answer:
            answer['assign'] = True
        else:
            answer['assign'] = False

    return answers
