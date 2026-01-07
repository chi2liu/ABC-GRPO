"""
Improved grader with better handling of scientific notation and numerical tolerance.
Based on the original grader.py but with enhanced evaluation logic.
"""

import re
import regex
import multiprocessing
from math import isclose
from typing import Union
from collections import defaultdict

from sympy import simplify, N
from sympy.parsing.sympy_parser import parse_expr
from sympy.parsing.latex import parse_latex
from latex2sympy2 import latex2sympy


def choice_answer_clean(pred: str):
    pred = pred.strip("\n").rstrip(".").rstrip("/").strip(" ").lstrip(":")
    # Clean the answer based on the dataset
    tmp = re.findall(r"\b(A|B|C|D|E)\b", pred.upper())
    if tmp:
        pred = tmp
    else:
        pred = [pred.strip().strip(".")]
    pred = pred[-1]
    # Remove the period at the end, again!
    pred = pred.rstrip(".").rstrip("/")
    return pred


def normalize_scientific_notation(s: str) -> float:
    """Convert various scientific notation formats to float"""
    if not s:
        return None

    s = str(s).strip()

    # Remove LaTeX formatting
    s = s.replace('\\times', '×').replace('\\cdot', '×')
    s = s.replace('\\,', '').replace(' ', '')
    s = s.replace('{', '').replace('}', '')

    # Patterns for scientific notation
    patterns = [
        # 1.23×10^45, 1.23x10^45
        (r'^([+-]?\d*\.?\d+)[×xX]?10\^([+-]?\d+)$',
         lambda m: float(m.group(1)) * (10 ** int(m.group(2)))),

        # 1.23e45, 1.23E45
        (r'^([+-]?\d*\.?\d+)[eE]([+-]?\d+)$',
         lambda m: float(m.group(1)) * (10 ** int(m.group(2)))),

        # Just a plain number
        (r'^([+-]?\d*\.?\d+)$',
         lambda m: float(m.group(1))),
    ]

    for pattern, converter in patterns:
        match = re.match(pattern, s)
        if match:
            try:
                return converter(match)
            except:
                pass

    return None


def parse_digits(num):
    """Enhanced digit parser with scientific notation support"""
    num = regex.sub(",", "", str(num))

    # First try scientific notation normalization
    result = normalize_scientific_notation(num)
    if result is not None:
        return result

    # Original parsing logic
    try:
        return float(num)
    except:
        if num.endswith("%"):
            num = num[:-1]
            if num.endswith("\\"):
                num = num[:-1]
            try:
                return float(num) / 100
            except:
                pass
    return None


def is_digit(num):
    # paired with parse_digits
    return parse_digits(num) is not None


def str_to_pmatrix(input_str):
    input_str = input_str.strip()
    matrix_str = re.findall(r"\{.*,.*\}", input_str)
    pmatrix_list = []

    for m in matrix_str:
        m = m.strip("{}")
        pmatrix = r"\begin{pmatrix}" + m.replace(",", "\\") + r"\end{pmatrix}"
        pmatrix_list.append(pmatrix)

    return ", ".join(pmatrix_list)


def numeric_equal(prediction: float, reference: float, rel_tol: float = 0.01):
    """
    Enhanced numeric comparison with configurable tolerance.
    Default relative tolerance increased to 1% to handle measurement uncertainties.
    """
    # For very small numbers, use absolute tolerance
    if abs(reference) < 1e-10:
        return isclose(reference, prediction, abs_tol=1e-8)

    # Use relative tolerance for normal numbers
    return isclose(reference, prediction, rel_tol=rel_tol)


def normalize_symbolic(s: str) -> str:
    """Normalize symbolic expressions for comparison"""
    if not s:
        return ""

    s = str(s).strip()

    # Remove common variations
    s = s.replace('\\', '')  # Remove LaTeX backslashes
    s = s.replace(' ', '')    # Remove spaces
    s = s.replace('_0', '0')  # x_0 -> x0
    s = s.replace('_{', '{')

    # Handle equation format: remove "variable=" part if it exists
    if '=' in s and len(s.split('=')) == 2:
        left, right = s.split('=')
        # If left side is just a variable or function notation (e.g., I(t), f(x))
        if re.match(r'^[a-zA-Z]\([^)]+\)$', left.strip()) or len(left.strip()) <= 3:
            s = right.strip()

    # Normalize common functions
    s = s.replace('sin(', 'sin(')
    s = s.replace('cos(', 'cos(')
    s = s.replace('exp(', 'exp(')
    s = s.replace('ln(', 'log(')

    # Normalize fractions
    s = re.sub(r'frac{([^}]+)}{([^}]+)}', r'(\1)/(\2)', s)

    return s


def math_equal(
    prediction: Union[bool, float, str],
    reference: Union[float, str],
    include_percentage: bool = True,
    is_close: bool = True,
    timeout: bool = False,
    rel_tol: float = 0.01,  # Increased default tolerance to 1%
) -> bool:
    """
    Improved math equality check with better handling of:
    1. Scientific notation formats
    2. Relative tolerance for numerical comparisons
    3. Symbolic expression normalization
    """
    if prediction is None or reference is None:
        return False

    # Direct string comparison (case-insensitive)
    if str(prediction).strip().lower() == str(reference).strip().lower():
        return True

    # Multiple choice handling
    if reference in ["A", "B", "C", "D", "E"] and choice_answer_clean(prediction) == reference:
        return True

    # Enhanced numerical comparison
    try:
        if is_digit(prediction) and is_digit(reference):
            prediction = parse_digits(prediction)
            reference = parse_digits(reference)

            # Handle percentage variations
            if include_percentage:
                gt_result = [reference / 100, reference, reference * 100]
            else:
                gt_result = [reference]

            for item in gt_result:
                try:
                    if is_close:
                        if numeric_equal(prediction, item, rel_tol=rel_tol):
                            return True
                    else:
                        if item == prediction:
                            return True
                except Exception:
                    continue
            return False
    except:
        pass

    if not prediction and prediction not in [0, False]:
        return False

    # Symbolic comparison with normalization
    reference = str(reference).strip()
    prediction = str(prediction).strip()

    # Normalize both expressions
    ref_norm = normalize_symbolic(reference)
    pred_norm = normalize_symbolic(prediction)

    # Check if normalized forms match
    if ref_norm == pred_norm:
        return True

    # Check alphanumeric-only version
    ref_clean = re.sub(r'[^a-zA-Z0-9]', '', ref_norm)
    pred_clean = re.sub(r'[^a-zA-Z0-9]', '', pred_norm)

    if ref_clean == pred_clean:
        return True

    ## pmatrix (amps)
    if "pmatrix" in prediction and not "pmatrix" in reference:
        reference = str_to_pmatrix(reference)

    ## deal with [], (), {}
    pred_str, ref_str = prediction, reference
    if (
        prediction.startswith("[")
        and prediction.endswith("]")
        and not reference.startswith("(")
    ) or (
        prediction.startswith("(")
        and prediction.endswith(")")
        and not reference.startswith("[")
    ):
        pred_str = pred_str.strip("[]()")
        ref_str = ref_str.strip("[]()")
    for s in ["{", "}", "(", ")"]:
        ref_str = ref_str.replace(s, "")
        pred_str = pred_str.replace(s, "")
    if pred_str.lower() == ref_str.lower():
        return True

    ## [a, b] vs. [c, d], return a==c and b==d
    if (
        regex.match(r"(\(|\[).+(\)|\])", prediction) is not None
        and regex.match(r"(\(|\[).+(\)|\])", reference) is not None
    ):
        pred_parts = prediction[1:-1].split(",")
        ref_parts = reference[1:-1].split(",")
        if len(pred_parts) == len(ref_parts):
            if all(
                [
                    math_equal(
                        pred_parts[i], ref_parts[i], include_percentage, is_close,
                        timeout, rel_tol
                    )
                    for i in range(len(pred_parts))
                ]
            ):
                return True

    # Matrix comparison
    if (
        (
            prediction.startswith("\\begin{pmatrix}")
            or prediction.startswith("\\begin{bmatrix}")
        )
        and (
            prediction.endswith("\\end{pmatrix}")
            or prediction.endswith("\\end{bmatrix}")
        )
        and (
            reference.startswith("\\begin{pmatrix}")
            or reference.startswith("\\begin{bmatrix}")
        )
        and (
            reference.endswith("\\end{pmatrix}") or reference.endswith("\\end{bmatrix}")
        )
    ):
        pred_lines = [
            line.strip()
            for line in prediction[
                len("\\begin{pmatrix}") : -len("\\end{pmatrix}")
            ].split("\\\\")
            if line.strip()
        ]
        ref_lines = [
            line.strip()
            for line in reference[
                len("\\begin{pmatrix}") : -len("\\end{pmatrix}")
            ].split("\\\\")
            if line.strip()
        ]
        matched = True
        if len(pred_lines) == len(ref_lines):
            for pred_line, ref_line in zip(pred_lines, ref_lines):
                pred_parts = pred_line.split("&")
                ref_parts = ref_line.split("&")
                if len(pred_parts) == len(ref_parts):
                    if not all(
                        [
                            math_equal(
                                pred_parts[i],
                                ref_parts[i],
                                include_percentage,
                                is_close,
                                timeout,
                                rel_tol
                            )
                            for i in range(len(pred_parts))
                        ]
                    ):
                        matched = False
                        break
                else:
                    matched = False
                if not matched:
                    break
        else:
            matched = False
        if matched:
            return True

    # Equation handling
    if prediction.count("=") == 1 and reference.count("=") == 1:
        pred = prediction.split("=")
        pred = f"{pred[0].strip()} - ({pred[1].strip()})"
        ref = reference.split("=")
        ref = f"{ref[0].strip()} - ({ref[1].strip()})"
        if symbolic_equal(pred, ref) or symbolic_equal(f"-({pred})", ref):
            return True
    elif (
        prediction.count("=") == 1
        and len(prediction.split("=")[0].strip()) <= 2
        and "=" not in reference
    ):
        if math_equal(
            prediction.split("=")[1], reference, include_percentage, is_close,
            timeout, rel_tol
        ):
            return True
    elif (
        reference.count("=") == 1
        and len(reference.split("=")[0].strip()) <= 2
        and "=" not in prediction
    ):
        if math_equal(
            prediction, reference.split("=")[1], include_percentage, is_close,
            timeout, rel_tol
        ):
            return True

    # symbolic equal with sympy
    if timeout:
        if call_with_timeout(symbolic_equal_process, prediction, reference):
            return True
    else:
        if symbolic_equal(prediction, reference):
            return True

    return False


def math_equal_process(param):
    """Process function for multiprocessing with improved tolerance"""
    return math_equal(param[-2], param[-1], rel_tol=0.01)


def symbolic_equal(a, b):
    def _parse(s):
        for f in [parse_latex, parse_expr, latex2sympy]:
            try:
                return f(s.replace("\\\\", "\\"))
            except:
                try:
                    return f(s)
                except:
                    pass
        return s

    a = _parse(a)
    b = _parse(b)

    # direct equal
    try:
        if str(a) == str(b) or a == b:
            return True
    except:
        pass

    # simplify equal
    try:
        if a.equals(b) or simplify(a - b) == 0:
            return True
    except:
        pass

    # equation equal
    try:
        if (abs(a.lhs - a.rhs)).equals(abs(b.lhs - b.rhs)):
            return True
    except:
        pass

    try:
        if numeric_equal(float(N(a)), float(N(b)), rel_tol=0.01):
            return True
    except:
        pass

    # matrix
    try:
        # if a and b are matrix
        if a.shape == b.shape:
            _a = a.applyfunc(lambda x: round(x, 3))
            _b = b.applyfunc(lambda x: round(x, 3))
            if _a.equals(_b):
                return True
    except:
        pass

    return False


def symbolic_equal_process(a, b, output_queue):
    result = symbolic_equal(a, b)
    output_queue.put(result)


def call_with_timeout(func, *args, timeout=1, **kwargs):
    output_queue = multiprocessing.Queue()
    process_args = args + (output_queue,)
    process = multiprocessing.Process(target=func, args=process_args, kwargs=kwargs)
    process.start()
    process.join(timeout)

    if process.is_alive():
        process.terminate()
        process.join()
        return False

    return output_queue.get()


def _test_math_equal():
    """Test cases for improved evaluator"""
    test_cases = [
        ("4.49", "4.482", True),  # Within 1% tolerance
        ("3.3e12", "3.275\\times10^{12}", True),  # Different notation
        ("2.7778e-6", "2.78\\times10^{-6}", True),  # Scientific notation
        ("0.192", "0.1919", True),  # Within tolerance
        ("7e37", "7.63\\times10^{33}", False),  # Order of magnitude error
        ("x_{0}\\cos(\\omega t)", "x_0\\cos(\\omega t)", True),  # LaTeX variations
        ("I(0)e^{-\\frac{t}{RC}}", "I(t)=I(0)e^{-\\frac{t}{RC}}", True),  # Equation format
    ]

    print("Testing Improved Math Equal:")
    print("="*60)

    for gt, pred, expected in test_cases:
        result = math_equal(pred, gt, timeout=False)
        status = "✓" if result == expected else "✗"
        print(f"{status} GT: '{gt}'")
        print(f"  Pred: '{pred}'")
        print(f"  Result: {result}, Expected: {expected}")
        print()


if __name__ == "__main__":
    _test_math_equal()