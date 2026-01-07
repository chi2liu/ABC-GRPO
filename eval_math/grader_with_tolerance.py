#!/usr/bin/env python3
"""
Improved grader with 2% relative error or 0.05 absolute error tolerance.
"""

import re
import math
from typing import Union, Optional, Tuple

def extract_numerical_value(text: str) -> Optional[float]:
    """
    Extract numerical value from text, handling various formats.

    Args:
        text: Input text containing a number

    Returns:
        Extracted numerical value or None if not found
    """
    if not text:
        return None

    text = str(text).strip()

    # Handle list format
    if isinstance(text, list) and text:
        text = str(text[0])

    # Remove common answer prefixes
    text = re.sub(r'^(the\s+)?answer\s+(is|:|=)\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^(therefore|thus|hence|so),?\s*', '', text, flags=re.IGNORECASE)

    # Handle scientific notation (e.g., 1.23e45, 1.23×10^45)
    scientific_patterns = [
        (r'([+-]?\d*\.?\d+)[×xX]?\s*10\^{?([+-]?\d+)}?', lambda m: float(m.group(1)) * (10 ** int(m.group(2)))),
        (r'([+-]?\d*\.?\d+)e([+-]?\d+)', lambda m: float(m.group(1)) * (10 ** int(m.group(2))))
    ]

    for pattern, converter in scientific_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                return converter(match)
            except:
                pass

    # Handle fractions (e.g., 1/2, 3/4)
    fraction_pattern = r'^([+-]?\d+)\s*/\s*([+-]?\d+)$'
    match = re.search(fraction_pattern, text)
    if match:
        try:
            numerator = float(match.group(1))
            denominator = float(match.group(2))
            if denominator != 0:
                return numerator / denominator
        except:
            pass

    # Handle percentages
    percent_pattern = r'^([+-]?\d*\.?\d+)\s*%$'
    match = re.search(percent_pattern, text)
    if match:
        try:
            return float(match.group(1)) / 100
        except:
            pass

    # Extract plain numbers
    number_patterns = [
        r'^([+-]?\d+\.?\d*)\s*$',  # Standard number
        r'^([+-]?\d+\.?\d*)[^\d]*$',  # Number followed by non-digits
        r'([+-]?\d+\.?\d*)',  # Any number in the text
    ]

    for pattern in number_patterns:
        match = re.search(pattern, text)
        if match:
            try:
                return float(match.group(1))
            except:
                continue

    return None


def check_numerical_equality_with_tolerance(
    prediction: Union[str, float],
    reference: Union[str, float],
    rel_tol: float = 0.02,  # 2% relative tolerance
    abs_tol: float = 0.05    # 0.05 absolute tolerance
) -> Tuple[bool, str]:
    """
    Check if two values are numerically equal within tolerance.

    Args:
        prediction: Predicted value (string or number)
        reference: Reference/correct value (string or number)
        rel_tol: Relative tolerance (default 2%)
        abs_tol: Absolute tolerance (default 0.05)

    Returns:
        Tuple of (is_equal, reason_message)
    """
    # Extract numerical values
    pred_num = extract_numerical_value(prediction) if isinstance(prediction, str) else prediction
    ref_num = extract_numerical_value(reference) if isinstance(reference, str) else reference

    # If we couldn't extract numbers, return False
    if pred_num is None or ref_num is None:
        return False, f"Could not extract numbers (pred={prediction}, ref={reference})"

    # Check for exact equality first
    if pred_num == ref_num:
        return True, f"Exact match: {pred_num}"

    # Check for special cases
    if math.isnan(pred_num) or math.isnan(ref_num):
        return False, "NaN values cannot be compared"

    if math.isinf(pred_num) or math.isinf(ref_num):
        return pred_num == ref_num, f"Infinity comparison: {pred_num} vs {ref_num}"

    # Calculate differences
    abs_diff = abs(pred_num - ref_num)

    # For small reference values, use absolute tolerance
    if abs(ref_num) < 1:
        is_equal = abs_diff <= abs_tol
        if is_equal:
            return True, f"Within absolute tolerance: |{pred_num} - {ref_num}| = {abs_diff:.6f} ≤ {abs_tol}"
        else:
            return False, f"Outside absolute tolerance: |{pred_num} - {ref_num}| = {abs_diff:.6f} > {abs_tol}"

    # For larger values, use relative tolerance
    rel_diff = abs_diff / abs(ref_num)
    is_equal = rel_diff <= rel_tol

    if is_equal:
        return True, f"Within relative tolerance: |{pred_num} - {ref_num}|/|{ref_num}| = {rel_diff:.4%} ≤ {rel_tol:.1%}"
    else:
        return False, f"Outside relative tolerance: |{pred_num} - {ref_num}|/|{ref_num}| = {rel_diff:.4%} > {rel_tol:.1%}"


def improved_math_equal(
    prediction: str,
    reference: str,
    rel_tol: float = 0.02,
    abs_tol: float = 0.05
) -> bool:
    """
    Main function to check if prediction equals reference with tolerance.

    Args:
        prediction: Model's prediction
        reference: Correct answer
        rel_tol: Relative tolerance (default 2%)
        abs_tol: Absolute tolerance (default 0.05)

    Returns:
        True if answers are equal within tolerance
    """
    is_equal, _ = check_numerical_equality_with_tolerance(
        prediction, reference, rel_tol, abs_tol
    )
    return is_equal


def test_tolerance_grader():
    """Test cases for the tolerance-based grader."""

    test_cases = [
        # (prediction, reference, expected_result, description)
        ("1.57", "1.6", True, "Within 2% tolerance"),
        ("1.57", "1.5708", True, "Very close values"),
        ("100", "98", True, "Within 2% for large numbers"),
        ("100", "97", False, "Outside 2% for large numbers"),
        ("0.01", "0.02", True, "Within absolute tolerance for small numbers"),
        ("0.01", "0.1", False, "Outside absolute tolerance for small numbers"),
        ("The answer is 1.57", "1.6", True, "Extract from text"),
        ("1.23e5", "123000", True, "Scientific notation"),
        ("1/2", "0.5", True, "Fraction conversion"),
        ("50%", "0.5", True, "Percentage conversion"),
    ]

    print("Testing Tolerance Grader")
    print("=" * 60)
    print(f"Settings: Relative Tolerance = 2%, Absolute Tolerance = 0.05")
    print("=" * 60)

    for pred, ref, expected, desc in test_cases:
        result = improved_math_equal(pred, ref)
        status = "✓" if result == expected else "✗"

        # Get detailed reason
        _, reason = check_numerical_equality_with_tolerance(pred, ref)

        print(f"\n{status} {desc}")
        print(f"  Prediction: {pred}")
        print(f"  Reference: {ref}")
        print(f"  Result: {result} (Expected: {expected})")
        print(f"  Reason: {reason}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    # Run tests
    test_tolerance_grader()

    # Interactive testing
    print("\nInteractive Testing Mode")
    print("Enter 'quit' to exit\n")

    while True:
        pred = input("Enter prediction: ").strip()
        if pred.lower() == 'quit':
            break

        ref = input("Enter reference: ").strip()
        if ref.lower() == 'quit':
            break

        is_equal, reason = check_numerical_equality_with_tolerance(pred, ref)

        print(f"\nResult: {'CORRECT' if is_equal else 'INCORRECT'}")
        print(f"Reason: {reason}")
        print("-" * 40)