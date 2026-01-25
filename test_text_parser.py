"""
Test the new text-based question parser with problematic cases.

This tests the improved parser that uses Qwen2.5 text model instead of LLaVA.
"""

from src.model_loader import load_text_model
from src.frame_extension import parse_question_with_llm


def test_parser():
    print("Loading text model for testing...")
    text_model, text_tokenizer = load_text_model()
    print()

    # Test cases that were previously misclassified
    test_questions = [
        # Should be CONDITIONAL
        ("When there is a cat, what color is the cat?", "conditional"),
        ("When there is a dog in the frame, what color is the dog?", "conditional"),
        ("Where the person is running, what are they wearing?", "conditional"),
        ("At frames with a car, is it red or blue?", "conditional"),
        ("While it's raining, what color is the umbrella?", "conditional"),

        # Should be TIMESTAMP
        ("What's at the beginning?", "timestamp"),
        ("What happens at 4 seconds?", "timestamp"),
        ("What's in the first 10 seconds?", "timestamp"),
        ("What occurs at the end?", "timestamp"),
        ("What is happening at 5 seconds?", "timestamp"),

        # Should be SIMPLE
        ("What is in this video?", "simple"),
        ("What objects appear in the video?", "simple"),
        ("What is the dominant color?", "simple"),
    ]

    print("=" * 70)
    print("TESTING QUESTION PARSER")
    print("=" * 70)
    print()

    passed = 0
    failed = 0

    for question, expected_type in test_questions:
        parsed = parse_question_with_llm(question, text_model, text_tokenizer)
        actual_type = parsed['type']

        status = "[PASS]" if actual_type == expected_type else "[FAIL]"
        if actual_type == expected_type:
            passed += 1
        else:
            failed += 1

        print(f"{status} Q: {question}")
        print(f"       Expected: {expected_type}, Got: {actual_type}")

        if actual_type == 'conditional':
            print(f"       Condition: {parsed['frame_condition']}")
            print(f"       Question: {parsed['answer_question']}")
        elif actual_type == 'timestamp':
            print(f"       Time: {parsed['timestamp']}")
            print(f"       Question: {parsed['answer_question']}")
        elif actual_type == 'simple':
            print(f"       Question: {parsed['answer_question']}")

        print()

    print("=" * 70)
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(test_questions)} tests")
    print("=" * 70)

    if failed == 0:
        print("[SUCCESS] All tests passed!")
    else:
        print(f"[WARNING] {failed} test(s) failed")

    return failed == 0


if __name__ == "__main__":
    success = test_parser()
    exit(0 if success else 1)
