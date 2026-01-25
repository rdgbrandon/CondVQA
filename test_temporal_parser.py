"""
Test script for temporal query parser.
Tests the parse_temporal_reference function without requiring a video.
"""

from src.frame_extension import parse_temporal_reference


def test_parser():
    """Test various temporal reference patterns."""

    video_duration = 60.0  # 60 second video for testing

    test_cases = [
        # (question, expected_start, expected_end, description)
        ("What is happening at 4 seconds?", 4.0, 4.0, "Specific time"),
        ("What's at 5.5 seconds?", 5.5, 5.5, "Decimal time"),
        ("What's in the 4th second?", 4.0, 4.0, "Ordinal"),
        ("What happens in the first 10 seconds?", 0.0, 10.0, "First N seconds"),
        ("What's in the last 5 seconds?", 55.0, 60.0, "Last N seconds"),
        ("What occurs between 5 and 8 seconds?", 5.0, 8.0, "Between X and Y"),
        ("What happens from 2 to 6 seconds?", 2.0, 6.0, "From X to Y"),
        ("What's at the beginning?", 0.0, 5.0, "Beginning (5s or 20%)"),
        ("What's in the middle of the video?", 27.5, 32.5, "Middle (5s window)"),
        ("What happens at the end?", 55.0, 60.0, "End (last 5s or 20%)"),
        ("What color is the car?", 0.0, 60.0, "No temporal reference"),
    ]

    print("=" * 80)
    print("TEMPORAL PARSER TEST RESULTS")
    print("=" * 80)
    print(f"Video Duration: {video_duration}s\n")

    passed = 0
    failed = 0

    for question, expected_start, expected_end, description in test_cases:
        start, end, core_q = parse_temporal_reference(question, video_duration)

        # Allow small floating point tolerance
        tolerance = 0.1
        start_ok = abs(start - expected_start) < tolerance
        end_ok = abs(end - expected_end) < tolerance

        if start_ok and end_ok:
            status = "[PASS]"
            passed += 1
        else:
            status = "[FAIL]"
            failed += 1

        print(f"{status} | {description}")
        print(f"  Q: {question}")
        print(f"  Expected: {expected_start:.1f}s - {expected_end:.1f}s")
        print(f"  Got:      {start:.1f}s - {end:.1f}s")
        print(f"  Core Q:   {core_q}")
        print()

    print("=" * 80)
    print(f"Results: {passed} passed, {failed} failed out of {len(test_cases)} tests")
    print("=" * 80)

    return passed, failed


if __name__ == "__main__":
    passed, failed = test_parser()

    if failed == 0:
        print("\n[SUCCESS] All tests passed! The temporal parser is working correctly.")
    else:
        print(f"\n[WARNING] {failed} test(s) failed. Review the output above.")
