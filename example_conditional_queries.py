"""
Example usage of conditional query VQA interpretation with LLM parsing.

This demonstrates how to ask complex conditional questions about videos.
"""

from src import load_model, conditional_query_vqa_interpret


def main():
    # Load the model
    print("Loading model...")
    model, processor = load_model()

    # Path to your video
    video_path = "path/to/your/video.mp4"

    # Complex conditional questions
    # The LLM will automatically parse these into:
    # 1. Frame identification condition
    # 2. Question to answer on matching frames

    conditional_questions = [
        # Conditional queries - find frames matching condition, then answer question
        "When there is a dog in the frame, what color is the dog?",
        "Where the person is running, what are they wearing?",
        "When there is a car visible, is it red or blue?",
        "At frames with a cat, what is the cat doing?",

        # Timestamp queries - still supported
        "What is happening at 5 seconds?",
        "What's in the first 10 seconds?",

        # Simple queries - analyze all frames
        "What objects appear in the video?",
    ]

    # Run conditional query analysis
    # The system will:
    # 1. Use LLM to parse each question
    # 2. For conditional questions:
    #    a. Scan all frames to find those matching the condition
    #    b. Answer the question only on matching frames
    # 3. Aggregate results

    results = conditional_query_vqa_interpret(
        video_path=video_path,
        questions=conditional_questions,
        model=model,
        processor=processor,
        fps_sample=1,  # Sample 1 frame per second
        aggregation_method='most_confident',  # Options: 'most_confident', 'consensus', 'average'
        confidence_threshold=0.5,  # Minimum confidence to consider frame as matching condition
        show_visualizations=True,
        save_results=True,
        include_text_attribution=False
    )

    # Access results
    print("\n" + "=" * 70)
    print("SUMMARY OF RESULTS")
    print("=" * 70)

    for question, result in results.items():
        if 'error' in result:
            print(f"\nQ: {question}")
            print(f"Error: {result['error']}")
            continue

        print(f"\nQ: {question}")

        # Show parsing result
        parsed = result['query_info']['parsed']
        print(f"Type: {parsed['type']}")

        if parsed['type'] == 'conditional':
            print(f"Condition: {parsed['frame_condition']}")
            print(f"Frames matched: {result['query_info']['num_matching_frames']}/{result['query_info']['total_frames']}")
            print(f"Matching frame indices: {result['query_info']['frame_indices']}")

        # Show answer
        print(f"Answer: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.2%}")


if __name__ == "__main__":
    main()
