"""
Example usage of temporal query VQA interpretation.

This demonstrates how to ask questions about specific timestamps in a video.
"""

from src import load_model, temporal_query_vqa_interpret


def main():
    # Load the model
    print("Loading model...")
    model, processor = load_model()

    # Path to your video
    video_path = "path/to/your/video.mp4"

    # Questions with temporal references
    temporal_questions = [
        "What is happening at 4 seconds?",
        "What's going on in the 5th second of the video?",
        "What happens in the first 10 seconds?",
        "What occurs between 5 and 8 seconds?",
        "What's at the beginning of the video?",
        "What's in the middle of the video?",
        "What happens at the end?",
    ]

    # Run temporal query analysis
    results = temporal_query_vqa_interpret(
        video_path=video_path,
        temporal_questions=temporal_questions,
        model=model,
        processor=processor,
        fps_sample=1,  # Sample 1 frame per second
        aggregation_method='most_confident',  # Options: 'most_confident', 'consensus', 'average'
        show_visualizations=True,
        save_results=True,
        include_text_attribution=False
    )

    # Access results
    for question, result in results.items():
        print(f"\nQuestion: {question}")
        print(f"Time Range: {result['temporal_info']['start_sec']:.2f}s - {result['temporal_info']['end_sec']:.2f}s")
        print(f"Answer: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Frames analyzed: {result['temporal_info']['num_frames']}")


if __name__ == "__main__":
    main()
