# Temporal Query VQA - Timestamp-Specific Video Analysis

This feature allows you to ask questions about specific timestamps in a video, automatically extracting and analyzing only the relevant frames.

## Overview

The `temporal_query_vqa_interpret()` function extends the temporal video analysis capabilities by:
1. **Parsing natural language temporal references** (e.g., "at 4 seconds", "first 10 seconds")
2. **Extracting only relevant frames** for the specified time range
3. **Computing Integrated Gradients attributions** on those frames
4. **Aggregating results** into a single answer with confidence score

## Features

### Supported Temporal References

The parser understands various temporal expressions:

| Pattern | Example | Parsed Range |
|---------|---------|--------------|
| Specific time | "What's at 4 seconds?" | 4.0s - 4.0s |
| Ordinal second | "What's in the 5th second?" | 5.0s - 5.0s |
| First N seconds | "What happens in the first 10 seconds?" | 0.0s - 10.0s |
| Last N seconds | "What's in the last 5 seconds?" | (end-5)s - end |
| Time range | "What occurs between 5 and 8 seconds?" | 5.0s - 8.0s |
| Time range (alt) | "What happens from 2 to 6 seconds?" | 2.0s - 6.0s |
| Beginning | "What's at the beginning?" | 0.0s - 5.0s* |
| Middle | "What's in the middle?" | (mid-2.5)s - (mid+2.5)s* |
| End | "What happens at the end?" | (end-5)s - end* |

*Adaptive based on video duration

### Aggregation Methods

When multiple frames fall within a time range, you can choose how to combine results:

1. **`most_confident`** (default)
   - Selects the frame with the highest confidence score
   - Best for getting the clearest answer

2. **`consensus`**
   - Uses the most common prediction across frames
   - Shows what percentage of frames agree
   - Best for robust answers across varying frames

3. **`average`**
   - Averages attribution maps and confidence scores
   - Uses the most common prediction
   - Best for smooth temporal transitions

## Usage

### Basic Example

```python
from src import load_model, temporal_query_vqa_interpret

# Load model
model, processor = load_model()

# Ask temporal questions
results = temporal_query_vqa_interpret(
    video_path="video.mp4",
    temporal_questions=[
        "What is happening at 4 seconds?",
        "What's in the first 10 seconds?",
    ],
    model=model,
    processor=processor,
    fps_sample=1,
    aggregation_method='most_confident'
)

# Access results
for question, result in results.items():
    print(f"Q: {question}")
    print(f"A: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.2%}")
```

### Advanced Example with Text Attribution

```python
results = temporal_query_vqa_interpret(
    video_path="video.mp4",
    temporal_questions=["What happens between 5 and 10 seconds?"],
    model=model,
    processor=processor,
    fps_sample=2,  # 2 frames per second for finer granularity
    aggregation_method='consensus',  # Use voting across frames
    show_visualizations=True,
    save_results=True,
    output_dir="my_analysis",
    include_text_attribution=True  # Also analyze question token importance
)
```

## API Reference

### `temporal_query_vqa_interpret()`

```python
def temporal_query_vqa_interpret(
    video_path: str,
    temporal_questions: List[str],
    model,
    processor,
    fps_sample: int = 1,
    output_dir: str = "temporal_query_analysis",
    aggregation_method: str = 'most_confident',
    show_visualizations: bool = True,
    save_results: bool = True,
    include_text_attribution: bool = False
) -> Dict
```

**Parameters:**
- `video_path`: Path to video file
- `temporal_questions`: List of questions with temporal references
- `model`: Vision-language model (from `load_model()`)
- `processor`: Model's processor (from `load_model()`)
- `fps_sample`: Frames per second to extract (default: 1)
- `output_dir`: Directory to save results
- `aggregation_method`: How to combine multi-frame results
  - `'most_confident'`: Use highest confidence frame
  - `'consensus'`: Use most common prediction
  - `'average'`: Average attributions and confidences
- `show_visualizations`: Display attribution heatmaps
- `save_results`: Save reports and visualizations to disk
- `include_text_attribution`: Compute text token importance

**Returns:**
Dictionary mapping each question to its result containing:
- `prediction`: The model's answer
- `confidence`: Confidence score (0-1)
- `attributions`: Attribution heatmap (for visualization)
- `original_image`: The analyzed frame
- `top_predictions`: Top-k alternative predictions
- `temporal_info`: Dict with:
  - `start_sec`: Start time
  - `end_sec`: End time
  - `core_question`: Question without temporal reference
  - `original_question`: Original question
  - `num_frames`: Number of frames analyzed
  - `individual_results`: Per-frame results

## Implementation Details

### Architecture

```
temporal_query_vqa_interpret()
├── extract_frames() - Extract all frames from video
├── parse_temporal_reference() - Parse temporal expressions
│   ├── Regex pattern matching
│   └── Fallback to full video if no match
├── get_frames_for_timerange() - Select relevant frames
├── process_frame_batch() - Compute IG for each frame
└── aggregate_temporal_results() - Combine results
    ├── most_confident
    ├── consensus
    └── average
```

### LLM Usage

**Primary Method: Regex-based parsing** (No LLM overhead)
- Fast pattern matching for common temporal expressions
- Deterministic and predictable behavior
- No additional API calls or model loading

**Fallback Option: LLM-based parsing** (Not implemented by default)
- Could be added for complex queries like:
  - "When does the car first appear?"
  - "Right before the explosion"
  - "During the conversation"

The current implementation uses **regex-only** for efficiency. If you need semantic temporal understanding, you could extend `parse_temporal_reference()` to use the LLaVA model for complex queries.

## Output Files

When `save_results=True`, the following files are created:

```
temporal_query_analysis/
├── frames/
│   ├── frame_0000.jpg
│   ├── frame_0001.jpg
│   └── ...
├── attribution_q1.png - Visualization for question 1
├── attribution_q2.png - Visualization for question 2
├── report_q1.txt - Detailed report for question 1
└── report_q2.txt - Detailed report for question 2
```

## Performance Considerations

### Frame Sampling Rate
- **Higher `fps_sample`** (e.g., 5): More precise temporal resolution, slower
- **Lower `fps_sample`** (e.g., 1): Faster processing, may miss quick events

### Aggregation Strategy
- **`most_confident`**: Fastest (no averaging)
- **`consensus`**: Fast (simple voting)
- **`average`**: Slowest (computes averages)

### Memory Usage
- Each frame processes independently
- GPU memory is cleared between frames
- No issues with long videos (frames processed sequentially)

## Comparison with Standard Temporal Analysis

| Feature | `temporal_vqa_interpret()` | `temporal_query_vqa_interpret()` |
|---------|----------------------------|----------------------------------|
| Use case | Analyze entire video | Answer timestamp-specific questions |
| Frame selection | All frames | Only relevant frames |
| Questions | Generic (no timestamps) | Temporal references required |
| Speed | Slower (processes all frames) | Faster (processes subset) |
| Output | Timeline + per-frame results | Aggregated answer + confidence |

## Examples

See [example_temporal_queries.py](example_temporal_queries.py) for a complete working example.

### Example Questions

```python
temporal_questions = [
    # Specific timestamps
    "What is the person doing at 3 seconds?",
    "What color is the car at 5.5 seconds?",

    # Time ranges
    "What happens in the first 10 seconds?",
    "What occurs between 15 and 20 seconds?",
    "What's visible from 5 to 8 seconds?",

    # Relative positions
    "What's at the beginning of the video?",
    "What's in the middle?",
    "What happens at the end?",

    # Ordinal references
    "What's in the 4th second?",
    "What happens in the 10th second of the video?",
]
```

## Limitations

1. **Temporal parsing is rule-based**: Complex semantic queries (e.g., "when the door opens") require manual inspection or could be extended with LLM parsing
2. **Frame granularity**: Limited by `fps_sample` - can't analyze sub-frame events
3. **No temporal interpolation**: Analyzes discrete frames, not continuous motion
4. **Single video format**: Expects standard video files readable by OpenCV

## Future Enhancements

Potential additions:
- [ ] LLM-based temporal parsing for complex queries
- [ ] Event detection ("when X happens")
- [ ] Multi-clip aggregation ("compare 2s and 10s")
- [ ] Temporal grounding visualization (highlighting timeline)
- [ ] Support for frame-accurate timestamps (not just seconds)
