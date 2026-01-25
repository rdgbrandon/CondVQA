# Conditional Query VQA - LLM-Based Intelligent Frame Selection

This feature uses LLM-based parsing to handle complex conditional video questions through intelligent two-stage analysis.

## Overview

The `conditional_query_vqa_interpret()` function enables you to ask questions like:
- **"When there is a dog in the frame, what color is the dog?"**
- **"Where the person is running, what are they wearing?"**
- **"At frames with a car, is it red or blue?"**

The system automatically:
1. **Parses** the question using the LLM to identify conditions and questions
2. **Scans** all frames to find those matching the condition
3. **Answers** the question only on matching frames
4. **Aggregates** results into a single answer

## How It Works

### Two-Stage Analysis Process

#### Example: *"When there is a dog in the frame, what color is the dog?"*

**Stage 1: Frame Identification**
- LLM parses question → Condition: "Is there a dog in the frame?"
- Scan all video frames
- Ask condition question on each frame
- Keep frames where answer is "yes" with high confidence

**Stage 2: Question Answering**
- Take only the frames with dogs (from Stage 1)
- Ask: "What color is the dog?"
- Compute Integrated Gradients attribution
- Aggregate answers (most confident / consensus / average)

### LLM-Based Parsing

The system uses the LLaVA model itself to parse questions into structured components:

```
Question: "When there is a dog in the frame, what color is the dog?"

LLM Parser Output:
├─ Type: CONDITIONAL
├─ Condition: "Is there a dog in the frame?"
└─ Question: "What color is the dog?"
```

```
Question: "What happens at 5 seconds?"

LLM Parser Output:
├─ Type: TIMESTAMP
├─ Time: 5.0
└─ Question: "What happens?"
```

```
Question: "What objects are in this video?"

LLM Parser Output:
├─ Type: SIMPLE
└─ Question: "What objects are in this video?"
```

## Supported Question Types

### 1. Conditional Questions

Questions with a condition clause that identifies specific frames:

| Pattern | Example | How it's parsed |
|---------|---------|-----------------|
| When + condition | "When there is a dog, what color is it?" | Condition: "Is there a dog?" |
| Where + condition | "Where the person is running, what are they wearing?" | Condition: "Is the person running?" |
| At frames with X | "At frames with a car, is it red?" | Condition: "Is there a car?" |
| While + condition | "While it's raining, what color is the sky?" | Condition: "Is it raining?" |

### 2. Timestamp Questions

Questions referencing specific times:

- "What happens at 5 seconds?"
- "What's at the 10th second?"
- "What's in the first 10 seconds?"

### 3. Simple Questions

Questions without conditions (analyzes all frames):

- "What objects appear in the video?"
- "What is the dominant color?"

## Usage

### Basic Example

```python
from src import load_model, conditional_query_vqa_interpret

# Load model
model, processor = load_model()

# Ask conditional questions
results = conditional_query_vqa_interpret(
    video_path="video.mp4",
    questions=["When there is a dog in the frame, what color is the dog?"],
    model=model,
    processor=processor,
    fps_sample=1
)

# Access results
for question, result in results.items():
    print(f"Q: {question}")
    print(f"A: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Frames matched: {result['query_info']['num_matching_frames']}")
```

### Advanced Example with Options

```python
results = conditional_query_vqa_interpret(
    video_path="video.mp4",
    questions=[
        "When there is a dog in the frame, what color is the dog?",
        "Where the person is running, what are they wearing?",
    ],
    model=model,
    processor=processor,

    # Frame sampling
    fps_sample=2,  # 2 frames per second for finer granularity

    # Condition matching
    confidence_threshold=0.6,  # Higher threshold = stricter matching

    # Result aggregation
    aggregation_method='consensus',  # Use voting across matching frames

    # Visualization
    show_visualizations=True,
    save_results=True,
    output_dir="my_conditional_analysis",

    # Text attribution
    include_text_attribution=True  # Also analyze question token importance
)
```

## API Reference

### `conditional_query_vqa_interpret()`

```python
def conditional_query_vqa_interpret(
    video_path: str,
    questions: List[str],
    model,
    processor,
    fps_sample: int = 1,
    output_dir: str = "conditional_query_analysis",
    aggregation_method: str = 'most_confident',
    confidence_threshold: float = 0.5,
    show_visualizations: bool = True,
    save_results: bool = True,
    include_text_attribution: bool = False
) -> Dict
```

**Parameters:**

- `video_path`: Path to video file
- `questions`: List of questions (can include conditional clauses, timestamps, or be simple)
- `model`: Vision-language model (from `load_model()`)
- `processor`: Model's processor
- `fps_sample`: Frames per second to extract (default: 1)
  - Higher = more frames to check, slower but more thorough
- `output_dir`: Directory to save results
- `aggregation_method`: How to combine multi-frame results
  - `'most_confident'`: Use frame with highest confidence (default)
  - `'consensus'`: Use most common prediction
  - `'average'`: Average attributions and confidences
- `confidence_threshold`: Minimum confidence for condition matching (default: 0.5)
  - Range: 0.0 to 1.0
  - Higher = stricter frame matching (fewer false positives)
  - Lower = more lenient (may include more frames)
- `show_visualizations`: Display attribution heatmaps
- `save_results`: Save reports and visualizations to disk
- `include_text_attribution`: Compute text token importance

**Returns:**

Dictionary mapping each question to its result:

```python
{
    "When there is a dog, what color is it?": {
        'prediction': 'brown',
        'confidence': 0.87,
        'attributions': <heatmap>,
        'original_image': <image>,
        'top_predictions': [('brown', 0.87), ('black', 0.08), ...],
        'query_info': {
            'parsed': {
                'type': 'conditional',
                'frame_condition': 'Is there a dog?',
                'answer_question': 'What color is it?'
            },
            'frame_indices': [5, 12, 18, 23],  # Frames with dogs
            'num_matching_frames': 4,
            'total_frames': 60,
            'individual_results': [...]  # Per-frame results
        }
    }
}
```

## Implementation Details

### Architecture

```
User Question: "When there is a dog, what color is it?"
                    ↓
┌──────────────────────────────────────────────────────────┐
│ conditional_query_vqa_interpret()                        │
├──────────────────────────────────────────────────────────┤
│ 1. Extract frames (fps_sample)                          │
│ 2. Parse question with LLM                              │ ← Uses LLaVA
│    → Type: CONDITIONAL                                   │
│    → Condition: "Is there a dog?"                        │
│    → Question: "What color is it?"                       │
│ 3. Scan all frames with condition question              │ ← Uses LLaVA
│    → Frame 5: "Yes" (conf=0.9) ✓                        │
│    → Frame 6: "No" (conf=0.2) ✗                         │
│    → Frame 12: "Yes" (conf=0.85) ✓                      │
│    ...                                                    │
│ 4. Answer question on matching frames only              │ ← Uses LLaVA + IG
│    → Frame 5: "brown" (conf=0.9)                        │
│    → Frame 12: "brown" (conf=0.8)                       │
│ 5. Aggregate results                                     │
│    → Final: "brown" (avg conf=0.87)                     │
└──────────────────────────────────────────────────────────┘
                    ↓
Result: "brown" (87% confidence) + attribution heatmap
```

### LLM Usage

**Where LLMs are used:**
1. **Question parsing** - Parse natural language into structured components
2. **Frame condition checking** - Determine if frame matches condition
3. **Question answering** - Answer the question on matching frames

**LLM Model:** LLaVA OneVision (same model used for VQA)

**Cost:** No additional API calls - uses the same model loaded for analysis

### Comparison: Regex vs LLM Parsing

| Feature | Regex (temporal_query_vqa_interpret) | LLM (conditional_query_vqa_interpret) |
|---------|--------------------------------------|---------------------------------------|
| Parsing method | Pattern matching | Natural language understanding |
| Timestamp queries | ✓ "at 4 seconds" | ✓ "at 4 seconds" |
| Conditional queries | ✗ Not supported | ✓ "when there is a dog" |
| Semantic understanding | ✗ Limited | ✓ Understands meaning |
| Speed | ⚡ Instant | 🐢 Requires LLM inference (~1-2s) |
| Flexibility | Fixed patterns only | Any natural language |

**When to use each:**
- **Regex parser** (`temporal_query_vqa_interpret`): Simple timestamp queries
- **LLM parser** (`conditional_query_vqa_interpret`): Complex conditional questions

## Performance

### Speed Considerations

**Conditional query workflow:**
1. Extract frames: ~1-5s (depends on video length)
2. Parse question (LLM): ~1-2s per question
3. Scan frames for condition: ~0.5s per frame
4. Answer question on matching frames: ~1-3s per frame (with IG)
5. Aggregate: <1s

**Example: 60-second video, 1 fps sampling, conditional question**
- Total frames: 60
- Scan phase: ~30s (60 frames × 0.5s)
- Matching frames found: 5
- Answer phase: ~10s (5 frames × 2s)
- **Total: ~45s**

**Optimization tips:**
- Lower `fps_sample` for faster scanning (e.g., 0.5 = 1 frame per 2 seconds)
- Increase `confidence_threshold` to reduce false positives
- Use `aggregation_method='most_confident'` (fastest)

### Accuracy Considerations

**Confidence Threshold Impact:**

| Threshold | Effect | Best for |
|-----------|--------|----------|
| 0.3-0.4 | Very lenient, many frames match | Rare events, avoid missing instances |
| 0.5 (default) | Balanced | General use |
| 0.6-0.7 | Stricter, fewer frames | Precise matching, avoid false positives |
| 0.8+ | Very strict | High-confidence matches only |

## Examples

### Example 1: Object-Specific Questions

```python
questions = [
    "When there is a dog in the frame, what color is the dog?",
    "When there is a cat in the frame, what is the cat doing?",
    "At frames with a car, what color is it?",
]
```

**Output:**
```
Q: When there is a dog in the frame, what color is the dog?
Frames matched: 4/60
Answer: brown (87% confidence)

Q: When there is a cat in the frame, what is the cat doing?
Frames matched: 0/60
Error: No matching frames found

Q: At frames with a car, what color is it?
Frames matched: 8/60
Answer: red (92% confidence)
```

### Example 2: Action-Based Questions

```python
questions = [
    "Where the person is running, what are they wearing?",
    "When someone is jumping, what is in the background?",
    "While it's raining, what color is the umbrella?",
]
```

### Example 3: Mixed Question Types

```python
questions = [
    "When there is a dog, what color is it?",  # Conditional
    "What happens at 5 seconds?",  # Timestamp
    "What objects appear in the video?",  # Simple (all frames)
]
```

## Output Files

When `save_results=True`, creates:

```
conditional_query_analysis/
├── frames/
│   ├── frame_0000.jpg
│   ├── frame_0001.jpg
│   └── ...
├── attribution_q1.png - Attribution heatmap for question 1
├── attribution_q2.png - Attribution heatmap for question 2
└── report_q1.txt - Detailed report for question 1
```

**Report Format:**
```
======================================================================
CONDITIONAL QUERY ANALYSIS REPORT
======================================================================

Original Question: When there is a dog in the frame, what color is the dog?
Question Type: conditional

Frame Condition: Is there a dog in the frame?
Matching Frames: 4 out of 60
Frame Indices: [5, 12, 18, 23]
Avg Condition Confidence: 87.5%
Answer Question: What color is the dog?
Aggregation Method: most_confident

----------------------------------------------------------------------
AGGREGATED RESULT
----------------------------------------------------------------------

Prediction: 'brown'
Confidence: 0.8700 (87.00%)

Consensus: 3/4 frames (75.0%)

Top Predictions:
  1. 'brown' - 0.8700 (87.00%)
  2. 'black' - 0.0800 (8.00%)
  3. 'white' - 0.0300 (3.00%)

----------------------------------------------------------------------
INDIVIDUAL FRAME RESULTS
----------------------------------------------------------------------

Frame 5 (condition conf: 90%):
  Prediction: 'brown'
  Confidence: 0.9000

Frame 12 (condition conf: 85%):
  Prediction: 'brown'
  Confidence: 0.8500

...
```

## Limitations

1. **Speed**: Slower than regex parsing due to LLM inference on every frame
2. **Condition accuracy**: Frame matching depends on VLM's yes/no accuracy
3. **Simple heuristic**: Confidence scoring for conditions is basic (looks for "yes"/"no")
4. **No negation**: "When there is NOT a dog" may not work reliably
5. **Single condition**: Currently supports one condition per question (not "when A and B")

## Future Enhancements

Potential improvements:

- [ ] **Better confidence scoring**: Use model logits for yes/no confidence
- [ ] **Multi-condition support**: "When there is a dog AND a cat"
- [ ] **Negation handling**: "When there is NO dog"
- [ ] **Temporal conditions**: "When the car starts moving"
- [ ] **Caching**: Save frame condition results for reuse
- [ ] **Parallel processing**: Check multiple frames simultaneously
- [ ] **Event detection integration**: Automatically detect key moments

## Comparison with Other Methods

### vs. Temporal Query (Regex)

| Feature | Conditional Query (LLM) | Temporal Query (Regex) |
|---------|------------------------|------------------------|
| Question type | Conditional + timestamps | Timestamps only |
| Parsing | LLM-based | Regex patterns |
| Flexibility | High (any phrasing) | Low (fixed patterns) |
| Speed | Slow (~1min) | Fast (~5s) |
| Use case | "When X, what Y?" | "At time T, what?" |

### vs. Full Video Analysis

| Feature | Conditional Query | Full Video Analysis |
|---------|------------------|---------------------|
| Frames analyzed | Only matching frames | All frames |
| Speed | Medium | Slow |
| Use case | Specific conditions | Overview/trends |
| Attribution | ✓ Per-frame IG | ✓ Per-frame IG |

## Best Practices

1. **Frame sampling**: Start with `fps_sample=1`, increase only if needed
2. **Confidence threshold**: Use default 0.5, adjust based on results
3. **Question phrasing**: Be explicit in conditions
   - Good: "When there is a dog in the frame"
   - Okay: "When a dog appears"
   - Avoid: "With dogs" (ambiguous)
4. **Aggregation method**:
   - `most_confident`: Best for clear single answer
   - `consensus`: Best when condition has many matches
   - `average`: Best for smooth transitions
5. **Check for no matches**: Handle `'error': 'No matching frames found'` in results

## Summary

✅ **Powerful**: Handles complex conditional questions
✅ **Intelligent**: LLM-based parsing understands natural language
✅ **Flexible**: Supports conditional, timestamp, and simple questions
✅ **Accurate**: Two-stage approach focuses on relevant frames
⚠️ **Slower**: Requires scanning all frames for conditions
⚠️ **Resource-intensive**: More LLM inference calls

**When to use this:**
- Questions with conditional clauses ("when", "where", "while")
- Object-specific or action-specific queries
- Need to focus on specific types of frames

**When NOT to use this:**
- Simple timestamp queries → Use `temporal_query_vqa_interpret` instead
- Single image → Use `vqa_interpret` instead
- Full video overview → Use `temporal_vqa_interpret` instead
