# Chapter 4: Evaluation Exercises

These exercises help you practice the evaluation concepts covered in Chapter 4.

## Prerequisites

- Complete reading of Chapter 4
- Working DSPy installation with API key
- Understanding of `dspy.Example`, metrics, and `dspy.Evaluate`

---

## Exercise 1: Building a Quality Dataset

**Difficulty**: ⭐⭐ Intermediate
**Estimated Time**: 30 minutes

### Objective

Create a well-balanced sentiment analysis dataset with proper train/dev/test splits.

### Requirements

1. Create 30+ examples with fields:
   - `text`: Review or comment text
   - `sentiment`: "positive", "negative", or "neutral"
   - `confidence`: Expected confidence (0.0-1.0)

2. Ensure balance across sentiment classes (roughly equal)

3. Include edge cases:
   - Sarcastic comments
   - Mixed sentiment
   - Very short texts
   - Questions

4. Split into train (60%), dev (20%), test (20%)

5. Validate the dataset for completeness

### Starter Code

See `solutions/exercise01.py` for starter code template.

### Success Criteria

- [ ] 30+ examples created
- [ ] Roughly balanced across 3 sentiment classes
- [ ] At least 5 edge cases included
- [ ] Proper train/dev/test splits with no overlap
- [ ] All examples pass validation

---

## Exercise 2: Designing a Comprehensive Metric

**Difficulty**: ⭐⭐⭐ Intermediate-Advanced
**Estimated Time**: 45 minutes

### Objective

Design a multi-aspect metric for evaluating question-answering quality.

### Requirements

Create a metric that evaluates:

1. **Correctness (40%)**: Does the answer contain the expected information?
2. **Completeness (30%)**: Are all key points addressed?
3. **Conciseness (20%)**: Is the answer appropriately brief?
4. **Format (10%)**: Is the answer well-formed?

The metric should:
- Return a float between 0 and 1
- Handle the `trace` parameter correctly
- Be robust to missing/empty fields
- Include helpful docstring

### Test Cases

Your metric should produce these approximate scores:

| Input | Expected Score |
|-------|---------------|
| Perfect match | ~0.95-1.0 |
| Correct but verbose | ~0.7-0.8 |
| Partially correct | ~0.4-0.6 |
| Wrong answer | ~0.0-0.2 |

### Starter Code

See `solutions/exercise02.py` for starter code template.

### Success Criteria

- [ ] Metric evaluates all four aspects
- [ ] Returns float between 0 and 1
- [ ] Handles trace parameter correctly
- [ ] Passes all test cases
- [ ] Robust to edge cases

---

## Exercise 3: Comprehensive Evaluation Pipeline

**Difficulty**: ⭐⭐⭐ Intermediate-Advanced
**Estimated Time**: 45 minutes

### Objective

Build a complete evaluation pipeline with detailed analysis and reporting.

### Requirements

Create a function `comprehensive_evaluation()` that:

1. Takes a module, dataset, and metric as input

2. Returns a results dictionary containing:
   - `aggregate_score`: Overall percentage
   - `by_category`: Scores broken down by category (if available)
   - `error_analysis`: Categorized failures
   - `best_examples`: Top 5 performing examples
   - `worst_examples`: Bottom 5 performing examples

3. Handles errors gracefully (doesn't crash on exceptions)

4. Provides progress updates during evaluation

### Output Format

```python
{
    'aggregate_score': 0.85,
    'total_examples': 100,
    'by_category': {
        'geography': 0.90,
        'science': 0.82,
        'history': 0.78
    },
    'error_analysis': {
        'wrong_answer': 10,
        'incomplete': 5,
        'format_error': 2
    },
    'best_examples': [...],
    'worst_examples': [...]
}
```

### Starter Code

See `solutions/exercise03.py` for starter code template.

### Success Criteria

- [ ] Function returns complete results dictionary
- [ ] Category breakdown works correctly
- [ ] Error analysis categorizes failures
- [ ] Best/worst examples are correctly identified
- [ ] Handles exceptions without crashing

---

## Exercise 4: Preventing Data Leakage

**Difficulty**: ⭐⭐⭐⭐ Advanced
**Estimated Time**: 60 minutes

### Objective

Implement a data splitting function that prevents various forms of data leakage.

### Requirements

Create `safe_split()` function that:

1. Removes exact duplicates
2. Groups similar items together (similarity threshold)
3. Ensures similar items don't appear across splits
4. Returns split statistics

### Similarity Detection

- Use text similarity (e.g., SequenceMatcher ratio)
- Default threshold: 0.85 (85% similar = same group)
- Group similar items and assign entire group to one split

### Verification

Create `verify_no_leakage()` function that:
- Checks all pairs across splits
- Reports any similar items found
- Returns list of issues

### Starter Code

See `solutions/exercise04.py` for starter code template.

### Success Criteria

- [ ] Duplicates are removed
- [ ] Similar items are grouped
- [ ] Groups stay in same split
- [ ] Verification detects cross-split leakage
- [ ] Statistics are accurate

---

## Exercise 5: Evaluation Report Generator

**Difficulty**: ⭐⭐⭐⭐ Advanced
**Estimated Time**: 60 minutes

### Objective

Create a function that generates a comprehensive, stakeholder-ready evaluation report in Markdown format.

### Requirements

Generate a report containing:

1. **Executive Summary**
   - Overall score
   - Pass/fail status
   - Key findings

2. **Performance Metrics**
   - Detailed metrics table
   - Score distribution

3. **Category Breakdown**
   - Performance by category
   - Comparison chart (ASCII)

4. **Trend Analysis** (if historical data provided)
   - Performance over time
   - Improvement/regression detection

5. **Error Analysis**
   - Error type breakdown
   - Example failures

6. **Recommendations**
   - Based on findings
   - Actionable improvements

### Output

Markdown file that can be shared with stakeholders.

### Starter Code

See `solutions/exercise05.py` for starter code template.

### Success Criteria

- [ ] Generates valid Markdown
- [ ] Includes all required sections
- [ ] Handles missing optional data gracefully
- [ ] Provides actionable recommendations
- [ ] Output is readable and professional

---

## Submission Guidelines

For each exercise:

1. Complete the implementation in the solutions folder
2. Test with the provided test cases
3. Ensure all success criteria are met
4. Add comments explaining your approach

## Resources

- [Chapter 4 Content](../../src/04-evaluation/)
- [Chapter 4 Examples](../../examples/chapter04/)
- [DSPy Documentation](https://dspy.ai)

## Getting Help

If you're stuck:
1. Review the relevant chapter section
2. Check the examples in `examples/chapter04/`
3. Look at the solution hints below

<details>
<summary>Exercise 1 Hints</summary>

- Use a loop to create examples for each sentiment class
- Remember to call `.with_inputs("text")` on each example
- Use `random.Random(42).shuffle()` for reproducible shuffling
- Consider using a dictionary to organize examples by sentiment before balancing
</details>

<details>
<summary>Exercise 2 Hints</summary>

- Calculate each aspect score separately, then combine
- Use `getattr(example, 'key_points', [])` to handle missing fields
- For conciseness, count words and compare to expected range
- Remember: `if trace is not None: return score >= threshold`
</details>

<details>
<summary>Exercise 3 Hints</summary>

- Use try/except around module calls to handle errors
- Store results in a list, then aggregate at the end
- Use `sorted(results, key=lambda x: x['score'])` for best/worst
- Use `defaultdict(list)` for category grouping
</details>

<details>
<summary>Exercise 4 Hints</summary>

- Build groups first, then shuffle groups (not individual items)
- `difflib.SequenceMatcher` calculates text similarity
- Keep track of which indices are already assigned to groups
- For verification, only need to check representative examples from each split
</details>

<details>
<summary>Exercise 5 Hints</summary>

- Build report sections as list of strings, join at end
- Use f-strings for dynamic content
- Create ASCII bar charts with simple character repetition
- Check for `None` values before including optional sections
</details>
