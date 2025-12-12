# Exercises

This directory contains practice exercises for each chapter of the DSPy ebook, complete with starter code and solutions.

---

## 📂 Directory Structure

```
exercises/
├── chapter01/              # Chapter 1: DSPy Fundamentals
│   ├── problems.md         # Exercise descriptions
│   └── solutions/          # Complete solutions
│       ├── exercise01.py
│       ├── exercise02.py
│       └── solutions.md    # Detailed explanations
│
├── chapter02/              # Chapter 2: Signatures
├── chapter03/              # Chapter 3: Modules
├── chapter04/              # Chapter 4: Evaluation
├── chapter05/              # Chapter 5: Optimizers
├── chapter06/              # Chapter 6: Real-World Applications
└── chapter07/              # Chapter 7: Advanced Topics
```

---

## 🎯 How to Use These Exercises

### 1. Read the Problem

Each chapter's `problems.md` file contains:
- **Exercise overview table**: Difficulty levels and topics
- **Detailed descriptions**: Clear objectives and requirements
- **Starter code**: Code to get you started (if applicable)
- **Hints**: Collapsible hints to guide you without spoilers
- **Expected output**: What your solution should produce

### 2. Attempt the Exercise

```bash
# Navigate to the exercises directory
cd exercises/chapter01

# Read the problems
cat problems.md

# Create your solution file
touch my_solution.py

# Work on your solution
```

**Tips for Success**:
- ✅ Try solving without hints first
- ✅ Use hints if stuck (they're designed to help, not give away the answer)
- ✅ Test your solution thoroughly
- ✅ Compare your output with expected output
- ✅ Only look at solutions after attempting

### 3. Check Your Solution

```bash
# Run your solution
python my_solution.py

# Compare with expected output
# Debug if needed
```

### 4. Study the Solution

Once you've attempted the exercise:

```bash
# View the solution code
cat solutions/exercise01.py

# Read the detailed explanation
cat solutions/solutions.md
```

**Learn from solutions**:
- Understand different approaches
- Study best practices
- Learn common pitfalls
- See code optimization techniques

---

## 🎓 Exercise Difficulty Levels

Exercises are marked with difficulty indicators:

### ⭐ Beginner
- **For**: New to DSPy or the chapter's concepts
- **Focus**: Following examples, basic understanding
- **Time**: 15-30 minutes
- **Help**: Lots of hints available

### ⭐⭐ Intermediate
- **For**: Comfortable with basics, ready to apply
- **Focus**: Combining concepts, problem-solving
- **Time**: 30-60 minutes
- **Help**: Some hints, requires independent thinking

### ⭐⭐⭐ Advanced
- **For**: Solid understanding, want a challenge
- **Focus**: Complex integration, optimization
- **Time**: 60-120 minutes
- **Help**: Minimal hints, self-directed learning

### ⭐⭐⭐⭐ Expert
- **For**: Deep understanding, pushing boundaries
- **Focus**: Open-ended, creative solutions
- **Time**: 2+ hours
- **Help**: High-level guidance only

---

## 📋 Exercise Format

Each exercise follows this structure:

### In problems.md

```markdown
## Exercise 1: Title

**Difficulty**: ⭐ Beginner

### Objective
What you'll learn from this exercise.

### Requirements
1. Specific task to complete
2. Another requirement
3. Constraints or rules

### Starter Code
Optional code to get you started.

### Hints
<details>
<summary>💡 Hint 1</summary>
Hidden hint here.
</details>

### Expected Output
Example of what success looks like.

### Solution
Link to solution file and explanation.
```

### In solutions/

- **exerciseXX.py**: Complete, working solution code
- **solutions.md**: Detailed explanations for all exercises

---

## ✅ Exercise Completion Checklist

For each exercise, ensure you:

- [ ] Read and understand the objective
- [ ] Reviewed all requirements
- [ ] Attempted solution without looking at hints
- [ ] Used hints if stuck (progressively)
- [ ] Got your code running without errors
- [ ] Output matches expected format
- [ ] Tested with different inputs (if applicable)
- [ ] Reviewed the official solution
- [ ] Read the detailed explanation
- [ ] Understood alternative approaches

---

## 🎯 Chapter-by-Chapter Exercise Guide

### Chapter 1: DSPy Fundamentals
**Focus**: Installation, basic concepts, first programs

**Exercises**:
- Exercise 1: Verify DSPy installation (Beginner)
- Exercise 2: Create a basic signature (Beginner)
- Exercise 3: Configure language models (Beginner)
- Exercise 4: Build a simple Q&A system (Intermediate)
- Exercise 5: Experiment with parameters (Intermediate)

### Chapter 2: Signatures
**Focus**: Creating and using signatures

**Exercises**:
- String-based signatures
- Typed signatures
- Multi-field signatures
- Advanced signature patterns
- Real-world signature design

### Chapter 3: Modules
**Focus**: Working with DSPy modules

**Exercises**:
- Using `dspy.Predict`
- Implementing Chain of Thought
- Building ReAct agents
- Creating custom modules
- Composing module pipelines

### Chapter 4: Evaluation
**Focus**: Metrics and evaluation loops

**Exercises**:
- Creating evaluation datasets
- Defining custom metrics
- Running evaluations
- Analyzing results
- Iterative improvement

### Chapter 5: Optimizers
**Focus**: Program optimization

**Exercises**:
- Using BootstrapFewShot
- Applying MIPRO
- Implementing KNNFewShot
- Comparing optimizers
- Optimization strategies

### Chapter 6: Real-World Applications
**Focus**: Building complete applications

**Exercises**:
- RAG system development
- Classification pipeline
- Entity extraction
- Agent development
- Integration challenges

### Chapter 7: Advanced Topics
**Focus**: Production patterns

**Exercises**:
- Tool integration
- Performance optimization
- Async implementation
- Deployment scenarios

---

## 🚀 Running Exercise Solutions

### Run a Specific Solution

```bash
# Navigate to project root
cd /path/to/Ebook_DSPy

# Activate virtual environment
source venv/bin/activate

# Run a solution
python exercises/chapter01/solutions/exercise01.py
```

### Validate All Solutions

```bash
# Use the validation script
python scripts/validate_code.py

# This checks:
# - Syntax validity
# - Import resolution
# - Code quality
```

---

## 💡 Tips for Success

### Before You Start

1. **Read the chapter first**: Understand concepts before exercises
2. **Set up your environment**: Virtual environment, API keys
3. **Block time**: Allocate focused time for exercises
4. **Have references ready**: Chapter content, DSPy docs

### While Solving

1. **Break it down**: Divide complex problems into steps
2. **Test incrementally**: Don't write all code at once
3. **Use print statements**: Debug by printing intermediate values
4. **Experiment**: Try different approaches
5. **Ask questions**: Write down what you don't understand

### After Solving

1. **Review your code**: Is it clear and well-structured?
2. **Compare with solution**: What did you do differently?
3. **Read explanations**: Understand the reasoning
4. **Try variations**: Modify the exercise for deeper learning
5. **Document learning**: Note key takeaways

---

## 🔧 Troubleshooting

### Exercise Won't Run

**Check**:
- Is your virtual environment activated?
- Are dependencies installed?
- Is your API key configured?
- Did you modify the starter code correctly?

### Output Doesn't Match

**Remember**:
- LLM outputs may vary slightly
- Focus on format and correctness, not exact wording
- Some variation is expected and acceptable

### Can't Solve an Exercise

**Try this progression**:
1. Re-read the chapter section
2. Review the code example for that topic
3. Use Hint 1
4. Use Hint 2
5. Use Hint 3
6. Look at the solution structure (not details)
7. Study the complete solution and explanation

---

## 📚 Additional Resources

### For Each Exercise

- **Chapter content**: Detailed explanations in `src/chapterXX/`
- **Code examples**: Reference implementations in `examples/chapterXX/`
- **Templates**: Exercise template in `assets/templates/`

### External Resources

- **DSPy Documentation**: [https://dspy.ai](https://dspy.ai)
- **DSPy Examples**: [GitHub Repository](https://github.com/stanfordnlp/dspy/tree/main/examples)
- **Community**: [GitHub Discussions](https://github.com/stanfordnlp/dspy/discussions)

---

## 🤝 Contributing

### Share Your Solutions

Have an alternative approach? Contribute it!

1. Create a new file: `alternate_solution_XX.py`
2. Document your approach
3. Submit a pull request

### Suggest New Exercises

Ideas for exercises?

1. Open an issue with tag "exercise-suggestion"
2. Describe the concept it would teach
3. Suggest difficulty level
4. Propose requirements

---

## 📊 Track Your Progress

Create a checklist to track completion:

```markdown
## Chapter 1: DSPy Fundamentals
- [ ] Exercise 1
- [ ] Exercise 2
- [ ] Exercise 3
- [ ] Exercise 4
- [ ] Exercise 5

## Chapter 2: Signatures
- [ ] Exercise 1
- [ ] Exercise 2
...
```

---

## 🎉 After Completing Exercises

Once you've worked through the exercises:

1. **Build a project**: Apply concepts to your own ideas
2. **Share your work**: Post to GitHub or community forums
3. **Help others**: Answer questions in discussions
4. **Give feedback**: Suggest improvements to exercises

---

## ❓ Getting Help

### Stuck on an Exercise?

1. **Review the chapter**: Re-read relevant sections
2. **Use progressive hints**: Don't skip straight to the solution
3. **Check discussion forums**: Others may have asked similar questions
4. **Ask for help**: Post in GitHub Discussions with:
   - Which exercise
   - What you've tried
   - Specific error or confusion

### Found an Issue?

- **Bug in exercise**: [Report it](https://github.com/dustinober1/Ebook_DSPy/issues)
- **Unclear instructions**: Suggest improvements
- **Missing information**: Let us know what would help

---

**Ready to practice?** Start with [Chapter 1 Exercises](chapter01/problems.md)!

Good luck! 🚀

*Remember: The goal is learning, not just completing. Take your time and understand each concept.*
