---
description: HTML component layout rules for ebook pages
---

# Icon + Text Component Layout Rules

When creating HTML components that have an icon with accompanying text (title + description), **ALWAYS** use the horizontal layout pattern with proper flexbox alignment.

## ✅ CORRECT Pattern - Use This

```html
<div class="chapter-sections-list">
    <div class="chapter-section-item">
        <span class="section-icon">📖</span>
        <div class="section-content">
            <h4>Title Here</h4>
            <p>Description text goes here.</p>
        </div>
    </div>
</div>
```

Key requirements:
- Icon and text content should be **side-by-side** (horizontal), NOT stacked vertically
- Use `display: flex` with `align-items: flex-start` on the container
- Icon should be in a `<span>` with fixed width
- Text content wrapped in a `<div>` that contains `<h4>` and `<p>`

## ❌ WRONG Pattern - Never Do This

```html
<div class="bad-item">
    <span class="icon">📖</span>
    <h4>Title</h4>
    <p>Description</p>
</div>
```

This causes icons to appear on separate lines from text.

## 🚫 Never Use Numbered Lists in Titles

**WRONG:**
```html
<h4>1. Signatures</h4>
<h4>2. Modules</h4>
<h4>3. Optimizers</h4>
```

**CORRECT:**
```html
<h4>Signatures</h4>
<h4>Modules</h4>
<h4>Optimizers</h4>
```

Numbers in titles create visual clutter and appear on separate lines. The visual hierarchy and icons provide sufficient organization.

## Icon Usage Guidelines

### When to Use Icons

✅ **Use icons for:**
- Feature lists (📝, 🧩, ⚡)
- Benefit sections (🔧, 📈, 📚)
- Stage/step descriptions (🎯, 🔍, 💡)
- Warning/problem sections (⚠️)

### When NOT to Use Icons

❌ **Skip icons for:**
- Philosophy/concept comparisons (use `.no-icons` class)
- Key takeaways lists (use `.no-icons` class)
- Simple text-only lists

### No-Icons Pattern

```html
<div class="chapter-sections-list no-icons">
    <div class="chapter-section-item">
        <div class="section-content">
            <h4>Title</h4>
            <p>Description without icon</p>
        </div>
    </div>
</div>
```

## Code Blocks and Lists Within Sections

Sections can contain code blocks and lists:

```html
<div class="chapter-section-item">
    <span class="section-icon">📝</span>
    <div class="section-content">
        <h4>Signatures</h4>
        <p>Task specifications:</p>
        <div class="command-box">
            <pre><code class="language-python">
class Example(dspy.Signature):
    input: str = dspy.InputField()
            </code></pre>
        </div>
        <p>Additional explanation here.</p>
    </div>
</div>
```

The CSS handles proper spacing for:
- `.command-box` within `.section-content`
- `<ul>` lists within `.section-content`
- Multiple `<p>` tags

## Problem/Warning Notes

For warning or problem notes, use the `.problem-note` class:

```html
<p class="problem-note">This is a warning or important note!</p>
```

This adds a subtle orange background with left border.

## Navigation Links

- Do NOT add hyperlinks (`<a>` tags) around section cards when navigation is available in the sidebar
- Use `<div>` elements instead of `<a>` tags for informational lists
- Hyperlinks cause unwanted underline styling on titles

## CSS Classes Reference

- `.chapter-sections-list` - Container for all section items
- `.chapter-section-item` - Individual item with icon + content
- `.section-icon` - Icon span (40x40px box)
- `.section-content` - Content wrapper for h4/p/code
- `.no-icons` - Modifier for lists without icons
- `.problem-note` - Warning/note styling with orange background

---

# Ebook-Specific Best Practices

## Content Structure

### Keep Sections Focused
- Each section should cover ONE concept or topic
- Break long explanations into multiple sections
- Use clear, descriptive headings (h2, h3, h4)

### Progressive Disclosure
- Start with simple concepts, build to complex
- Use "Key Takeaways" sections to summarize
- Provide "What You'll Learn" at chapter start
- Include "Continue to..." navigation at chapter end

### Code Examples
- **Always** include working, runnable code
- Add comments explaining key lines
- Show both input and expected output when relevant
- Use realistic examples, not just "foo/bar"

```html
<!-- GOOD: Realistic example with context -->
<div class="command-box">
    <pre><code class="language-python"># Create a question-answering module
class QA(dspy.Signature):
    """Answer questions based on context."""
    context: str = dspy.InputField()
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

qa = dspy.ChainOfThought(QA)
result = qa(context="Paris is the capital of France", 
            question="What is the capital of France?")
print(result.answer)  # Output: Paris</code></pre>
</div>
```

## Readability Guidelines

### Typography
- Use short paragraphs (2-4 sentences max)
- Break up walls of text with:
  - Subheadings
  - Lists
  - Code examples
  - Visual callouts (tip boxes, warning boxes)

### Lists
- Use bulleted lists for unordered items
- Use the `.chapter-sections-list` component for feature/benefit lists
- Avoid numbered lists in titles (use icons instead)

### Emphasis
- Use `<strong>` for important terms on first use
- Use `<em>` for emphasis or technical terms
- Use `<code>` for inline code, class names, function names

## Visual Hierarchy

### Headings
- **h1**: Page title only (in hero section)
- **h2**: Major section headings
- **h3**: Subsection headings
- **h4**: Component titles (in cards/lists)
- **h5**: Comparison labels (Traditional vs DSPy)

### Spacing
- Use `content-block` class for major sections
- This provides consistent spacing and bottom borders
- Don't add manual spacing with `<br>` tags

## Interactive Elements

### Navigation
- Every chapter page should have:
  - Sidebar navigation to all sections
  - "Continue to..." button at the end
  - Breadcrumb or chapter label in hero
  
### Links
- External links should open in new tab: `target="_blank"`
- Internal navigation uses sidebar (no inline links to other sections)
- Use descriptive link text, not "click here"

## Accessibility

### Semantic HTML
- Use proper heading hierarchy (don't skip levels)
- Use `<code>` for code, not just styling
- Use `<pre><code>` for code blocks
- Use semantic elements: `<article>`, `<section>`, `<aside>`

### Alt Text and Labels
- All images need descriptive alt text
- Buttons need aria-labels when icon-only
- Code examples should specify language for syntax highlighting

## Performance

### Code Highlighting
- Use Prism.js for syntax highlighting
- Always specify language: `class="language-python"`
- Supported languages: python, javascript, bash, json

### Images
- Optimize images before adding
- Use appropriate formats (WebP for photos, SVG for diagrams)
- Lazy load images below the fold

## Consistency Checklist

Before completing a chapter page, verify:

- [ ] All sections use `chapter-sections-list` for icon+text layouts
- [ ] No numbered lists in titles (1., 2., 3.)
- [ ] Icons are used consistently (features, benefits, stages)
- [ ] Code examples are complete and runnable
- [ ] Navigation works (sidebar, continue button)
- [ ] Headings follow proper hierarchy
- [ ] No hyperlinks in section cards (use sidebar instead)
- [ ] Syntax highlighting is applied to code blocks
- [ ] Page has meta description and title
- [ ] Reading time estimate is accurate

## Common Patterns

### Comparison Sections
Use side-by-side comparison for Traditional vs DSPy:

```html
<div class="comparison-container">
    <div class="comparison-side traditional">
        <h5>Traditional Approach</h5>
        <!-- content -->
    </div>
    <div class="comparison-side dspy">
        <h5>DSPy Approach</h5>
        <!-- content -->
    </div>
</div>
```

### Tip/Warning Boxes
```html
<div class="tip-box">
    <span class="tip-icon">💡</span>
    <p><strong>Pro Tip:</strong> Your tip here</p>
</div>
```

### Summary Sections
Use `.no-icons` pattern for key takeaways:

```html
<div class="content-block summary-block">
    <h2>📝 Key Takeaways</h2>
    <div class="chapter-sections-list no-icons">
        <!-- items here -->
    </div>
</div>
```
