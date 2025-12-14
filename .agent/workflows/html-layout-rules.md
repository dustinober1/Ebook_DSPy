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
