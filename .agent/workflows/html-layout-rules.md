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

## Navigation Links

- Do NOT add hyperlinks (`<a>` tags) around section cards when navigation is available in the sidebar
- Use `<div>` elements instead of `<a>` tags for informational lists
- Hyperlinks cause unwanted underline styling on titles

## CSS Class to Use

The `.chapter-sections-list` and `.chapter-section-item` classes are defined in `content.css` and should be reused for all icon + text list patterns throughout the ebook.
