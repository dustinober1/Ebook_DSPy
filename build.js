#!/usr/bin/env node

const fs = require('fs-extra');
const path = require('path');
const MarkdownIt = require('markdown-it');
const anchor = require('markdown-it-anchor');
const toc = require('markdown-it-table-of-contents');
const hljs = require('markdown-it-highlightjs');
const fm = require('front-matter');

// Initialize markdown-it with plugins
const md = new MarkdownIt({
  html: true,
  linkify: true,
  typographer: true
})
  .use(anchor, {
    level: [1, 2, 3, 4],
    slugify: (s) => s.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/(^-|-$)/g, '')
  })
  .use(toc, {
    includeLevel: [2, 3],
    containerClass: 'table-of-contents',
    slugify: (s) => s.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/(^-|-$)/g, '')
  })
  .use(hljs);

// HTML template
const HTML_TEMPLATE = `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{TITLE}}</title>
    <link rel="stylesheet" href="/assets/css/main.css">
    <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.8.0/styles/github.min.css">
</head>
<body>
    <div id="app">
        {{CONTENT}}
    </div>
    <script src="/assets/js/navigation.js"></script>
    <script src="/assets/js/main.js"></script>
</body>
</html>`;

// Chapters navigation
const CHAPTERS = [
  // Frontmatter
  { title: "Preface", path: "00-frontmatter/00-preface.html" },
  { title: "How to Use This Book", path: "00-frontmatter/01-how-to-use-this-book.html" },
  { title: "Prerequisites", path: "00-frontmatter/02-prerequisites.html" },
  { title: "Setup Instructions", path: "00-frontmatter/03-setup-instructions.html" },
  // Part I: Foundations
  { title: "1. What is DSPy?", path: "01-fundamentals/01-what-is-dspy.html" },
  { title: "2. Getting Started", path: "01-fundamentals/02-getting-started.html" },
  { title: "3. Your First Program", path: "01-fundamentals/03-first-program.html" },
  // Part II: Core Concepts
  { title: "4. Signatures", path: "02-core-concepts/01-signatures.html" },
  { title: "5. Modules", path: "02-core-concepts/02-modules.html" },
  { title: "6. Composition", path: "02-core-concepts/03-composition.html" },
  // Add more chapters as needed
];

async function buildSite() {
  console.log('🚀 Building static DSPy ebook...');

  // Copy assets
  await copyAssets();

  // Convert markdown files
  await convertMarkdownFiles();

  // Create index.html
  await createIndex();

  console.log('✅ Build complete! Run "npm run serve" to preview locally.');
}

async function copyAssets() {
  console.log('📁 Verifying assets...');

  // Ensure directories exist
  await fs.ensureDir('assets/css');
  await fs.ensureDir('assets/js');

  // Copy navigation JS
  if (await fs.pathExists('assets/navigation.js')) {
    await fs.copy('assets/navigation.js', 'assets/js/navigation.js');
  }

  // Create main.js with navigation logic
  const mainJS = `
// Initialize navigation when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
  // The navigation is loaded from navigation.js
  console.log('DSPy Ebook loaded');
});
  `;
  await fs.writeFile('assets/js/main.js', mainJS);

  // Copy any images if they exist
  const imagesDir = 'assets/images';
  if (await fs.pathExists(imagesDir)) {
    // Images are already in the right place
  }
}

async function convertMarkdownFiles() {
  console.log('📝 Converting markdown files...');

  const sourceDir = 'src';
  const targetDir = '.';

  // Process each markdown file
  for (const chapter of CHAPTERS) {
    const sourcePath = path.join(sourceDir, chapter.path.replace('.html', '.md'));
    const targetPath = path.join(targetDir, chapter.path);

    if (await fs.pathExists(sourcePath)) {
      await convertMarkdownToHTML(sourcePath, targetPath, chapter.title);
      console.log(`  ✓ Converted: ${chapter.title}`);
    } else {
      console.log(`  ⚠️ Not found: ${sourcePath}`);
      // Create placeholder HTML
      await createPlaceholderHTML(targetPath, chapter.title);
    }
  }
}

async function convertMarkdownToHTML(sourcePath, targetPath, title) {
  const content = await fs.readFile(sourcePath, 'utf8');
  const { data: frontMatter, body: markdown } = fm(content);

  // Convert markdown to HTML
  const htmlContent = md.render(markdown);

  // Build navigation
  const navHTML = buildNavigation(title);

  // Build page content
  const pageContent = `
    <nav class="page-sidebar">
      ${navHTML}
    </nav>

    <div class="page-wrapper">
      <div class="chapter-progress">
        <div class="bar"></div>
      </div>

      <div class="content">
        <h1>${title}</h1>
        ${htmlContent}

        <div class="page-nav-buttons">
          ${getPrevNextButtons(title)}
        </div>
      </div>
    </div>
  `;

  // Insert into template
  const finalHTML = HTML_TEMPLATE
    .replace('{{TITLE}}', `${title} - DSPy: A Practical Guide`)
    .replace('{{CONTENT}}', pageContent);

  // Ensure directory exists
  await fs.ensureDir(path.dirname(targetPath));

  // Write HTML file
  await fs.writeFile(targetPath, finalHTML);
}

function buildNavigation(currentTitle) {
  let html = `
    <div class="logo">
      <div class="logo-icon">
        <img src="/assets/logos/logo-512.png" alt="DSPy Logo" />
      </div>
      <h1>DSPy</h1>
      <div class="subtitle">A Practical Guide</div>
    </div>
    <nav class="sidebar-nav">
  `;

  const parts = [
    {
      title: 'Frontmatter',
      chapters: CHAPTERS.filter(c => c.path.startsWith('00-frontmatter'))
    },
    {
      title: 'Part I: Foundations',
      chapters: CHAPTERS.filter(c => c.path.startsWith('01-fundamentals'))
    },
    {
      title: 'Part II: Core Concepts',
      chapters: CHAPTERS.filter(c => c.path.startsWith('02-core-concepts'))
    }
  ];

  for (const part of parts) {
    if (part.chapters.length > 0) {
      html += `
        <div class="nav-part">
          <div class="nav-part-title">${part.title}</div>
          <ul class="nav-chapters">
      `;

      for (const chapter of part.chapters) {
        const active = chapter.title === currentTitle ? 'active' : '';
        const chapterNumber = extractChapterNumber(chapter.title);
        html += `
          <li>
            <a href="/${chapter.path}" class="${active}">
              <span class="chapter-number">${chapterNumber}</span>
              ${chapter.title}
            </a>
          </li>
        `;
      }

      html += `
          </ul>
        </div>
      `;
    }
  }

  html += `
    </nav>
  `;

  return html;
}

function extractChapterNumber(title) {
  // Extract number from titles like "1. What is DSPy?"
  const match = title.match(/^(\d+|[A-Z])\./);
  return match ? match[1] : '•';
}

function getPrevNextButtons(currentTitle) {
  const currentIndex = CHAPTERS.findIndex(c => c.title === currentTitle);
  let buttons = '';

  // Previous button
  if (currentIndex > 0) {
    const prev = CHAPTERS[currentIndex - 1];
    buttons += `
      <a href="/${prev.path}" class="nav-button prev">
        <span class="direction">Previous</span>
        <span class="arrow">←</span>
      </a>
    `;
  }

  // Next button
  if (currentIndex < CHAPTERS.length - 1) {
    const next = CHAPTERS[currentIndex + 1];
    buttons += `
      <a href="/${next.path}" class="nav-button next">
        <span class="arrow">→</span>
        <span class="direction">Next</span>
      </a>
    `;
  }

  return buttons;
}

async function createPlaceholderHTML(targetPath, title) {
  const placeholderContent = `
    <div class="page-sidebar">
      <div class="logo">
        <h1>DSPy</h1>
        <div class="subtitle">A Practical Guide</div>
      </div>
    </div>

    <div class="page-wrapper">
      <div class="chapter-progress">
        <div class="bar"></div>
      </div>

      <div class="content">
        <h1>${title}</h1>
        <div class="callout note">
          <p>This chapter is coming soon! The DSPy team is working hard to bring you the latest content.</p>
        </div>
        <p>In the meantime, check out the <a href="https://github.com/stanfordnlp/dspy">official DSPy documentation</a>.</p>
      </div>
    </div>
  `;

  const finalHTML = HTML_TEMPLATE
    .replace('{{TITLE}}', `${title} - DSPy: A Practical Guide`)
    .replace('{{CONTENT}}', placeholderContent);

  await fs.ensureDir(path.dirname(targetPath));
  await fs.writeFile(targetPath, finalHTML);
}

async function createIndex() {
  console.log('🏠 Creating index.html...');

  const indexContent = `
    <div class="page-wrapper">
      <div class="content" style="text-align: center; padding-top: 4rem;">
        <div class="cover-container">
          <img src="/assets/covers/cover.jpg" alt="DSPy: A Practical Guide - Cover" class="cover-image" />
        </div>

        <h1 style="font-size: 3rem; margin-bottom: 1rem; margin-top: 2rem;">DSPy: A Practical Guide</h1>
        <p style="font-size: 1.25rem; color: var(--text-secondary); margin-bottom: 3rem;">
          Programming Foundation Models
        </p>

        <div class="callout tip" style="max-width: 600px; margin: 2rem auto; text-align: left;">
          <p><strong>Welcome!</strong> This comprehensive guide will take you from DSPy basics to advanced applications.</p>
        </div>

        <h2 style="margin-top: 3rem;">Start Reading</h2>
        <div style="display: grid; gap: 1rem; max-width: 500px; margin: 0 auto; text-align: left;">
          ${CHAPTERS.slice(0, 5).map(chapter => `
            <a href="/${chapter.path}" style="display: flex; align-items: center; padding: 1rem; background: white; border-radius: 8px; text-decoration: none; box-shadow: 0 2px 4px rgba(0,0,0,0.1); transition: all 0.2s;">
              <span style="background: var(--color-primary); color: white; padding: 0.5rem; border-radius: 4px; margin-right: 1rem; font-weight: bold;">
                ${extractChapterNumber(chapter.title)}
              </span>
              <span style="color: var(--color-text-primary); font-weight: 500;">${chapter.title}</span>
            </a>
          `).join('')}
        </div>

        <div style="margin-top: 3rem;">
          <a href="/00-frontmatter/00-preface.html" class="nav-button">
            <span class="direction">Start Reading</span>
            <span class="arrow">→</span>
          </a>
        </div>
      </div>
    </div>
  `;

  const finalHTML = HTML_TEMPLATE
    .replace('{{TITLE}}', 'DSPy: A Practical Guide')
    .replace('{{CONTENT}}', indexContent);

  await fs.writeFile('index.html', finalHTML);
}

// Run the build
buildSite().catch(console.error);