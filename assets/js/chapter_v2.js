/**
 * Chapter Page JavaScript
 * Handles markdown loading, syntax highlighting, navigation, and animations
 */

// Initialize mermaid with custom configuration
if (typeof mermaid !== 'undefined') {
    mermaid.initialize({
        startOnLoad: false,
        theme: 'base',
        themeVariables: {
            primaryColor: '#084887',
            primaryTextColor: '#fff',
            primaryBorderColor: '#084887',
            lineColor: '#666370',
            secondaryColor: '#f8fafc',
            tertiaryColor: '#fffdf7'
        },
        securityLevel: 'loose',
        fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'
    });
}

/**
 * Custom Markdown Renderer Configuration
 */
function configureMarked() {
    if (typeof marked === 'undefined') return;

    const renderer = new marked.Renderer();

    // Custom code block rendering with copy button
    // Modern marked.js passes an object with text and lang properties
    renderer.code = function (codeBlock) {
        // Handle both old API (code, language) and new API (object with text, lang)
        let code, lang;
        if (typeof codeBlock === 'object' && codeBlock !== null) {
            code = codeBlock.text || codeBlock.code || '';
            lang = codeBlock.lang || codeBlock.language || 'text';
        } else {
            code = codeBlock || '';
            lang = arguments[1] || 'text';
        }

        const escapedCode = escapeHtml(String(code));

        // Check if it's a mermaid diagram
        if (lang === 'mermaid') {
            return `<div class="mermaid">${code}</div>`;
        }

        return `
            <div class="code-block-wrapper">
                <button class="copy-code-btn" onclick="copyCode(this)">Copy</button>
                <pre><code class="language-${lang}">${escapedCode}</code></pre>
            </div>
        `;
    };

    // Enhanced table rendering
    renderer.table = function (header, body) {
        // Handle new API format
        if (typeof header === 'object' && header !== null && header.header !== undefined) {
            return `
                <div class="table-wrapper">
                    <table>
                        <thead>${header.header}</thead>
                        <tbody>${header.body}</tbody>
                    </table>
                </div>
            `;
        }
        return `
            <div class="table-wrapper">
                <table>
                    <thead>${header}</thead>
                    <tbody>${body}</tbody>
                </table>
            </div>
        `;
    };

    // Configure marked options
    marked.setOptions({
        renderer: renderer,
        gfm: true,
        breaks: false,
        headerIds: true,
        mangle: false
    });
}

/**
 * Escape HTML characters
 */
function escapeHtml(text) {
    // Ensure text is a string
    if (typeof text !== 'string') {
        text = String(text || '');
    }
    const map = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#039;'
    };
    return text.replace(/[&<>"']/g, m => map[m]);
}

/**
 * Copy code to clipboard
 */
function copyCode(button) {
    const wrapper = button.closest('.code-block-wrapper');
    const code = wrapper.querySelector('code').textContent;

    navigator.clipboard.writeText(code).then(() => {
        button.textContent = 'Copied!';
        button.classList.add('copied');

        setTimeout(() => {
            button.textContent = 'Copy';
            button.classList.remove('copied');
        }, 2000);
    }).catch(err => {
        console.error('Failed to copy:', err);
        button.textContent = 'Failed';
    });
}

/**
 * Load markdown content from file
 */
async function loadMarkdownContent(url, targetId) {
    const targetElement = document.getElementById(targetId);
    if (!targetElement) return;

    try {
        const response = await fetch(url);

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const markdown = await response.text();

        // Parse markdown to HTML
        const html = marked.parse(markdown);
        targetElement.innerHTML = html;

        // Process mermaid diagrams
        await processMermaidDiagrams(targetElement);

        // Apply syntax highlighting
        applySyntaxHighlighting(targetElement);

        // Add visible class for animation
        const section = targetElement.closest('.content-section');
        if (section) {
            section.classList.add('is-visible');
        }

    } catch (error) {
        console.error(`Error loading ${url}:`, error);
        targetElement.innerHTML = `
            <div class="error-message">
                <p><strong>Unable to load content.</strong></p>
                <p>Please try refreshing the page or view the content directly:</p>
                <p><a href="${url}" target="_blank">Open markdown file →</a></p>
            </div>
        `;
    }
}

/**
 * Load HTML content from file (for professional pre-rendered content)
 */
async function loadHtmlContent(url, targetId) {
    const targetElement = document.getElementById(targetId);
    if (!targetElement) return;

    try {
        const response = await fetch(url);

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const html = await response.text();
        targetElement.innerHTML = html;

        // Apply syntax highlighting to any code blocks
        applySyntaxHighlighting(targetElement);

        // Add visible class for animation
        const section = targetElement.closest('.content-section');
        if (section) {
            section.classList.add('is-visible');
        }

    } catch (error) {
        console.error(`Error loading ${url}:`, error);
        targetElement.innerHTML = `
            <div class="error-message">
                <p><strong>Unable to load content.</strong></p>
                <p class="error-details" style="font-family: monospace; font-size: 0.8em; color: #d32f2f; margin-top: 5px;">${error.message}</p>
                <p>Please try refreshing the page.</p>
            </div>
        `;
    }
}

/**
 * Process Mermaid diagrams
 */
async function processMermaidDiagrams(container) {
    if (typeof mermaid === 'undefined') return;

    const mermaidDivs = container.querySelectorAll('.mermaid');

    for (let i = 0; i < mermaidDivs.length; i++) {
        const div = mermaidDivs[i];
        const id = `mermaid-${Date.now()}-${i}`;

        try {
            const { svg } = await mermaid.render(id, div.textContent);
            div.innerHTML = svg;
        } catch (error) {
            console.warn('Mermaid rendering failed:', error);
            div.innerHTML = `<pre class="mermaid-error">${div.textContent}</pre>`;
        }
    }
}

/**
 * Apply Prism.js syntax highlighting
 */
function applySyntaxHighlighting(container) {
    if (typeof Prism !== 'undefined') {
        Prism.highlightAllUnder(container);
    }
}

/**
 * Reading progress bar
 */
function updateReadingProgress() {
    const progressBar = document.querySelector('.progress-bar');
    if (!progressBar) return;

    const scrollTop = window.scrollY;
    const docHeight = document.documentElement.scrollHeight - window.innerHeight;
    const progress = (scrollTop / docHeight) * 100;

    progressBar.style.width = `${Math.min(100, Math.max(0, progress))}%`;
}

/**
 * Sidebar section highlighting based on scroll position
 */
function updateActiveSection() {
    const sections = document.querySelectorAll('.content-section');
    const sectionItems = document.querySelectorAll('.section-item');

    let currentSection = null;
    const scrollPosition = window.scrollY + 150;

    sections.forEach(section => {
        const sectionTop = section.offsetTop;
        const sectionHeight = section.offsetHeight;

        if (scrollPosition >= sectionTop && scrollPosition < sectionTop + sectionHeight) {
            currentSection = section.dataset.section;
        }
    });

    sectionItems.forEach(item => {
        const itemSection = item.dataset.section;
        item.classList.remove('active');

        if (itemSection === currentSection) {
            item.classList.add('active');
        }
    });
}

/**
 * Sidebar toggle for mobile
 */
function initSidebarToggle() {
    const toggle = document.getElementById('sidebar-toggle');
    const sidebar = document.getElementById('sidebar');

    if (!toggle || !sidebar) return;

    toggle.addEventListener('click', () => {
        sidebar.classList.toggle('open');
        toggle.classList.toggle('active');
    });

    // Close sidebar when clicking outside
    document.addEventListener('click', (e) => {
        if (sidebar.classList.contains('open') &&
            !sidebar.contains(e.target) &&
            !toggle.contains(e.target)) {
            sidebar.classList.remove('open');
            toggle.classList.remove('active');
        }
    });

    // Close sidebar when clicking a link
    sidebar.querySelectorAll('a').forEach(link => {
        link.addEventListener('click', () => {
            if (window.innerWidth <= 1024) {
                sidebar.classList.remove('open');
                toggle.classList.remove('active');
            }
        });
    });
}

/**
 * Smooth scroll to section
 */
function initSmoothScroll() {
    document.querySelectorAll('.section-item a').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            const href = this.getAttribute('href');
            if (href.startsWith('#')) {
                e.preventDefault();
                const target = document.querySelector(href);
                if (target) {
                    target.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                }
            }
        });
    });
}

/**
 * Scroll to top button
 */
function initScrollToTop() {
    // Create scroll indicator if it doesn't exist
    let scrollIndicator = document.querySelector('.scroll-indicator');

    if (!scrollIndicator) {
        scrollIndicator = document.createElement('button');
        scrollIndicator.className = 'scroll-indicator';
        scrollIndicator.innerHTML = `
            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <polyline points="18 15 12 9 6 15"></polyline>
            </svg>
        `;
        scrollIndicator.setAttribute('aria-label', 'Scroll to top');
        document.body.appendChild(scrollIndicator);
    }

    scrollIndicator.addEventListener('click', () => {
        window.scrollTo({
            top: 0,
            behavior: 'smooth'
        });
    });

    // Show/hide based on scroll position
    window.addEventListener('scroll', () => {
        if (window.scrollY > 500) {
            scrollIndicator.classList.add('visible');
        } else {
            scrollIndicator.classList.remove('visible');
        }
    });
}

/**
 * Intersection Observer for section animations
 */
function initSectionAnimations() {
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('is-visible');
            }
        });
    }, {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    });

    document.querySelectorAll('.content-section').forEach(section => {
        observer.observe(section);
    });
}

/**
 * Track viewed sections
 */
function trackViewedSections() {
    const viewedSections = new Set(
        JSON.parse(localStorage.getItem('viewedSections') || '[]')
    );

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting && entry.intersectionRatio > 0.5) {
                const sectionId = entry.target.dataset.section;
                if (sectionId) {
                    viewedSections.add(sectionId);
                    localStorage.setItem('viewedSections', JSON.stringify([...viewedSections]));

                    // Update sidebar
                    const sidebarItem = document.querySelector(`.section-item[data-section="${sectionId}"]`);
                    if (sidebarItem) {
                        sidebarItem.classList.add('viewed');
                    }
                }
            }
        });
    }, {
        threshold: 0.5
    });

    document.querySelectorAll('.content-section').forEach(section => {
        observer.observe(section);

        // Check if already viewed
        const sectionId = section.dataset.section;
        if (viewedSections.has(sectionId)) {
            const sidebarItem = document.querySelector(`.section-item[data-section="${sectionId}"]`);
            if (sidebarItem) {
                sidebarItem.classList.add('viewed');
            }
        }
    });
}

/**
 * Initialize chapter page
 */
function initChapter(config) {
    // Configure marked
    configureMarked();

    // Load all sections
    if (config && config.sections) {
        config.sections.forEach((section, index) => {
            // Stagger loading for smoother appearance
            setTimeout(() => {
                // Support both HTML and markdown content types
                if (section.type === 'html') {
                    loadHtmlContent(section.file, section.contentId);
                } else {
                    loadMarkdownContent(section.file, section.contentId);
                }
            }, index * 100);
        });
    }

    // Initialize UI components
    initSidebarToggle();
    initSmoothScroll();
    initScrollToTop();
    initSectionAnimations();
    trackViewedSections();

    // Scroll event listeners
    let scrollTimeout;
    window.addEventListener('scroll', () => {
        updateReadingProgress();

        // Debounce active section updates
        clearTimeout(scrollTimeout);
        scrollTimeout = setTimeout(updateActiveSection, 50);
    });

    // Initial calls
    updateReadingProgress();
    updateActiveSection();
}

// Make functions globally available
window.initChapter = initChapter;
window.copyCode = copyCode;
