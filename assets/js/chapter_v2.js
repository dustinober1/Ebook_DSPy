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

    // Set initial aria-expanded state
    toggle.setAttribute('aria-expanded', 'false');

    const toggleSidebar = (open) => {
        const isOpen = open !== undefined ? open : !sidebar.classList.contains('open');

        if (isOpen) {
            sidebar.classList.add('open');
            toggle.classList.add('active');
            toggle.setAttribute('aria-expanded', 'true');
        } else {
            sidebar.classList.remove('open');
            toggle.classList.remove('active');
            toggle.setAttribute('aria-expanded', 'false');
        }
    };

    toggle.addEventListener('click', () => toggleSidebar());

    // Keyboard support for sidebar toggle
    toggle.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            toggleSidebar();
        }
    });

    // Close sidebar when clicking outside
    document.addEventListener('click', (e) => {
        if (sidebar.classList.contains('open') &&
            !sidebar.contains(e.target) &&
            !toggle.contains(e.target)) {
            toggleSidebar(false);
        }
    });

    // Close sidebar when pressing Escape
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && sidebar.classList.contains('open')) {
            toggleSidebar(false);
            toggle.focus(); // Return focus to toggle button
        }
    });

    // Close sidebar when clicking a link
    sidebar.querySelectorAll('a').forEach(link => {
        link.addEventListener('click', () => {
            if (window.innerWidth <= 1024) {
                toggleSidebar(false);
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
 * Swipe Gestures for Mobile Sidebar
 */
function initSwipeGestures() {
    const sidebar = document.getElementById('sidebar');
    const toggle = document.getElementById('sidebar-toggle');
    if (!sidebar || !toggle) return;

    let touchStartX = 0;
    // const touchStartY = 0; // Unused
    const SWIPE_THRESHOLD = 75;
    const EDGE_THRESHOLD = 50;
    const SCROLL_THRESHOLD = 50;

    document.addEventListener('touchstart', (e) => {
        touchStartX = e.changedTouches[0].screenX;
        // touchStartY = e.changedTouches[0].screenY;
    }, { passive: true });

    document.addEventListener('touchend', (e) => {
        const touchEndX = e.changedTouches[0].screenX;
        const touchEndY = e.changedTouches[0].screenY;

        // We need startY for scroll check
        // Ideally capture startY in touchstart too
    }, { passive: true });
}
// Retrying Swipe Gestures with correct variable scope
function initSwipeGesturesCorrected() {
    const sidebar = document.getElementById('sidebar');
    const toggle = document.getElementById('sidebar-toggle');
    if (!sidebar || !toggle) return;

    let touchStartX = 0;
    let touchStartY = 0;
    const SWIPE_THRESHOLD = 75;
    const EDGE_THRESHOLD = 50;
    const SCROLL_THRESHOLD = 50;

    document.addEventListener('touchstart', (e) => {
        touchStartX = e.changedTouches[0].screenX;
        touchStartY = e.changedTouches[0].screenY;
    }, { passive: true });

    document.addEventListener('touchend', (e) => {
        const touchEndX = e.changedTouches[0].screenX;
        const touchEndY = e.changedTouches[0].screenY;

        const diffX = touchEndX - touchStartX;
        const diffY = touchEndY - touchStartY;

        if (Math.abs(diffY) > SCROLL_THRESHOLD) return;

        // Swipe Right (Open)
        if (diffX > SWIPE_THRESHOLD && touchStartX < EDGE_THRESHOLD) {
            if (!sidebar.classList.contains('open')) {
                sidebar.classList.add('open');
                toggle.classList.add('active');
                toggle.setAttribute('aria-expanded', 'true');
            }
        }

        // Swipe Left (Close)
        if (diffX < -SWIPE_THRESHOLD) {
            if (sidebar.classList.contains('open')) {
                sidebar.classList.remove('open');
                toggle.classList.remove('active');
                toggle.setAttribute('aria-expanded', 'false');
                toggle.focus();
            }
        }
    }, { passive: true });
}

/**
 * Dynamic Breadcrumb Generation
 */
function initBreadcrumbs() {
    const heroContent = document.querySelector('.chapter-hero-content');
    if (!heroContent) return;

    const chapterTitleElement = document.querySelector('.chapter-title');
    const sectionTitleElement = document.querySelector('.section-item.active .section-title');

    const chapterTitle = document.querySelector('.sidebar-header h2')?.textContent.trim() ||
        chapterTitleElement?.textContent.trim() || 'Chapter';

    const currentSectionTitle = sectionTitleElement?.textContent.trim() || 'Introduction';

    const breadcrumbNav = document.createElement('nav');
    breadcrumbNav.className = 'breadcrumbs is-visible fade-in-up';
    breadcrumbNav.setAttribute('aria-label', 'Breadcrumb');
    breadcrumbNav.style.marginBottom = '1rem';
    breadcrumbNav.style.fontSize = '0.9rem';
    breadcrumbNav.style.color = 'var(--secondary-color)';

    breadcrumbNav.innerHTML = `
        <ol style="list-style: none; padding: 0; margin: 0; display: flex; flex-wrap: wrap; align-items: center; gap: 8px;">
            <li><a href="../../index.html" style="color: var(--secondary-color); text-decoration: none;">Home</a></li>
            <li aria-hidden="true" style="opacity: 0.5;">/</li>
            <li><a href="index.html" style="color: var(--secondary-color); text-decoration: none;">${chapterTitle}</a></li>
            ${currentSectionTitle && currentSectionTitle !== 'Preface' && currentSectionTitle !== 'Chapter Overview' ? `
                <li aria-hidden="true" style="opacity: 0.5;">/</li>
                <li style="color: var(--primary-color); font-weight: 500;" aria-current="page">${currentSectionTitle}</li>
            ` : ''}
        </ol>
    `;

    heroContent.insertBefore(breadcrumbNav, heroContent.firstChild);
}

/**
 * Dynamic Content Navigation
 */
function initContentNavigation() {
    const mainContent = document.querySelector('.chapter-content');
    if (!mainContent) return;
    if (document.querySelector('.chapter-nav')) return;

    const currentPath = window.location.pathname.split('/').pop() || 'index.html';
    const sidebarLinks = Array.from(document.querySelectorAll('.section-list a'));
    const currentIndex = sidebarLinks.findIndex(link => {
        const href = link.getAttribute('href');
        return href === currentPath || (currentPath === 'index.html' && href.startsWith('index.html'));
    });

    if (currentIndex === -1) return;

    const navContainer = document.createElement('div');
    navContainer.className = 'chapter-nav is-visible fade-in-up';

    let prevHtml = '';
    if (currentIndex > 0) {
        const prevLink = sidebarLinks[currentIndex - 1];
        const prevTitle = prevLink.querySelector('.section-title').textContent.trim();
        const prevHref = prevLink.getAttribute('href');
        prevHtml = `<a href="${prevHref}" class="nav-link prev"><span class="nav-direction">← Previous</span><span class="nav-title">${prevTitle}</span></a>`;
    }

    let nextHtml = '';
    if (currentIndex < sidebarLinks.length - 1) {
        const nextLink = sidebarLinks[currentIndex + 1];
        const nextTitle = nextLink.querySelector('.section-title').textContent.trim();
        const nextHref = nextLink.getAttribute('href');
        nextHtml = `<a href="${nextHref}" class="nav-link next"><span class="nav-direction">Next →</span><span class="nav-title">${nextTitle}</span></a>`;
    } else {
        const nextChapterBtn = document.querySelector('.next-chapter-btn');
        if (nextChapterBtn) {
            const nextHref = nextChapterBtn.getAttribute('href');
            const nextText = nextChapterBtn.textContent.replace('Next: ', '').replace(' →', '').trim();
            nextHtml = `<a href="${nextHref}" class="nav-link next chapter-step"><span class="nav-direction">Next Chapter →</span><span class="nav-title">${nextText}</span></a>`;
        }
    }

    if (prevHtml || nextHtml) {
        navContainer.innerHTML = `${prevHtml}${nextHtml}`;
        const contentWrapper = document.querySelector('.content-wrapper');
        if (contentWrapper) contentWrapper.appendChild(navContainer);
    }
}

/**
 * Search Functionality
 */
function initSearch() {
    const sidebarHeader = document.querySelector('.sidebar-header');
    if (!sidebarHeader) return;

    // Avoid duplicate search bars
    if (document.querySelector('.sidebar-search')) return;

    const searchContainer = document.createElement('div');
    searchContainer.className = 'sidebar-search';
    // Using inline styles for now as we haven't updated CSS
    searchContainer.style.marginTop = '10px';
    searchContainer.style.position = 'relative';

    searchContainer.innerHTML = `
        <div class="search-input-wrapper" style="position: relative; display: flex; align-items: center; background: rgba(255,255,255,0.15); border-radius: 6px; padding: 4px 8px;">
            <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="opacity: 0.7; margin-right: 6px;">
                <circle cx="11" cy="11" r="8"></circle>
                <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
            </svg>
            <input type="text" id="search-input" placeholder="Search..." aria-label="Search" autocomplete="off" style="background: transparent; border: none; font-size: 0.9rem; color: white; width: 100%; outline: none;">
            <button id="clear-search" aria-label="Clear search" style="display: none; background: none; border: none; cursor: pointer; color: white; padding: 0;">
                <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <line x1="18" y1="6" x2="6" y2="18"></line>
                    <line x1="6" y1="6" x2="18" y2="18"></line>
                </svg>
            </button>
        </div>
        <div id="search-results" class="search-results" style="display: none; position: absolute; top: 105%; left: 0; right: 0; background: white; border: 1px solid var(--border-color); border-radius: 6px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); max-height: 400px; overflow-y: auto; z-index: 1000; color: var(--text-color);"></div>
    `;

    sidebarHeader.appendChild(searchContainer);

    // Placeholder styling
    // Since we can't easily do ::placeholder in inline styles, we rely on browser default or inherited

    const searchInput = document.getElementById('search-input');
    const resultsContainer = document.getElementById('search-results');
    const clearBtn = document.getElementById('clear-search');
    let searchIndex = null;
    let isFetching = false;

    searchInput.addEventListener('focus', async () => {
        if (!searchIndex && !isFetching) {
            isFetching = true;
            try {
                const pathParts = window.location.pathname.split('/');
                let prefix = '';
                if (pathParts.includes('chapters')) {
                    prefix = '../../';
                }
                const response = await fetch(`${prefix}assets/js/search_index.json`);
                searchIndex = await response.json();
            } catch (err) {
                console.error('Failed to load search index:', err);
                // Creating search index failed?
            } finally {
                isFetching = false;
            }
        }
    });

    searchInput.addEventListener('input', (e) => {
        const query = e.target.value.toLowerCase().trim();
        if (query.length > 0) {
            clearBtn.style.display = 'block';
            resultsContainer.style.display = 'block';
            if (!searchIndex) {
                resultsContainer.innerHTML = '<div style="padding: 12px; font-size: 0.85rem; color: #666;">Loading index...</div>';
                return;
            }
            const results = searchIndex.filter(page => {
                return page.title.toLowerCase().includes(query) || page.content.toLowerCase().includes(query);
            }).slice(0, 50);
            displayResults(results, query);
        } else {
            clearBtn.style.display = 'none';
            resultsContainer.style.display = 'none';
        }
    });

    clearBtn.addEventListener('click', () => {
        searchInput.value = '';
        resultsContainer.style.display = 'none';
        clearBtn.style.display = 'none';
        searchInput.focus();
    });

    function displayResults(results, query) {
        if (results.length === 0) {
            resultsContainer.innerHTML = '<div style="padding: 12px; font-size: 0.85rem; color: #666;">No results found</div>';
            return;
        }
        const pathParts = window.location.pathname.split('/');
        let prefix = '';
        if (pathParts.includes('chapters')) {
            prefix = '../../';
        }

        const html = results.map(result => {
            const contentLower = result.content.toLowerCase();
            const queryIdx = contentLower.indexOf(query);
            let snippet = '';
            if (queryIdx > -1) {
                const start = Math.max(0, queryIdx - 40);
                const end = Math.min(result.content.length, queryIdx + 60);
                snippet = '...' + result.content.substring(start, end) + '...';
            } else {
                snippet = result.content.substring(0, 100) + '...';
            }
            snippet = snippet.replace(new RegExp(query, 'gi'), match => `<mark style="background: rgba(255, 235, 59, 0.5); border-radius: 2px;">${match}</mark>`);

            return `
                <a href="${prefix}${result.url}" style="display: block; padding: 10px; border-bottom: 1px solid #eee; text-decoration: none; color: inherit; transition: background 0.2s;">
                    <div style="font-weight: 600; font-size: 0.9rem; color: var(--primary-color); margin-bottom: 4px;">${result.title}</div>
                    <div style="font-size: 0.8rem; color: #666; line-height: 1.4;">${snippet}</div>
                </a>
            `;
        }).join('');
        resultsContainer.innerHTML = html;

        // Add hover effects via JS since inline
        resultsContainer.querySelectorAll('a').forEach(a => {
            a.addEventListener('mouseenter', () => a.style.background = '#f8fafc');
            a.addEventListener('mouseleave', () => a.style.background = 'transparent');
        });
    }

    document.addEventListener('click', (e) => {
        if (!searchContainer.contains(e.target)) {
            resultsContainer.style.display = 'none';
        }
    });
}

function initChapter(config) {
    configureMarked();

    if (config && config.sections) {
        config.sections.forEach((section, index) => {
            setTimeout(() => {
                if (section.type === 'html') {
                    loadHtmlContent(section.file, section.contentId);
                } else {
                    loadMarkdownContent(section.file, section.contentId);
                }
            }, index * 100);
        });
    }

    initSidebarToggle();
    initSmoothScroll();
    initScrollToTop();
    initSectionAnimations();
    trackViewedSections();

    // Init new features
    initSwipeGesturesCorrected();
    initBreadcrumbs();
    initContentNavigation();
    initSearch(); // Initialize search

    let scrollTimeout;
    window.addEventListener('scroll', () => {
        updateReadingProgress();
        clearTimeout(scrollTimeout);
        scrollTimeout = setTimeout(updateActiveSection, 50);
    });

    updateReadingProgress();
    updateActiveSection();
}

window.initChapter = initChapter;
window.copyCode = copyCode;
