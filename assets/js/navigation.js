// Professional Navigation for DSPy Ebook

document.addEventListener('DOMContentLoaded', function() {
    // Create sidebar navigation
    createSidebarNavigation();

    // Add page navigation buttons
    addPageNavigationButtons();

    // Add chapter progress indicator
    addChapterProgress();

    // Highlight current chapter in sidebar
    highlightCurrentChapter();
});

function createSidebarNavigation() {
    const sidebar = document.createElement('div');
    sidebar.className = 'page-sidebar';

    sidebar.innerHTML = `
        <div class="logo">
            <h1>DSPy</h1>
            <div class="subtitle">A Practical Guide</div>
        </div>
        <nav class="sidebar-nav">
            <div class="nav-part">
                <div class="nav-part-title">Frontmatter</div>
                <ul class="nav-chapters">
                    <li><a href="../00-frontmatter/00-preface.html"><span class="chapter-number">P</span>Preface</a></li>
                    <li><a href="../00-frontmatter/01-how-to-use-this-book.html"><span class="chapter-number">1</span>How to Use This Book</a></li>
                    <li><a href="../00-frontmatter/02-prerequisites.html"><span class="chapter-number">2</span>Prerequisites</a></li>
                    <li><a href="../00-frontmatter/03-setup-instructions.html"><span class="chapter-number">3</span>Setup Instructions</a></li>
                </ul>
            </div>
            <div class="nav-part">
                <div class="nav-part-title">Part I: Foundations</div>
                <ul class="nav-chapters">
                    <li><a href="../01-fundamentals/01-what-is-dspy.html"><span class="chapter-number">1</span>What is DSPy?</a></li>
                    <li><a href="../01-fundamentals/02-getting-started.html"><span class="chapter-number">2</span>Getting Started</a></li>
                    <li><a href="../01-fundamentals/03-first-program.html"><span class="chapter-number">3</span>Your First Program</a></li>
                </ul>
            </div>
            <div class="nav-part">
                <div class="nav-part-title">Part II: Core Concepts</div>
                <ul class="nav-chapters">
                    <li><a href="../02-core-concepts/01-signatures.html"><span class="chapter-number">4</span>Signatures</a></li>
                    <li><a href="../02-core-concepts/02-modules.html"><span class="chapter-number">5</span>Modules</a></li>
                    <li><a href="../02-core-concepts/03-composition.html"><span class="chapter-number">6</span>Composition</a></li>
                </ul>
            </div>
            <div class="nav-part">
                <div class="nav-part-title">Part III: Optimization</div>
                <ul class="nav-chapters">
                    <li><a href="../03-optimization/01-evaluation.html"><span class="chapter-number">7</span>Evaluation</a></li>
                    <li><a href="../03-optimization/02-optimizers.html"><span class="chapter-number">8</span>Optimizers</a></li>
                    <li><a href="../03-optimization/03-compilation.html"><span class="chapter-number">9</span>Compilation</a></li>
                </ul>
            </div>
            <div class="nav-part">
                <div class="nav-part-title">Part IV: Applications</div>
                <ul class="nav-chapters">
                    <li><a href="../04-applications/01-rag-systems.html"><span class="chapter-number">10</span>RAG Systems</a></li>
                    <li><a href="../04-applications/02-agents.html"><span class="chapter-number">11</span>Agents</a></li>
                    <li><a href="../04-applications/03-pipelines.html"><span class="chapter-number">12</span>Pipelines</a></li>
                </ul>
            </div>
            <div class="nav-part">
                <div class="nav-part-title">Part V: Case Studies</div>
                <ul class="nav-chapters">
                    <li><a href="../05-case-studies/01-healthcare.html"><span class="chapter-number">13</span>Healthcare</a></li>
                    <li><a href="../05-case-studies/02-finance.html"><span class="chapter-number">14</span>Finance</a></li>
                    <li><a href="../05-case-studies/03-legal.html"><span class="chapter-number">15</span>Legal</a></li>
                </ul>
            </div>
            <div class="nav-part">
                <div class="nav-part-title">Appendices</div>
                <ul class="nav-chapters">
                    <li><a href="../09-appendices/01-api-reference.html"><span class="chapter-number">A</span>API Reference</a></li>
                    <li><a href="../09-appendices/02-troubleshooting.html"><span class="chapter-number">B</span>Troubleshooting</a></li>
                    <li><a href="../09-appendices/03-glossary.html"><span class="chapter-number">C</span>Glossary</a></li>
                </ul>
            </div>
        </nav>
    `;

    document.body.insertBefore(sidebar, document.body.firstChild);

    // Wrap page content in wrapper
    const content = document.querySelector('.content');
    if (content) {
        const wrapper = document.createElement('div');
        wrapper.className = 'page-wrapper';
        content.parentNode.insertBefore(wrapper, content);
        wrapper.appendChild(content);
    }
}

function addPageNavigationButtons() {
    const navigation = document.querySelector('.nav-chapters');
    if (!navigation) return;

    const links = Array.from(navigation.querySelectorAll('a'));
    const currentPath = window.location.pathname;
    const currentIndex = links.findIndex(link => link.href.endsWith(currentPath) || link.href.includes(currentPath));

    const buttonsContainer = document.createElement('div');
    buttonsContainer.className = 'page-nav-buttons';

    // Previous button
    if (currentIndex > 0) {
        const prevLink = links[currentIndex - 1];
        const prevButton = document.createElement('a');
        prevButton.href = prevLink.href;
        prevButton.className = 'nav-button prev';
        prevButton.innerHTML = `
            <span class="direction">Previous</span>
            <span class="arrow">←</span>
        `;
        buttonsContainer.appendChild(prevButton);
    }

    // Next button
    if (currentIndex < links.length - 1) {
        const nextLink = links[currentIndex + 1];
        const nextButton = document.createElement('a');
        nextButton.href = nextLink.href;
        nextButton.className = 'nav-button next';
        nextButton.innerHTML = `
            <span class="arrow">→</span>
            <span class="direction">Next</span>
        `;
        buttonsContainer.appendChild(nextButton);
    }

    // Add buttons at the end of content
    const content = document.querySelector('.content');
    if (content && buttonsContainer.children.length > 0) {
        content.appendChild(buttonsContainer);
    }
}

function addChapterProgress() {
    const progressBar = document.createElement('div');
    progressBar.className = 'chapter-progress';
    progressBar.innerHTML = '<div class="bar"></div>';
    document.body.appendChild(progressBar);

    // Update progress on scroll
    window.addEventListener('scroll', updateProgress);

    function updateProgress() {
        const scrollTop = window.pageYOffset;
        const docHeight = document.documentElement.scrollHeight - window.innerHeight;
        const scrollPercent = (scrollTop / docHeight) * 100;
        progressBar.querySelector('.bar').style.width = scrollPercent + '%';
    }
}

function highlightCurrentChapter() {
    const currentPath = window.location.pathname;
    const links = document.querySelectorAll('.sidebar-nav a');

    links.forEach(link => {
        if (link.href.includes(currentPath)) {
            link.classList.add('active');
        } else {
            link.classList.remove('active');
        }
    });
}

// Mobile menu toggle
function addMobileMenuToggle() {
    const toggle = document.createElement('button');
    toggle.innerHTML = '☰';
    toggle.style.cssText = `
        position: fixed;
        top: 1rem;
        left: 1rem;
        z-index: 1001;
        background: var(--color-primary);
        color: white;
        border: none;
        padding: 0.5rem;
        border-radius: 0.25rem;
        cursor: pointer;
        font-size: 1.25rem;
    `;

    document.body.appendChild(toggle);

    toggle.addEventListener('click', () => {
        const sidebar = document.querySelector('.page-sidebar');
        sidebar.style.transform = sidebar.style.transform === 'translateX(0)' ? 'translateX(-100%)' : 'translateX(0)';
    });
}

// Add mobile toggle for screens < 1024px
if (window.innerWidth < 1024) {
    addMobileMenuToggle();
}