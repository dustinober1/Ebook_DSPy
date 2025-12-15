document.addEventListener('DOMContentLoaded', () => {
    // 1. Header Scroll State
    const header = document.querySelector('header');

    const handleScroll = () => {
        if (window.scrollY > 50) {
            header.classList.add('scrolled');
        } else {
            header.classList.remove('scrolled');
        }
    };

    window.addEventListener('scroll', handleScroll);

    // 2. Intersection Observer for Scroll Animations
    const observerOptions = {
        root: null,
        rootMargin: '0px',
        threshold: 0.1
    };

    const observer = new IntersectionObserver((entries, observer) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('is-visible');
                observer.unobserve(entry.target); // Only animate once
            }
        });
    }, observerOptions);

    const fadeElements = document.querySelectorAll('.fade-in-up, .card');
    fadeElements.forEach(el => observer.observe(el));

    // 3. Staggered Animation for Grid Items
    const grid = document.querySelector('.grid');
    if (grid) {
        const cards = grid.querySelectorAll('.card');
        cards.forEach((card, index) => {
            card.style.transitionDelay = `${index * 100}ms`;
        });
    }
    // 4. Dark Mode Toggle
    initTheme();
});

function initTheme() {
    const savedTheme = localStorage.getItem('theme');
    const systemDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    const initialTheme = savedTheme || (systemDark ? 'dark' : 'light');
    document.documentElement.setAttribute('data-theme', initialTheme);

    const navList = document.querySelector('nav ul');
    if (navList && !document.getElementById('theme-toggle-li')) {
        const li = document.createElement('li');
        li.id = 'theme-toggle-li';

        const btn = document.createElement('button');
        btn.setAttribute('aria-label', 'Toggle Dark Mode');
        btn.style.background = 'transparent';
        btn.style.border = 'none';
        btn.style.cursor = 'pointer';
        btn.style.color = 'inherit';
        btn.style.padding = '4px';
        btn.style.display = 'flex';
        btn.style.alignItems = 'center';

        btn.innerHTML = `
            <svg class="sun-icon" xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display: ${initialTheme === 'dark' ? 'none' : 'block'}">
                <circle cx="12" cy="12" r="5"></circle>
                <line x1="12" y1="1" x2="12" y2="3"></line>
                <line x1="12" y1="21" x2="12" y2="23"></line>
                <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line>
                <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line>
                <line x1="1" y1="12" x2="3" y2="12"></line>
                <line x1="21" y1="12" x2="23" y2="12"></line>
                <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line>
                <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line>
            </svg>
            <svg class="moon-icon" xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display: ${initialTheme === 'dark' ? 'block' : 'none'}">
                <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path>
            </svg>
        `;

        btn.addEventListener('click', () => {
            const current = document.documentElement.getAttribute('data-theme');
            const next = current === 'dark' ? 'light' : 'dark';
            document.documentElement.setAttribute('data-theme', next);
            localStorage.setItem('theme', next);

            const sun = btn.querySelector('.sun-icon');
            const moon = btn.querySelector('.moon-icon');
            if (next === 'dark') {
                sun.style.display = 'none';
                moon.style.display = 'block';
            } else {
                sun.style.display = 'block';
                moon.style.display = 'none';
            }
        });

        li.appendChild(btn);
        navList.appendChild(li);
    }
}
