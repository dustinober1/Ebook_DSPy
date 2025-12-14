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
});
