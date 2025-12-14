// Populate the sidebar
//
// This is a script, and not included directly in the page, to control the total size of the book.
// The TOC contains an entry for each page, so if each page includes a copy of the TOC,
// the total size of the page becomes O(n**2).
class MDBookSidebarScrollbox extends HTMLElement {
    constructor() {
        super();
    }
    connectedCallback() {
        this.innerHTML = '<ol class="chapter"><li class="chapter-item "><span class="chapter-link-wrapper"><a href="../index.html">Introduction</a></span></li><li class="chapter-item "><li class="spacer"></li></li><li class="chapter-item "><li class="part-title">Front Matter</li></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="00-frontmatter/00-preface.html"><strong aria-hidden="true">1.</strong> Preface</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="00-frontmatter/01-how-to-use-this-book.html"><strong aria-hidden="true">2.</strong> How to Use This Book</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="00-frontmatter/02-prerequisites.html"><strong aria-hidden="true">3.</strong> Prerequisites</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="00-frontmatter/03-setup-instructions.html"><strong aria-hidden="true">4.</strong> Setup Instructions</a></span></li><li class="chapter-item "><li class="spacer"></li></li><li class="chapter-item "><li class="part-title">Part I: Foundations</li></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="01-fundamentals/00-chapter-intro.html"><strong aria-hidden="true">5.</strong> Chapter 1: DSPy Fundamentals</a><a class="chapter-fold-toggle"><div>❱</div></a></span><ol class="section"><li class="chapter-item "><span class="chapter-link-wrapper"><a href="01-fundamentals/01-what-is-dspy.html"><strong aria-hidden="true">5.1.</strong> What is DSPy?</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="01-fundamentals/02-programming-vs-prompting.html"><strong aria-hidden="true">5.2.</strong> Programming vs. Prompting</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="01-fundamentals/03-installation-setup.html"><strong aria-hidden="true">5.3.</strong> Installation and Setup</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="01-fundamentals/04-first-dspy-program.html"><strong aria-hidden="true">5.4.</strong> Your First DSPy Program</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="01-fundamentals/05-language-models.html"><strong aria-hidden="true">5.5.</strong> Language Models</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="01-fundamentals/06-exercises.html"><strong aria-hidden="true">5.6.</strong> Exercises</a></span></li></ol><li class="chapter-item "><li class="spacer"></li></li><li class="chapter-item "><li class="part-title">Part II: Core Concepts</li></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="02-signatures/00-chapter-intro.html"><strong aria-hidden="true">6.</strong> Chapter 2: Signatures</a><a class="chapter-fold-toggle"><div>❱</div></a></span><ol class="section"><li class="chapter-item "><span class="chapter-link-wrapper"><a href="02-signatures/01-understanding-signatures.html"><strong aria-hidden="true">6.1.</strong> Understanding Signatures</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="02-signatures/02-signature-syntax.html"><strong aria-hidden="true">6.2.</strong> Signature Syntax</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="02-signatures/03-typed-signatures.html"><strong aria-hidden="true">6.3.</strong> Typed Signatures</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="02-signatures/04-advanced-signatures.html"><strong aria-hidden="true">6.4.</strong> Advanced Signatures</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="02-signatures/05-practical-examples.html"><strong aria-hidden="true">6.5.</strong> Practical Examples</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="02-signatures/06-exercises.html"><strong aria-hidden="true">6.6.</strong> Exercises</a></span></li></ol><li class="chapter-item "><span class="chapter-link-wrapper"><a href="03-modules/00-chapter-intro.html"><strong aria-hidden="true">7.</strong> Chapter 3: Modules</a><a class="chapter-fold-toggle"><div>❱</div></a></span><ol class="section"><li class="chapter-item "><span class="chapter-link-wrapper"><a href="03-modules/01-module-basics.html"><strong aria-hidden="true">7.1.</strong> Module Basics</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="03-modules/02-predict-module.html"><strong aria-hidden="true">7.2.</strong> Predict Module</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="03-modules/02a-typed-predictor.html"><strong aria-hidden="true">7.3.</strong> TypedPredictor</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="03-modules/03-chainofthought.html"><strong aria-hidden="true">7.4.</strong> Chain of Thought</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="03-modules/04-react-agents.html"><strong aria-hidden="true">7.5.</strong> ReAct Agents</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="03-modules/05-custom-modules.html"><strong aria-hidden="true">7.6.</strong> Custom Modules</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="03-modules/06-composing-modules.html"><strong aria-hidden="true">7.7.</strong> Composing Modules</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="03-modules/08-assertions.html"><strong aria-hidden="true">7.8.</strong> Assertions</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="03-modules/07-exercises.html"><strong aria-hidden="true">7.9.</strong> Exercises</a></span></li></ol><li class="chapter-item "><li class="spacer"></li></li><li class="chapter-item "><li class="part-title">Part III: Evaluation and Optimization</li></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="04-evaluation/00-chapter-intro.html"><strong aria-hidden="true">8.</strong> Chapter 4: Evaluation</a><a class="chapter-fold-toggle"><div>❱</div></a></span><ol class="section"><li class="chapter-item "><span class="chapter-link-wrapper"><a href="04-evaluation/01-why-evaluation-matters.html"><strong aria-hidden="true">8.1.</strong> Why Evaluation Matters</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="04-evaluation/02-creating-datasets.html"><strong aria-hidden="true">8.2.</strong> Creating Datasets</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="04-evaluation/03-defining-metrics.html"><strong aria-hidden="true">8.3.</strong> Defining Metrics</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="04-evaluation/04-evaluation-loops.html"><strong aria-hidden="true">8.4.</strong> Evaluation Loops</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="04-evaluation/05-best-practices.html"><strong aria-hidden="true">8.5.</strong> Best Practices</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="04-evaluation/07-structured-prompting.html"><strong aria-hidden="true">8.6.</strong> Structured Prompting</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="04-evaluation/08-llm-as-a-judge.html"><strong aria-hidden="true">8.7.</strong> LLM-as-a-Judge</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="04-evaluation/09-human-aligned-evaluation.html"><strong aria-hidden="true">8.8.</strong> Human-Aligned Evaluation</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="04-evaluation/06-exercises.html"><strong aria-hidden="true">8.9.</strong> Exercises</a></span></li></ol><li class="chapter-item "><span class="chapter-link-wrapper"><a href="05-optimizers/00-chapter-intro.html"><strong aria-hidden="true">9.</strong> Chapter 5: Optimizers and Compilation</a><a class="chapter-fold-toggle"><div>❱</div></a></span><ol class="section"><li class="chapter-item "><span class="chapter-link-wrapper"><a href="05-optimizers/01-compilation-concept.html"><strong aria-hidden="true">9.1.</strong> The Compilation Concept</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="05-optimizers/02-bootstrapfewshot.html"><strong aria-hidden="true">9.2.</strong> BootstrapFewShot</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="05-optimizers/02a-copro.html"><strong aria-hidden="true">9.3.</strong> COPRO: Cost-aware Prompt Optimization</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="05-optimizers/03-mipro.html"><strong aria-hidden="true">9.4.</strong> MIPRO</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="05-optimizers/04-knnfewshot.html"><strong aria-hidden="true">9.5.</strong> KNNFewShot</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="05-optimizers/05-finetuning.html"><strong aria-hidden="true">9.6.</strong> Fine-tuning</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="05-optimizers/07-constraint-driven-optimization.html"><strong aria-hidden="true">9.7.</strong> Constraint-Driven Optimization</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="05-optimizers/08-reflective-prompt-evolution.html"><strong aria-hidden="true">9.8.</strong> Reflective Prompt Evolution</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="05-optimizers/09-copa-method.html"><strong aria-hidden="true">9.9.</strong> COPA Method</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="05-optimizers/10-joint-optimization.html"><strong aria-hidden="true">9.10.</strong> Joint Optimization</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="05-optimizers/11-monte-carlo-optimization.html"><strong aria-hidden="true">9.11.</strong> Monte Carlo Optimization</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="05-optimizers/12-bayesian-optimization.html"><strong aria-hidden="true">9.12.</strong> Bayesian Optimization</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="05-optimizers/13-comprehensive-examples.html"><strong aria-hidden="true">9.13.</strong> Comprehensive Examples</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="05-optimizers/06-choosing-optimizers.html"><strong aria-hidden="true">9.14.</strong> Choosing Optimizers</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="05-optimizers/14-multistage-optimization-theory.html"><strong aria-hidden="true">9.15.</strong> Multi-stage Optimization Theory</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="05-optimizers/15-instruction-tuning-frameworks.html"><strong aria-hidden="true">9.16.</strong> Instruction Tuning Frameworks</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="05-optimizers/16-demonstration-optimization.html"><strong aria-hidden="true">9.17.</strong> Demonstration Optimization</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="05-optimizers/17-multistage-architectures.html"><strong aria-hidden="true">9.18.</strong> Multi-stage Program Architectures</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="05-optimizers/18-complex-pipeline-optimization.html"><strong aria-hidden="true">9.19.</strong> Complex Pipeline Optimization</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="05-optimizers/19-instruction-demonstration-interactions.html"><strong aria-hidden="true">9.20.</strong> Instruction-Demonstration Interactions</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="05-optimizers/20-prompts-as-hyperparameters.html"><strong aria-hidden="true">9.21.</strong> Prompts as Auto-Optimized Hyperparameters</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="05-optimizers/21-minimal-data-pipelines.html"><strong aria-hidden="true">9.22.</strong> Minimal Data Training Pipelines</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="05-optimizers/22-gepa-genetic-pareto-optimization.html"><strong aria-hidden="true">9.23.</strong> GEPA: Genetic-Pareto Optimization</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="05-optimizers/23-state-space-prompt-optimization.html"><strong aria-hidden="true">9.24.</strong> State-Space Search for Prompt Optimization</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="05-optimizers/24-inpars-plus-synthetic-data-ir.html"><strong aria-hidden="true">9.25.</strong> InPars+: Advanced Synthetic Data Generation for Information Retrieval</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="05-optimizers/25-custom-mipro-enhanced-optimization.html"><strong aria-hidden="true">9.26.</strong> CustomMIPROv2: Enhanced Multi-Stage Prompt Optimization</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="05-optimizers/26-automatic-prompt-optimization-research.html"><strong aria-hidden="true">9.27.</strong> Automatic Prompt Optimization Research</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="05-optimizers/07-exercises.html"><strong aria-hidden="true">9.28.</strong> Exercises</a></span></li></ol><li class="chapter-item "><li class="spacer"></li></li><li class="chapter-item "><li class="part-title">Part IV: Real-World Applications</li></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="06-real-world-applications/00-chapter-intro.html"><strong aria-hidden="true">10.</strong> Chapter 6: Building Real-World Applications</a><a class="chapter-fold-toggle"><div>❱</div></a></span><ol class="section"><li class="chapter-item "><span class="chapter-link-wrapper"><a href="06-real-world-applications/01-rag-systems.html"><strong aria-hidden="true">10.1.</strong> RAG Systems</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="06-real-world-applications/02-multi-hop-search.html"><strong aria-hidden="true">10.2.</strong> Multi-hop Search</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="06-real-world-applications/03-classification-tasks.html"><strong aria-hidden="true">10.3.</strong> Classification Tasks</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="06-real-world-applications/04-entity-extraction.html"><strong aria-hidden="true">10.4.</strong> Entity Extraction</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="06-real-world-applications/05-intelligent-agents.html"><strong aria-hidden="true">10.5.</strong> Intelligent Agents</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="06-real-world-applications/06-code-generation.html"><strong aria-hidden="true">10.6.</strong> Code Generation</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="06-real-world-applications/07-perspective-driven-research.html"><strong aria-hidden="true">10.7.</strong> Perspective-Driven Research</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="06-real-world-applications/08-extreme-multilabel-classification.html"><strong aria-hidden="true">10.8.</strong> Extreme Multi-Label Classification</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="06-real-world-applications/08-long-form-generation.html"><strong aria-hidden="true">10.9.</strong> Long-Form Generation</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="06-real-world-applications/09-outline-generation.html"><strong aria-hidden="true">10.10.</strong> Outline Generation</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="06-real-world-applications/11-extreme-few-shot-learning.html"><strong aria-hidden="true">10.11.</strong> Extreme Few-Shot Learning: Training with 10 Gold Labels</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="06-real-world-applications/12-ir-model-training-scratch.html"><strong aria-hidden="true">10.12.</strong> IR Model Training from Scratch</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="06-real-world-applications/13-lingvarbench-healthcare-synthetic-data.html"><strong aria-hidden="true">10.13.</strong> LingVarBench: Synthetic Healthcare Transcript Generation</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="06-real-world-applications/14-scientific-figure-caption-generation.html"><strong aria-hidden="true">10.14.</strong> Scientific Figure Caption Generation with DSPy</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="06-real-world-applications/15-retrieval-augmented-guardrails.html"><strong aria-hidden="true">10.15.</strong> Retrieval-Augmented Guardrails for AI Systems</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="06-real-world-applications/16-graphrag-wikipedia-tidb-tutorial.html"><strong aria-hidden="true">10.16.</strong> GraphRAG from Wikipedia with TiDB</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="06-real-world-applications/17-framework-comparisons-dspy-ecosystem.html"><strong aria-hidden="true">10.17.</strong> Framework Comparisons in the DSPy Ecosystem</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="06-real-world-applications/18-multi-agent-rag-systems.html"><strong aria-hidden="true">10.18.</strong> Multi-Agent RAG Systems</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="06-real-world-applications/07-exercises.html"><strong aria-hidden="true">10.19.</strong> Exercises</a></span></li></ol><li class="chapter-item "><span class="chapter-link-wrapper"><a href="07-advanced-topics/00-chapter-intro.html"><strong aria-hidden="true">11.</strong> Chapter 7: Advanced Topics</a><a class="chapter-fold-toggle"><div>❱</div></a></span><ol class="section"><li class="chapter-item "><span class="chapter-link-wrapper"><a href="07-advanced-topics/01-adapters-tools.html"><strong aria-hidden="true">11.1.</strong> Adapters and Tools</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="07-advanced-topics/02-caching-performance.html"><strong aria-hidden="true">11.2.</strong> Caching and Performance</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="07-advanced-topics/03-async-streaming.html"><strong aria-hidden="true">11.3.</strong> Async and Streaming</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="07-advanced-topics/04-debugging-tracing.html"><strong aria-hidden="true">11.4.</strong> Debugging and Tracing</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="07-advanced-topics/05-deployment-strategies.html"><strong aria-hidden="true">11.5.</strong> Deployment Strategies</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="07-advanced-topics/07-self-refining-pipelines.html"><strong aria-hidden="true">11.6.</strong> Self-Refining Pipelines</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="07-advanced-topics/08-declarative-compilation.html"><strong aria-hidden="true">11.7.</strong> Declarative Compilation</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="07-advanced-topics/06-exercises.html"><strong aria-hidden="true">11.8.</strong> Exercises</a></span></li></ol><li class="chapter-item "><li class="spacer"></li></li><li class="chapter-item "><li class="part-title">Part V: Case Studies</li></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="08-case-studies/00-introduction.html"><strong aria-hidden="true">12.</strong> Chapter 8: Case Studies</a><a class="chapter-fold-toggle"><div>❱</div></a></span><ol class="section"><li class="chapter-item "><span class="chapter-link-wrapper"><a href="08-case-studies/01-enterprise-rag-system.html"><strong aria-hidden="true">12.1.</strong> Enterprise RAG System</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="08-case-studies/02-customer-support-chatbot.html"><strong aria-hidden="true">12.2.</strong> Customer Support Chatbot</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="08-case-studies/03-ai-code-assistant.html"><strong aria-hidden="true">12.3.</strong> AI Code Assistant</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="08-case-studies/04-automated-data-analysis.html"><strong aria-hidden="true">12.4.</strong> Automated Data Analysis</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="08-case-studies/05-storm-writing-assistant.html"><strong aria-hidden="true">12.5.</strong> STORM Writing Assistant</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="08-case-studies/06-assertion-driven-applications.html"><strong aria-hidden="true">12.6.</strong> Assertion-Driven Applications</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="08-case-studies/07-databricks-jetblue-llm-optimization.html"><strong aria-hidden="true">12.7.</strong> Databricks &amp; JetBlue LLM Optimization</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="08-case-studies/08-replit-code-repair-dspy.html"><strong aria-hidden="true">12.8.</strong> Replit Code Repair with DSPy</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="08-case-studies/09-databricks-dspy-platform-integration.html"><strong aria-hidden="true">12.9.</strong> Databricks Platform Integration</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="08-case-studies/10-ddi-behavioral-simulation-automation.html"><strong aria-hidden="true">12.10.</strong> DDI Behavioral Simulation Automation</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="08-case-studies/11-salomatic-medical-report-generation.html"><strong aria-hidden="true">12.11.</strong> Salomatic Medical Report Generation</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="08-case-studies/05-exercises.html"><strong aria-hidden="true">12.12.</strong> Exercises</a></span></li></ol><li class="chapter-item "><li class="spacer"></li></li><li class="chapter-item "><li class="part-title">Appendices</li></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="09-appendices/00-introduction.html"><strong aria-hidden="true">13.</strong> Chapter 9: Appendices</a><a class="chapter-fold-toggle"><div>❱</div></a></span><ol class="section"><li class="chapter-item "><span class="chapter-link-wrapper"><a href="09-appendices/01-api-reference-quick.html"><strong aria-hidden="true">13.1.</strong> API Reference Quick Guide</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="09-appendices/02-troubleshooting.html"><strong aria-hidden="true">13.2.</strong> Troubleshooting</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="09-appendices/03-resources.html"><strong aria-hidden="true">13.3.</strong> Additional Resources</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="09-appendices/04-glossary.html"><strong aria-hidden="true">13.4.</strong> Glossary</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="09-appendices/05-community-resources.html"><strong aria-hidden="true">13.5.</strong> Community Resources</a></span></li></ol></li></ol>';
        // Set the current, active page, and reveal it if it's hidden
        let current_page = document.location.href.toString().split('#')[0].split('?')[0];
        if (current_page.endsWith('/')) {
            current_page += 'index.html';
        }
        const links = Array.prototype.slice.call(this.querySelectorAll('a'));
        const l = links.length;
        for (let i = 0; i < l; ++i) {
            const link = links[i];
            const href = link.getAttribute('href');
            if (href && !href.startsWith('#') && !/^(?:[a-z+]+:)?\/\//.test(href)) {
                link.href = path_to_root + href;
            }
            // The 'index' page is supposed to alias the first chapter in the book.
            if (link.href === current_page
                || i === 0
                && path_to_root === ''
                && current_page.endsWith('/index.html')) {
                link.classList.add('active');
                let parent = link.parentElement;
                while (parent) {
                    if (parent.tagName === 'LI' && parent.classList.contains('chapter-item')) {
                        parent.classList.add('expanded');
                    }
                    parent = parent.parentElement;
                }
            }
        }
        // Track and set sidebar scroll position
        this.addEventListener('click', e => {
            if (e.target.tagName === 'A') {
                const clientRect = e.target.getBoundingClientRect();
                const sidebarRect = this.getBoundingClientRect();
                sessionStorage.setItem('sidebar-scroll-offset', clientRect.top - sidebarRect.top);
            }
        }, { passive: true });
        const sidebarScrollOffset = sessionStorage.getItem('sidebar-scroll-offset');
        sessionStorage.removeItem('sidebar-scroll-offset');
        if (sidebarScrollOffset !== null) {
            // preserve sidebar scroll position when navigating via links within sidebar
            const activeSection = this.querySelector('.active');
            if (activeSection) {
                const clientRect = activeSection.getBoundingClientRect();
                const sidebarRect = this.getBoundingClientRect();
                const currentOffset = clientRect.top - sidebarRect.top;
                this.scrollTop += currentOffset - parseFloat(sidebarScrollOffset);
            }
        } else {
            // scroll sidebar to current active section when navigating via
            // 'next/previous chapter' buttons
            const activeSection = document.querySelector('#mdbook-sidebar .active');
            if (activeSection) {
                activeSection.scrollIntoView({ block: 'center' });
            }
        }
        // Toggle buttons
        const sidebarAnchorToggles = document.querySelectorAll('.chapter-fold-toggle');
        function toggleSection(ev) {
            ev.currentTarget.parentElement.parentElement.classList.toggle('expanded');
        }
        Array.from(sidebarAnchorToggles).forEach(el => {
            el.addEventListener('click', toggleSection);
        });
    }
}
window.customElements.define('mdbook-sidebar-scrollbox', MDBookSidebarScrollbox);


// ---------------------------------------------------------------------------
// Support for dynamically adding headers to the sidebar.

(function() {
    // This is used to detect which direction the page has scrolled since the
    // last scroll event.
    let lastKnownScrollPosition = 0;
    // This is the threshold in px from the top of the screen where it will
    // consider a header the "current" header when scrolling down.
    const defaultDownThreshold = 150;
    // Same as defaultDownThreshold, except when scrolling up.
    const defaultUpThreshold = 300;
    // The threshold is a virtual horizontal line on the screen where it
    // considers the "current" header to be above the line. The threshold is
    // modified dynamically to handle headers that are near the bottom of the
    // screen, and to slightly offset the behavior when scrolling up vs down.
    let threshold = defaultDownThreshold;
    // This is used to disable updates while scrolling. This is needed when
    // clicking the header in the sidebar, which triggers a scroll event. It
    // is somewhat finicky to detect when the scroll has finished, so this
    // uses a relatively dumb system of disabling scroll updates for a short
    // time after the click.
    let disableScroll = false;
    // Array of header elements on the page.
    let headers;
    // Array of li elements that are initially collapsed headers in the sidebar.
    // I'm not sure why eslint seems to have a false positive here.
    // eslint-disable-next-line prefer-const
    let headerToggles = [];
    // This is a debugging tool for the threshold which you can enable in the console.
    let thresholdDebug = false;

    // Updates the threshold based on the scroll position.
    function updateThreshold() {
        const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
        const windowHeight = window.innerHeight;
        const documentHeight = document.documentElement.scrollHeight;

        // The number of pixels below the viewport, at most documentHeight.
        // This is used to push the threshold down to the bottom of the page
        // as the user scrolls towards the bottom.
        const pixelsBelow = Math.max(0, documentHeight - (scrollTop + windowHeight));
        // The number of pixels above the viewport, at least defaultDownThreshold.
        // Similar to pixelsBelow, this is used to push the threshold back towards
        // the top when reaching the top of the page.
        const pixelsAbove = Math.max(0, defaultDownThreshold - scrollTop);
        // How much the threshold should be offset once it gets close to the
        // bottom of the page.
        const bottomAdd = Math.max(0, windowHeight - pixelsBelow - defaultDownThreshold);
        let adjustedBottomAdd = bottomAdd;

        // Adjusts bottomAdd for a small document. The calculation above
        // assumes the document is at least twice the windowheight in size. If
        // it is less than that, then bottomAdd needs to be shrunk
        // proportional to the difference in size.
        if (documentHeight < windowHeight * 2) {
            const maxPixelsBelow = documentHeight - windowHeight;
            const t = 1 - pixelsBelow / Math.max(1, maxPixelsBelow);
            const clamp = Math.max(0, Math.min(1, t));
            adjustedBottomAdd *= clamp;
        }

        let scrollingDown = true;
        if (scrollTop < lastKnownScrollPosition) {
            scrollingDown = false;
        }

        if (scrollingDown) {
            // When scrolling down, move the threshold up towards the default
            // downwards threshold position. If near the bottom of the page,
            // adjustedBottomAdd will offset the threshold towards the bottom
            // of the page.
            const amountScrolledDown = scrollTop - lastKnownScrollPosition;
            const adjustedDefault = defaultDownThreshold + adjustedBottomAdd;
            threshold = Math.max(adjustedDefault, threshold - amountScrolledDown);
        } else {
            // When scrolling up, move the threshold down towards the default
            // upwards threshold position. If near the bottom of the page,
            // quickly transition the threshold back up where it normally
            // belongs.
            const amountScrolledUp = lastKnownScrollPosition - scrollTop;
            const adjustedDefault = defaultUpThreshold - pixelsAbove
                + Math.max(0, adjustedBottomAdd - defaultDownThreshold);
            threshold = Math.min(adjustedDefault, threshold + amountScrolledUp);
        }

        if (documentHeight <= windowHeight) {
            threshold = 0;
        }

        if (thresholdDebug) {
            const id = 'mdbook-threshold-debug-data';
            let data = document.getElementById(id);
            if (data === null) {
                data = document.createElement('div');
                data.id = id;
                data.style.cssText = `
                    position: fixed;
                    top: 50px;
                    right: 10px;
                    background-color: 0xeeeeee;
                    z-index: 9999;
                    pointer-events: none;
                `;
                document.body.appendChild(data);
            }
            data.innerHTML = `
                <table>
                  <tr><td>documentHeight</td><td>${documentHeight.toFixed(1)}</td></tr>
                  <tr><td>windowHeight</td><td>${windowHeight.toFixed(1)}</td></tr>
                  <tr><td>scrollTop</td><td>${scrollTop.toFixed(1)}</td></tr>
                  <tr><td>pixelsAbove</td><td>${pixelsAbove.toFixed(1)}</td></tr>
                  <tr><td>pixelsBelow</td><td>${pixelsBelow.toFixed(1)}</td></tr>
                  <tr><td>bottomAdd</td><td>${bottomAdd.toFixed(1)}</td></tr>
                  <tr><td>adjustedBottomAdd</td><td>${adjustedBottomAdd.toFixed(1)}</td></tr>
                  <tr><td>scrollingDown</td><td>${scrollingDown}</td></tr>
                  <tr><td>threshold</td><td>${threshold.toFixed(1)}</td></tr>
                </table>
            `;
            drawDebugLine();
        }

        lastKnownScrollPosition = scrollTop;
    }

    function drawDebugLine() {
        if (!document.body) {
            return;
        }
        const id = 'mdbook-threshold-debug-line';
        const existingLine = document.getElementById(id);
        if (existingLine) {
            existingLine.remove();
        }
        const line = document.createElement('div');
        line.id = id;
        line.style.cssText = `
            position: fixed;
            top: ${threshold}px;
            left: 0;
            width: 100vw;
            height: 2px;
            background-color: red;
            z-index: 9999;
            pointer-events: none;
        `;
        document.body.appendChild(line);
    }

    function mdbookEnableThresholdDebug() {
        thresholdDebug = true;
        updateThreshold();
        drawDebugLine();
    }

    window.mdbookEnableThresholdDebug = mdbookEnableThresholdDebug;

    // Updates which headers in the sidebar should be expanded. If the current
    // header is inside a collapsed group, then it, and all its parents should
    // be expanded.
    function updateHeaderExpanded(currentA) {
        // Add expanded to all header-item li ancestors.
        let current = currentA.parentElement;
        while (current) {
            if (current.tagName === 'LI' && current.classList.contains('header-item')) {
                current.classList.add('expanded');
            }
            current = current.parentElement;
        }
    }

    // Updates which header is marked as the "current" header in the sidebar.
    // This is done with a virtual Y threshold, where headers at or below
    // that line will be considered the current one.
    function updateCurrentHeader() {
        if (!headers || !headers.length) {
            return;
        }

        // Reset the classes, which will be rebuilt below.
        const els = document.getElementsByClassName('current-header');
        for (const el of els) {
            el.classList.remove('current-header');
        }
        for (const toggle of headerToggles) {
            toggle.classList.remove('expanded');
        }

        // Find the last header that is above the threshold.
        let lastHeader = null;
        for (const header of headers) {
            const rect = header.getBoundingClientRect();
            if (rect.top <= threshold) {
                lastHeader = header;
            } else {
                break;
            }
        }
        if (lastHeader === null) {
            lastHeader = headers[0];
            const rect = lastHeader.getBoundingClientRect();
            const windowHeight = window.innerHeight;
            if (rect.top >= windowHeight) {
                return;
            }
        }

        // Get the anchor in the summary.
        const href = '#' + lastHeader.id;
        const a = [...document.querySelectorAll('.header-in-summary')]
            .find(element => element.getAttribute('href') === href);
        if (!a) {
            return;
        }

        a.classList.add('current-header');

        updateHeaderExpanded(a);
    }

    // Updates which header is "current" based on the threshold line.
    function reloadCurrentHeader() {
        if (disableScroll) {
            return;
        }
        updateThreshold();
        updateCurrentHeader();
    }


    // When clicking on a header in the sidebar, this adjusts the threshold so
    // that it is located next to the header. This is so that header becomes
    // "current".
    function headerThresholdClick(event) {
        // See disableScroll description why this is done.
        disableScroll = true;
        setTimeout(() => {
            disableScroll = false;
        }, 100);
        // requestAnimationFrame is used to delay the update of the "current"
        // header until after the scroll is done, and the header is in the new
        // position.
        requestAnimationFrame(() => {
            requestAnimationFrame(() => {
                // Closest is needed because if it has child elements like <code>.
                const a = event.target.closest('a');
                const href = a.getAttribute('href');
                const targetId = href.substring(1);
                const targetElement = document.getElementById(targetId);
                if (targetElement) {
                    threshold = targetElement.getBoundingClientRect().bottom;
                    updateCurrentHeader();
                }
            });
        });
    }

    // Takes the nodes from the given head and copies them over to the
    // destination, along with some filtering.
    function filterHeader(source, dest) {
        const clone = source.cloneNode(true);
        clone.querySelectorAll('mark').forEach(mark => {
            mark.replaceWith(...mark.childNodes);
        });
        dest.append(...clone.childNodes);
    }

    // Scans page for headers and adds them to the sidebar.
    document.addEventListener('DOMContentLoaded', function() {
        const activeSection = document.querySelector('#mdbook-sidebar .active');
        if (activeSection === null) {
            return;
        }

        const main = document.getElementsByTagName('main')[0];
        headers = Array.from(main.querySelectorAll('h2, h3, h4, h5, h6'))
            .filter(h => h.id !== '' && h.children.length && h.children[0].tagName === 'A');

        if (headers.length === 0) {
            return;
        }

        // Build a tree of headers in the sidebar.

        const stack = [];

        const firstLevel = parseInt(headers[0].tagName.charAt(1));
        for (let i = 1; i < firstLevel; i++) {
            const ol = document.createElement('ol');
            ol.classList.add('section');
            if (stack.length > 0) {
                stack[stack.length - 1].ol.appendChild(ol);
            }
            stack.push({level: i + 1, ol: ol});
        }

        // The level where it will start folding deeply nested headers.
        const foldLevel = 3;

        for (let i = 0; i < headers.length; i++) {
            const header = headers[i];
            const level = parseInt(header.tagName.charAt(1));

            const currentLevel = stack[stack.length - 1].level;
            if (level > currentLevel) {
                // Begin nesting to this level.
                for (let nextLevel = currentLevel + 1; nextLevel <= level; nextLevel++) {
                    const ol = document.createElement('ol');
                    ol.classList.add('section');
                    const last = stack[stack.length - 1];
                    const lastChild = last.ol.lastChild;
                    // Handle the case where jumping more than one nesting
                    // level, which doesn't have a list item to place this new
                    // list inside of.
                    if (lastChild) {
                        lastChild.appendChild(ol);
                    } else {
                        last.ol.appendChild(ol);
                    }
                    stack.push({level: nextLevel, ol: ol});
                }
            } else if (level < currentLevel) {
                while (stack.length > 1 && stack[stack.length - 1].level > level) {
                    stack.pop();
                }
            }

            const li = document.createElement('li');
            li.classList.add('header-item');
            li.classList.add('expanded');
            if (level < foldLevel) {
                li.classList.add('expanded');
            }
            const span = document.createElement('span');
            span.classList.add('chapter-link-wrapper');
            const a = document.createElement('a');
            span.appendChild(a);
            a.href = '#' + header.id;
            a.classList.add('header-in-summary');
            filterHeader(header.children[0], a);
            a.addEventListener('click', headerThresholdClick);
            const nextHeader = headers[i + 1];
            if (nextHeader !== undefined) {
                const nextLevel = parseInt(nextHeader.tagName.charAt(1));
                if (nextLevel > level && level >= foldLevel) {
                    const toggle = document.createElement('a');
                    toggle.classList.add('chapter-fold-toggle');
                    toggle.classList.add('header-toggle');
                    toggle.addEventListener('click', () => {
                        li.classList.toggle('expanded');
                    });
                    const toggleDiv = document.createElement('div');
                    toggleDiv.textContent = '❱';
                    toggle.appendChild(toggleDiv);
                    span.appendChild(toggle);
                    headerToggles.push(li);
                }
            }
            li.appendChild(span);

            const currentParent = stack[stack.length - 1];
            currentParent.ol.appendChild(li);
        }

        const onThisPage = document.createElement('div');
        onThisPage.classList.add('on-this-page');
        onThisPage.append(stack[0].ol);
        const activeItemSpan = activeSection.parentElement;
        activeItemSpan.after(onThisPage);
    });

    document.addEventListener('DOMContentLoaded', reloadCurrentHeader);
    document.addEventListener('scroll', reloadCurrentHeader, { passive: true });
})();

