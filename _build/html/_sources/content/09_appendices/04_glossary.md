# Glossary

A comprehensive guide to terminology used throughout this DSPy ebook and the broader AI/ML ecosystem.

## A

**Adapter**
A component that connects DSPy to external tools, APIs, or systems. Adapters enable integration with vector databases, knowledge bases, and custom tools while maintaining the DSPy abstraction.

**Agent**
An autonomous system that can perceive its environment, make decisions, and take actions toward goals. In DSPy, agents like ReAct combine reasoning and tool use.

**API (Application Programming Interface)**
A set of protocols and tools that enables different software components to communicate. In this ebook, API often refers to the DSPy interface for creating programs.

**Augmented Generation**
The process of enhancing LLM outputs with external information, typically from databases or knowledge sources. See also Retrieval-Augmented Generation (RAG).

**AutoML (Automated Machine Learning)**
Techniques for automatically designing, optimizing, and deploying machine learning models. DSPy's compilation process is a form of AutoML for LLM programs.

## B

**Baseline**
A reference performance level, typically the performance of the original unoptimized program or a simple baseline approach.

**Batch Processing**
Processing multiple inputs together rather than one at a time. Useful for efficiency and sometimes enables optimization opportunities.

**Benchmark**
A standardized test or dataset used to evaluate system performance, often compared across different approaches.

**Bootstrap** (BootstrapFewShot)
A DSPy optimization technique that automatically finds and selects good examples for in-context learning by searching for demonstrations that lead to correct predictions.

**Byte Pair Encoding (BPE)**
A tokenization technique that breaks text into subwords, commonly used by modern language models.

## C

**Cache/Caching**
Storing previously computed results to avoid recomputation. DSPy supports caching to reduce API calls and improve performance.

**Chain-of-Thought**
A prompting technique that asks the model to explain its reasoning steps before arriving at an answer, often improving accuracy.

**Chatbot**
An AI system that engages in conversation with users. In this ebook, chatbots are built as DSPy modules using dialogue modules.

**Classification**
The task of assigning input text to predefined categories or labels.

**Compilation**
The process of optimizing a DSPy program using training data and a metric, similar to compiling code in traditional programming.

**Confidence Score**
A numerical value (typically 0-1) indicating how confident a model is in its prediction.

**Configuration**
Setting up DSPy with a language model, API keys, and other parameters using `dspy.configure()`.

**Contextualized Embeddings**
Vector representations of text that capture meaning based on the surrounding context, as opposed to static embeddings.

**Convergence**
In optimization, reaching a point where further iterations don't significantly improve performance.

## D

**Dataset**
A collection of examples used for training, validation, or testing. DSPy expects datasets with Example objects.

**Demonstration**
Example input-output pairs shown to the model to improve performance through in-context learning.

**Deployment**
Making a trained or optimized DSPy program available for production use.

**DevSet (Development Set)**
A set of examples used during development for iterative improvement. Often used for evaluation and optimization.

**Dialogue System**
A system that engages in multi-turn conversations, maintaining context across multiple exchanges.

**Diarization**
In the context of dialogue, identifying which speaker produced which utterance.

## E

**Embedding**
A vector representation of text or data that captures semantic meaning in a high-dimensional space.

**Entity Extraction**
The task of identifying and extracting specific entities (people, locations, organizations) from text.

**Evaluation**
Measuring how well a program performs on a test set using a metric function.

**Example**
A single data point consisting of input fields and sometimes expected outputs, used for training or testing.

**Expert**
In the context of mixture-of-experts or multi-expert systems, a specialized model handling a specific task type.

**Exploration-Exploitation**
In optimization, the trade-off between exploring new solutions (exploration) and refining known good solutions (exploitation).

## F

**Few-Shot Learning**
Learning from a small number of examples, typically through in-context learning.

**Fine-tuning**
Adapting a pre-trained model to a specific task by training on task-specific data.

**Forward Pass**
In DSPy modules, the computation performed by the `forward()` method, which defines the module's behavior.

**Frozen Parameters**
Parameters that are not updated during optimization, often used to preserve certain model behaviors.

## G

**Generation**
The process of producing new text or outputs, as opposed to analyzing existing text.

**Gradient**
A measure of how a function changes with respect to its inputs, used in optimization.

**Greedy Decoding**
A text generation strategy that always selects the most likely next token.

## H

**Hallucination**
When an LLM generates plausible-sounding but false information.

**Hard Negative**
A negative example that is difficult to distinguish from positive examples, useful for robust evaluation.

**Hierarchical Module**
A module composed of other modules in a nested structure.

**Hyperparameter**
A parameter set before training/optimization, such as learning rate or number of examples. Distinguished from model parameters.

## I

**In-Context Learning**
Learning from examples provided in the prompt itself, without updating model weights.

**InputField**
In DSPy, a field specification defining an input to a signature.

**Instruction**
The prompt or guidance given to an LLM, often optimized during DSPy compilation.

**Intent**
In dialogue systems, the user's underlying goal or request.

**Intermediary Output**
An output from an intermediate step in a multi-step pipeline, not the final result.

## J

**JSON**
JavaScript Object Notation, a common format for structured data often used with LLMs.

## K

**KNN (K-Nearest Neighbors)**
A technique that finds the most similar examples to a query, used in KNNFewShot optimization.

**KNNFewShot**
A DSPy optimization technique that selects in-context examples based on similarity to the query.

## L

**LLM (Large Language Model)**
A large neural network trained on vast amounts of text, capable of generating coherent and contextually relevant text.

**Loss Function**
A function measuring the difference between predicted and expected outputs, minimized during optimization.

**Low-Rank Adaptation (LoRA)**
A technique for efficient fine-tuning that adapts a small number of additional parameters.

## M

**Metric**
A function that evaluates prediction quality, returning a score or boolean indicating correctness.

**MIPRO (Multi-Prompt In-Context Program Optimization)**
An advanced DSPy optimization technique that jointly optimizes instructions and demonstrations.

**Module**
In DSPy, a composable unit that performs a task, analogous to functions in traditional programming.

**Multi-hop**
A reasoning process requiring multiple steps, each building on previous results.

**Multi-task Learning**
Training a single model on multiple related tasks simultaneously.

## N

**Named Entity Recognition (NER)**
The task of identifying and classifying named entities in text.

**Natural Language Processing (NLP)**
The field of AI focused on understanding and generating human language.

**Negative Sampling**
Selecting negative examples for training or evaluation to make the task more challenging.

## O

**Optimization**
The process of improving a program's performance using training data and a metric, typically via DSPy compilation.

**OutputField**
In DSPy, a field specification defining an output from a signature.

**Overfitting**
When a model learns training data too well and performs poorly on new data.

## P

**Parameter**
A learnable value in a model, distinguished from hyperparameters which are set beforehand.

**Prompt**
The input text or instructions given to an LLM to elicit a response.

**Prompt Engineering**
Carefully crafting prompts to improve LLM performance on specific tasks.

## Q

**Query**
The input request or question, typically what a user asks the system.

**Question Answering (QA)**
The task of providing accurate answers to questions, often from a document or knowledge base.

## R

**RAG (Retrieval-Augmented Generation)**
A technique combining document retrieval with generation, where relevant documents are fetched and used to improve generation quality.

**RankBM25**
A ranking function for information retrieval based on term frequency and document length.

**ReAct (Reasoning + Acting)**
A DSPy module and technique combining reasoning steps with action execution, enabling agent behavior.

**Recall**
In evaluation, the proportion of relevant items successfully retrieved or identified.

**Recommendation System**
A system that suggests items to users based on preferences, behavior, or similarity.

**Retrieval**
The process of finding relevant documents or information from a corpus.

**Routing**
Directing queries to different specialized modules or systems based on content or intent.

## S

**Scaling Laws**
Empirical observations about how model performance improves with more data, parameters, or compute.

**Semantic Similarity**
How similar the meaning or concepts of two pieces of text are.

**Signature**
In DSPy, a specification of a task's input/output contract, using either string syntax or Python classes.

**Soft Prompt**
Learnable embeddings placed before a prompt to guide model behavior, used in some optimization techniques.

**Sparse Retrieval**
Retrieval using sparse representations (like BM25), as opposed to dense vector similarity.

**Streaming**
Processing data incrementally as it arrives, rather than waiting for all data before processing.

**String Signature**
A DSPy signature written as a simple string, e.g., "question -> answer".

## T

**Targeted Generation**
Generating output constrained to specific formats, structures, or vocabularies.

**Temperature**
A parameter controlling randomness in LLM outputs (0 = deterministic, higher = more random).

**Test Set**
Data used to evaluate final model performance, kept separate from training and validation data.

**Token**
A unit of text, typically a word or subword (character-level or byte-pair), that serves as input to LLMs.

**Tokenization**
The process of breaking text into tokens for LLM processing.

**Top-k Sampling**
A sampling strategy that considers only the top k most likely next tokens.

**Top-p Sampling (Nucleus Sampling)**
A sampling strategy that considers tokens until cumulative probability reaches p.

**Trace**
A record of intermediate values and computations during a DSPy program's execution, useful for debugging.

**TrainSet (Training Set)**
A set of examples used to train or optimize a program.

**Transformer**
The neural network architecture underlying modern LLMs, using self-attention mechanisms.

**Typed Signature**
A DSPy signature using Python type hints and classes, providing more control than string signatures.

## U

**Underfitting**
When a model is too simple to capture the patterns in training data, performing poorly.

**Utility Function**
A function measuring the value or quality of an outcome, used in optimization.

## V

**Validation Set**
Data used to tune hyperparameters and make optimization decisions during training.

**Vector Database**
A specialized database optimized for storing and searching high-dimensional vectors (embeddings).

**Vectorization**
Converting text to numerical vectors (embeddings) for processing.

**Vocabulary**
The set of all tokens (words/subwords) that an LLM can handle.

## W

**Weighted Sampling**
Sampling where some options are more likely than others, based on assigned weights.

**Word Embedding**
A vector representation of a word capturing semantic meaning.

**Workflow**
A sequence of steps or modules executed in order to accomplish a task.

## X

**Extraction**
The task of pulling specific information from text, like entity extraction or relation extraction.

## Y

**Zero-Shot Learning**
Performing a task without any examples or task-specific training data.

## Z

**Zero-Shot Prompting**
Asking an LLM to perform a task using only the instructions in the prompt, without examples.

---

## Related Glossary Resources

- **[NLTK Glossary](https://www.nltk.org/howto/portuguese_en.html)** - Natural Language Processing terms
- **[ML Glossary](https://ml-cheatsheet.readthedocs.io/)** - Machine Learning concepts
- **[DeepLearning.AI Glossary](https://www.deeplearning.ai/glossary/)** - AI terminology
- **[Papers with Code Glossary](https://paperswithcode.com/glossary/)** - ML research terms

---

**Note:** This glossary covers terms used in this ebook and DSPy-specific concepts. For language model and NLP specifics, refer to the [NLTK Glossary](https://www.nltk.org/howto/portuguese_en.html) or academic papers.
