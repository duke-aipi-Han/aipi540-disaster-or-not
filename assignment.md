Mini Hackathon #2: Can Machines Understand Us Reliably?

In domains ranging from healthcare to public policy to community support services, organizations increasingly rely on NLP systems that can interpret language accurately and consistently. But subtle failures in understanding (bias, misclassification, hallucination, misinterpretation) can have real consequences. Your task is to prototype a system that explores how well a model understands language, with evaluation at the core.

Choose a text-based problem where misinterpretation has meaningful downstream impact (e.g., misinformation detection, summarizing patient concerns, analyzing sentiment for nonprofits, identifying safety-related content).

Build a quick prototype that:

Uses any NLP approach (classification, NER, summarization, simple prompting, or a small RAG)
Includes an evaluation plan using at least two complementary metrics or stress tests
Demonstrates how evaluation reveals limitations and guides model improvement

Possible Project Ideas

Misinformation Classifier: Build a simple misinformation detector and perform an error analysis.
Medical Complaint Summarizer: Summarize synthetic patient statements and evaluate hallucination using factuality checks.
RAG System: Build a RAG over a corpus of information and evaluate retrieval accuracy with a curated query set.
Wildfire Alert Summarizer: Summarize citizen-reported wildfire sightings and evaluate factual consistency, missing details, and hallucinated geographic information.
Campus Safety Message Prioritizer: Categorize student-submitted safety tips into “informational,” “monitor,” and “urgent,” and evaluate robustness to slang, emojis, and incomplete phrasing.
Mental Health Support Triage Assistant: Classify messages into “informational,” “emotional support needed,” or “escalation required,” and evaluate false negatives on subtle distress cues.
 

1. Pitch (5 minutes max, hard stop)

Your presentation will take place in person (residential cohort) or via video submission (online cohort only; accessible link required).

Your pitch should include:

1. Your problem statement

2. What pre-trained model you used + transfer learning approach

3. What data augmentation techniques you used

4. Preliminary results

 

2. Code & Deployment

You must submit:

A link to a GitHub repository (public or private) containing your full codebase (We will not be evaluating your code for quality for the hackathon, but we will evaluate if you used GitHub best practices e.g., branches, PRs).
A link to a live, deployed web or mobile application