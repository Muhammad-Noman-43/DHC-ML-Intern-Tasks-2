# Task 5: Auto-Tagging Support Tickets Using LLM

## Overview

This notebook automatically assigns the top 3 most relevant tags to customer support tickets using the Groq API (GPT-OSS-120B model).

## Approach

- **Zero-Shot Prompting**: Model predicts without examples
- **Few-Shot Prompting**: Model predicts with 5 labeled examples provided in the prompt

## Implementation

1. Load customer support tickets dataset (50 tickets for testing)
2. Define classification function supporting both zero-shot and few-shot modes
3. Generate predictions for each ticket via Groq API
4. Extract primary prediction (top tag) for evaluation
5. Evaluate accuracy and generate classification report using scikit-learn

## Key Features

- Extracts top-3 tags per ticket, ordered by relevance
- Uses only predefined categories from dataset
- Rate limiting (2-second pause between API calls)
- Evaluation metrics: Accuracy, Precision, Recall, F1-Score

## Output

Classification report comparing LLM predictions against actual ticket types, including overall accuracy and per-category metrics.
