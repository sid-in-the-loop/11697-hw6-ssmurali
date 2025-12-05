#!/usr/bin/env python3
"""
Evaluation script for RAG QA predictions.

Implements:
1. LLM-as-a-judge (using GPT-5-mini via CMU Gateway) - required metric
2. F1 Score (token-level overlap) - chosen metric
3. Recall Score (token-level overlap) - chosen metric
4. Exact Match (case-insensitive, whitespace-normalized) - additional metric

Usage:
    python src/evaluate.py --prediction <pred_file> --reference <answer_file> --output <eval_file>
"""

import argparse
import sys
import os
from pathlib import Path
from typing import List, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from openai import OpenAI
except ImportError:
    print("Error: openai package required. Install with: pip install openai")
    sys.exit(1)

# CMU Gateway configuration
CMU_GATEWAY_BASE_URL = "https://ai-gateway.andrew.cmu.edu"
GATEWAY_API_KEY = "sk-gx6H5IC6DMWM311Ef2Gdyw"

def init_openai_client():
    """Initialize OpenAI client with CMU Gateway."""
    env_key = os.getenv("CMU_GATEWAY_API_KEY") or os.getenv("OPENAI_API_KEY")
    api_key = env_key if env_key and env_key.startswith('sk-') else GATEWAY_API_KEY
    
    return OpenAI(
        api_key=api_key,
        base_url=CMU_GATEWAY_BASE_URL
    )

def llm_judge(client: OpenAI, question: str, reference_answer: str, prediction: str) -> float:
    """
    Use LLM-as-a-judge to score prediction against reference.
    Returns binary score: 1.0 if match, 0.0 if no match.
    """
    prompt = f"""You are an expert evaluator for question-answering systems.

Question: {question}

Reference Answer (ground truth): {reference_answer}

Predicted Answer (to evaluate): {prediction}

Task: Compare the predicted answer against the reference answer and determine if they match.

You must:
1. Compare the predicted answer DIRECTLY with the reference answer
2. Check if the predicted answer contains the same key information as the reference
3. Consider if the predicted answer is factually correct compared to the reference
4. Assess completeness - does the prediction cover all important points from the reference?

Output ONLY 1 or 0:
- 1 = The predicted answer matches the reference answer (same facts, complete, correct)
- 0 = The predicted answer does NOT match the reference answer (different facts, missing information, or incorrect)

Be strict: Only return 1 if the prediction is essentially equivalent to the reference answer. Return 0 for any significant differences.

Return ONLY the number 1 or 0, nothing else."""

    try:
        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": "You are an expert evaluator. Return only a numeric score."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        
        score_text = response.choices[0].message.content.strip()
        # Extract number from response (should be 0 or 1)
        import re
        match = re.search(r'\b([01])\b', score_text)
        if match:
            score = int(match.group(1))
            return float(score)
        else:
            # Fallback: try to parse any number and convert to binary
            num_match = re.search(r'(\d+)', score_text)
            if num_match:
                num = int(num_match.group(1))
                # If number is > 0, treat as 1, else 0
                return 1.0 if num > 0 else 0.0
            print(f"Warning: Could not parse binary score from: {score_text}, defaulting to 0")
            return 0.0
    except Exception as e:
        print(f"Error in LLM judge: {e}, defaulting to 0")
        return 0.0

def exact_match(reference: str, prediction: str) -> float:
    """
    Exact match metric (case-insensitive, whitespace-normalized).
    Returns 1.0 if match, 0.0 otherwise.
    """
    ref_normalized = ' '.join(reference.lower().split())
    pred_normalized = ' '.join(prediction.lower().split())
    return 1.0 if ref_normalized == pred_normalized else 0.0

def tokenize(text: str) -> set:
    """Tokenize text into lowercase word set."""
    return set(text.lower().split())

def f1_score(reference: str, prediction: str) -> float:
    """
    Calculate F1 score based on token overlap.
    Returns F1 score between 0.0 and 1.0.
    """
    ref_tokens = tokenize(reference)
    pred_tokens = tokenize(prediction)
    
    if len(ref_tokens) == 0 and len(pred_tokens) == 0:
        return 1.0
    if len(ref_tokens) == 0 or len(pred_tokens) == 0:
        return 0.0
    
    # Calculate overlap
    overlap = len(ref_tokens & pred_tokens)
    
    # Precision: overlap / prediction tokens
    precision = overlap / len(pred_tokens) if len(pred_tokens) > 0 else 0.0
    
    # Recall: overlap / reference tokens
    recall = overlap / len(ref_tokens) if len(ref_tokens) > 0 else 0.0
    
    # F1: harmonic mean of precision and recall
    if precision + recall == 0:
        return 0.0
    f1 = 2 * (precision * recall) / (precision + recall)
    
    return f1

def recall_score(reference: str, prediction: str) -> float:
    """
    Calculate Recall score based on token overlap.
    Returns Recall score between 0.0 and 1.0.
    """
    ref_tokens = tokenize(reference)
    pred_tokens = tokenize(prediction)
    
    if len(ref_tokens) == 0:
        return 1.0 if len(pred_tokens) == 0 else 0.0
    
    # Calculate overlap
    overlap = len(ref_tokens & pred_tokens)
    
    # Recall: overlap / reference tokens
    recall = overlap / len(ref_tokens) if len(ref_tokens) > 0 else 0.0
    
    return recall

def load_tsv_file(filepath: str) -> List[str]:
    """Load TSV file, return list of values (first column).
    Handles two formats:
    1. Multi-line predictions with tab separator (predictions)
    2. Simple line-by-line format (references)
    """
    # First, check if file has tabs (multi-line format)
    with open(filepath, 'r', encoding='utf-8') as f:
        sample = f.read(1000)  # Sample first 1KB
        has_tabs = '\t' in sample
        f.seek(0)
        
        if has_tabs:
            # Multi-line format: read until tab separator
            predictions = []
            current_lines = []
            
            for line in f:
                if '\t' in line:
                    # This line contains the tab separator - end of prediction
                    parts = line.split('\t', 1)
                    current_lines.append(parts[0])
                    prediction = '\n'.join(current_lines)
                    predictions.append(prediction)
                    current_lines = []
                elif line.strip():
                    # Continuation of multi-line prediction
                    current_lines.append(line.rstrip('\n'))
                # Empty lines are ignored
            
            # Handle case where file ends without tab
            if current_lines:
                prediction = '\n'.join(current_lines)
                predictions.append(prediction)
            
            return predictions
        else:
            # Simple format: one entry per line
            lines = []
            for line in f:
                line = line.strip()
                if line:
                    lines.append(line)
            return lines

def evaluate(prediction_file: str, reference_file: str, output_file: str, use_llm_judge: bool = True):
    """
    Evaluate predictions against references.
    
    Args:
        prediction_file: Path to predictions TSV (one per line)
        reference_file: Path to reference answers TSV (one per line)
        output_file: Path to write evaluation scores TSV
        use_llm_judge: Whether to use LLM-as-a-judge (slower but more accurate)
    """
    print(f"Loading predictions from {prediction_file}...")
    predictions = load_tsv_file(prediction_file)
    
    print(f"Loading references from {reference_file}...")
    references = load_tsv_file(reference_file)
    
    # Load questions for LLM judge context (handle multi-line MCQA format)
    question_file = "data/question.tsv"
    questions = []
    if os.path.exists(question_file):
        # Parse questions using same logic as rag_pipeline.py
        current_question_lines = []
        with open(question_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.rstrip('\n\r')
                if not line.strip():
                    continue
                parts = line.split('\t')
                if len(parts) >= 2 and parts[-1] in ['factoid', 'multiple_choice', 'list']:
                    current_question_lines.append(parts[0] if len(parts) > 1 else line)
                    full_question = '\n'.join(current_question_lines)
                    questions.append(full_question)
                    current_question_lines = []
                else:
                    current_question_lines.append(line)
        if current_question_lines:
            full_question = '\n'.join(current_question_lines)
            questions.append(full_question)
        print(f"Loaded {len(questions)} questions for context")
    else:
        print(f"Warning: {question_file} not found, using generic questions for LLM judge")
        questions = [""] * len(predictions)
    
    if len(predictions) != len(references):
        raise ValueError(f"Mismatch: {len(predictions)} predictions vs {len(references)} references")
    
    if len(questions) != len(predictions):
        print(f"Warning: {len(questions)} questions vs {len(predictions)} predictions, using generic questions")
        questions = [""] * len(predictions)
    
    print(f"Evaluating {len(predictions)} predictions...")
    
    # Initialize LLM client if needed
    client = None
    if use_llm_judge:
        print("Initializing LLM-as-a-judge (GPT-5-mini)...")
        client = init_openai_client()
    
    scores = []
    f1_scores = []
    recall_scores = []
    em_scores = []
    
    for i, (pred, ref, q) in enumerate(zip(predictions, references, questions)):
        if (i + 1) % 10 == 0:
            print(f"  Processing {i + 1}/{len(predictions)}...")
        
        # All metrics
        em = exact_match(ref, pred)
        f1 = f1_score(ref, pred)
        recall = recall_score(ref, pred)
        em_scores.append(em)
        f1_scores.append(f1)
        recall_scores.append(recall)
        
        # LLM judge
        if use_llm_judge:
            question = q if q else "Question not provided"
            llm_score = llm_judge(client, question, ref, pred)
            scores.append((llm_score, f1, recall, em))
        else:
            scores.append((0.0, f1, recall, em))  # No LLM judge
    
    # Write results
    print(f"Writing results to {output_file}...")
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for llm_score, f1, recall, em in scores:
            f.write(f"{llm_score:.4f}\t{f1:.4f}\t{recall:.4f}\t{em:.4f}\n")
    
    # Print summary
    avg_llm = sum(s[0] for s in scores) / len(scores) if scores else 0.0
    avg_f1 = sum(s[1] for s in scores) / len(scores) if scores else 0.0
    avg_recall = sum(s[2] for s in scores) / len(scores) if scores else 0.0
    avg_em = sum(s[3] for s in scores) / len(scores) if scores else 0.0
    
    print(f"\n✅ Evaluation complete!")
    print(f"Average LLM-as-a-judge score: {avg_llm:.4f}")
    print(f"Average F1 score: {avg_f1:.4f}")
    print(f"Average Recall score: {avg_recall:.4f}")
    print(f"Average Exact Match score: {avg_em:.4f}")
    print(f"Results written to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate QA predictions')
    parser.add_argument('--prediction', type=str, required=True,
                       help='Path to prediction TSV file')
    parser.add_argument('--reference', type=str, required=True,
                       help='Path to reference answer TSV file')
    parser.add_argument('--output', type=str, required=True,
                       help='Path to output evaluation scores TSV file')
    parser.add_argument('--no-llm-judge', action='store_true',
                       help='Skip LLM-as-a-judge, use only exact match')
    
    args = parser.parse_args()
    
    evaluate(
        prediction_file=args.prediction,
        reference_file=args.reference,
        output_file=args.output,
        use_llm_judge=not args.no_llm_judge
    )

if __name__ == "__main__":
    main()

