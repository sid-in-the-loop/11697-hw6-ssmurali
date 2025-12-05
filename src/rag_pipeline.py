#!/usr/bin/env python3
"""
Main RAG Pipeline for Airline Policy Q&A

This script provides a command-line interface for running RAG-based question answering.
Supports multiple retriever and generator combinations as required by HW6.
"""

import argparse
import sys
import os
from pathlib import Path
from typing import List

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# RAG import is conditional - only import when needed (not for no-retrieval mode)
# from rag.rag import RAG  # Moved to conditional import
from llm.llm import LLM, LLaMa, GPT5Mini, Qwen
from config import (
    Config, RAGConfig, LLMConfig, RetrieverConfig,
    Mode, EmbedderType, RetrieverType
)

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class RAGPipeline:
    """Main RAG pipeline orchestrator."""
    
    def __init__(self, config: Config, no_retrieval: bool = False, refine_summary: bool = False, refine_k: int = 3):
        # Only initialize RAG if not in no-retrieval mode
        if no_retrieval:
            self.rag = None
        else:
            # Lazy import RAG to avoid dependency issues in no-retrieval mode
            from rag.rag import RAG
            self.rag = RAG(config.rag_config)
        
        # Initialize appropriate LLM based on model name
        model_name = config.llm_config.model.lower()
        if model_name == 'llama3':
            self.llm = LLaMa(config.llm_config)
        elif model_name in ['gpt-5-mini', 'gpt-5', 'gpt-4o-mini', 'gpt-4o', 'gpt-4', 'gpt-3.5-turbo']:
            self.llm = GPT5Mini(config.llm_config)
        elif model_name in ['qwen', 'qwen3-4b-instruct-2507']:
            self.llm = Qwen(config.llm_config)
        else:
            logger.warning(f"Unknown model {model_name}, using base LLM")
            self.llm = LLM(config.llm_config)
        
        self.config = config
        self.no_retrieval = no_retrieval
        self.use_refine_summary = refine_summary
        self.refine_k = refine_k
        
        # Initialize GPT-5-mini client for summarization if refine_summary is enabled
        if refine_summary:
            from llm.llm import GPT5Mini
            # Create a separate GPT5Mini instance for summarization
            self.summarizer = GPT5Mini(config.llm_config)
        else:
            self.summarizer = None

    def refine_summary(self, query: str, chunks: List[str], k: int = 3) -> str:
        """Summarize top-k retrieved chunks using GPT-5-mini.
        
        Args:
            query: The user's question
            chunks: List of retrieved context chunks
            k: Number of top chunks to summarize (default: 3)
        
        Returns:
            Summarized context string
        """
        if self.summarizer is None:
            raise ValueError("Summarizer not initialized. Set refine_summary=True in RAGPipeline.")
        
        # Take top-k chunks
        top_k_chunks = chunks[:k]
        
        # Combine chunks into context
        context_text = "\n\n".join([f"Chunk {i+1}:\n{chunk}" for i, chunk in enumerate(top_k_chunks)])
        
        # Create summarization prompt
        messages = [
            {
                'role': 'system',
                'content': 'You are a helpful assistant that summarizes retrieved context passages to answer questions. Create a clear, concise summary that focuses on information relevant to the question. Preserve important details, facts, numbers, and specific information that might answer the question.'
            },
            {
                'role': 'user',
                'content': f"Question: {query}\n\nRetrieved Context:\n{context_text}\n\nPlease summarize the above context passages in a clear and understandable manner, focusing on information that helps answer the question. Keep the summary concise but complete."
            }
        ]
        
        # Call GPT-5-mini API for summarization
        response = self.summarizer.client.chat.completions.create(
            model=self.summarizer.model_name,
            messages=messages,
            temperature=0
        )
        
        summary = response.choices[0].message.content
        logger.debug(f"Summarized {k} chunks into summary of length {len(summary)}")
        
        return summary
    
    def generate(self, query: str) -> str:
        """Generate answer for a query using RAG."""
        assert self.llm is not None
        
        # Retrieve relevant context (or empty list for no-retrieval)
        if self.rag is not None:
            context = self.rag.retrieve(query)
        else:
            context = []
        
        # Optionally refine/summarize context using GPT-5-mini
        if self.use_refine_summary and context:
            try:
                refined_context = self.refine_summary(query, context, k=self.refine_k)
                # Replace context with refined summary (single string)
                context = [refined_context]
                logger.debug(f"Context refined: {len(context)} summary chunks")
            except Exception as e:
                logger.warning(f"Failed to refine context: {e}. Using original context.")
        
        # Generate answer using LLM
        answer = self.llm.generate(query, context)
        
        return answer
    
    def process_questions(self, questions_file: str, output_file: str, max_questions: int = None):
        """Process questions from file and write predictions.
        
        Args:
            questions_file: Path to TSV file with questions
            output_file: Path to output TSV file
            max_questions: Maximum number of questions to process (None = all)
        """
        logger.info(f"Processing questions from {questions_file}")
        
        # Read questions - handle multi-line MCQA format
        questions = []
        current_question_lines = []
        
        with open(questions_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.rstrip('\n\r')  # Keep trailing spaces, only remove newlines
                if not line.strip():
                    # Empty line - skip
                    continue
                
                # Check if this line ends with a question type (factoid, multiple_choice, list)
                parts = line.split('\t')
                if len(parts) >= 2 and parts[-1] in ['factoid', 'multiple_choice', 'list']:
                    # This is the end of a question - combine all lines
                    current_question_lines.append(parts[0] if len(parts) > 1 else line)
                    full_question = '\n'.join(current_question_lines)
                    questions.append(full_question)
                    current_question_lines = []
                    
                    # Limit to max_questions if specified
                    if max_questions and len(questions) >= max_questions:
                        break
                else:
                    # This is part of a multi-line question (e.g., MCQA options)
                    current_question_lines.append(line)
        
        # Handle case where file ends without a question type marker
        if current_question_lines:
            full_question = '\n'.join(current_question_lines)
            questions.append(full_question)
        
        logger.info(f"Found {len(questions)} questions to process")
        
        # Generate answers with progress bar
        try:
            from tqdm import tqdm
        except ImportError:
            logger.warning("tqdm not available, using basic progress logging")
            tqdm = lambda x, **kwargs: x
        
        predictions = []
        for question in tqdm(questions, desc="Processing questions", unit="q"):
            try:
                answer = self.generate(question)
                predictions.append(f"{answer}\t")
            except Exception as e:
                logger.error(f"Error processing question: {e}")
                predictions.append(f"ERROR: {str(e)}\t")
        
        # Write predictions
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            for pred in predictions:
                f.write(pred + '\n')
        
        logger.info(f"Predictions written to {output_file}")


def create_config(
    retriever_type: str,
    generator_type: str,
    embedder_type: str,
    db_path: str,
    data_path: str = None,
    mode: str = 'infer',
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
    topk: int = 20,
    rerank: bool = False,
    rerank_model: str = None
) -> Config:
    """Create configuration from command-line arguments."""
    
    # Map string to enum
    retriever_enum = RetrieverType(retriever_type.lower())
    embedder_enum = EmbedderType(embedder_type.lower())
    mode_enum = Mode(mode.lower())
    
    # Retriever config
    retriever_config = RetrieverConfig(
        retriever_type=retriever_enum,
        normalize_embeddings=True,
        faiss_factory='Flat',
        topk=topk,
        rrf_subsystem_k=100,
        k_rrf=60
    )
    
    # RAG config
    rag_config = RAGConfig(
        embedder_type=embedder_enum,
        retriever_config=retriever_config,
        db_path=db_path,
        data_path=data_path,
        mode=mode_enum,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        rerank=rerank,
        rerank_model=rerank_model,
        rerank_shortlist=50
    )
    
    # LLM config (placeholder - will be extended)
    llm_config = LLMConfig()
    llm_config.model = generator_type
    llm_config.key = ""  # Add API key if needed
    
    return Config(rag_config=rag_config, llm_config=llm_config)


def main():
    parser = argparse.ArgumentParser(description='RAG Pipeline for Airline Policy Q&A')
    
    # Retriever options
    parser.add_argument('--retriever', type=str, required=True,
                       choices=['sparse', 'dense', 'rrf', 'none'],
                       help='Retriever type: sparse (BM25), dense (FAISS), rrf (hybrid), or none')
    
    # Generator options (optional for embed mode, required for infer mode)
    parser.add_argument('--generator', type=str, required=False,
                       choices=['qwen', 'gpt-5-mini', 'llama3'],
                       help='Generator/LLM type: qwen (open-weight), gpt-5-mini (API), llama3 (open-weight). Required for infer mode.')
    
    # Embedder options
    parser.add_argument('--embedder', type=str, default='minilm12',
                       choices=['minilm6', 'minilm12', 'azure'],
                       help='Embedder type: minilm6/minilm12 (open-weight), azure (API-based)')
    
    # Paths
    parser.add_argument('--db_path', type=str, required=True,
                       help='Path to vector database (for inference) or where to save (for embedding)')
    parser.add_argument('--data_path', type=str, default=None,
                       help='Path to documents.json (required for embedding mode)')
    parser.add_argument('--questions', type=str, default='data/question.tsv',
                       help='Path to questions file (default: data/question.tsv)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file path for predictions (required for inference mode)')
    
    # Mode
    parser.add_argument('--mode', type=str, default='infer',
                       choices=['embed', 'infer'],
                       help='Mode: embed (build database) or infer (query)')
    
    # RAG parameters
    parser.add_argument('--chunk_size', type=int, default=1000,
                       help='Chunk size for document splitting (default: 1000)')
    parser.add_argument('--chunk_overlap', type=int, default=100,
                       help='Chunk overlap size (default: 100)')
    parser.add_argument('--topk', type=int, default=20,
                       help='Number of retrieved chunks (default: 20)')
    
    # Reranking
    parser.add_argument('--rerank', action='store_true',
                       help='Enable reranking with cross-encoder')
    parser.add_argument('--rerank_model', type=str, default='cross-encoder/ms-marco-MiniLM-L6-v2',
                       help='Cross-encoder model for reranking')
    
    # Context summarization
    parser.add_argument('--refine-summary', type=str, default='False',
                       choices=['True', 'False', 'true', 'false'],
                       help='Enable context summarization using GPT-5-mini (True/False)')
    parser.add_argument('--refine-k', type=int, default=3,
                       choices=[1, 3, 5],
                       help='Number of top-k chunks to summarize (1, 3, or 5)')
    
    args = parser.parse_args()
    
    # Validate arguments based on mode
    if args.mode == 'infer':
        if not args.generator:
            parser.error("--generator is required in infer mode")
        if not args.output:
            parser.error("--output is required in infer mode")
    elif args.mode == 'embed':
        if not args.data_path:
            parser.error("--data_path is required in embed mode")
    
    # Handle no-retrieval case
    no_retrieval = (args.retriever == 'none')
    if no_retrieval:
        logger.info("Running in no-retrieval mode (direct generation)")
        retriever_type = 'dense'  # Still need a retriever type for config, but won't be used
    else:
        retriever_type = args.retriever
    
    # Use default generator for embed mode (not actually used)
    generator_type = args.generator if args.generator else 'gpt-5-mini'
    
    # Create config
    config = create_config(
        retriever_type=retriever_type,
        generator_type=generator_type,
        embedder_type=args.embedder,
        db_path=args.db_path,
        data_path=args.data_path,
        mode=args.mode,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        topk=args.topk,
        rerank=args.rerank,
        rerank_model=args.rerank_model if args.rerank else None
    )
    
    # Initialize pipeline
    if config.rag_config.mode == Mode.EMBED:
        logger.info("Building vector database...")
        # For embed mode, we don't need LLM, just RAG
        from rag.rag import RAG
        rag = RAG(config.rag_config)
        logger.info("Database built successfully!")
    else:
        logger.info("Initializing RAG pipeline for inference...")
        # Parse refine_summary flag
        refine_summary = args.refine_summary.lower() == 'true'
        if refine_summary:
            logger.info(f"Context summarization enabled: summarizing top-{args.refine_k} chunks with GPT-5-mini")
        
        pipeline = RAGPipeline(config, no_retrieval=no_retrieval, 
                               refine_summary=refine_summary, refine_k=args.refine_k)
        
        # Process questions
        if os.path.exists(args.questions):
            pipeline.process_questions(args.questions, args.output)
        else:
            logger.warning(f"Questions file not found: {args.questions}")
            logger.info("Pipeline initialized. Use pipeline.generate(query) to answer questions.")


if __name__ == "__main__":
    main()


