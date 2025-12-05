#!/usr/bin/env python3
"""
Document processor for airline policy knowledge base.
Handles cleaning, categorization, and duplicate detection.
"""

import os
import re
import json
import hashlib
from collections import defaultdict
from difflib import SequenceMatcher
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DocumentProcessor:
    def __init__(self, min_word_count=50):
        self.min_word_count = min_word_count
        self.category_keywords = {
            "Baggage Policies": [
                "baggage", "luggage", "carry-on", "checked", "allowance", "weight", 
                "size", "dimensions", "restrictions", "prohibited", "items"
            ],
            "Fees and Charges": [
                "fee", "charge", "cost", "price", "payment", "refund", "penalty",
                "extra", "additional", "overweight", "oversized"
            ],
            "Ticket Changes and Cancellations": [
                "change", "modify", "cancel", "refund", "exchange", "rebook",
                "reschedule", "no-show", "penalty", "deadline"
            ],
            "General Policies": [
                "policy", "terms", "conditions", "rules", "regulations", "contract",
                "carriage", "passenger", "rights", "responsibilities"
            ]
        }
    
    def clean_text(self, text):
        """Clean and normalize text content."""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common noise patterns
        noise_patterns = [
            r'cookie|privacy|terms|conditions',
            r'click here|read more|learn more',
            r'subscribe|newsletter|email',
            r'follow us|social media',
            r'copyright|all rights reserved',
            r'javascript|enable javascript',
            r'loading|please wait'
        ]
        
        for pattern in noise_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        return text.strip()
    
    def get_category(self, title, content, url):
        """Determine document category based on content analysis."""
        text = f"{title} {content} {url}".lower()
        
        # Baggage keywords
        baggage_keywords = ['baggage', 'luggage', 'carry-on', 'checked', 'allowance', 'weight', 'size', 'dimensions', 'restrictions', 'prohibited', 'items']
        baggage_score = sum(1 for word in baggage_keywords if word in text)
        
        # Fees keywords
        fees_keywords = ['fee', 'charge', 'cost', 'price', 'payment', 'refund', 'penalty', 'extra', 'additional', 'overweight', 'oversized']
        fees_score = sum(1 for word in fees_keywords if word in text)
        
        # Ticket changes keywords
        changes_keywords = ['change', 'modify', 'cancel', 'refund', 'exchange', 'rebook', 'reschedule', 'no-show', 'penalty', 'deadline']
        changes_score = sum(1 for word in changes_keywords if word in text)
        
        # General policy keywords
        general_keywords = ['policy', 'terms', 'conditions', 'rules', 'regulations', 'contract', 'carriage', 'passenger', 'rights', 'responsibilities']
        general_score = sum(1 for word in general_keywords if word in text)
        
        # Return category with highest score
        scores = {
            "Baggage Policies": baggage_score,
            "Fees and Charges": fees_score,
            "Ticket Changes and Cancellations": changes_score,
            "General Policies": general_score
        }
        
        # Return the category with the highest score, default to General if tied
        best_category = max(scores, key=scores.get)
        return best_category if scores[best_category] > 0 else "General Policies"
    
    def process_web_documents(self, raw_data_dir):
        """Process web documents from raw data."""
        documents = []
        html_dir = os.path.join(raw_data_dir, 'html')
        
        if not os.path.exists(html_dir):
            logger.warning(f"HTML directory not found: {html_dir}")
            return documents
        
        for filename in os.listdir(html_dir):
            if not filename.endswith('.html'):
                continue
                
            filepath = os.path.join(html_dir, filename)
            
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract title, URL, and content
            title_match = re.search(r'<title>(.*?)</title>', content)
            url_match = re.search(r'<url>(.*?)</url>', content)
            content_match = re.search(r'<content>(.*?)</content>', content)
            
            if not all([title_match, url_match, content_match]):
                continue
            
            title = title_match.group(1)
            url = url_match.group(1)
            raw_content = content_match.group(1)
            
            # Clean content
            cleaned_content = self.clean_text(raw_content)
            
            # Filter by word count
            word_count = len(cleaned_content.split())
            if word_count < self.min_word_count:
                continue
            
            # Determine category
            category = self.get_category(title, cleaned_content, url)
            
            documents.append({
                'id': f"doc_{len(documents) + 1:03d}",
                'content': cleaned_content,
                'metadata': {
                    'source': f"raw_data/html/{filename}",
                    'type': 'html',
                    'category': category,
                    'title': title,
                    'url': url,
                    'word_count': word_count
                }
            })
        
        logger.info(f"Processed {len(documents)} web documents")
        return documents
    
    def process_pdf_documents(self, raw_data_dir):
        """Process PDF documents from raw data."""
        documents = []
        pdf_dir = os.path.join(raw_data_dir, 'pdf')
        
        if not os.path.exists(pdf_dir):
            logger.warning(f"PDF directory not found: {pdf_dir}")
            return documents
        
        for filename in os.listdir(pdf_dir):
            if not filename.endswith('.json'):
                continue
                
            filepath = os.path.join(pdf_dir, filename)
            
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            title = data.get('title', filename.replace('.json', ''))
            raw_content = data.get('content', '')
            
            # Clean content
            cleaned_content = self.clean_text(raw_content)
            
            # Filter by word count
            word_count = len(cleaned_content.split())
            if word_count < self.min_word_count:
                continue
            
            # Determine category
            category = self.get_category(title, cleaned_content, filepath)
            
            documents.append({
                'id': f"doc_{len(documents) + 1:03d}",
                'content': cleaned_content,
                'metadata': {
                    'source': f"raw_data/pdf/{filename}",
                    'type': 'pdf',
                    'category': category,
                    'title': title,
                    'word_count': word_count
                }
            })
        
        logger.info(f"Processed {len(documents)} PDF documents")
        return documents


class DuplicateDetector:
    def __init__(self, similarity_threshold=0.85):
        self.similarity_threshold = similarity_threshold
    
    def normalize_text(self, text):
        """Normalize text for comparison."""
        if not text:
            return ""
        return re.sub(r'\s+', ' ', text).strip().lower()
    
    def get_content_hash(self, content):
        """Generate hash for exact duplicate detection."""
        normalized = self.normalize_text(content)
        return hashlib.md5(normalized.encode('utf-8')).hexdigest()
    
    def is_similar(self, content1, content2):
        """Check if two contents are similar (near duplicates)."""
        # Use first 1000 characters for performance
        sample1 = self.normalize_text(content1)[:1000]
        sample2 = self.normalize_text(content2)[:1000]
        
        if not sample1 or not sample2:
            return False
        
        ratio = SequenceMatcher(None, sample1, sample2).ratio()
        return ratio >= self.similarity_threshold
    
    def remove_duplicates(self, documents):
        """Remove duplicate and near-duplicate documents."""
        unique_docs = []
        content_hashes = set()
        processed_indices = set()
        
        logger.info(f"Starting duplicate detection on {len(documents)} documents...")
        
        # First pass: exact duplicates
        for i, doc in enumerate(documents):
            content = doc.get('content', '')
            content_hash = self.get_content_hash(content)
            
            if content_hash not in content_hashes:
                content_hashes.add(content_hash)
                unique_docs.append(doc)
            else:
                logger.info(f"Removed exact duplicate: {doc.get('metadata', {}).get('title', 'Untitled')}")
        
        # Second pass: near duplicates
        final_docs = []
        for i, doc1 in enumerate(unique_docs):
            if i in processed_indices:
                continue
            
            is_duplicate = False
            for j, doc2 in enumerate(unique_docs):
                if i == j or j in processed_indices:
                    continue
                
                if self.is_similar(doc1.get('content', ''), doc2.get('content', '')):
                    # Keep the longer document
                    if len(doc1.get('content', '')) >= len(doc2.get('content', '')):
                        processed_indices.add(j)
                        logger.info(f"Removed near duplicate: {doc2.get('metadata', {}).get('title', 'Untitled')}")
                    else:
                        is_duplicate = True
                        processed_indices.add(i)
                        logger.info(f"Removed near duplicate: {doc1.get('metadata', {}).get('title', 'Untitled')}")
                        break
            
            if not is_duplicate:
                final_docs.append(doc1)
                processed_indices.add(i)
        
        removed_count = len(documents) - len(final_docs)
        logger.info(f"Removed {removed_count} duplicates. Final count: {len(final_docs)}")
        
        return final_docs


def create_knowledge_base(documents, output_file):
    """Create final knowledge base JSON file."""
    # Calculate statistics
    total_documents = len(documents)
    total_word_count = sum(doc.get('metadata', {}).get('word_count', 0) for doc in documents)
    
    # Category distribution
    category_counts = defaultdict(int)
    for doc in documents:
        category = doc.get('metadata', {}).get('category', 'Uncategorized')
        category_counts[category] += 1
    
    knowledge_base = {
        'documents': documents,
        'summary': {
            'total_documents': total_documents,
            'total_word_count': total_word_count,
            'categories': dict(category_counts)
        }
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(knowledge_base, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Knowledge base saved to {output_file}")
    logger.info(f"Total documents: {total_documents}, Total words: {total_word_count:,}")


def main():
    """Main processing function."""
    raw_data_dir = 'raw_data'
    output_file = 'documents.json'
    
    # Initialize processors
    doc_processor = DocumentProcessor()
    duplicate_detector = DuplicateDetector()
    
    # Process documents
    logger.info("Processing web documents...")
    web_docs = doc_processor.process_web_documents(raw_data_dir)
    
    logger.info("Processing PDF documents...")
    pdf_docs = doc_processor.process_pdf_documents(raw_data_dir)
    
    # Combine all documents
    all_documents = web_docs + pdf_docs
    logger.info(f"Total documents before deduplication: {len(all_documents)}")
    
    # Remove duplicates
    unique_documents = duplicate_detector.remove_duplicates(all_documents)
    
    # Create knowledge base
    create_knowledge_base(unique_documents, output_file)


if __name__ == "__main__":
    main()


