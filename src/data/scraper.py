#!/usr/bin/env python3
"""
Web and PDF scraper for airline policy knowledge base.
Handles crawling websites and extracting text from PDFs.
"""

import os
import re
import time
import hashlib
import requests
import pdfplumber
import pypdf
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class WebScraper:
    def __init__(self, max_urls=200, delay=1.0):
        self.max_urls = max_urls
        self.delay = delay
        self.visited_urls = set()
        self.crawled_data = []
        
    def get_safe_filename(self, url):
        """Generate human-readable filename from URL."""
        parsed = urlparse(url)
        domain = parsed.netloc.replace('www.', '')
        path = parsed.path.strip('/').replace('/', '_')
        
        if not path:
            path = 'index'
        
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        filename = f"{domain}_{path}_{url_hash}"
        
        return filename[:100]  # Limit length
    
    def extract_links(self, html_content, base_url, max_links=50):
        """Extract relevant links from HTML content, prioritizing keywords."""
        soup = BeautifulSoup(html_content, 'html.parser')
        links = []
        
        for link in soup.find_all('a', href=True):
            href = link.get('href')
            if not href:
                continue
                
            full_url = urljoin(base_url, href)
            parsed = urlparse(full_url)
            
            # Only same domain
            if parsed.netloc != urlparse(base_url).netloc:
                continue
                
            # Skip non-HTML files
            if any(full_url.lower().endswith(ext) for ext in ['.pdf', '.jpg', '.png', '.gif', '.css', '.js']):
                continue
                
            link_text = link.get_text(strip=True)
            links.append((full_url, link_text))
        
        # Score links by relevance
        scored_links = []
        for url, text in links:
            score = 0
            url_lower = url.lower()
            text_lower = text.lower()
            
            # Keywords that indicate relevant content (airline policies)
            keywords = [
                'baggage', 'luggage', 'fee', 'fees', 'policy', 'policies', 
                'ticket', 'change', 'cancel', 'cancellation', 'refund', 
                'allowance', 'carry-on', 'carryon', 'checked', 'restrictions',
                'size', 'weight', 'dimensions', 'charges', 'optional', 'services'
            ]
            for keyword in keywords:
                if keyword in url_lower or keyword in text_lower:
                    score += 2  # Increased weight for keyword matches
            
            # URL patterns that indicate policy pages
            policy_patterns = ['policy', 'policies', 'terms', 'conditions', 'rules', 'guide', 'information', 'info']
            for pattern in policy_patterns:
                if pattern in url_lower:
                    score += 3  # Higher weight for policy-related URLs
                
            scored_links.append((url, text, score))
        
        # Sort by score and filter: only include links with score > 0 (have relevant keywords)
        scored_links.sort(key=lambda x: x[2], reverse=True)
        # Return top links that have at least some relevance (score > 0)
        relevant_links = [url for url, text, score in scored_links if score > 0]
        return relevant_links[:max_links]
    
    def crawl_url(self, url):
        """Crawl a single URL and extract content."""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extract title
            title = soup.find('title')
            title = title.get_text(strip=True) if title else "Untitled"
            
            # Extract main content
            content = soup.get_text()
            content = re.sub(r'\s+', ' ', content).strip()
            
            return {
                'url': url,
                'title': title,
                'content': content,
                'links': self.extract_links(response.content, url)
            }
            
        except Exception as e:
            logger.warning(f"Failed to crawl {url}: {e}")
            return None
    
    def crawl_websites(self, urls, max_depth=2):
        """
        Crawl multiple websites using BFS with depth tracking.
        Goes up to max_depth layers deep from seed URLs, prioritizing relevant links.
        """
        # Initialize queue with seed URLs at depth 0
        queue = [(url, 0) for url in urls]  # (url, depth) tuples
        self.visited_urls = set()
        self.crawled_data = []
        
        # Track URLs by domain to ensure we go deeper per domain
        domain_depths = {}  # domain -> max depth reached
        
        with tqdm(total=self.max_urls, desc="Crawling websites") as pbar:
            while queue and len(self.visited_urls) < self.max_urls:
                current_url, depth = queue.pop(0)
                
                if current_url in self.visited_urls:
                    continue
                
                # Skip if we've exceeded max depth
                if depth > max_depth:
                    continue
                    
                self.visited_urls.add(current_url)
                
                # Track domain depth
                domain = urlparse(current_url).netloc
                if domain not in domain_depths:
                    domain_depths[domain] = 0
                domain_depths[domain] = max(domain_depths[domain], depth)
                
                result = self.crawl_url(current_url)
                if result:
                    self.crawled_data.append(result)
                    
                    # Only follow links if we haven't exceeded max depth
                    if depth < max_depth:
                        # Add new relevant links to queue with incremented depth
                        # Only follow same-domain links to go deeper into each URL structure
                    for link in result['links']:
                        if link not in self.visited_urls and len(self.visited_urls) < self.max_urls:
                                link_domain = urlparse(link).netloc
                                # Only follow links from the same domain (go deeper into each URL)
                                if link_domain == domain:
                                    # Same domain: add with incremented depth
                                    queue.append((link, depth + 1))
                
                pbar.update(1)
                time.sleep(self.delay)
        
        logger.info(f"Crawled {len(self.crawled_data)} pages across {len(domain_depths)} domains")
        logger.info(f"Max depth reached per domain: {dict(sorted(domain_depths.items(), key=lambda x: x[1], reverse=True)[:10])}")
        return self.crawled_data


class PDFScraper:
    def __init__(self):
        self.scraped_data = []
    
    def extract_pdf_text(self, pdf_path):
        """Extract text from PDF file."""
        try:
            # Try pdfplumber first
            with pdfplumber.open(pdf_path) as pdf:
                text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                
                if text.strip():
                    return {
                        'method': 'pdfplumber',
                        'text': text.strip(),
                        'pages': len(pdf.pages)
                    }
        except Exception:
            pass
        
        try:
            # Fallback to pypdf
            with open(pdf_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                
                return {
                    'method': 'pypdf',
                    'text': text.strip(),
                    'pages': len(pdf_reader.pages)
                }
        except Exception as e:
            logger.warning(f"Failed to extract text from {pdf_path}: {e}")
            return None
    
    def scrape_pdfs(self, pdf_paths):
        """Scrape multiple PDF files."""
        failed_pdfs = []
        for pdf_path in tqdm(pdf_paths, desc="Processing PDFs"):
            if not os.path.exists(pdf_path):
                logger.warning(f"PDF file not found: {pdf_path}")
                failed_pdfs.append(pdf_path)
                continue
            
            result = self.extract_pdf_text(pdf_path)
            if result and result['text']:
                filename = os.path.basename(pdf_path)
                self.scraped_data.append({
                    'source': pdf_path,
                    'title': filename.replace('.pdf', ''),
                    'content': result['text'],
                    'pages': result['pages'],
                    'method': result['method']
                })
            else:
                logger.warning(f"Failed to extract text from PDF: {pdf_path}")
                failed_pdfs.append(pdf_path)
        
        logger.info(f"Processed {len(self.scraped_data)} PDFs successfully")
        if failed_pdfs:
            logger.warning(f"Failed to process {len(failed_pdfs)} PDFs: {failed_pdfs}")
        return self.scraped_data


def sanitize_filename(filename, max_length=200):
    """Sanitize filename by removing/replacing invalid characters."""
    # Replace invalid filename characters with underscores
    invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|', '\n', '\r', '\t']
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    # Replace multiple spaces/underscores with single underscore
    filename = re.sub(r'[_\s]+', '_', filename)
    
    # Remove leading/trailing dots and spaces
    filename = filename.strip('. ')
    
    # Limit length
    if len(filename) > max_length:
        filename = filename[:max_length]
    
    # If filename is empty after sanitization, use a default
    if not filename:
        filename = 'untitled'
    
    return filename


def save_raw_data(web_data, pdf_data, output_dir):
    """Save raw scraped data to files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save web data
    web_dir = os.path.join(output_dir, 'html')
    os.makedirs(web_dir, exist_ok=True)
    
    for item in web_data:
        # Sanitize title for filename
        safe_title = sanitize_filename(item['title'])
        url_hash = hashlib.md5(item['url'].encode()).hexdigest()[:8]
        filename = f"{safe_title}_{url_hash}.html"
        filepath = os.path.join(web_dir, filename)
        
        try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"<title>{item['title']}</title>\n")
            f.write(f"<url>{item['url']}</url>\n")
            f.write(f"<content>{item['content']}</content>")
        except Exception as e:
            logger.error(f"Failed to save web data for {item['url']}: {e}")
    
    # Save PDF data
    pdf_dir = os.path.join(output_dir, 'pdf')
    os.makedirs(pdf_dir, exist_ok=True)
    
    for item in pdf_data:
        # Sanitize title for filename
        safe_title = sanitize_filename(item['title'])
        filename = f"{safe_title}.json"
        filepath = os.path.join(pdf_dir, filename)
        
        try:
        import json
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(item, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save PDF data for {item.get('source', 'unknown')}: {e}")


def main():
    """Main scraping function."""
    from urls_config import URLS, PDFS
    
    # Initialize scrapers
    # Increased max_urls to allow for deeper crawling (114 seed URLs * ~2-3 pages per URL = ~300-500 URLs)
    web_scraper = WebScraper(max_urls=500, delay=1.0)
    pdf_scraper = PDFScraper()
    
    # Collect all URLs
    all_urls = []
    for category, urls in URLS.items():
        all_urls.extend(urls)
    
    logger.info(f"Starting web scraping with {len(all_urls)} seed URLs...")
    logger.info("Will crawl up to 2 layers deep, prioritizing links with relevant keywords (baggage, fees, policy, etc.)")
    web_data = web_scraper.crawl_websites(all_urls, max_depth=2)
    
    # Scrape PDFs
    logger.info("Starting PDF scraping...")
    pdf_data = pdf_scraper.scrape_pdfs(PDFS)
    
    # Save raw data
    save_raw_data(web_data, pdf_data, 'raw_data')
    
    logger.info(f"Scraping complete. Web: {len(web_data)}, PDF: {len(pdf_data)}")


if __name__ == "__main__":
    main()


