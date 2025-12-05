#!/usr/bin/env python3
"""
Quick script to process and save PDFs only.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from scraper import PDFScraper, save_raw_data
from urls_config import PDFS

def main():
    print("Processing PDFs only...")
    pdf_scraper = PDFScraper()
    pdf_data = pdf_scraper.scrape_pdfs(PDFS)
    
    # Save only PDF data (empty web_data list)
    save_raw_data([], pdf_data, 'raw_data')
    print(f"Saved {len(pdf_data)} PDFs to raw_data/pdf/")

if __name__ == "__main__":
    main()



