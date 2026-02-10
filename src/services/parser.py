import fitz
from typing import List, Tuple
from src.services.ocr import DextoraOCR

class PDFParser:
    @staticmethod
    def parse(pdf_bytes: bytes) -> str:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        markdown_content = ""
        
        print(f"Parsing PDF with {len(doc)} pages...")
        
        for i, page in enumerate(doc):
            print(f"Processing page {i+1}...")
            # Render page to image (2x zoom for better OCR)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img_bytes = pix.tobytes("png")
            
            # OCR
            result = DextoraOCR.scan_bytes(img_bytes, f"page_{i+1}.png")
            
            if result.get("success", False):
                text = result.get("scanned_text", "")
                markdown_content += f"\n\n## Page {i+1}\n\n{text}"
            else:
                print(f"OCR failed for page {i+1}, falling back to text extraction")
                text = page.get_text()
                markdown_content += f"\n\n## Page {i+1} (Fallback)\n\n{text}"
                
        return markdown_content
