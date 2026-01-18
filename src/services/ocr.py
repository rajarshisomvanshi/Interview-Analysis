import requests
from typing import Dict, Any

class DextoraOCR:
    API_URL = "https://ocr.dextora.org/scan"
    
    @classmethod
    def scan_bytes(cls, image_bytes: bytes, filename: str = "image.png") -> Dict[str, Any]:
        try:
            files = {'image': (filename, image_bytes)}
            response = requests.post(cls.API_URL, files=files)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error calling Dextora OCR: {e}")
            return {"success": False, "scanned_text": "", "error": str(e)}
