from fastapi import APIRouter, UploadFile, File
from src.services.parser import PDFParser
from src.services.extractor import QuestionExtractor
from src.services.generator import QuestionGenerator

router = APIRouter()

@router.post("/process")
async def process_paper(file: UploadFile = File(...)):
    try:
        # 1. Read File
        contents = await file.read()
        
        # 2. Parse PDF to Markdown (OCR)
        markdown_content = PDFParser.parse(contents)
        
        # 3. Extract Questions
        questions = QuestionExtractor.extract(markdown_content)
        
        # 4. Generate Mimic Questions
        mimic_results = await QuestionGenerator.generate(questions)
        
        return {
            "status": 1,
            "data": {
                "parsed_text_length": len(markdown_content),
                "questions_found": len(questions),
                "generated_results": mimic_results
            }
        }
        
    except Exception as e:
        print(f"Processing error: {e}")
        return {
            "status": 0,
            "data": {
                "error": str(e)
            }
        }
