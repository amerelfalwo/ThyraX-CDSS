from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
import base64
import json
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from app.core.config import settings
from app.core.security import verify_internal_api_key
from app.schemas.labs import ExtractedLabsResponse

router = APIRouter(prefix="/labs", tags=["Premium AI OCR"], dependencies=[Depends(verify_internal_api_key)])

@router.post("/extract", response_model=ExtractedLabsResponse)
async def extract_labs_from_image(file: UploadFile = File(...)):
    """
    Use Claude 3 Vision API to OCR and extract TSH, T3, T4, and notes from an image.
    Does NOT save to database - used for frontend preview/autofill.
    """
    if not settings.GOOGLE_API_KEY_LABS:
        raise HTTPException(status_code=500, detail="Google API Key for Labs not configured.")
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    try:
        image_bytes = await file.read()
        b64_img = base64.b64encode(image_bytes).decode("utf-8")
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            api_key=settings.GOOGLE_API_KEY_LABS
        )
        
        prompt = (
            "Analyze this medical laboratory report image. Extract the specific values "
            "for the following features and return them as a valid JSON object ONLY. "
            "Use these keys: \"TSH\" (Thyroid Stimulating Hormone), \"T3\" (Triiodothyronine), "
            "\"TT4\" (Total T4), \"FTI\" (Free Thyroxine Index), \"T4U\" (T4 Uptake), "
            "and \"test_date\" (YYYY-MM-DD). If any value is missing, return null for that key."
        )

        messages = [
            SystemMessage(content="You are a medical OCR assistant. You strictly return JSON."),
            HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{file.content_type};base64,{b64_img}"}}
                ]
            )
        ]
        
        response = llm.invoke(messages)
        response_text = response.content.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        
        data = json.loads(response_text)
        
        return ExtractedLabsResponse(
            TSH=data.get("TSH"),
            T3=data.get("T3"),
            TT4=data.get("TT4"),
            FTI=data.get("FTI"),
            T4U=data.get("T4U"),
            test_date=data.get("test_date"),
        )
            
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Failed to parse Claude extraction output.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vision extraction error: {e}")
