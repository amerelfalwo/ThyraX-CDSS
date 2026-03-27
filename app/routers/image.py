from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
import base64
import json
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from app.core.config import settings
from app.core.security import verify_internal_api_key
from app.schemas.image import ImageValidationResponse
from app.segmentation.model import process_full_pipeline

router = APIRouter(prefix="/image", tags=["Image Pipeline"], dependencies=[Depends(verify_internal_api_key)])

@router.post("/validate", response_model=ImageValidationResponse)
async def validate_ultrasound_image(file: UploadFile = File(...)):
    """Gatekeeper checking if the image is a valid ultrasound."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")
        
    if not settings.GOOGLE_API_KEY_VISION:
        # If no key, skip validation and assume true
        return ImageValidationResponse(is_ultrasound=True)
        
    try:
        image_bytes = await file.read()
        b64_img = base64.b64encode(image_bytes).decode("utf-8")
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            api_key=settings.GOOGLE_API_KEY_VISION
        )
        
        prompt = 'Analyze this image. Is it a valid medical ultrasound scan? Return ONLY a valid JSON object: {"is_ultrasound": true/false}'
        
        messages = [
            SystemMessage(content="You are an image validation gatekeeper. You reply with strictly valid JSON only."),
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
            
        val_data = json.loads(response_text)
        is_us = val_data.get("is_ultrasound", False)
        
        return ImageValidationResponse(is_ultrasound=is_us)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image validation error: {str(e)}")


@router.post("/predict")
async def predict_ultrasound_image(file: UploadFile = File(...)):
    """Runs the ONNX segmentation/classification models."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")
    try:
        image_bytes = await file.read()
        result = process_full_pipeline(image_bytes)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image processing error: {str(e)}")
