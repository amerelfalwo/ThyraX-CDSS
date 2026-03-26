from pydantic import BaseModel, Field

class ThyroidInput(BaseModel):
    TT4: float = Field(..., ge=0, description="Total T4 الكلي - مستوى هرمون الغدة الدرقية (طبيعي ~ 4.5-12.0 µg/dL)")
    TSH: float = Field(..., ge=0, description="Thyroid Stimulating Hormone - هرمون منشط الغدة الدرقية (طبيعي ~ 0.4-4.0 µIU/mL)")
    T3: float = Field(..., ge=0, description="Triiodothyronine - الهرمون النشط للتمثيل الغذائي (طبيعي ~ 0.8-2.0 ng/mL)")
    FTI: float = Field(..., ge=0, description="Free Thyroxine Index مؤشر - يعكس كمية T4 الفعالة")
    T4U: float = Field(..., ge=0, description="T4 Uptake - قدرة البروتينات على ربط T4")
    age: int = Field(..., ge=0, le=120, description="عمر المريض بالسنوات")
    on_thyroxine: int = Field(..., ge=0, le=1, description="هل المريض بياخد دواء Thyroxine؟ 1=نعم, 0=لا")
    thyroid_surgery: int = Field(..., ge=0, le=1, description="هل المريض عمل جراحة للغدة الدرقية؟ 1=نعم, 0=لا")
    query_hyperthyroid: int = Field(..., ge=0, le=1, description="هل فيه شك في فرط نشاط الغدة؟ 1=نعم, 0=لا")