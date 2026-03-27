from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload
import datetime

from app.core.database import get_db
from app.core.security import verify_internal_api_key
from app.models.patient import Patient, Visit, ImagingResult
from app.schemas.patient import PatientCreate, VisitCreate, DashboardResponse, PatientDemographics, VisitResponse, ImagingResponse

router = APIRouter(prefix="/patient", tags=["Patient Dashboard & Data"], dependencies=[Depends(verify_internal_api_key)])

@router.post("/create", response_model=PatientDemographics)
async def create_patient(req: PatientCreate, db: AsyncSession = Depends(get_db)):
    """Create a new patient profile."""
    try:
        patient = Patient(name=req.name, age=req.age, gender=req.gender)
        db.add(patient)
        await db.commit()
        await db.refresh(patient)
        return patient
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {e}")

@router.post("/{patient_id}/visit", response_model=VisitResponse)
async def save_visit(patient_id: int, req: VisitCreate, db: AsyncSession = Depends(get_db)):
    """Save the final confirmed visit data (labs + clinical assessment results) to the database."""
    patient = await db.get(Patient, patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail=f"Patient ID {patient_id} not found")

    try:
        visit_date = datetime.datetime.utcnow()
        if req.test_date:
            try:
                visit_date = datetime.datetime.strptime(req.test_date, "%Y-%m-%d")
            except ValueError:
                pass 

        visit = Visit(
            patient_id=patient.id,
            visit_date=visit_date,
            tsh=req.tsh,
            t3=req.t3,
            t4=req.t4,
            tt4=req.t4,
            notes=req.notes,
            functional_status=req.functional_status,
            clinical_recommendation=req.clinical_recommendation,
            next_step=req.next_step,
            imaging_result_id=req.imaging_result_id
        )
        db.add(visit)
        await db.commit()
        await db.refresh(visit)
        return visit
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {e}")

@router.get("/{patient_id}/dashboard", response_model=DashboardResponse)
async def get_patient_dashboard(patient_id: int, db: AsyncSession = Depends(get_db)):
    """
    Master endpoint for the frontend.
    Returns demographics, chronological visits, imaging results, and latest recommendation.
    """
    patient = await db.get(Patient, patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail=f"Patient ID {patient_id} not found")

    # Fetch Visits (ordered chronological for trending)
    stmt_visits = select(Visit).where(Visit.patient_id == patient_id).order_by(Visit.visit_date.asc())
    visits_result = await db.execute(stmt_visits)
    visits = visits_result.scalars().all()

    # Fetch Imaging Results (using joins or direct query)
    stmt_imaging = select(ImagingResult).where(ImagingResult.patient_id == patient_id).order_by(ImagingResult.processed_at.desc())
    imaging_result = await db.execute(stmt_imaging)
    images = imaging_result.scalars().all()

    latest_rec = None
    if visits:
        latest_rec = visits[-1].clinical_recommendation

    return DashboardResponse(
        patient=patient,
        visits=visits,
        imaging=images,
        latest_clinical_recommendation=latest_rec,
        status="success"
    )
