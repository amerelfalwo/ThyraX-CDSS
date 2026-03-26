"""
SQLAlchemy ORM models for the patient history database.
"""
import datetime
from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime, Text, ForeignKey, JSON
)
from sqlalchemy.orm import relationship
from app.core.database import Base


class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(200), nullable=False)
    age = Column(Integer, nullable=False)
    gender = Column(String(10), nullable=True)
    medical_record_number = Column(String(50), unique=True, index=True, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    visits = relationship("Visit", back_populates="patient", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Patient(id={self.id}, name='{self.name}', age={self.age})>"


class Visit(Base):
    __tablename__ = "visits"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)
    visit_date = Column(DateTime, default=datetime.datetime.utcnow)

    # ── Lab Results ──
    tsh = Column(Float, nullable=True)
    t3 = Column(Float, nullable=True)
    t4 = Column(Float, nullable=True)
    tt4 = Column(Float, nullable=True)
    fti = Column(Float, nullable=True)
    t4u = Column(Float, nullable=True)

    # ── Clinical Fields ──
    on_thyroxine = Column(Boolean, default=False)
    thyroid_surgery = Column(Boolean, default=False)
    query_hyperthyroid = Column(Boolean, default=False)
    nodule_present = Column(Boolean, default=False)

    # ── AI Assessment Results ──
    functional_status = Column(String(20), nullable=True)  # normal / hypothyroid / hyperthyroid
    clinical_recommendation = Column(Text, nullable=True)
    next_step = Column(String(50), nullable=True)  # radionuclide_scan / upload_ultrasound / routine_followup

    # ── Doctor Notes ──
    notes = Column(Text, nullable=True)

    patient = relationship("Patient", back_populates="visits")
    imaging_result = relationship("ImagingResult", back_populates="visit", uselist=False, cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Visit(id={self.id}, patient_id={self.patient_id}, status='{self.functional_status}')>"


class ImagingResult(Base):
    __tablename__ = "imaging_results"

    id = Column(Integer, primary_key=True, index=True)
    visit_id = Column(Integer, ForeignKey("visits.id"), nullable=False, unique=True)

    classification_label = Column(String(20), nullable=True)   # benign / malignant
    confidence = Column(Float, nullable=True)
    tirads_stage = Column(String(10), nullable=True)           # TR1-TR5
    bbox = Column(JSON, nullable=True)                         # [x_min, y_min, x_max, y_max]
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    visit = relationship("Visit", back_populates="imaging_result")

    def __repr__(self):
        return f"<ImagingResult(id={self.id}, label='{self.classification_label}', tirads='{self.tirads_stage}')>"
