import hashlib
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from database.models import Document
from database.session import SessionLocal


def calculate_file_hash(file_path):
    """
    Calculate SHA-256 hash without loading the entire file into memory.
    """

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def save_document_metadata(
    file_path,
    pdf_url,
    source_name,
    title=None,
    abstract=None,
    authors=None,
    published_date=None,
    external_id=None,
    source_url=None,
    matched_keywords=None,
):
    """
    Save downloaded PDF metadata in PostgreSQL.

    Returns:
        Dictionary containing status and document ID.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Downloaded file not found: {file_path}"
        )

    if file_path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files can be registered.")

    file_hash = calculate_file_hash(file_path)
    file_size = file_path.stat().st_size

    with SessionLocal() as session:

        # Check whether this exact PDF already exists.
        existing_document = session.scalar(
            select(Document).where(
                Document.file_hash == file_hash
            )
        )

        if existing_document:
            return {
                "status": "duplicate",
                "document_id": str(existing_document.id),
            }

        document = Document(
            title=title,
            abstract=abstract,
            authors=authors,
            published_date=published_date,
            source_name=source_name,
            external_id=external_id,
            source_url=source_url,
            pdf_url=pdf_url,
            original_filename=file_path.name,
            file_path=str(file_path.resolve()),
            file_hash=file_hash,
            file_size=file_size,
            content_type="application/pdf",
            matched_keywords=matched_keywords,
        )

        try:
            session.add(document)
            session.commit()

            return {
                "status": "saved",
                "document_id": str(document.id),
            }

        except IntegrityError:
            session.rollback()

            # Handles a duplicate inserted concurrently.
            existing_document = session.scalar(
                select(Document).where(
                    Document.file_hash == file_hash
                )
            )

            if existing_document:
                return {
                    "status": "duplicate",
                    "document_id": str(existing_document.id),
                }

            raise