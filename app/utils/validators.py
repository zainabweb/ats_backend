import os

from fastapi import UploadFile

from app.config import get_settings
from app.core.exceptions import FileTooLarge, UnsupportedFileType

settings = get_settings()


def validate_upload(file: UploadFile, size_bytes: int) -> None:
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise UnsupportedFileType(f"'{ext or 'unknown'}' is not supported. Use PDF or DOCX.")

    max_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024
    if size_bytes > max_bytes:
        raise FileTooLarge(f"{file.filename} exceeds the {settings.MAX_FILE_SIZE_MB}MB limit.")
