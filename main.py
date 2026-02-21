import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Fix for PyTorch 2.6+ weights_only=True default - MUST be before any other imports
import torch
import torch.serialization
import functools

# Patch the internal _legacy_load function which is what torch.load ultimately calls
# if hasattr(torch.serialization, '_legacy_load'):
#     _original_legacy_load = torch.serialization._legacy_load
#     @functools.wraps(_original_legacy_load)
#     def _patched_legacy_load(*args, **kwargs):
#         if 'weights_only' not in kwargs:
#             kwargs['weights_only'] = False
#         return _original_legacy_load(*args, **kwargs)
#     torch.serialization._legacy_load = _patched_legacy_load

# Also patch torch.load directly
# _original_torch_load = torch.load
# @functools.wraps(_original_torch_load)
# def _patched_torch_load(*args, **kwargs):
#     if "weights_only" not in kwargs:
#         kwargs["weights_only"] = False
#     return _original_torch_load(*args, **kwargs)
# torch.load = _patched_torch_load


_original_torch_load = torch.load

def _trusted_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)

torch.load = _trusted_load

# Additionally, add common omegaconf classes to safe globals as a fallback
if hasattr(torch.serialization, 'add_safe_globals'):
    try:
        from omegaconf import DictConfig, ListConfig
        from omegaconf.base import ContainerMetadata, Metadata
        torch.serialization.add_safe_globals([DictConfig, ListConfig, ContainerMetadata, Metadata])
    except ImportError:
        pass

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from config import settings
from app.routes.qa_report import router as qa_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(settings.reports_dir, exist_ok=True)
    os.makedirs(settings.uploads_dir, exist_ok=True)
    yield


app = FastAPI(
    title="Call Center QA Agent",
    description="AI-powered quality assurance agent for call center recordings",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(qa_router, prefix="/api/v1")

@app.get("/api/v1/health")
async def health_check():
    return {"status": "healthy", "version": "1.0.0"}
