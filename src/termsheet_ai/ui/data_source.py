"""Single place where the UI gets a PipelinePayload (mock file or the real pipeline)."""

import os
import tempfile
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from termsheet_ai.pipeline.runner import MockIngestor, TermSheetPipeline
from termsheet_ai.schemas.contracts import PipelinePayload

MOCK_PATH = Path(__file__).resolve().parent.parent / "data" / "mock_payload.json"


def load_mock_payload(path: Path = MOCK_PATH) -> PipelinePayload:
    """Load and validate the bundled demo result."""
    return PipelinePayload.model_validate_json(path.read_text(encoding="utf-8"))


def resolve_gemini_api_key() -> Optional[str]:
    """Resolve Gemini API key from environment, .env, or Streamlit secrets."""
    load_dotenv()
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        try:
            import streamlit as st

            key = st.secrets.get("GEMINI_API_KEY")
        except Exception:
            pass
    return key.strip() if key and key.strip() else None


def build_pipeline(use_cloud_llm: bool = False) -> TermSheetPipeline:
    """Construct the pipeline, wiring live Google Gemini Cloud LLM when requested and available."""
    if use_cloud_llm:
        api_key = resolve_gemini_api_key()
        if api_key:
            try:
                from termsheet_ai.explain.explainer import PlainEnglishExplainer
                from termsheet_ai.extract.extractor import FactExtractor
                from termsheet_ai.extract.llm_client import make_gemini_callable

                llm = make_gemini_callable(api_key=api_key)
                extractor = FactExtractor.with_gemini(api_key=api_key)
                explainer = PlainEnglishExplainer(llm_callable=llm)
                return TermSheetPipeline(extractor=extractor, explainer=explainer)
            except Exception:
                # If cloud client initialization fails, fall back to offline mode
                pass

    return TermSheetPipeline()


def uses_plaintext_ingestor(pipeline: TermSheetPipeline) -> bool:
    return isinstance(pipeline.ingestor, MockIngestor)


def run_pipeline_on_upload(
    filename: str,
    data: bytes,
    pipeline: Optional[TermSheetPipeline] = None,
    use_cloud_llm: bool = False,
) -> PipelinePayload:
    """Write the upload to a temp file (keeping its name) and run the pipeline on it."""
    safe_name = Path(filename).name or "upload.txt"
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / safe_name
        path.write_bytes(data)
        pipe = pipeline or build_pipeline(use_cloud_llm=use_cloud_llm)
        return pipe.run(path)
