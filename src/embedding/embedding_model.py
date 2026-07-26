from typing import List, Union, Optional
import os
import logging
import numpy as np


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
DEFAULT_MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", DEFAULT_MODELS_DIR)
os.environ.setdefault("TRANSFORMERS_CACHE", DEFAULT_MODELS_DIR)
# =======================================================================

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None
    raise ImportError("Cần cài đặt: pip install sentence-transformers")

_default_model = None

def _detect_device(preferred: Optional[str] = None) -> str:
    if preferred and preferred.lower() == "cpu":
        return "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"

class EmbeddingModel:
    def __init__(self, model_name: str = "all-mpnet-base-v2", device: Optional[str] = None):
        self.model_name = model_name
        self.device = _detect_device(device)
        print(f"Loading model '{model_name}' on {self.device}...")
        
        self.model = SentenceTransformer(model_name, device=self.device)

        # Lấy dimension
        try:
            self.dim = self.model.get_sentence_embedding_dimension()
        except Exception:
            self.dim = 768

    def get_embedding(self, text: Union[str, List[str]]) -> np.ndarray:
        if isinstance(text, str):
            text = [text]
            is_single = True
        else:
            is_single = False

        emb = self.model.encode(text, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        emb = np.asarray(emb, dtype=np.float32)
        
        return emb[0] if is_single else emb

    def encode_batch(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        return self.model.encode(
            texts, 
            batch_size=batch_size, 
            convert_to_numpy=True, 
            normalize_embeddings=True, 
            show_progress_bar=True
        ).astype(np.float32)

def get_default_model(model_name: str = "all-mpnet-base-v2", device: Optional[str] = None) -> EmbeddingModel:
    global _default_model
    if _default_model is None or _default_model.model_name != model_name:
        _default_model = EmbeddingModel(model_name=model_name, device=device)
    return _default_model