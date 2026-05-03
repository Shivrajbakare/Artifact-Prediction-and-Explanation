import torch
import PIL.Image as Image
import pandas as pd
from transformers import AutoModel
import torch.nn.functional as F
from utils import common_parser
from constants import artifact_index_dict

# --- HUGGINGFACE BUG PATCHES ---

import transformers.models.clip.modeling_clip
if not hasattr(transformers.models.clip.modeling_clip, 'clip_loss'):
    def dummy_clip_loss(*args, **kwargs):
        pass
    transformers.models.clip.modeling_clip.clip_loss = dummy_clip_loss

def load_clip():
    """Loads Jina CLIP with patches to survive dependency conflicts."""
    print("[INFO] Loading Jina-CLIP-v2 (Applying Meta Tensor Patch)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    _original_linspace = torch.linspace
    def _safe_linspace(*args, **kwargs):
        kwargs['device'] = torch.device('cpu') 
        return _original_linspace(*args, **kwargs)
    torch.linspace = _safe_linspace
    
    try:
        model = AutoModel.from_pretrained(
            'jinaai/jina-clip-v2',
            trust_remote_code=True,
            low_cpu_mem_usage=False,
            device_map=None
        )
    finally:
        torch.linspace = _original_linspace

    if device == "cuda":
        model = model.half().to(device)
    
    model.eval()
    return model

def run_clip(clip_model, image_path: str, object_class: int, dataframe: pd.DataFrame, clip_prediction_list_limit: int, threshold: int) -> list:
    image = Image.open(image_path).convert("RGB")
    
    with torch.no_grad():
        image_embeddings = clip_model.encode_image(image)
        artifacts_keys = list(artifact_index_dict.keys())
        text_embeddings = clip_model.encode_text(artifacts_keys)

        # Ensure embeddings are tensors and correctly shaped for math
        if not isinstance(image_embeddings, torch.Tensor):
            image_embeddings = torch.tensor(image_embeddings)
        if not isinstance(text_embeddings, torch.Tensor):
            text_embeddings = torch.tensor(text_embeddings)

        # Normalize the embeddings to compute cosine similarity correctly
        image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)
        text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)

        # SAFE DOT PRODUCT: Handles varying array shapes from Jina output
        # result is a 1D tensor/array of scores
        logits = torch.matmul(text_embeddings, image_embeddings.T).view(-1)
        
    results = logits.tolist()

    clip_total = set(
        common_parser(dataframe["CLIP Accurate"][object_class]) +
        common_parser(dataframe["CLIP Approximate"][object_class]) +
        common_parser(dataframe["CLIP Miscellaneous"][object_class])
    )

    clip_results_sorted = sorted(enumerate(results), key=lambda x: x[1], reverse=True)
    clip_prediction = []

    for idx, score in clip_results_sorted:
        artifact_id = artifact_index_dict[artifacts_keys[idx]]
        if artifact_id in clip_total:
            if score > threshold:
                clip_prediction.append(artifact_id)
            if len(clip_prediction) == clip_prediction_list_limit:
                break

    return clip_prediction
