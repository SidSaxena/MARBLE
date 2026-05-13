# marble/encoders/CLaMP3/model.py

import os
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from typing import Union, List, Optional
from einops import rearrange
from transformers import BertConfig, AutoTokenizer

from marble.core.base_encoder import BaseEncoder
from marble.encoders.CLaMP3.clamp3_util import CLaMP3Model, M3Patchilizer
from marble.encoders.CLaMP3.mert_util import load_audio
from marble.encoders.MERT.model import MERT_v1_95M_Encoder, MERT_v1_95M_FeatureExtractor


# --- Constants and Configuration ---
DEFAULT_PRE_TRAINED_FOLDER = os.path.expanduser("~/.cache/clamp3/")
CLAMP3_CKPT_NAME = "weights_clamp3_saas_h_size_768_t_model_FacebookAI_xlm-roberta-base_t_length_128_a_size_768_a_layers_12_a_length_128_s_size_768_s_layers_12_p_size_64_p_length_512.pth"
CLAMP3_LINK = f"https://huggingface.co/sander-wood/clamp3/resolve/main/{CLAMP3_CKPT_NAME}"

class CLaMP3Config:
    """Configuration class for CLaMP3 model parameters."""
    # Text Model Config
    TEXT_MODEL_NAME = "FacebookAI/xlm-roberta-base"
    MAX_TEXT_LENGTH = 128

    # Audio Model Config
    AUDIO_HIDDEN_SIZE = 768
    AUDIO_NUM_LAYERS = 12
    MAX_AUDIO_LENGTH = 128
    
    # Symbolic (M3) Model Config
    M3_HIDDEN_SIZE = 768
    PATCH_SIZE = 64
    PATCH_LENGTH = 512
    PATCH_NUM_LAYERS = 12
    
    # CLaMP3 Model Config
    CLAMP3_HIDDEN_SIZE = 768
    CLAMP3_LOAD_M3 = True # Inferred from infer_test3, can be configurable
    LOGIT_SCALE = 1.0


# --- Helper Functions ---

def download_checkpoint_if_needed(folder: str, filename: str, url: str):
    """Downloads the checkpoint file if it doesn't exist (cross-platform)."""
    import urllib.request

    os.makedirs(folder, exist_ok=True)

    filepath = os.path.join(folder, filename)
    if not os.path.exists(filepath):
        print(f"Downloading pre-trained CLaMP3 model to {filepath} ...")
        tmp = filepath + ".part"
        try:
            def _progress(block, block_size, total):
                if total > 0:
                    pct = min(100, block * block_size * 100 // total)
                    print(f"\r  {pct}%", end="", flush=True)
            urllib.request.urlretrieve(url, tmp, reporthook=_progress)
            print()
            os.replace(tmp, filepath)
        except Exception:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise
        print(f"Download complete. File saved to: {filepath}")
    return filepath

def extract_mert_features_batch(
    waveforms: torch.Tensor,
    feature_extractor: 'MERT_v1_95M_Encoder',
    device: Union[str, torch.device]
) -> torch.Tensor:
    """
    Extracts per-layer, time-averaged features from a batch of audio waveforms
    using a sliding window approach.

    This function processes full-length chunks and the final partial chunk in separate
    inference calls to avoid padding the last chunk, ensuring feature fidelity.

    Args:
        waveforms (torch.Tensor): A batch of audio waveforms.
            - Shape: (batch_size, num_samples).
            - Assumption: All waveforms in the batch have the same length and have
              already been resampled to the model's target sampling rate.
        feature_extractor (MERT_v1_95M_Encoder): The MERT encoder instance.
        device (Union[str, torch.device]): The device ('cuda' or 'cpu') to run computations on.

    Returns:
        torch.Tensor: A tensor containing the extracted features. Returns None if no
                      valid chunks are found.
            - Shape: (batch_size, num_layers, num_chunks, feature_dim).
    """
    # --- Configuration ---
    target_sr = 24000
    sliding_window_size_in_sec = 5.0
    
    wavs = waveforms.to(device)
    
    window_size_samples = int(target_sr * sliding_window_size_in_sec)
    
    # --- Step 1: Chunking ---
    all_chunks = list(wavs.split(window_size_samples, dim=-1))
    
    min_len_samples = int(target_sr * 1)
    if all_chunks and all_chunks[-1].shape[-1] < min_len_samples:
        all_chunks = all_chunks[:-1]

    if not all_chunks:
        return None

    # --- Step 2: Separate full chunks from the last partial chunk ---
    last_chunk = None
    if all_chunks[-1].shape[-1] < window_size_samples:
        last_chunk = all_chunks[-1]
        full_chunks = all_chunks[:-1]
    else:
        full_chunks = all_chunks
    
    all_features = []

    # --- Step 3a: Process all full-sized chunks ---
    if full_chunks:
        full_chunks_tensor = torch.cat(full_chunks, dim=0)
        o = feature_extractor(full_chunks_tensor).hidden_states
        time_averaged_features = torch.stack(o).mean(-2)
        batch_size = waveforms.size(0)
        full_features = rearrange(time_averaged_features, 'l (c b) h -> b c l h', b=batch_size)
        all_features.append(full_features)

    # --- Step 3b: Process the final, shorter chunk ---
    if last_chunk is not None:
        o = feature_extractor(last_chunk).hidden_states
        time_averaged_features = torch.stack(o, dim=0).mean(dim=-2)
        last_feature = rearrange(time_averaged_features, 'l b h -> b 1 l h')
        all_features.append(last_feature)
        
    # --- Step 4: Concatenate results ---
    if not all_features:
        return None
    
    final_features = torch.cat(all_features, dim=1)
    final_features = rearrange(final_features, 'b c l h -> b l c h')
    
    return final_features


# --- Core Encoder Class ---
class CLaMP3_FeatureExtractor(MERT_v1_95M_FeatureExtractor):
    pass
    
class CLaMP3_Encoder(BaseEncoder):
    """
    CLaMP3 Encoder for generating joint text, audio, and symbolic embeddings.
    This implementation extracts MERT features on-the-fly and supports batching.
    """
    NAME = "CLaMP3"
    SAMPLING_RATE = 24000
    NUM_FEATURES = 768
    TOKEN_RATE = 1

    def __init__(self, train_mode: str = "freeze", pre_trained_folder: str = None,) -> None:
        super().__init__()

        self.config = CLaMP3Config()
        self.sample_rate = self.SAMPLING_RATE

        # 1. Download CLaMP3 checkpoint
        pre_trained_folder = pre_trained_folder or DEFAULT_PRE_TRAINED_FOLDER
        checkpoint_path = download_checkpoint_if_needed(
            pre_trained_folder, CLAMP3_CKPT_NAME, CLAMP3_LINK
        )
        
        # 2. Initialize MERT models for feature extraction
        self.mert_preprocessor = MERT_v1_95M_FeatureExtractor()
        self.mert_encoder = MERT_v1_95M_Encoder()
        
        # 3. Initialize CLaMP3 components
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.TEXT_MODEL_NAME)
        
        audio_config = BertConfig(
            vocab_size=1, hidden_size=self.config.AUDIO_HIDDEN_SIZE,
            num_hidden_layers=self.config.AUDIO_NUM_LAYERS,
            num_attention_heads=self.config.AUDIO_HIDDEN_SIZE // 64,
            intermediate_size=self.config.AUDIO_HIDDEN_SIZE * 4,
            max_position_embeddings=self.config.MAX_AUDIO_LENGTH
        )
        symbolic_config = BertConfig(
            vocab_size=1, hidden_size=self.config.M3_HIDDEN_SIZE,
            num_hidden_layers=self.config.PATCH_NUM_LAYERS,
            num_attention_heads=self.config.M3_HIDDEN_SIZE // 64,
            intermediate_size=self.config.M3_HIDDEN_SIZE * 4,
            max_position_embeddings=self.config.PATCH_LENGTH
        )
        self.model = CLaMP3Model(
            audio_config=audio_config, symbolic_config=symbolic_config,
            hidden_size=self.config.CLAMP3_HIDDEN_SIZE,
            load_m3=self.config.CLAMP3_LOAD_M3
        )
        
        # 4. Load pretrained weights
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        print(f"Loading CLaMP3 Checkpoint from Epoch {checkpoint['epoch']} with loss {checkpoint['min_eval_loss']}")
        self.model.load_state_dict(checkpoint['model'])

        # 5. Patchiliser for symbolic input (MIDI → M3 patches).  Used by both
        #    the symbolic forward path (CLaMP3_Symbolic_Encoder) and the
        #    cross-modal embed_symbolic helper.  Stateless and cheap to
        #    construct, so eager init is fine.
        from marble.encoders.CLaMP3.clamp3_util import M3Patchilizer
        self.patchilizer = M3Patchilizer()

        # 6. Set training mode
        if train_mode == "freeze":
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.mert_encoder.parameters():
                param.requires_grad = False
            self.model.eval()
            self.mert_encoder.eval()
        else:
            raise NotImplementedError(f"Train mode '{train_mode}' is not supported for CLaMP3_Encoder.")

    def _prepare_segments(self, data: torch.Tensor, max_len: int) -> List[torch.Tensor]:
        """Replicates the segmentation strategy from the original inference script."""
        if len(data) <= max_len:
            return [data]
        
        segments = list(data.split(max_len, dim=0))
        # Ensure the last segment is also max_len by overlapping
        if len(segments) > 1 and segments[-1].shape[0] < max_len:
             segments[-1] = data[-max_len:]
        return segments

    @torch.no_grad()
    def _get_embedding_from_segments(
        self,
        input_data: torch.Tensor,
        max_len: int,
        data_type: str,
        device: torch.device
    ) -> torch.Tensor:
        """Processes segmented data to get a final global embedding."""
        segments = self._prepare_segments(input_data, max_len)
        hidden_states_list = []

        for segment in segments:
            seg_len = segment.size(0)
            mask = torch.ones(seg_len, device=device)
            pad_len = max_len - seg_len
            mask = F.pad(mask, (0, pad_len), 'constant', 0)
            
            if data_type == 'text':
                segment = F.pad(segment, (0, pad_len), 'constant', self.tokenizer.pad_token_id)
                features = self.model.get_text_features(
                    text_inputs=segment.unsqueeze(0), text_masks=mask.unsqueeze(0), get_global=True
                )
            elif data_type == 'audio':
                pad_tensor = torch.zeros(pad_len, self.config.AUDIO_HIDDEN_SIZE, device=device)
                segment = torch.cat((segment, pad_tensor), 0)
                features = self.model.get_audio_features(
                    audio_inputs=segment.unsqueeze(0), audio_masks=mask.unsqueeze(0), get_global=True
                )
            else:
                raise ValueError(f"Unsupported data_type: {data_type}")
            
            hidden_states_list.append(features)

        # Weighted average of segment features to get the final embedding
        full_chunks = len(input_data) // max_len
        rem_len = len(input_data) % max_len
        weights = [max_len] * full_chunks
        if rem_len > 0:
            weights.append(rem_len)
        
        feature_weights = torch.tensor(weights, device=device).view(-1, 1)
        all_features = torch.cat(hidden_states_list, dim=0)
        final_embedding = (all_features * feature_weights).sum(dim=0) / feature_weights.sum()
            
        return final_embedding
        
    @torch.no_grad()
    def _get_layer_embeddings_from_segments(
        self,
        input_data: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Extract per-layer BERT embeddings for a single audio item, aggregated
        across segments with weighted average by segment length.

        Parameters
        ----------
        input_data : torch.Tensor
            Shape ``(S, AUDIO_HIDDEN_SIZE)`` — the full sequence of MERT chunk
            features for one item (with leading/trailing zero vectors prepended
            by the caller).
        device : torch.device

        Returns
        -------
        torch.Tensor
            Shape ``(13, H)`` — one averaged vector per BERT hidden state
            (embedding layer + 12 transformer layers).
        """
        segments = self._prepare_segments(input_data, self.config.MAX_AUDIO_LENGTH)
        per_layer_accum = None
        total_weight = 0.0

        for segment in segments:
            seg_len = segment.size(0)
            pad_len = self.config.MAX_AUDIO_LENGTH - seg_len
            mask = F.pad(
                torch.ones(seg_len, device=device),
                (0, pad_len),
                value=0.0,
            )
            pad_tensor = torch.zeros(pad_len, self.config.AUDIO_HIDDEN_SIZE, device=device)
            segment_padded = torch.cat((segment, pad_tensor), dim=0)

            bert_out = self.model.audio_model(
                inputs_embeds=segment_padded.unsqueeze(0).to(device),
                attention_mask=mask.unsqueeze(0).to(device),
                output_hidden_states=True,
            )

            # mask_3d: (1, seq_len, 1) for broadcasting over hidden dim
            mask_3d = mask.view(1, -1, 1)
            valid_tokens = mask_3d.sum(dim=1)  # (1, 1)

            # Stack all 13 hidden states and mean-pool over valid token positions
            # bert_out.hidden_states: tuple of 13 tensors, each (1, seq_len, H)
            layer_feats = torch.stack(
                [(hs * mask_3d).sum(dim=1) / valid_tokens for hs in bert_out.hidden_states],
                dim=0,
            ).squeeze(1)  # (13, H)

            weight = float(seg_len)
            per_layer_accum = (
                layer_feats * weight
                if per_layer_accum is None
                else per_layer_accum + layer_feats * weight
            )
            total_weight += weight

        return per_layer_accum / total_weight  # (13, H)

    @torch.no_grad()
    def forward(
        self,
        wavs: Optional[torch.Tensor] = None,
        texts: Optional[Union[str, List[str]]] = None
    ) -> tuple:
        """
        Generates embeddings for a batch of audio waveforms or texts.

        Audio path now returns a **tuple of 13 tensors**, one per BERT
        hidden state (embedding layer + 12 transformer layers), each of
        shape ``(B, 1, H)``.  This is compatible with
        ``marble.modules.transforms.LayerSelector``.

        Text path is unchanged and still returns a single ``(B, 1, H)`` tensor
        wrapped in a 1-element tuple.
        """
        # Determine device from input tensor or model parameters
        if wavs is not None:
            device = wavs.device
        elif texts is not None:
            device = next(self.model.parameters()).device
        else:
            raise ValueError("Either 'wavs' or 'texts' must be provided.")

        self.model.to(device)
        self.mert_encoder.to(device)

        # --- Audio Path ---
        if wavs is not None:
            batch_size = wavs.size(0)

            # 1. Preprocess audio batch for MERT
            processed_wavs_list = [
                self.mert_preprocessor({'input_features': wav, 'sampling_rate': self.SAMPLING_RATE})['input_features']
                for wav in wavs
            ]
            processed_wavs = torch.stack(processed_wavs_list, dim=0)

            # 2. Extract batched MERT features (B, L, C, H)
            mert_features = extract_mert_features_batch(processed_wavs, self.mert_encoder, device)

            # 3. Average over MERT layers → (B, C, H)
            mert_chunk_features = mert_features.mean(dim=1)

            # 4. For each item in the batch, collect all 13 BERT layer embeddings
            all_layer_embeddings = []
            for i in range(batch_size):
                item_features = mert_chunk_features[i].to(device)  # (C, H)

                # Add zero vectors at start/end to match original script
                zero_vec = torch.zeros((1, item_features.size(-1)), device=device)
                input_data = torch.cat((zero_vec, item_features, zero_vec), dim=0)

                layer_embeds = self._get_layer_embeddings_from_segments(input_data, device)
                all_layer_embeddings.append(layer_embeds)  # (13, H)

            stacked = torch.stack(all_layer_embeddings, dim=0)  # (B, 13, H)
            num_layers = stacked.shape[1]
            # Return tuple of 13 tensors, each (B, 1, H)
            return tuple(stacked[:, l, :].unsqueeze(1) for l in range(num_layers))

        # --- Text Path ---
        if texts is not None:
            if isinstance(texts, str): texts = [texts]
            
            embeddings = []
            for text_item in texts:
                # Prepare text input
                items = list(set(text_item.split("\n")))
                items = "\n".join(items).split("\n")
                items = [c for c in items if len(c) > 0]
                item_str = self.tokenizer.sep_token.join(items)
                input_data = self.tokenizer(item_str, return_tensors="pt")['input_ids'].squeeze(0).to(device)

                emb = self._get_embedding_from_segments(
                    input_data=input_data,
                    max_len=self.config.MAX_TEXT_LENGTH,
                    data_type='text',
                    device=device
                )
                embeddings.append(emb)

            output = torch.stack(embeddings, dim=0)
            return (output.unsqueeze(1),) # Return shape (B, 1, H)


# ──────────────────────────────────────────────────────────────────────────────
# Symbolic-music encoder (M3 path)
# ──────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────
# Cross-modal embedding API
# ──────────────────────────────────────────────────────────────────────────────
# CLaMP3 was trained with contrastive loss across three modality branches —
# audio, symbolic (M3), text — each with a projection head that lands the
# pooled embedding in a SHARED 768-dim space.  These helpers expose those
# shared-space embeddings so user code can:
#
#   • Compare an audio clip and a MIDI motif via cosine similarity
#   • Use a MIDI motif as a query against a full audio track
#   • Use a text description as a query against either modality
#
# The shape contract is uniform across all three: (B, H_shared) where
# H_shared = CLAMP3_HIDDEN_SIZE (768 for the public checkpoint).
# These are *separate* from the layer-probe forward() — they always use the
# final projection head and a global pooled embedding, never a per-layer
# hidden state.


class CLaMP3CrossModalMixin:
    """Mix-in adding ``embed_audio`` / ``embed_symbolic`` / ``embed_text``
    helpers to CLaMP3 encoder classes.

    Each method returns ``(B, H_shared)`` L2-normalised tensors that live in
    the same space — so ``embed_audio(wavs) @ embed_symbolic(patches).T``
    yields a valid cross-modal similarity matrix.
    """

    @torch.no_grad()
    def embed_audio(self, wavs: torch.Tensor) -> torch.Tensor:
        """Audio waveform → ``(B, H_shared)`` L2-normalised shared embeddings."""
        device = wavs.device
        self.model.to(device)
        self.mert_encoder.to(device)

        # Step 1: MERT preprocessing + feature extraction (same as forward())
        processed = torch.stack([
            self.mert_preprocessor(
                {'input_features': w, 'sampling_rate': self.SAMPLING_RATE}
            )['input_features']
            for w in wavs
        ], dim=0)
        mert_features = extract_mert_features_batch(processed, self.mert_encoder, device)
        mert_chunks   = mert_features.mean(dim=1)                  # (B, C, H_mert)

        # Step 2: pad with zero vectors at start/end (matches infer_test3)
        embeddings = []
        for b in range(mert_chunks.size(0)):
            zero_vec  = torch.zeros((1, mert_chunks.size(-1)), device=device)
            seq       = torch.cat((zero_vec, mert_chunks[b], zero_vec), dim=0)
            emb       = self._get_embedding_from_segments(
                input_data=seq,
                max_len=self.config.MAX_AUDIO_LENGTH,
                data_type='audio',
                device=device,
            )
            embeddings.append(emb)
        out = torch.stack(embeddings, dim=0)                       # (B, H_shared)
        return F.normalize(out, dim=-1)

    @torch.no_grad()
    def embed_symbolic(self, patches: torch.Tensor) -> torch.Tensor:
        """MIDI patches → ``(B, H_shared)`` L2-normalised shared embeddings.

        Parameters
        ----------
        patches : Tensor, shape ``(B, P, PATCH_SIZE)``
            Pre-tokenised patches as produced by
            ``M3Patchilizer`` (see the VGMIDITVar symbolic datamodule).
            Padding rows must carry ``pad_token_id == 0``.
        """
        device = patches.device
        self.model.to(device)

        embeddings = []
        for b in range(patches.size(0)):
            item = patches[b].to(device).long()
            # Strip fully-padded trailing rows
            non_pad = (item != self.patchilizer.pad_token_id).any(dim=-1)
            real_len = int(non_pad.sum().item())
            if real_len == 0:
                real_len = 1
            real = item[:real_len]

            # The CLaMP3Model.get_symbolic_features wrapper handles batching
            # but expects already-padded input; we segment and pad manually
            # so very long pieces are length-weighted correctly.
            max_len = self.config.PATCH_LENGTH
            segments = self._prepare_segments(real, max_len)
            feats = []
            weights = []
            for seg in segments:
                seg_len = seg.size(0)
                pad_len = max_len - seg_len
                mask = F.pad(
                    torch.ones(seg_len, device=device), (0, pad_len), value=0.0,
                )
                pad_tok = torch.full(
                    (pad_len, self.config.PATCH_SIZE),
                    self.patchilizer.pad_token_id,
                    dtype=torch.long,
                    device=device,
                )
                seg_padded = torch.cat((seg.to(device), pad_tok), dim=0)

                # get_symbolic_features(..., get_global=True) applies symbolic_proj
                emb = self.model.get_symbolic_features(
                    symbolic_inputs=seg_padded.unsqueeze(0),
                    symbolic_masks=mask.unsqueeze(0),
                    get_global=True,
                )                                             # (1, H_shared)
                feats.append(emb.squeeze(0))
                weights.append(float(seg_len))

            stacked = torch.stack(feats, dim=0)              # (S, H_shared)
            w = torch.tensor(weights, device=device).unsqueeze(-1)
            embeddings.append((stacked * w).sum(0) / w.sum())

        out = torch.stack(embeddings, dim=0)                  # (B, H_shared)
        return F.normalize(out, dim=-1)

    @torch.no_grad()
    def embed_text(self, texts) -> torch.Tensor:
        """Text strings → ``(B, H_shared)`` L2-normalised shared embeddings.

        Parameters
        ----------
        texts : str | list[str]
            One or more natural-language descriptions, e.g.
            ``["triumphant orchestral fanfare in C major"]``.
        """
        if isinstance(texts, str):
            texts = [texts]
        device = next(self.model.parameters()).device

        embeddings = []
        for text_item in texts:
            items = list(set(text_item.split("\n")))
            items = "\n".join(items).split("\n")
            items = [c for c in items if len(c) > 0]
            item_str = self.tokenizer.sep_token.join(items)
            input_data = self.tokenizer(item_str, return_tensors="pt")['input_ids']
            input_data = input_data.squeeze(0).to(device)
            emb = self._get_embedding_from_segments(
                input_data=input_data,
                max_len=self.config.MAX_TEXT_LENGTH,
                data_type='text',
                device=device,
            )
            embeddings.append(emb)
        out = torch.stack(embeddings, dim=0)                  # (B, H_shared)
        return F.normalize(out, dim=-1)


# Attach the cross-modal helpers to CLaMP3_Encoder directly.
#
# Why attribute assignment instead of multiple inheritance:
#   CLaMP3_Encoder already extends BaseEncoder, and changing its MRO to
#   include CLaMP3CrossModalMixin would require touching the parent
#   class declaration — invasive, and risky given BaseEncoder's own
#   subclass conventions used elsewhere in marble/encoders/.
#   Attaching the methods as class attributes here is equivalent
#   functionally: Python's descriptor protocol binds `self` to the
#   instance the same way it would for an inherited method.  Both
#   CLaMP3_Encoder and its CLaMP3_Symbolic_Encoder subclass pick them up.
#
# Do NOT "clean up" by moving these methods inline into CLaMP3_Encoder's
# class body — keeping them in the mixin keeps the cross-modal API
# discoverable as a single coherent group.
CLaMP3_Encoder.embed_audio    = CLaMP3CrossModalMixin.embed_audio
CLaMP3_Encoder.embed_symbolic = CLaMP3CrossModalMixin.embed_symbolic
CLaMP3_Encoder.embed_text     = CLaMP3CrossModalMixin.embed_text


class CLaMP3_Symbolic_Encoder(CLaMP3_Encoder):
    """CLaMP3 encoder that consumes pre-tokenised MIDI patches instead of audio.

    Shape contract
    --------------
    Input  : Tensor ``(B, P, PATCH_SIZE)`` of int64 patch token IDs.  Padding
             rows must be filled with the patchiliser's ``pad_token_id`` (0)
             so the attention mask can be inferred at forward time.
    Output : tuple of 13 tensors, each ``(B, 1, H=768)`` — one per BERT
             hidden state (embedding layer + 12 transformer layers).  Same
             contract as the audio path, so ``LayerSelector`` and
             ``TimeAvgPool`` work unchanged.

    Notes
    -----
    * MIDI → patches conversion is the dataset's responsibility (see
      ``marble.tasks.VGMIDITVar.datamodule.VGMIDITVarSymbolic*``).  This
      encoder is concerned only with mapping patches → embeddings.
    * Segments longer than ``CLaMP3Config.PATCH_LENGTH`` (default 512) are
      chunked and length-weighted, mirroring the audio path's behaviour.
    """

    # No __init__ override needed — CLaMP3_Encoder.__init__ now eagerly
    # instantiates self.patchilizer.  This subclass only adds a new
    # forward() that consumes pre-tokenised patches instead of audio.

    @torch.no_grad()
    def forward(self, patches: torch.Tensor) -> tuple:    # type: ignore[override]
        device = patches.device
        self.model.to(device)

        pad_id = self.patchilizer.pad_token_id
        batch_outputs = []

        for b in range(patches.size(0)):
            item = patches[b]                        # (P, PATCH_SIZE)

            # Drop fully-padded trailing rows so we only feed real patches.
            non_pad_mask = (item != pad_id).any(dim=-1)   # (P,)
            real_len = int(non_pad_mask.sum().item())
            if real_len == 0:
                # All padding → return zeros for every layer.  Skips BERT
                # entirely so downstream cosine similarities can't see
                # phantom matches from an all-pad input that happens to
                # produce a non-zero embedding in BERT.
                H = self.config.M3_HIDDEN_SIZE
                n_layers = self.config.PATCH_NUM_LAYERS + 1
                layer_embeds = torch.zeros(n_layers, H, device=device)
            else:
                real = item[:real_len].to(device)            # (real_len, PATCH_SIZE)
                layer_embeds = self._get_symbolic_layer_embeddings(real, device)
            batch_outputs.append(layer_embeds)             # (13, H)

        stacked = torch.stack(batch_outputs, dim=0)        # (B, 13, H)
        return tuple(stacked[:, l, :].unsqueeze(1) for l in range(stacked.size(1)))

    # ── helpers ──────────────────────────────────────────────────────────────

    def _get_symbolic_layer_embeddings(
        self,
        patches: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Return per-layer pooled embeddings for one item's symbolic patches.

        Mirrors ``_get_layer_embeddings_from_segments`` (the audio helper)
        but uses the symbolic_model's *inner* BertModel directly so we can
        request ``output_hidden_states=True``.  We replicate the
        one-hot → linear patch embedding step that ``M3PatchEncoder.forward``
        normally performs before BERT.
        """
        sym = self.model.symbolic_model           # M3PatchEncoder
        bert = sym.base                            # HF BertModel
        patch_embed = sym.patch_embedding          # Linear(PATCH_SIZE*128 → H)
        H = self.config.M3_HIDDEN_SIZE

        max_len = self.config.PATCH_LENGTH
        segments = self._prepare_segments(patches, max_len)

        per_layer_accum = None
        total_weight = 0.0

        for segment in segments:
            seg_len = segment.size(0)
            pad_len = max_len - seg_len

            # Token-level mask: 1 for real patches, 0 for padding.
            mask = F.pad(
                torch.ones(seg_len, device=device),
                (0, pad_len),
                value=0.0,
            )
            pad_token = torch.full(
                (pad_len, self.config.PATCH_SIZE),
                self.patchilizer.pad_token_id,
                dtype=segment.dtype,
                device=device,
            )
            seg_padded = torch.cat((segment.to(device), pad_token), dim=0).long()
            # → (P, PATCH_SIZE) of int patch token IDs in [0, 128)

            # One-hot → linear projection, same as M3PatchEncoder.forward.
            oh   = F.one_hot(seg_padded, num_classes=128).float()  # (P, PS, 128)
            flat = oh.reshape(seg_padded.size(0), -1)              # (P, PS*128)
            emb  = patch_embed(flat).unsqueeze(0)                   # (1, P, H)

            bert_out = bert(
                inputs_embeds=emb,
                attention_mask=mask.unsqueeze(0),
                output_hidden_states=True,
            )
            # bert_out.hidden_states: tuple of 13 × (1, P, H)
            # Weighted-mean each layer over the real (mask==1) positions.
            mask_b = mask.unsqueeze(0).unsqueeze(-1)       # (1, P, 1)
            denom = mask_b.sum().clamp_min(1.0)
            pooled = [
                (hs * mask_b).sum(dim=1).squeeze(0) / denom
                for hs in bert_out.hidden_states
            ]
            stacked = torch.stack(pooled, dim=0)            # (13, H)

            weight = float(seg_len)
            if per_layer_accum is None:
                per_layer_accum = stacked * weight
            else:
                per_layer_accum = per_layer_accum + stacked * weight
            total_weight += weight

        return per_layer_accum / total_weight              # (13, H)


if __name__ == '__main__':
    # --- GTZAN Demo for Verification ---
    print("--- Running GTZAN Demo with CLaMP3_Encoder ---")
    
    # 1. Setup paths and audio file
    demo_dir = "tests"
    audio_path = os.path.join(demo_dir, "blues.00000.wav")
    if not os.path.exists(audio_path):
        print(f"'{audio_path}' not found. Please ensure it exists.")
        # Create a dummy file if it doesn't exist, similar to infer_test3.py
        if not os.path.exists(demo_dir): os.makedirs(demo_dir)
        import wave, struct
        sample_rate = 22050.0; duration = 30; n_samples = int(duration * sample_rate)
        with wave.open(audio_path, 'w') as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sample_rate)
            for _ in range(n_samples): wf.writeframes(struct.pack('<h', 0))
        print(f"Created a dummy silent WAV file at '{audio_path}'.")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 2. Instantiate the encoder
    # This will automatically handle model downloading and setup
    encoder = CLaMP3_Encoder().to(device)

    # 3. Load and prepare audio data (as a batch of size 2 for testing)
    waveform = load_audio(audio_path, target_sr=encoder.SAMPLING_RATE, is_mono=True)
    wavs_batch = torch.stack([waveform, waveform], dim=0).to(device)
    print(f"Audio loaded and prepared as a batch of shape: {wavs_batch.shape}")

    # 4. Get audio features (embedding)
    print("\nExtracting audio features...")
    # The forward pass handles all intermediate steps (MERT extraction, CLaMP projection)
    audio_output = encoder(wavs=wavs_batch)
    print(f"Audio features extracted with shape: {audio_output[0].shape}")  # Should be (2, 1, 768)
    
    # We only need the embedding for the first item in the batch for this demo
    audio_feature = audio_output[0][0] # Shape: (1, 768)
    print("Audio feature extracted.")

    # 5. Calculate similarities with GTZAN genres
    gtzan_genres = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
    similarities = {}

    print("\nCalculating similarities with GTZAN genres...")
    audio_feature_norm = audio_feature / audio_feature.norm(dim=-1, keepdim=True)
    
    for genre in tqdm(gtzan_genres, desc="Genres"):
        # Get text embedding for each genre
        text_output = encoder(texts=genre)
        print(f"Text feature for genre '{genre}': {text_output[0].shape}")
        text_feature = text_output[0].squeeze(1) # Shape: (1, 768)
        text_feature_norm = text_feature / text_feature.norm(dim=-1, keepdim=True)
        
        similarity = (audio_feature_norm * text_feature_norm).sum().item()
        similarities[genre] = similarity

    # 6. Print results for verification
    print(f"\n--- Similarity Results for '{os.path.basename(audio_path)}' ---")
    sorted_genres = sorted(similarities.items(), key=lambda item: item[1], reverse=True)
    for genre, score in sorted_genres:
        print(f"{genre:<10}: {score:.4f}")

    print("\n✅ Verification complete. Compare these scores with the original script's output.")