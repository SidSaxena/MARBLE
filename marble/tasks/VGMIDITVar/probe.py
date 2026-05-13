# marble/tasks/VGMIDITVar/probe.py
"""
VGMIDI-TVar theme-and-variation retrieval probe.

Reuses ``CoverRetrievalTask`` from Covers80 — the MAP computation and
evaluation pipeline are identical.  The only difference is what counts
as a positive match: in Covers80 it's "same musical work, different
artist"; here it's "same theme group, different variation".

Evaluation procedure (test stage):
  1. Encode every rendered audio → mean-pool over time → L2-normalise.
  2. Mean-pool all clip embeddings that share the same audio file.
  3. Build an N×N cosine-similarity matrix (N = total rendered files).
  4. For each query, rank the other N-1 files by similarity.
  5. AP = average precision over the positions of the same-work hits.
  6. MAP = mean(AP) over all N queries.

A high MAP here means: when the encoder sees a variation, it ranks
other variations of the same theme highly — which is the property we
need for leitmotif detection in arrangement-rich settings.

Note
----
``CoverRetrievalTask`` requires a JSONL with ``work_id`` and ``audio_path``
fields and clips returning ``(waveform, work_id, audio_path)`` — the
VGMIDITVar datamodule provides exactly that.
"""

from marble.tasks.Covers80.probe import CoverRetrievalTask

# Re-export so configs can use marble.tasks.VGMIDITVar.probe.CoverRetrievalTask
__all__ = ["CoverRetrievalTask"]
