# marble/tasks/VGMLoopStructure/probe.py
#
# ProbeAudioTask is fully generic (wires encoder / decoders / losses / metrics
# from config; contains no HookTheoryStructure-specific logic).  Re-export it
# rather than duplicating it so that any future fixes in the upstream class
# propagate automatically.
from marble.tasks.HookTheoryStructure.probe import ProbeAudioTask  # noqa: F401

__all__ = ["ProbeAudioTask"]
