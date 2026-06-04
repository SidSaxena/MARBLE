Function vocabulary:
In (intro), Lp (loop body), Ln (linear body), Ou (outro), Br (bridge), St (stinger).
Frequencies: Lp 1237, In 356, Ln 165, St 64, Ou 54, Br 41.

Section vocabulary:
A–G (thematic, 1370 total with A dominant), X (non-thematic placeholder for In/Ou/Br/St, 547 instances).

Companion dataset: metadata/pairs.csv — 3,304 section-pair rows with pre-computed similarity scores using the compound metric 0.4·chroma_KS + 0.3·duration + 0.1·register + 0.1·density (Moonbeam tokens, 480 TPQ, 120 BPM).
This is what enables "similarity-bucket conditioning" in the upstream paper.

MIDI input completely fails on minority classes: outro F1 = 0.000, stinger F1 = 0.000 at every single layer (0..12) and meanall. The MIDI representation never gives the probe any signal to discriminate these classes.
