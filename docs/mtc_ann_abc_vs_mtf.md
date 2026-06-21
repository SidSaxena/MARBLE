# MTC-ANN TuneFamily — ABC vs MTF (per-layer MAP, identical pool)

Identical occurrence pool (same melodies, same work_id / relevance). Zero-shot CLaMP3-symbolic, no CV folds. Δ = ABC − MTF (positive ⇒ notation-preserving ABC helps).

## raw test/map

| layer | ABC | MTF | Δ (ABC−MTF) |
|------:|----:|----:|------------:|
| 0 | 0.5514 | 0.3378 | 0.2136 |
| 1 | 0.5880 | 0.3493 | 0.2387 |
| 2 | 0.6014 | 0.3649 | 0.2366 |
| 3 | 0.6160 | 0.3913 | 0.2246 |
| 4 | 0.6272 | 0.4141 | 0.2131 |
| 5 | 0.6334 | 0.4400 | 0.1934 |
| 6 | 0.6382 | 0.4599 | 0.1783 |
| 7 ⭐ABC | 0.6443 | 0.4803 | 0.1640 |
| 8 | 0.6359 | 0.4751 | 0.1608 |
| 9 | 0.6442 | 0.4748 | 0.1694 |
| 10 | 0.6324 | 0.4529 | 0.1795 |
| 11 ⭐MTF | 0.6271 | 0.4921 | 0.1350 |
| 12 | 0.5729 | 0.4562 | 0.1167 |
| **meanall** | 0.6400 | 0.4997 | 0.1402 |

## centered test/map

| layer | ABC | MTF | Δ (ABC−MTF) |
|------:|----:|----:|------------:|
| 0 | 0.5649 | 0.3344 | 0.2305 |
| 1 | 0.6092 | 0.3406 | 0.2686 |
| 2 | 0.6269 | 0.3559 | 0.2710 |
| 3 | 0.6462 | 0.3844 | 0.2617 |
| 4 | 0.6553 | 0.4112 | 0.2441 |
| 5 | 0.6596 | 0.4415 | 0.2181 |
| 6 | 0.6671 | 0.4682 | 0.1989 |
| 7 | 0.6776 | 0.4933 | 0.1844 |
| 8 | 0.6785 | 0.4869 | 0.1916 |
| 9 | 0.6897 | 0.4917 | 0.1980 |
| 10 | 0.6772 | 0.4741 | 0.2031 |
| 11 | 0.6645 | 0.5110 | 0.1535 |
| 12 | 0.6051 | 0.4736 | 0.1316 |

## Verdict (raw MAP)

- **ABC best layer = 7** (MAP 0.6443)
- **MTF best layer = 11** (MAP 0.4921)
- peak Δ (ABC_best − MTF_best) = 0.1523
- same-layer Δ at ABC's peak (L7): 0.1640

## recall@K at each arm's best layer (raw)

| K | ABC@L7 | MTF@L11 |
|---:|----:|----:|
| 1 | 0.0686 | 0.0615 |
| 5 | 0.2990 | 0.2472 |
| 10 | 0.5022 | 0.3833 |
| 50 | 0.7547 | 0.6576 |

# MTC-ANN Motif — ABC vs MTF (per-layer MAP, identical pool)

Identical occurrence pool (same melodies, same work_id / relevance). Zero-shot CLaMP3-symbolic, no CV folds. Δ = ABC − MTF (positive ⇒ notation-preserving ABC helps).

## raw test/map

| layer | ABC | MTF | Δ (ABC−MTF) |
|------:|----:|----:|------------:|
| 0 | 0.5354 | 0.4916 | 0.0439 |
| 1 | 0.5698 | 0.5171 | 0.0527 |
| 2 | 0.5749 | 0.5214 | 0.0534 |
| 3 | 0.5866 | 0.5223 | 0.0643 |
| 4 | 0.5831 | 0.5331 | 0.0500 |
| 5 | 0.5816 | 0.5390 | 0.0426 |
| 6 | 0.5836 | 0.5447 | 0.0389 |
| 7 | 0.5893 | 0.5485 | 0.0408 |
| 8 | 0.5974 | 0.5517 | 0.0457 |
| 9 ⭐ABC ⭐MTF | 0.5981 | 0.5714 | 0.0267 |
| 10 | 0.5855 | 0.5693 | 0.0162 |
| 11 | 0.5517 | 0.5470 | 0.0047 |
| 12 | 0.5349 | 0.5469 | -0.0120 |
| **meanall** | 0.5837 | 0.5706 | 0.0131 |

## centered test/map

| layer | ABC | MTF | Δ (ABC−MTF) |
|------:|----:|----:|------------:|
| 0 | 0.5648 | 0.4830 | 0.0817 |
| 1 | 0.6017 | 0.5118 | 0.0900 |
| 2 | 0.6062 | 0.5157 | 0.0904 |
| 3 | 0.6215 | 0.5167 | 0.1048 |
| 4 | 0.6049 | 0.5306 | 0.0743 |
| 5 | 0.6044 | 0.5345 | 0.0698 |
| 6 | 0.6047 | 0.5453 | 0.0595 |
| 7 | 0.6142 | 0.5485 | 0.0658 |
| 8 | 0.6254 | 0.5598 | 0.0656 |
| 9 | 0.6276 | 0.5783 | 0.0493 |
| 10 | 0.6178 | 0.5780 | 0.0399 |
| 11 | 0.5818 | 0.5579 | 0.0239 |
| 12 | 0.5624 | 0.5602 | 0.0022 |

## Verdict (raw MAP)

- **ABC best layer = 9** (MAP 0.5981)
- **MTF best layer = 9** (MAP 0.5714)
- peak Δ (ABC_best − MTF_best) = 0.0267
- same-layer Δ at ABC's peak (L9): 0.0267

## recall@K at each arm's best layer (raw)

| K | ABC@L9 | MTF@L9 |
|---:|----:|----:|
| 1 | 0.0453 | 0.0459 |
| 5 | 0.2007 | 0.1926 |
| 10 | 0.3461 | 0.3284 |
| 50 | 0.6790 | 0.6762 |

## Confound-free Motif metrics (per layer)

- `test/map` = realistic full-gallery MAP (CONFOUNDED by tune-family similarity for the Motif task).
- **`test/map_samefamily` = the CONFOUND-FREE discriminative metric** (gallery restricted to the query's own tune family → measures motif identity, not tune-family identity). Read the verdict off THIS column.
- `test/map_len_le3` / `test/map_len_gt3` = MAP for short (≤3-note) vs long (>3-note) motif queries.

| layer | arm | full | same-fam | ≤3 | >3 |
|------:|:----|----:|----:|----:|----:|
| 0 | ABC | 0.5354 | 0.7759 | 0.5121 | 0.5576 |
| 0 | MTF | 0.4916 | 0.7685 | 0.4676 | 0.5145 |
| 1 | ABC | 0.5698 | 0.7820 | 0.5440 | 0.5953 |
| 1 | MTF | 0.5171 | 0.7727 | 0.4920 | 0.5448 |
| 2 | ABC | 0.5749 | 0.7751 | 0.5405 | 0.6093 |
| 2 | MTF | 0.5214 | 0.7711 | 0.4904 | 0.5525 |
| 3 | ABC | 0.5866 | 0.7864 | 0.5456 | 0.6287 |
| 3 | MTF | 0.5223 | 0.7685 | 0.4900 | 0.5561 |
| 4 | ABC | 0.5831 | 0.8100 | 0.5408 | 0.6253 |
| 4 | MTF | 0.5331 | 0.7805 | 0.4917 | 0.5751 |
| 5 | ABC | 0.5816 | 0.8036 | 0.5336 | 0.6297 |
| 5 | MTF | 0.5390 | 0.7848 | 0.4890 | 0.5912 |
| 6 | ABC | 0.5836 | 0.8040 | 0.5376 | 0.6321 |
| 6 | MTF | 0.5447 | 0.7913 | 0.4858 | 0.6077 |
| 7 | ABC | 0.5893 | 0.8104 | 0.5390 | 0.6423 |
| 7 | MTF | 0.5485 | 0.7973 | 0.4933 | 0.6059 |
| 8 | ABC | 0.5974 | 0.8130 | 0.5527 | 0.6436 |
| 8 | MTF | 0.5517 | 0.8017 | 0.5000 | 0.6049 |
| 9 | ABC | 0.5981 | 0.8103 | 0.5553 | 0.6436 |
| 9 | MTF | 0.5714 | 0.8108 | 0.5220 | 0.6224 |
| 10 | ABC | 0.5855 | 0.7910 | 0.5453 | 0.6258 |
| 10 | MTF | 0.5693 | 0.8017 | 0.5220 | 0.6199 |
| 11 | ABC | 0.5517 | 0.7537 | 0.5165 | 0.5870 |
| 11 | MTF | 0.5470 | 0.7393 | 0.4981 | 0.5971 |
| 12 | ABC | 0.5349 | 0.7389 | 0.5018 | 0.5669 |
| 12 | MTF | 0.5469 | 0.7485 | 0.4974 | 0.5982 |

### Verdict on the CONFOUND-FREE metric (map_samefamily)

- **ABC best layer = 8** (map_samefamily 0.8130)
- **MTF best layer = 9** (map_samefamily 0.8108)
- peak Δ (ABC_best − MTF_best) = 0.0022
