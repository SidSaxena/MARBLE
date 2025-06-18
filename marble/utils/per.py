#!/usr/bin/env python3
"""
Compute Phoneme Error Rate (PER) between reference and hypothesis texts.

This script provides a robust PER calculation with a clear fallback strategy for
phoneme conversion across multiple languages.

Prerequisites:
apt-get update && apt-get install espeak-ng espeak-ng-data -y

pip install phonemizer unidecode editdistance langid epitran pyopenjtalk pycantonese pypinyin

python3 -c "import epitran.download; epitran.download.cedict()"
"""

import re
import warnings
import sys
from typing import List, Tuple

# Attempt to import all libraries and provide helpful error messages
try:
    from unidecode import unidecode
    from phonemizer import phonemize
    from phonemizer.separator import Separator
    import editdistance
    import langid
    import epitran
    import pyopenjtalk
    import pycantonese
    import pypinyin
    from pypinyin.style._utils import get_finals, get_initials
except ImportError as e:
    print(f"Error: A required library is not installed: {e.name}")
    print("Please install all required libraries from the script's docstring.")
    sys.exit(1)


# Mapping from langid codes to phonemizer/internal codes
LANGUAGE_MAPPING = {
    'en': 'en-us', # Specify US English for phonemizer
    'fr': 'fr-fr',
    'es': 'es',
    'de': 'de',
    'zh': 'zh',   # Mandarin
    'ja': 'ja',
    'ru': 'ru',
    'yue': 'yue', # Cantonese (Custom code for our script)
}

# Create a list of languages that the `langid` library actually supports.
LANGID_DETECTABLE_LANGS = [lang for lang in LANGUAGE_MAPPING.keys() if lang != 'yue']
langid.set_languages(LANGID_DETECTABLE_LANGS)


# Epitran language codes for fallback (Chinese 'zh' is now handled by pypinyin)
EPITRAN_MAPPING = {
    'ko': 'kor-Hang',      # Korean (Hangul) -> IPA
    'ru': 'rus-Cyrl',      # Russian (Cyrillic) -> IPA
}


def normalize_text(text: str, language_code: str) -> str:
    """Apply language-appropriate text normalization."""
    text = text.lower().strip()
    if language_code in {'en-us', 'fr-fr', 'es', 'de'}:
        text = unidecode(text)
    punctuation_to_remove = ".,?!:;\"()[]-"
    text = re.sub(f'[{re.escape(punctuation_to_remove)}]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def text_to_phonemes(
    text: str,
    language_code: str,
    verbose: bool = False,
    pinyin_with_tone: bool = True
) -> List[str]:
    """
    Convert text to a list of phonemes with a clear fallback strategy.
    For Chinese, allows toggling tones in Pinyin output.
    """
    if not text:
        return []

    # --- 1. Specialized Libraries ---

    # Handle Mandarin Chinese with pypinyin for accurate Pinyin syllables
    if language_code == 'zh':
        try:
            # Choose the pypinyin style based on the pinyin_with_tone flag
            style = pypinyin.Style.TONE3 if pinyin_with_tone else pypinyin.Style.NORMAL
            pinyin_list = pypinyin.pinyin(text, style=style)
            # Flatten the list of lists and remove any empty strings
            phonemes = [item for sublist in pinyin_list for item in sublist if item.strip()]
            if phonemes:
                return phonemes
        except Exception as e:
            warnings.warn(f"pypinyin failed for 'zh', falling back. Error: {e}")

    if language_code == 'ja':
        try:
            phonemes = pyopenjtalk.g2p(text, kana=False).strip().split()
            return [p for p in phonemes if p.lower() != 'pau']
        except Exception as e:
            warnings.warn(f"pyopenjtalk failed for 'ja', falling back. Error: {e}")

    # if language_code == 'yue':
    #     try:
    #         text_no_space = text.replace(' ', '')
    #         jp_tuples = pycantonese.characters_to_jyutping(text_no_space)
    #         phonemes = [jp[1] for jp in jp_tuples if jp[1]]
    #         if phonemes:
    #             return phonemes
    #     except Exception as e:
    #         warnings.warn(
    #             f"pycantonese failed for 'yue'. Is 'hkcancor' data installed? Error: {e}"
    #         )

    # --- 2. Primary Method: phonemizer (espeak-ng) ---
    try:
        phon_str = phonemize(
            text,
            language=language_code,
            backend='espeak',
            separator=Separator(phone=' ', word='|'),
            strip=True,
            preserve_punctuation=False,
            with_stress=True,
            njobs=1
        )
        phonemes = [p for p in phon_str.replace('|', ' ').split() if p]
        if phonemes:
            return phonemes
    except RuntimeError as e:
        if verbose:
            warnings.warn(f"phonemizer (espeak) does not support '{language_code}', trying epitran. Error: {e}")
    except Exception as e:
        if verbose:
            warnings.warn(f"phonemizer (espeak) failed unexpectedly, trying epitran. Error: {e}")

    # --- 3. Secondary Fallback: epitran ---
    epitran_code = EPITRAN_MAPPING.get(language_code)
    if epitran_code:
        try:
            epi = epitran.Epitran(epitran_code)
            phon_str = epi.transliterate(text)
            phonemes = [p for p in phon_str.split() if p]
            if phonemes:
                return phonemes
        except Exception as e:
            warnings.warn(f"Epitran fallback failed for '{language_code}'. Error: {e}")
            if "cedict.script" in str(e):
                print("Hint: You may need to download epitran data. See script docstring for the command.")

    # --- 4. Final Fallback: Character-level ---
    warnings.warn(
        f"All phoneme backends failed for language '{language_code}'. "
        f"Falling back to character-level representation. "
        f"The result will be Character Error Rate (CER), not PER."
    )
    if language_code in ['zh', 'ja', 'yue']:
        return list(text.replace(' ', ''))
    return list(text.replace(' ', ''))


def phoneme_error_rate(
    ref_text: str,
    hyp_text: str,
    language_code: str,
    verbose: bool = False,
    pinyin_with_tone: bool = True
) -> float:
    """
    Calculate Phoneme Error Rate (PER).

    Args:
        ref_text (str): The reference text.
        hyp_text (str): The hypothesis text.
        language_code (str): The language code (e.g., 'en-us', 'zh').
        verbose (bool): If True, prints detailed intermediate steps.
        pinyin_with_tone (bool): For Chinese ('zh'), specifies whether to
                                 include tones in Pinyin conversion.
                                 Defaults to True.
    """
    ref_norm = normalize_text(ref_text, language_code)
    hyp_norm = normalize_text(hyp_text, language_code)

    if verbose:
        print(f"Normalized REF: '{ref_norm}'")
        print(f"Normalized HYP: '{hyp_norm}'")

    ref_phs = text_to_phonemes(ref_norm, language_code, verbose, pinyin_with_tone)
    hyp_phs = text_to_phonemes(hyp_norm, language_code, verbose, pinyin_with_tone)

    if verbose:
        print(f"Reference phonemes: {' '.join(ref_phs)}")
        print(f"Hypothesis phonemes: {' '.join(hyp_phs)}")

    distance = editdistance.eval(ref_phs, hyp_phs)

    if not ref_phs:
        return 0.0 if not hyp_phs else 1.0

    return distance / len(ref_phs)


if __name__ == "__main__":
    test_cases: List[Tuple[str, str, str]] = [
        # (Reference, Hypothesis, Language Code)
        ("Cats are sitting on the mats.", "Cats are siting on the mat.", 'en-us'),
        ("Hello world!", "Hello world!", 'en-us'),
        ("Phoneme error rate.", "Phoneme eror rate.", 'en-us'),
        ("One two three four", "One two three", 'en-us'),
        ("Sphinx of black quartz, judge my vow.", "Sphinx of black quartz judge my vow", 'en-us'),
        ("", "", 'en-us'),
        ("Non empty ref", "", 'en-us'),
        ("你好 世界", "你好 思杰", 'zh'),  # Mandarin: 'shi jie' vs 'si jie'
        ("唔該 世界", "你好 世界", 'yue'), # Cantonese vs Mandarin
        ("こんにちは 世界", "こんいちは せかい", 'ja'),
        ("Bonjour le monde", "Bonjor le monde", 'fr-fr'),
        ("Привет мир", "Привет мир", 'ru'),
        ("Hola mundo", "Hola mundo!", 'es')
    ]

    for idx, (r, h, lang_code) in enumerate(test_cases, 1):
        print(f"--- Test Case {idx} ---")
        print(f"REF: {r}\nHYP: {h}")
        print(f"Language specified: {lang_code}")

        if not r and not h:
            print("Both reference and hypothesis are empty. PER is 0.00%\n")
            continue

        # Special handling to demonstrate the pinyin tone switch
        if lang_code == 'zh':
            print("\n--- Testing Chinese with Tones (Default) ---")
            try:
                per_val = phoneme_error_rate(r, h, language_code=lang_code, verbose=True, pinyin_with_tone=True)
                print(f"PER (with tones): {per_val:.2%}\n")
            except Exception as e:
                print(f"Error computing PER: {e}\n")
            
            print("--- Testing Chinese without Tones ---")
            try:
                per_val = phoneme_error_rate(r, h, language_code=lang_code, verbose=True, pinyin_with_tone=False)
                print(f"PER (without tones): {per_val:.2%}\n")
            except Exception as e:
                print(f"Error computing PER: {e}\n")
        else:
            try:
                per_val = phoneme_error_rate(r, h, language_code=lang_code, verbose=True)
                print(f"PER: {per_val:.2%}\n")
            except Exception as e:
                print(f"Error computing PER: {e}\n")