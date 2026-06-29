import eng_to_ipa as ipa


# 1. Define a hardcoded, deterministic dictionary mapping individual IPA phonemes to their phonological binary features.
# This serves as our linguistic knowledge base.
IPA_FEATURES_TABLE = {
    # =========================================================================
    # A. CONSONANTS - 24 Phonemes
    # =========================================================================

    # Plosives (Stop Consonants): Formed by completely obstructing the airflow and then releasing it
    'p': {'voiced': 0, 'fricative': 0, 'plosive': 1, 'nasal': 0, 'long_vowel': 0},  # e.g., "pin" (Unvoiced)
    'b': {'voiced': 1, 'fricative': 0, 'plosive': 1, 'nasal': 0, 'long_vowel': 0},  # e.g., "bat" (Voiced)
    't': {'voiced': 0, 'fricative': 0, 'plosive': 1, 'nasal': 0, 'long_vowel': 0},  # e.g., "toe" (Unvoiced)
    'd': {'voiced': 1, 'fricative': 0, 'plosive': 1, 'nasal': 0, 'long_vowel': 0},  # e.g., "dog" (Voiced)
    'k': {'voiced': 0, 'fricative': 0, 'plosive': 1, 'nasal': 0, 'long_vowel': 0},  # e.g., "cat" (Unvoiced)
    'g': {'voiced': 1, 'fricative': 0, 'plosive': 1, 'nasal': 0, 'long_vowel': 0},  # e.g., "go"  (Voiced)

    # Fricatives: Formed by forcing air through a narrow channel, creating friction noise
    'f': {'voiced': 0, 'fricative': 1, 'plosive': 0, 'nasal': 0, 'long_vowel': 0},  # e.g., "fan" (Unvoiced)
    'v': {'voiced': 1, 'fricative': 1, 'plosive': 0, 'nasal': 0, 'long_vowel': 0},  # e.g., "van" (Voiced)
    'θ': {'voiced': 0, 'fricative': 1, 'plosive': 0, 'nasal': 0, 'long_vowel': 0},  # e.g., "thin" (Unvoiced TH)
    'ð': {'voiced': 1, 'fricative': 1, 'plosive': 0, 'nasal': 0, 'long_vowel': 0},  # e.g., "this" (Voiced TH)
    's': {'voiced': 0, 'fricative': 1, 'plosive': 0, 'nasal': 0, 'long_vowel': 0},  # e.g., "sit" (Unvoiced)
    'z': {'voiced': 1, 'fricative': 1, 'plosive': 0, 'nasal': 0, 'long_vowel': 0},  # e.g., "zoo" (Voiced)
    'ʃ': {'voiced': 0, 'fricative': 1, 'plosive': 0, 'nasal': 0, 'long_vowel': 0},  # e.g., "she" (Unvoiced SH)
    'ʒ': {'voiced': 1, 'fricative': 1, 'plosive': 0, 'nasal': 0, 'long_vowel': 0},  # e.g., "vision" (Voiced ZH)
    'h': {'voiced': 0, 'fricative': 1, 'plosive': 0, 'nasal': 0, 'long_vowel': 0},  # e.g., "hat" (Unvoiced)

    # Affricates: Complex consonants that begin as a plosive and release as a fricative
    'tʃ': {'voiced': 0, 'fricative': 1, 'plosive': 1, 'nasal': 0, 'long_vowel': 0},  # e.g., "church" (Unvoiced)
    'dʒ': {'voiced': 1, 'fricative': 1, 'plosive': 1, 'nasal': 0, 'long_vowel': 0},  # e.g., "judge" (Voiced)

    # Nasals: Formed by lowering the velum, allowing air to escape through the nasal cavity (All inherently voiced)
    'm': {'voiced': 1, 'fricative': 0, 'plosive': 0, 'nasal': 1, 'long_vowel': 0},  # e.g., "man"
    'n': {'voiced': 1, 'fricative': 0, 'plosive': 0, 'nasal': 1, 'long_vowel': 0},  # e.g., "no"
    'ŋ': {'voiced': 1, 'fricative': 0, 'plosive': 0, 'nasal': 1, 'long_vowel': 0},  # e.g., "sing" (NG sound)

    # Approximants & Liquids: Glides and fluid sounds where articulators approach but do not touch (All inherently voiced)
    'l': {'voiced': 1, 'fricative': 0, 'plosive': 0, 'nasal': 0, 'long_vowel': 0},  # e.g., "leg" (Lateral liquid)
    'r': {'voiced': 1, 'fricative': 0, 'plosive': 0, 'nasal': 0, 'long_vowel': 0},  # e.g., "red" (Rhotic liquid)
    'j': {'voiced': 1, 'fricative': 0, 'plosive': 0, 'nasal': 0, 'long_vowel': 0},  # e.g., "yes" (Palatal approximant)
    'w': {'voiced': 1, 'fricative': 0, 'plosive': 0, 'nasal': 0, 'long_vowel': 0},
    # e.g., "wet" (Labio-velar approximant)

    # =========================================================================
    # B. VOWELS - ~20 Phonemes (All vowels are inherently voiced, non-consonantal)
    # =========================================================================

    # Short Monophthongs: Steady, single vowel sounds with a short acoustic duration
    'ɪ': {'voiced': 1, 'fricative': 0, 'plosive': 0, 'nasal': 0, 'long_vowel': 0},  # e.g., "pit"
    'ɛ': {'voiced': 1, 'fricative': 0, 'plosive': 0, 'nasal': 0, 'long_vowel': 0},  # e.g., "pet"
    'æ': {'voiced': 1, 'fricative': 0, 'plosive': 0, 'nasal': 0, 'long_vowel': 0},  # e.g., "pat"
    'ʊ': {'voiced': 1, 'fricative': 0, 'plosive': 0, 'nasal': 0, 'long_vowel': 0},  # e.g., "put"
    'ʌ': {'voiced': 1, 'fricative': 0, 'plosive': 0, 'nasal': 0, 'long_vowel': 0},  # e.g., "cut"
    'ɒ': {'voiced': 1, 'fricative': 0, 'plosive': 0, 'nasal': 0, 'long_vowel': 0},  # e.g., "pot"
    'ə': {'voiced': 1, 'fricative': 0, 'plosive': 0, 'nasal': 0, 'long_vowel': 0},  # e.g., "about" (Unstressed Schwa)

    # Long Monophthongs (With Explicit Length Colon Symbol ':')
    'i:': {'voiced': 1, 'fricative': 0, 'plosive': 0, 'nasal': 0, 'long_vowel': 1},  # e.g., "keep"
    'ɑ:': {'voiced': 1, 'fricative': 0, 'plosive': 0, 'nasal': 0, 'long_vowel': 1},  # e.g., "father"
    'ɔ:': {'voiced': 1, 'fricative': 0, 'plosive': 0, 'nasal': 0, 'long_vowel': 1},  # e.g., "caught"
    'u:': {'voiced': 1, 'fricative': 0, 'plosive': 0, 'nasal': 0, 'long_vowel': 1},  # e.g., "boot"
    'ɜ:': {'voiced': 1, 'fricative': 0, 'plosive': 0, 'nasal': 0, 'long_vowel': 1},  # e.g., "bird"

    # Long Monophthongs Fallbacks (Without Colon, in case the dictionary library omits them)
    'i': {'voiced': 1, 'fricative': 0, 'plosive': 0, 'nasal': 0, 'long_vowel': 1},
    'ɑ': {'voiced': 1, 'fricative': 0, 'plosive': 0, 'nasal': 0, 'long_vowel': 1},
    'ɔ': {'voiced': 1, 'fricative': 0, 'plosive': 0, 'nasal': 0, 'long_vowel': 1},
    'u': {'voiced': 1, 'fricative': 0, 'plosive': 0, 'nasal': 0, 'long_vowel': 1},
    'ɜ': {'voiced': 1, 'fricative': 0, 'plosive': 0, 'nasal': 0, 'long_vowel': 1},

    # Diphthongs: Complex vowels where the sound glides dynamically from one target to another
    'eɪ': {'voiced': 1, 'fricative': 0, 'plosive': 0, 'nasal': 0, 'long_vowel': 1},  # e.g., "say"
    'aɪ': {'voiced': 1, 'fricative': 0, 'plosive': 0, 'nasal': 0, 'long_vowel': 1},  # e.g., "my"
    'ɔɪ': {'voiced': 1, 'fricative': 0, 'plosive': 0, 'nasal': 0, 'long_vowel': 1},  # e.g., "boy"
    'aʊ': {'voiced': 1, 'fricative': 0, 'plosive': 0, 'nasal': 0, 'long_vowel': 1},  # e.g., "now"
    'oʊ': {'voiced': 1, 'fricative': 0, 'plosive': 0, 'nasal': 0, 'long_vowel': 1},  # e.g., "go"
    'ɪə': {'voiced': 1, 'fricative': 0, 'plosive': 0, 'nasal': 0, 'long_vowel': 1},  # e.g., "near"
    'eə': {'voiced': 1, 'fricative': 0, 'plosive': 0, 'nasal': 0, 'long_vowel': 1},  # e.g., "hair"
    'ʊə': {'voiced': 1, 'fricative': 0, 'plosive': 0, 'nasal': 0, 'long_vowel': 1},  # e.g., "poor"
}


def extract_phonetic_vector(word_phonemes):
    """
    Automatically normalize the linguistic features of a list of phonemes into a single numerical vector.
    """
    # Defensive check to prevent ZeroDivisionError if the phoneme list is empty
    if not word_phonemes:
        return [0.0, 0.0, 0.0, 0.0, 0.0]

    # Initialize a blank dictionary representing the baseline (all features absent).
    features_appearance_count = {'voiced': 0, 'fricative': 0, 'plosive': 0, 'nasal': 0, 'long_vowel': 0}

    # Iterate through each phoneme extracted from the word and count the appearance of each feature
    for phoneme in word_phonemes:
        # Check if the phoneme exists in our predefined linguistic lookup table.
        if phoneme in IPA_FEATURES_TABLE:
            features = IPA_FEATURES_TABLE[phoneme]
            # If a feature is present (equal to 1) in the current phoneme, increment its count
            for key in features_appearance_count.keys():
                if features[key] == 1:
                    features_appearance_count[key] += 1  # FIXED: Correctly incrementing the dictionary value

    # normalize the weight of each feature
    normalized_features_weight = {'voiced': 0.0, 'fricative': 0.0, 'plosive': 0.0, 'nasal': 0.0, 'long_vowel': 0.0}
    for key in features_appearance_count.keys():
        #Dividing the counts by the total number of phonemes
        normalized_features_weight[key] = features_appearance_count[key] / len(word_phonemes)

    # Return the final normalized features as a flat numerical list (vector) suitable for ML models.
    return [
        normalized_features_weight['voiced'],
        normalized_features_weight['fricative'],
        normalized_features_weight['plosive'],
        normalized_features_weight['nasal'],
        normalized_features_weight['long_vowel']
    ]

def get_phonetic_features(audio_path, ground_truth_text=None, allow_asr=True):
    """
    A hybrid pipeline function that extracts phonetic features depending on data availability and constraints.
    It adapts to whether the spoken text is known in advance, unknown, or if speed is prioritized.
    """

    # -----------------------------------------------------------------------------------
    # SCENARIO 1: The text is known in advance (Ideal for datasets like Speech Accent Archive).
    # -----------------------------------------------------------------------------------
    if ground_truth_text:
        # Convert the standard English text directly into its theoretical IPA string representation.
        raw_ipa = ipa.convert(ground_truth_text)

        # Clean up any unexpected characters (like unresolved asterisks) and split into a list of characters.
        phonemes = list(raw_ipa.replace("*", ""))

        # Pass the extracted phonemes to our encoder to get the final numerical vector.
        return extract_phonetic_vector(phonemes)

    # -----------------------------------------------------------------------------------
    # SCENARIO 2: The text is unknown, but automated transcription (ASR) is permitted.
    # -----------------------------------------------------------------------------------
    elif allow_asr:
        # In a real environment, you would invoke a fast, lightweight local model here.
        # Example: text_from_whisper = whisper_tiny_model.transcribe(audio_path)
        text_from_whisper = "stella"  # Simulated lightweight Whisper-tiny transcript output.

        # Convert the transcribed text into its theoretical IPA representation.
        raw_ipa = ipa.convert(text_from_whisper)

        # Convert the IPA characters into our standard numerical feature vector.
        return extract_phonetic_vector(list(raw_ipa))

    # -----------------------------------------------------------------------------------
    # SCENARIO 3: The text is unknown and ASR is disabled to maximize processing speed.
    # -----------------------------------------------------------------------------------
    else:
        # Return a fallback neutral vector (all zeros) to ensure the model pipeline doesn't crash.
        # This allows the classifier to rely solely on the acoustic features (MFCC) when text is missing.
        return [0, 0, 0, 0, 0]

