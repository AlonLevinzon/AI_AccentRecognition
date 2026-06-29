# # The expected golden text for validation
# EXPECTED_TEXT = (
#     "Please call Stella. Ask her to bring these things with her from the store: "
#     "Six spoons of fresh snow peas, five thick slabs of blue cheese, and maybe a snack for her brother Bob. "
#     "We also need a small plastic snake and a big toy frog for the kids. "
#     "She can scoop these things into three red bags, and we will go meet her Wednesday at the train station."
# )
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
import gc
import whisperx

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================
DEVICE = "cpu"  # Change to "cuda" if you have an Nvidia GPU

# The exact 69 words of the Stella Paragraph in chronological order
STELLA_WORDS = [
    "please",
    "call",
    "stella",
    "ask",
    "her",
    "to",
    "bring",
    "these",
    "things",
    "with",
    "her",
    "from",
    "the",
    "store",
    "six",
    "spoons",
    "of",
    "fresh",
    "snow",
    "peas",
    "five",
    "thick",
    "slabs",
    "of",
    "blue",
    "cheese",
    "and",
    "maybe",
    "a",
    "snack",
    "for",
    "her",
    "brother",
    "bob",
    "we",
    "also",
    "need",
    "a",
    "small",
    "plastic",
    "snake",
    "and",
    "a",
    "big",
    "toy",
    "frog",
    "for",
    "the",
    "kids",
    "she",
    "can",
    "scoop",
    "these",
    "things",
    "into",
    "three",
    "red",
    "bags",
    "and",
    "we",
    "will",
    "go",
    "meet",
    "her",
    "wednesday",
    "at",
    "the",
    "train",
    "station",
]
stella_sentences = ["Please call Stella. Ask her to bring these things" ,

"with her from the store: Six spoons of fresh snow peas," ,

"five thick slabs of blue cheese, and maybe a snack for her brother Bob." ,

"We also need a small plastic snake and a big toy frog for the kids." ,

"She can scoop these things into three red bags," ,

"and we will go meet her Wednesday at the train station."]

# ==========================================
# MODEL INITIALIZATION (Happens once)
# ==========================================
print("Loading WhisperX models into memory (this happens only once)...")
WHISPER_MODEL = whisperx.load_model("small", device=DEVICE)
# Forcing English alignment model as we know the text is always English
ALIGN_MODEL, ALIGN_METADATA = whisperx.load_align_model(
    language_code="en", device=DEVICE
)
print("Models loaded successfully!\n")


def get_sentences_time(audio_file, target_segments):
    """
    Transcribes the audio using WhisperX, aligns word-level timestamps,
    and extracts the exact start and end times for the 6 target segments
    based on their first and last words.

    Args:
        audio_file (str): Path to the input audio file.
        target_segments (list): A list of the 6 text phrases to extract.

    Returns:
        list of tuples: [(index, segment_text, start_time, end_time), ...]
    """
    try:
        audio_data = whisperx.load_audio(audio_file)

        # to guide the model and improve alignment accuracy with accented speech.
        full_stella_text = (
            "Please call Stella. Ask her to bring these things with her from the store: "
            "Six spoons of fresh snow peas, five thick slabs of blue cheese, and maybe a snack for her brother Bob. "
            "We also need a small plastic snake and a big toy frog for the kids. "
            "She can scoop these things into three red bags, and we will go meet her Wednesday at the train station."
        )

        # 1. Transcribe audio with only the absolute necessary arguments
        result = WHISPER_MODEL.transcribe(
            audio_data,
            batch_size=16,
            language="en"
        )

        # 2. Align timestamps at the word level
        aligned_result = whisperx.align(
            result["segments"], ALIGN_MODEL, ALIGN_METADATA, audio_data, DEVICE, return_char_alignments=False
        )

        # Force garbage collection to prevent memory overhead on CPU/GPU
        gc.collect()

        # 3. Flatten all detected words with valid timestamps into a single list
        all_words = []
        for segment in aligned_result["segments"]:
            for word_info in segment.get("words", []):
                if "start" in word_info and "end" in word_info:
                    all_words.append({
                        "word": word_info["word"].lower().strip(".,;:?!"),
                        "start": word_info["start"],
                        "end": word_info["end"]
                    })

        final_segments_timestamps = []
        word_pointer = 0

        # 4. Map the 6 target text segments to the detected word timestamps
        for idx, segment_text in enumerate(target_segments):
            segment_words = [w.lower().strip(".,;:?!") for w in segment_text.split()]

            start_time = None
            end_time = None

            # Locate the start time using the first word of the target segment
            while word_pointer < len(all_words):
                if all_words[word_pointer]["word"] == segment_words[0]:
                    start_time = all_words[word_pointer]["start"]
                    break
                word_pointer += 1

            # Locate the end time using the last word of the target segment
            target_end_word = segment_words[-1]
            search_limit = min(word_pointer + len(segment_words) + 5, len(all_words))

            for j in range(word_pointer, search_limit):
                if all_words[j]["word"] == target_end_word:
                    end_time = all_words[j]["end"]
                    word_pointer = j + 1  # Move pointer forward for the next segment
                    break

            # Fallback: If the exact last word was missed by alignment, use the estimated window end
            if end_time is None and start_time is not None:
                fallback_idx = min(word_pointer + len(segment_words) - 1, len(all_words) - 1)
                end_time = all_words[fallback_idx]["end"]

            # 5. Apply acoustic padding suitable for feature extraction (3-5s clips)
            if start_time is not None and end_time is not None:
                padded_start = max(0.0, start_time - 0.1)  # 100ms pre-padding to preserve initial consonants
                padded_end = end_time + 0.2  # 200ms post-padding to prevent clipping tail ends

                final_segments_timestamps.append((idx, segment_text, padded_start, padded_end))

        return final_segments_timestamps

    except Exception as e:
        print(f"Error in get_sentences_time: {e}")
        return []
# ==========================================
# CORE TIMESTAMP EXTRACTION FUNCTION
# ==========================================
def get_words_time(audio_file):
    """
    Transcribes the audio, aligns timestamps, and pairs them with the
    correct expected Stella word based on its index/position.

    Returns:
        A list of tuples: [(index, correct_word, start_time, end_time), ...]
    """
    try:
        # 1. Load audio and transcribe
        audio_data = whisperx.load_audio(audio_file)
        result = WHISPER_MODEL.transcribe(
            audio_data, batch_size=16, language="en"
        )

        # 2. Align timestamps
        aligned_result = whisperx.align(
            result["segments"],
            ALIGN_MODEL,
            ALIGN_METADATA,
            audio_data,
            DEVICE,
            return_char_alignments=False,
        )

        # Force garbage collection to prevent memory leaks on CPU
        gc.collect()

        # 3. Extract raw timestamps where both start and end exist
        detected_timestamps = []
        for segment in aligned_result["segments"]:
            for word_info in segment["words"]:
                if "start" in word_info and "end" in word_info:
                    detected_timestamps.append(
                        (word_info["start"], word_info["end"])
                    )

        # 4. Zip the expected words with the timestamps based on index
        final_numbered_list = []
        matched_elements = min(len(STELLA_WORDS), len(detected_timestamps))

        start_padding = 0.06  # 60ms is usually perfect for the start cutoff

        for index in range(matched_elements):
            correct_word = STELLA_WORDS[index]
            start_time, end_time = detected_timestamps[index]

            # Base padding for the end (default to 250ms if there's enough space)
            end_padding = 0.25

            # If there is a next word, calculate the available gap between them
            if index < matched_elements - 1:
                next_word_start = detected_timestamps[index + 1][0]
                gap_to_next_word = next_word_start - end_time

                if gap_to_next_word > 0:
                    # Don't take more than 40% of the gap to ensure we NEVER touch the next word
                    end_padding = min(end_padding, gap_to_next_word * 0.4)
                else:
                    # If words overlap in timestamps, don't add padding at all
                    end_padding = 0.0

            # Apply the calculated padding
            padded_start = max(0.0, start_time - start_padding)
            padded_end = end_time + end_padding

            # Append as a 4-element tuple
            final_numbered_list.append((index, correct_word, padded_start, padded_end))

        return final_numbered_list

    except Exception as e:
        print(f"Error in get_words_time for {audio_file}: {e}")
        return []
