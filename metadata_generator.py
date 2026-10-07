import sys
from pathlib import Path

# Add project root directory to sys.path to ensure module imports
WORKING_DIR = Path.cwd()
if str(WORKING_DIR) not in sys.path:
    sys.path.insert(0, str(WORKING_DIR))

# Import metadata builder subroutines from the K_fold module
from K_fold import (
    build_free_text_only_metadata,
    build_mixed_metadata,
    build_archive_only_metadata
)

# =====================================================================
# CONFIGURATION
# =====================================================================
# Set the desired mode: "free_text", "mixed", or "archive_only"
RUN_MODE = "free_text"


def generate_metadata_by_mode(mode: str):
    """
    Triggers the specific metadata generation pipeline in K_fold based on mode.
    Returns the generated DataFrame or raises ValueError if mode is unknown.
    """
    print(f"====================================================")
    print(f"🚀 Starting Metadata Generation Pipeline (Mode: '{mode}')")
    print(f"====================================================")

    if mode == "free_text":
        df_metadata = build_free_text_only_metadata()
    elif mode == "mixed":
        df_metadata = build_mixed_metadata()
    elif mode == "archive_only":
        df_metadata = build_archive_only_metadata()
    else:
        raise ValueError(f"❌ Invalid mode: '{mode}'. Expected 'free_text', 'mixed', or 'archive_only'.")

    if df_metadata is not None and not df_metadata.empty:
        print(f"\n✅ Successfully generated metadata for '{mode}' with {len(df_metadata)} rows.")
    else:
        print(f"\n⚠️ Warning: Metadata generation for '{mode}' returned an empty DataFrame.")

    return df_metadata


if __name__ == "__main__":
    generate_metadata_by_mode(RUN_MODE)