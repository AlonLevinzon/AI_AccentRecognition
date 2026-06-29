import os
import pandas as pd

accent_mapping = {
    0: "Dutch",
    1: "German",
    2: "Czech",
    3: "Polish",
    4: "French",
    5: "Hungarian",
    6: "Finnish",
    7: "Romanian",
    8: "Slovak",
    9: "Spanish",
    10: "Italian",
    11: "Estonian",
    12: "Lithuanian",
    13: "Croatian",
    14: "Slovene",
    15: "English",
    16: "Scottish",
    17: "Irish",
    18: "NorthernIrish",
    19: "Indian",
    20: "Vietnamese",
    21: "Canadian",
    22: "American"
}
group_folder=["train" ,"validation" ,"test" ]
for group in group_folder:
    metadata_path = f"metadata_{group}.csv"
    output_excel_path = f"{group}_accent_counts.csv"
    df = pd.read_csv(metadata_path)

    counts = df["accent"].value_counts().reset_index()
    counts.columns = ["Accent Number", "Count"]

    counts["Accent Name"] = counts["Accent Number"].map(accent_mapping)

    counts = counts[["Accent Number", "Accent Name", "Count"]]
    counts = counts.sort_values(by="Accent Number").reset_index(drop=True)

    # counts.to_csv(output_excel_path, index=False, encoding="utf-8-sig")
    #
    # print(f"Excel file created successfully at: {output_excel_path}")
    print(f"\nPreview of the data for {group}:")
    print(counts)