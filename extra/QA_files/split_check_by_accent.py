# # Extra Block: Detailed Accent Breakdown across 10 Outer Folds (Train / Val / Test)
# # Prints a breakdown table of sample counts and percentages per accent for each fold split.
#
# import pandas as pd
# from sklearn.model_selection import GroupKFold
#
# print("\n" + "=" * 80)
# print("Detailed Accent Breakdown across 10 Folds (Train / Val / Test)")
# print("=" * 80)
#
# gkf_outer = GroupKFold(n_splits=10)
# groups = df_master_features["speaker_id"]
#
# for fold, (train_val_idx, test_idx) in enumerate(gkf_outer.split(df_master_features, groups=groups), 1):
#     train_val_df = df_master_features.iloc[train_val_idx].reset_index(drop=True)
#     test_df = df_master_features.iloc[test_idx].reset_index(drop=True)
#
#     # Split train_val set to extract inner Validation set
#     gkf_inner = GroupKFold(n_splits=9)
#     train_idx_inner, val_idx_inner = next(gkf_inner.split(train_val_df, groups=train_val_df["speaker_id"]))
#
#     train_df = train_val_df.iloc[train_idx_inner].reset_index(drop=True)
#     val_df = train_val_df.iloc[val_idx_inner].reset_index(drop=True)
#
#     print(f"\n--- FOLD {fold} DISTRIBUTION ---")
#
#     train_counts = train_df['accent'].value_counts()
#     val_counts = val_df['accent'].value_counts()
#     test_counts = test_df['accent'].value_counts()
#
#     fold_dist = pd.DataFrame({
#         'Train_Count': train_counts,
#         'Train_%': (train_counts / len(train_df) * 100).round(2),
#         'Val_Count': val_counts,
#         'Val_%': (val_counts / len(val_df) * 100).round(2),
#         'Test_Count': test_counts,
#         'Test_%': (test_counts / len(test_df) * 100).round(2)
#     }).fillna(0)
#
#     # Cast count columns back to integer formatting for clean output
#     count_cols = ['Train_Count', 'Val_Count', 'Test_Count']
#     fold_dist[count_cols] = fold_dist[count_cols].astype(int)
#
#     print(fold_dist.to_string())
#     print(f"Total Samples in Fold {fold}: Train={len(train_df)} | Val={len(val_df)} | Test={len(test_df)}")
#
# print("\n" + "=" * 80)