import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_classif
#
# # 1. Load data
# csv_path = "master_extracted_features_free_text.csv"
# if not os.path.exists(csv_path):
#     raise FileNotFoundError(f"Missing master file: {csv_path}")
#
# df = pd.read_csv(csv_path)
#
# # 2. Exclude non-phonetic and absolute time/volume columns as decided
# exclude_cols = ['filename', 'file_path', 'speaker_id', 'text', 'accent',
#                 'total_duration', 'phonation_time', 'phonation_ratio',
#                 'vowel_consonant_ratio'] # Useless features
# feature_cols = [c for c in df.columns if c not in exclude_cols]
#
# print(f"📊 Analyzing {len(feature_cols)} acoustic features over {len(df)} samples...")
#
# # ==========================================
# # G1: ANOVA F-SCORE RANKING
# # ==========================================
# X = df[feature_cols].fillna(0)
# le = LabelEncoder()
# y = le.fit_transform(df['accent'])
#
# f_scores, p_values = f_classif(X, y)
# anova_df = pd.DataFrame({
#     'Feature': feature_cols,
#     'F_Score': f_scores,
#     'P_Value': p_values
# }).sort_values(by='F_Score', ascending=False)
#
# print("\n🏆 Top 10 Features by ANOVA F-Score:")
# print(anova_df.head(10).to_string(index=False))
#
# # ==========================================
# # G2: CORRELATION HEATMAP (Top 15 features)
# # ==========================================
# top_15_features = anova_df['Feature'].head(15).tolist()
# plt.figure(figsize=(12, 10))
# sns.heatmap(df[top_15_features].corr(), annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
# plt.title("Correlation Heatmap of Top 15 ANOVA Features", fontsize=14)
# plt.tight_layout()
# plt.savefig("top_features_correlation_heatmap.png", dpi=300)
# plt.close()
# print("💾 Saved: top_features_correlation_heatmap.png")
#
# # ==========================================
# # G3: VIOLIN PLOT (For the top feature)
# # ==========================================
# top_feature = anova_df['Feature'].iloc[0]
# plt.figure(figsize=(10, 6))
# sns.violinplot(x='accent', y=top_feature, data=df, palette="muted")
# plt.title(f"Acoustic Distribution of '{top_feature}' across Accents", fontsize=14)
# plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.tight_layout()
# plt.savefig("top_feature_distribution_violin.png", dpi=300)
# plt.close()
# print(f"💾 Saved: top_feature_distribution_violin.png (Based on {top_feature})")
#
# # ==========================================
# # G4: PCA - 2D VISUALIZATION
# # ==========================================
# X_scaled = StandardScaler().fit_transform(X)
# pca = PCA(n_components=2, random_state=42)
# X_pca = pca.fit_transform(X_scaled)
#
# plt.figure(figsize=(11, 8))
# sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df['accent'], palette="Set1", alpha=0.6, edgecolor=None)
# plt.title(f"2D PCA Projection of Accent Acoustic Space\n(Explained Variance: {np.sum(pca.explained_variance_ratio_)*100:.1f}%)", fontsize=14)
# plt.xlabel("Principal Component 1")
# plt.ylabel("Principal Component 2")
# plt.grid(True, linestyle='--', alpha=0.5)
# plt.tight_layout()
# plt.savefig("dataset_acoustic_pca_2d.png", dpi=300)
# plt.close()
# print("💾 Saved: dataset_acoustic_pca_2d.png")
#
# print("\n✅ Exploratory Data Analysis complete! Use the generated figures for your official report.")


def plot_top_features_multidimensional_pca(df, feature_columns, target_column='accent', top_k=15,
                                           save_path='top_features_pca_2d.png'):
    """
    Identifies the top K statistically significant acoustic features using ANOVA F-Score,
    projecting them onto a 2D space using Principal Component Analysis (PCA) to visualize
    accent clustering.
    """
    X = df[feature_columns].fillna(0)
    y = df[target_column]

    # Compute ANOVA F-scores
    f_scores, _ = f_classif(X, y)
    anova_df = pd.DataFrame({'Feature': feature_columns, 'F_Score': f_scores})

    # Sort and extract top K features
    top_features = anova_df.sort_values(by='F_Score', ascending=False)['Feature'].head(top_k).tolist()

    # Standardize and apply PCA
    X_subset = df[top_features].fillna(0)
    X_scaled = StandardScaler().fit_transform(X_subset)
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    # Plot multidimensional visualization
    plt.figure(figsize=(11, 8))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df[target_column], palette="Set1", alpha=0.6, edgecolor=None)

    variance_explained = np.sum(pca.explained_variance_ratio_) * 100
    plt.title(f"2D PCA Projection of Top {top_k} ANOVA Features\n(Explained Variance: {variance_explained:.1f}%)",
              fontsize=14)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    print(f"💾 Saved TOP features multidimensional PCA plot to: {save_path}")
    plt.show()


def print_weakest_anova_features(df, feature_columns, target_column='accent', bottom_k=15):
    """
    Computes ANOVA F-Scores across all clean features and prints the bottom K
    weakest features (lowest variance between class scores) directly to the console.
    """
    X = df[feature_columns].fillna(0)
    y = df[target_column]

    # Compute ANOVA F-scores
    f_scores, p_values = f_classif(X, y)
    anova_df = pd.DataFrame({
        'Feature': feature_columns,
        'F_Score': f_scores,
        'P_Value': p_values
    })

    # Sort in ascending order to isolate the weakest performing features
    sorted_anova = anova_df.sort_values(by='F_Score', ascending=True)

    # Print the resulting worst feature list to terminal
    print(f"\n❌ Top {bottom_k} WEAKEST Acoustic Features (Lowest ANOVA F-Scores):")
    print("----------------------------------------------------------------------")
    print(sorted_anova.head(bottom_k).to_string(index=False))
    print("----------------------------------------------------------------------")


def plot_top_features_3d_pca(df, feature_columns, target_column='accent', top_k=15):
    """
    Projects the top K features into a 3D space using PCA components 1, 2, and 3,
    generating a 3D scatter plot to visualize multidimensional acoustic separation.
    """
    X = df[feature_columns].fillna(0)
    y = df[target_column]

    # 1. Extract Top K features using ANOVA
    f_scores, _ = f_classif(X, y)
    anova_df = pd.DataFrame({'Feature': feature_columns, 'F_Score': f_scores})
    top_features = anova_df.sort_values(by='F_Score', ascending=False)['Feature'].head(top_k).tolist()

    # 2. Scale and apply PCA with 3 components instead of 2
    X_subset = df[top_features].fillna(0)
    X_scaled = StandardScaler().fit_transform(X_subset)
    pca = PCA(n_components=3, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    # 3. Create a 3D Plot
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Map target labels to colors
    unique_labels = df[target_column].unique()
    colors = plt.cm.get_cmap('Set1', len(unique_labels))

    for i, label in enumerate(unique_labels):
        mask = df[target_column] == label
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], X_pca[mask, 2],
                   label=label, alpha=0.6, s=30, color=colors(i))

    variance_explained = np.sum(pca.explained_variance_ratio_) * 100
    ax.set_title(
        f"3D PCA Projection of Top {top_k} ANOVA Features\n(Total Explained Variance: {variance_explained:.1f}%)",
        fontsize=14)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_zlabel("PC 3")
    ax.legend()

    plt.tight_layout()
    plt.show()

# ==========================================
# MAIN EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    csv_name = "master_extracted_features_free_text.csv"

    if not os.path.exists(csv_name):
        print(f"❌ Error: Could not find '{csv_name}' in the current directory.")
    else:
        df_master = pd.read_csv(csv_name)

        # Define clean acoustic features (excluding metadata and static/leaked metrics)
        exclude_list = ['filename', 'file_path', 'speaker_id', 'text', 'accent',
                        'total_duration', 'phonation_time', 'phonation_ratio',
                        'vowel_consonant_ratio']
        clean_features = [col for col in df_master.columns if col not in exclude_list]

        print("🚀 Starting statistical feature analysis...")

        # # 1. Generate and display the Top 15 Features Multidimensional PCA Plot
        # plot_top_features_multidimensional_pca(df_master, clean_features, top_k=15)
        #
        # # 2. Extract and print the 15 weakest features directly to the console
        # print_weakest_anova_features(df_master, clean_features, bottom_k=15)
        plot_top_features_3d_pca(df_master, clean_features, target_column='accent', top_k=15)

        print("\n✅ Analysis complete.")