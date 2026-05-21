import os
import numpy as np
import pandas as pd


def generate_simulated_pte_features(
    n_subjects: int = 300,
    random_state: int = 42,
    output_path: str = "data/simulated_pte_features.csv"
) -> pd.DataFrame:
    """
    Generate a simulated imaging-derived feature dataset inspired by
    neuroimaging ML workflows.

    This does NOT use real patient data.
    This does NOT reproduce any clinical study.
    It only creates synthetic subject-level features to demonstrate:
    - lesion-style ROI features
    - ALFF-style signal features
    - functional connectivity-style features
    - supervised ML-ready labels

    label:
        0 = non-PTE-like feature pattern
        1 = PTE-like feature pattern
    """

    np.random.seed(random_state)

    if n_subjects % 2 != 0:
        raise ValueError("n_subjects should be even to maintain class balance.")

    n_per_class = n_subjects // 2

    # -----------------------------
    # Non-PTE-like simulated group
    # -----------------------------

    non_pte = pd.DataFrame({
        "subject_id": [f"SUBJ_{i:03d}" for i in range(1, n_per_class + 1)],

        # Lesion-style ROI features
        # Lower average lesion burden
        "right_temporal_lesion_volume": np.random.normal(1.2, 0.40, n_per_class),
        "left_temporal_lesion_volume": np.random.normal(1.1, 0.40, n_per_class),
        "right_occipital_lesion_volume": np.random.normal(0.9, 0.35, n_per_class),
        "left_occipital_lesion_volume": np.random.normal(0.8, 0.35, n_per_class),
        "cerebellum_lesion_volume": np.random.normal(0.7, 0.30, n_per_class),
        "right_parietal_lesion_volume": np.random.normal(0.9, 0.35, n_per_class),

        # ALFF-style features
        # Baseline low-frequency signal amplitude values
        "right_temporal_alff": np.random.normal(0.45, 0.08, n_per_class),
        "left_temporal_alff": np.random.normal(0.44, 0.08, n_per_class),
        "right_occipital_alff": np.random.normal(0.42, 0.07, n_per_class),
        "left_occipital_alff": np.random.normal(0.41, 0.07, n_per_class),
        "right_parietal_alff": np.random.normal(0.40, 0.07, n_per_class),

        # Functional connectivity-style features
        # Simulated correlation-like values
        "temporal_occipital_connectivity": np.random.normal(0.25, 0.12, n_per_class),
        "temporal_cerebellum_connectivity": np.random.normal(0.20, 0.12, n_per_class),
        "frontal_temporal_connectivity": np.random.normal(0.30, 0.12, n_per_class),
        "parietal_occipital_connectivity": np.random.normal(0.28, 0.12, n_per_class),

        "label": 0
    })

    # -----------------------------
    # PTE-like simulated group
    # -----------------------------

    pte = pd.DataFrame({
        "subject_id": [f"SUBJ_{i:03d}" for i in range(n_per_class + 1, n_subjects + 1)],

        # Lesion-style ROI features
        # Higher values in regions inspired by the paper:
        # temporal, occipital, cerebellum, and right parietal areas
        "right_temporal_lesion_volume": np.random.normal(1.8, 0.50, n_per_class),
        "left_temporal_lesion_volume": np.random.normal(1.6, 0.50, n_per_class),
        "right_occipital_lesion_volume": np.random.normal(1.3, 0.45, n_per_class),
        "left_occipital_lesion_volume": np.random.normal(1.1, 0.45, n_per_class),
        "cerebellum_lesion_volume": np.random.normal(1.1, 0.40, n_per_class),
        "right_parietal_lesion_volume": np.random.normal(1.2, 0.40, n_per_class),

        # ALFF-style features
        # Slightly altered low-frequency amplitude values
        "right_temporal_alff": np.random.normal(0.52, 0.10, n_per_class),
        "left_temporal_alff": np.random.normal(0.49, 0.10, n_per_class),
        "right_occipital_alff": np.random.normal(0.48, 0.09, n_per_class),
        "left_occipital_alff": np.random.normal(0.46, 0.09, n_per_class),
        "right_parietal_alff": np.random.normal(0.45, 0.09, n_per_class),

        # Functional connectivity-style features
        # Simulated altered connectivity pattern
        "temporal_occipital_connectivity": np.random.normal(0.32, 0.15, n_per_class),
        "temporal_cerebellum_connectivity": np.random.normal(0.28, 0.15, n_per_class),
        "frontal_temporal_connectivity": np.random.normal(0.26, 0.15, n_per_class),
        "parietal_occipital_connectivity": np.random.normal(0.33, 0.15, n_per_class),

        "label": 1
    })

    df = pd.concat([non_pte, pte], ignore_index=True)

    # Prevent impossible negative values for lesion and ALFF-style features
    non_negative_columns = [
        col for col in df.columns
        if "lesion_volume" in col or "alff" in col
    ]
    df[non_negative_columns] = df[non_negative_columns].clip(lower=0)

    # Keep connectivity values within correlation-like range
    connectivity_columns = [
        col for col in df.columns
        if "connectivity" in col
    ]
    df[connectivity_columns] = df[connectivity_columns].clip(lower=-1, upper=1)

    # Shuffle rows
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Save file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    return df


if __name__ == "__main__":
    df = generate_simulated_pte_features()
    print("Simulated feature dataset created successfully.")
    print(f"Shape: {df.shape}")
    print(df.head())
    print("\nClass balance:")
    print(df["label"].value_counts())