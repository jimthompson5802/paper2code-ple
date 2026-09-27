"""Generate random numeric values and display their raw PLE encodings."""

import numpy as np
import pandas as pd
import torch

from models.embeddings import PiecewiseLinearEmbedding


NUM_ROWS = 1_000
PLE_BINS = 10
SAMPLE_SIZE = 5
RANDOM_SEED = 42


def main() -> None:
    """Create a numeric/PLE DataFrame and print a reproducible random sample."""
    rng = np.random.default_rng(RANDOM_SEED)
    values = rng.uniform(0.0, 100.0, size=NUM_ROWS).astype(np.float32)
    numeric_df = pd.DataFrame({"numeric_value": values})

    boundaries = torch.linspace(0.0, 100.0, steps=PLE_BINS + 1)
    ple_embedding = PiecewiseLinearEmbedding(
        num_features=1,
        output_dim=0,
        num_bins=PLE_BINS,
    ).set_boundaries(boundaries)
    print(f"PLE bin boundaries: {boundaries.tolist()}")
    numeric_tensor = torch.from_numpy(values).reshape(-1, 1)

    with torch.no_grad():
        ple_vectors = ple_embedding(numeric_tensor).cpu().numpy()

    result_df = numeric_df.assign(ple=[vector.tolist() for vector in ple_vectors])
    sample = result_df.sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED)
    print(sample.to_string(index=False, max_colwidth=100))


if __name__ == "__main__":
    main()
