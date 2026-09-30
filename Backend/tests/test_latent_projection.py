"""FR11, the latent projection explorer.

The requirement-to-test matrix recorded FR11 as having no dedicated backend
test: the route and the reduction dependency both existed, and the frontend had
the panel, but nothing on the backend asserted the projection was correct. The
gap was confirmed by searching the test directory, not assumed.

These cover `reduce_dimensions`, which is the part a wrong answer would be
invisible in: a projection that silently returns the wrong shape, or reorders
its rows, still renders as a perfectly plausible scatter plot. Only the
relationship between input and output can catch that.

The route itself is deliberately not exercised here. It needs real model
weights to produce embeddings, which the offline suite does not download; the
reduction is the part that belongs to AudioLIT rather than to the model.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.domain.model_loader_service import reduce_dimensions


def _embeddings(n: int = 12, dim: int = 64, seed: int = 0) -> list:
    """n distinct high-dimensional vectors, as the extractor would return."""
    rng = np.random.default_rng(seed)
    return [rng.normal(size=dim).astype("float32") for _ in range(n)]


class TestOutputShape:
    """The projection must return one row per input at the requested width."""

    @pytest.mark.parametrize("n_components", [2, 3])
    def test_pca_returns_one_row_per_sample(self, n_components):
        embs = _embeddings(n=12)
        out = reduce_dimensions(embs, method="pca", n_components=n_components)

        assert out.shape == (12, n_components), (
            f"projected {len(embs)} embeddings to {out.shape}; the plot binds "
            "each row to a file, so a row count mismatch mislabels every point"
        )

    def test_an_empty_input_is_not_an_error(self):
        # The panel calls this before any file is selected.
        out = reduce_dimensions([], method="pca", n_components=2)
        assert out.size == 0

    def test_output_is_finite(self):
        out = reduce_dimensions(_embeddings(), method="pca", n_components=2)
        assert np.isfinite(out).all(), "a NaN coordinate silently drops a point from the plot"


class TestMethodSelection:
    """The method the caller asks for is the method that runs."""

    def test_unsupported_method_raises_rather_than_falling_back(self):
        # A silent fallback to PCA would show the user a t-SNE-labelled plot
        # that is not t-SNE, which is exactly the class of defect this project
        # treats as most serious.
        with pytest.raises(ValueError, match="Unsupported reduction method"):
            reduce_dimensions(_embeddings(), method="definitely-not-a-method", n_components=2)

    def test_pca_and_tsne_do_not_produce_the_same_projection(self):
        embs = _embeddings(n=20, seed=3)
        pca = reduce_dimensions(embs, method="pca", n_components=2)
        tsne = reduce_dimensions(embs, method="tsne", n_components=2)

        assert pca.shape == tsne.shape
        # Two different algorithms on the same input must differ. Equality here
        # would mean one label is lying about which ran.
        assert not np.allclose(pca, tsne), (
            "PCA and t-SNE returned the same coordinates, so the method "
            "selection is not reaching the reducer"
        )


class TestDeterminism:
    """FR4.4's reproducibility expectation extends to the projection: the same
    embeddings must plot the same way twice, or a user comparing two screenshots
    of one dataset sees movement that is not in the data."""

    def test_pca_is_reproducible(self):
        embs = _embeddings(n=15, seed=7)
        first = reduce_dimensions(embs, method="pca", n_components=3)
        second = reduce_dimensions(embs, method="pca", n_components=3)
        np.testing.assert_allclose(first, second, rtol=1e-6, atol=1e-6)

    def test_tsne_is_seeded(self):
        # t-SNE is stochastic, so this only holds because random_state is fixed.
        # If someone removes the seed, this is what catches it.
        embs = _embeddings(n=15, seed=8)
        first = reduce_dimensions(embs, method="tsne", n_components=2)
        second = reduce_dimensions(embs, method="tsne", n_components=2)
        np.testing.assert_allclose(first, second, rtol=1e-5, atol=1e-5)


class TestStructureIsPreserved:
    """A projection is only useful if it reflects the input. These assert the
    weakest property that still has teeth: well-separated clusters in the input
    stay separated in the output."""

    def test_two_distant_clusters_stay_separate_under_pca(self):
        rng = np.random.default_rng(11)
        dim = 32
        a = [rng.normal(loc=-8.0, scale=0.3, size=dim).astype("float32") for _ in range(8)]
        b = [rng.normal(loc=+8.0, scale=0.3, size=dim).astype("float32") for _ in range(8)]

        out = reduce_dimensions(a + b, method="pca", n_components=2)
        first, second = out[:8], out[8:]

        within = max(
            np.linalg.norm(first - first.mean(axis=0), axis=1).max(),
            np.linalg.norm(second - second.mean(axis=0), axis=1).max(),
        )
        between = float(np.linalg.norm(first.mean(axis=0) - second.mean(axis=0)))

        assert between > within * 2, (
            f"cluster separation {between:.2f} is not clearly greater than the "
            f"within-cluster spread {within:.2f}; the projection is not carrying "
            "the structure that makes the panel meaningful"
        )

    def test_row_order_is_preserved(self):
        """Row i of the output must be the projection of row i of the input.

        The panel maps output rows back to filenames by position, so a reorder
        would attach every point to the wrong file with no visible symptom.
        """
        rng = np.random.default_rng(5)
        dim = 32
        far = rng.normal(loc=20.0, scale=0.1, size=dim).astype("float32")
        near = [rng.normal(loc=0.0, scale=0.1, size=dim).astype("float32") for _ in range(6)]

        # Put the outlier last; it must still be the last output row.
        out = reduce_dimensions(near + [far], method="pca", n_components=2)
        distances = np.linalg.norm(out - out[:-1].mean(axis=0), axis=1)

        assert int(np.argmax(distances)) == len(out) - 1, (
            "the outlier did not land in the last output row, so the projection "
            "reordered its input relative to the caller's file list"
        )
