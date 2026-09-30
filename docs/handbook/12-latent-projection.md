# Chapter 12 — Latent projection: seeing a model's internal space

Attribution explains one prediction. The latent projection explorer shows the
*structure the model has learned* across many clips — which ones it considers
similar, and whether that similarity matches the labels you care about.

---

## 12.1 What an embedding is

A neural network's hidden layers represent their input as vectors. For
Wav2Vec2's encoder the last hidden state is `[time_frames, 1024]` — a
1024-dimensional vector per frame.

Mean-pool over time and you get **one 1024-dimensional vector per clip**
(§6.7). That vector is the model's internal representation of the clip.

The useful property: **similar inputs get nearby vectors**, under whatever
notion of similarity the model learned. Two angry clips should land near each
other in an emotion model's space. If they do not, the model has not learned
what you think it has.

That last sentence is the diagnostic value. You are not looking at the audio;
you are looking at the model's opinion about the audio.

---

## 12.2 Why 1024 dimensions must become 2

You cannot plot 1024 dimensions. **Dimensionality reduction** maps
high-dimensional points to 2 or 3 while preserving as much structure as
possible.

"As much as possible" is where the methods differ, and the differences are not
cosmetic — they answer different questions.

---

## 12.3 The three methods

```python
def reduce_dimensions(embeddings_list: list, method: str = "pca", n_components: int = 2) -> np.ndarray:
    """
    Reduce dimensionality of embeddings for visualization.

    Args:
        embeddings_list: List of embedding arrays
        method: "pca", "tsne", or "umap"
        n_components: Number of output dimensions (2 or 3)

    Returns:
        Reduced embeddings as numpy array [n_samples, n_components]
    """
    if not embeddings_list:
        return np.array([])

    X = np.vstack(embeddings_list)

    if method.lower() == "pca":
        reducer = PCA(n_components=n_components, random_state=42)
    elif method.lower() == "tsne":
        reducer = TSNE(n_components=n_components, random_state=42,
                       perplexity=min(30, len(embeddings_list)-1))
    elif method.lower() == "umap":
        reducer = umap.UMAP(n_components=n_components, random_state=42,
                           n_neighbors=min(15, len(embeddings_list)-1))
    else:
        raise ValueError(f"Unsupported reduction method: {method}")

    reduced = reducer.fit_transform(X)
    return reduced
```

### PCA — Principal Component Analysis

**Linear.** Finds the orthogonal directions of greatest variance and projects
onto the top *n*.

Mechanically: centre the data, compute the covariance matrix, take its
eigenvectors. The eigenvector with the largest eigenvalue is the direction
along which the data varies most.

| Property | Value |
|---|---|
| Preserves | global structure, large distances |
| Deterministic | yes |
| Speed | fast |
| Interpretable axes | yes — each is a fixed linear combination of input dimensions |
| Invertible | yes (approximately) |

**Use it when** you want a faithful, trustworthy overview. If two groups are
separated in PCA, they are genuinely far apart in the original space.

**Limitation:** it can only find linear structure. Data on a curved manifold —
a spiral, a sphere — gets flattened and superimposed.

### t-SNE — t-distributed Stochastic Neighbour Embedding

**Non-linear, neighbourhood-preserving.** It converts distances to
probabilities ("how likely is *j* to be *i*'s neighbour?") in both the high-
and low-dimensional spaces, then moves the low-dimensional points to minimise
the divergence between the two distributions.

| Property | Value |
|---|---|
| Preserves | local neighbourhoods |
| Deterministic | only with a fixed seed |
| Speed | slow (O(n²) naively) |
| Interpretable axes | **no** |
| Invertible | no |

**Three things you must know before reading a t-SNE plot**, because they are
routinely misinterpreted:

1. **Distances between clusters are meaningless.** Two clusters far apart may
   be adjacent in the original space. t-SNE optimises local structure and
   sacrifices global.
2. **Cluster sizes are meaningless.** t-SNE expands dense regions and contracts
   sparse ones. A visually large cluster is not a more variable one.
3. **`perplexity` changes the answer.** It is roughly "how many neighbours
   count", and different values produce genuinely different plots. There is no
   single correct value.

**Use it when** you want to see whether distinct groups exist at all. Do not
measure anything off the plot.

```python
perplexity=min(30, len(embeddings_list)-1)
```

30 is the usual default, but perplexity must be less than the sample count —
with 10 clips, perplexity 30 is an error. `min(30, n-1)` adapts. This is a
required guard: a user selecting five clips must not get a crash.

### UMAP — Uniform Manifold Approximation and Projection

**Non-linear, manifold-based.** Builds a fuzzy topological representation of
the data — a weighted neighbourhood graph — and finds a low-dimensional layout
with a similar structure.

| Property | Value |
|---|---|
| Preserves | local **and** some global structure |
| Deterministic | with a seed |
| Speed | much faster than t-SNE |
| Interpretable axes | no |
| Invertible | partially |

**Use it as the default non-linear method.** It keeps more global structure than
t-SNE, so inter-cluster distances carry *some* meaning — more than t-SNE, less
than PCA.

```python
n_neighbors=min(15, len(embeddings_list)-1)
```

`n_neighbors` is UMAP's local/global dial: small values emphasise fine local
structure, large values emphasise the big picture. Same adaptation as
perplexity, for the same reason.

---

## 12.4 Determinism, and why it is a requirement here

```python
reducer = PCA(n_components=n_components, random_state=42)
reducer = TSNE(..., random_state=42, ...)
reducer = umap.UMAP(..., random_state=42, ...)
```

`random_state=42` on all three, including PCA, which is deterministic anyway
(some solvers use randomised SVD, so it is not redundant).

The reproducibility requirement applies here, and the test explains the
user-facing reason:

```python
class TestDeterminism:
    """FR4.4's reproducibility expectation extends to the projection: the same
    embeddings must plot the same way twice, or a user comparing two screenshots
    of one dataset sees movement that is not in the data."""

    def test_tsne_is_seeded(self):
        # t-SNE is stochastic, so this only holds because random_state is fixed.
        # If someone removes the seed, this is what catches it.
        embs = _embeddings(n=15, seed=8)
        first = reduce_dimensions(embs, method="tsne", n_components=2)
        second = reduce_dimensions(embs, method="tsne", n_components=2)
        np.testing.assert_allclose(first, second, rtol=1e-5, atol=1e-5)
```

*"a user comparing two screenshots of one dataset sees movement that is not in
the data"* — that is the harm. Unseeded t-SNE gives a different layout every
run. A researcher would interpret the difference as a finding.

---

## 12.5 No silent fallback

```python
    else:
        raise ValueError(f"Unsupported reduction method: {method}")
```

And the test that pins it:

```python
def test_unsupported_method_raises_rather_than_falling_back(self):
    # A silent fallback to PCA would show the user a t-SNE-labelled plot
    # that is not t-SNE, which is exactly the class of defect this project
    # treats as most serious.
    with pytest.raises(ValueError, match="Unsupported reduction method"):
        reduce_dimensions(_embeddings(), method="definitely-not-a-method", n_components=2)
```

Plus a test that the labels are not interchangeable:

```python
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
```

That second test is the interesting one. It does not verify the algorithms are
*correct* — that is scikit-learn's job. It verifies the **method selection
actually reaches the reducer**. If a refactor broke the dispatch and everything
fell through to PCA, the shapes would still match, determinism would still
hold, cluster separation would still work. Only this test would fail.

**Testing that two things differ is a real technique**, and it is the only way
to catch "the parameter is being ignored".

---

## 12.6 The test that catches an invisible bug

```python
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
            "the outlier did not reorder its input relative to the caller's file list"
        )
```

The frontend binds output row *i* to filename *i* **by position**. There is no
id travelling with the vector. So if a reducer reordered its rows, every point
on the scatter plot would be attached to the wrong file.

And the symptom would be: a perfectly plausible scatter plot. Clusters, spread,
everything looking right — with every label wrong. Clicking a point would play
the wrong audio.

The test constructs a detectable situation: six clustered vectors plus one
far-away outlier, with the outlier placed **last**. If row order survives, the
most distant output row is the last one.

This is the single best example in the codebase of a test designed around *what
a wrong answer would look like* rather than around what the function does. A
reordered projection passes every shape, determinism and separation test.

---

## 12.7 The other guards

```python
    def test_an_empty_input_is_not_an_error(self):
        # The panel calls this before any file is selected.
        out = reduce_dimensions([], method="pca", n_components=2)
        assert out.size == 0
```

The panel renders before the user picks anything. Empty input must return an
empty array, not raise.

```python
    def test_output_is_finite(self):
        out = reduce_dimensions(_embeddings(), method="pca", n_components=2)
        assert np.isfinite(out).all(), "a NaN coordinate silently drops a point from the plot"
```

A NaN coordinate does not error in a plotting library — the point is simply not
drawn. So a clip silently vanishes from the visualisation, and the user has no
way to know one is missing.

```python
    def test_two_distant_clusters_stay_separate_under_pca(self):
        ...
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
```

The weakest property with real teeth: two clusters that are far apart in 32
dimensions must remain separated in 2. `between > within * 2` is a loose
threshold, chosen because a tight one would be flaky across scikit-learn
versions while still failing if the projection stopped carrying structure at
all.

---

## 12.8 Why the route is not tested

```python
"""The route itself is deliberately not exercised here. It needs real model
weights to produce embeddings, which the offline suite does not download; the
reduction is the part that belongs to AudioLIT rather than to the model.
"""
```

Two reasons, and the second is the principle:

- The route needs real weights, and the offline suite downloads nothing.
- **The reduction is AudioLIT's code. The embeddings are the model's.** Testing
  that Wav2Vec2 produces good embeddings is testing Wav2Vec2.

Draw the line at your own code, and say where you drew it. The alternative —
either an untested gap or a slow, network-dependent test — is worse than a
documented boundary.

---

## 12.9 Colour coding, and the point of the panel

The requirement is that points be colour-codable by emotion label, by
bona-fide/synthetic, and by accent or speaker group.

This is where the panel earns its place. Project a set of clips, then colour
them by ground-truth label:

- **Colours form clean clusters** → the model's internal space separates the
  classes. It has learned the distinction.
- **Colours are mixed** → the model does not represent the distinction, and
  whatever accuracy it reports comes from somewhere else.
- **Colours cluster by *speaker* rather than by *emotion*** → the model has
  learned to identify speakers, and is inferring emotion from speaker identity.
  That will collapse on unseen speakers.

That third case is a genuine and common failure that aggregate accuracy cannot
reveal. On a test set with the same speakers as training, a speaker-identity
shortcut scores well. The projection shows it immediately.

This is interpretability of the *model*, not of a prediction — and it is why
the panel is not merely a pretty visualisation.

---

## 12.10 The embedding family trap, once more

From §2.10 and §6.8, because it lands here:

```python
# ``wav2vec2-add`` contains "wav2vec", so the deepfake models were
# handed to the emotion extractor - same 1024 dims, no error, wrong latent
# space. And a custom SER checkpoint whose name says neither "whisper" nor
# "wav2vec" (``myorg/custom-ser``) fell through to the *Whisper* extractor,
# which is how a custom emotion model ended up plotted in Whisper's space.
```

Both models produce 1024-dimensional vectors. The shapes match. The projection
runs. The plot renders. The points are in a space belonging to a model the user
did not select.

Nothing in the visualisation can reveal this. It is only catchable by getting
the routing right, which is why the fix asks the registry rather than parsing
the name.

---

## 12.11 Summary

- An embedding is the model's internal representation of a clip: mean-pooled
  encoder hidden states, one vector per clip.
- PCA is linear, deterministic, interpretable, and preserves global structure —
  distances mean something.
- t-SNE preserves local neighbourhoods only. **Inter-cluster distances and
  cluster sizes are meaningless**, and perplexity changes the answer.
- UMAP is the better non-linear default: faster than t-SNE and keeps more global
  structure.
- `perplexity` and `n_neighbors` must be less than the sample count;
  `min(default, n-1)` is a required guard, not politeness.
- Seed everything. Unseeded t-SNE makes two screenshots of one dataset differ,
  and a researcher will read that as a finding.
- An unsupported method raises. A test asserts PCA and t-SNE *differ*, which is
  the only way to catch a dispatch that silently falls through.
- Row order must be preserved, because the frontend binds rows to filenames by
  position. A reorder produces a perfect-looking plot with every label wrong —
  tested by planting a detectable outlier last.
- NaN coordinates silently drop points from a plot rather than erroring.
- The route is untested on purpose, and the reason is stated: the reduction is
  this project's code, the embeddings are the model's.
- Colouring by label turns the panel into a real diagnostic — especially for
  spotting a model that clusters by speaker rather than by the class it claims
  to predict.

Next: [Chapter 13 — Evaluation](13-evaluation.md).
