# The AudioLIT Handbook

A complete, from-first-principles account of what AudioLIT is, the science it
rests on, and how every part of it is built — written so that someone with
basic programming knowledge and no machine-learning background can rebuild the
whole system from scratch using only this handbook and the official
documentation of the libraries it names.

It is deliberately long. The goal is that after reading it you have no
questions left that the code itself could answer.

---

## Who this is for

You can write a function, a loop and a class in some language. You have used a
terminal. You may never have trained a model, never have touched a Fourier
transform, never have written a web backend. Everything past that is explained
here when it is first needed.

Where a topic is genuinely deep — the mathematics of Integrated Gradients, say
— the handbook explains what it computes, why it computes that, what it
guarantees, what it does *not* guarantee, and what the code actually does. It
does not prove theorems. It tells you where the proofs live.

---

## How to read it

The chapters are ordered as a dependency chain. Chapter *n* assumes chapters
*1..n-1*. If you read them in order, nothing will refer forward to something
unexplained.

| # | Chapter | What it gives you |
|---|---------|-------------------|
| 1 | [Digital audio and the machine learning behind it](01-foundations-audio-ml.md) | Sampling, spectrograms, neural networks, transformers, and the three model tasks. Start here even if you know some ML — it fixes the vocabulary the rest of the handbook uses. |
| 2 | [Explainable AI for audio: the theory](02-xai-theory.md) | What an "explanation" of a model even is, the four attribution families AudioLIT implements, why attention is not an explanation, and how you measure whether an explanation is honest. |
| 3 | [System architecture](03-architecture.md) | The five layers, why heavy work never runs on the web request, the two cache schemes, the three storage tiers, and the shape of every request. |
| 4 | [An audio file's journey: the upload path](04-upload-path.md) | Line-by-line trace from the browser's file picker to bytes on disk, including every validation, library and failure mode. |
| 5 | [The model registry: loading a model safely](05-model-registry.md) | Safetensors, version pinning, hook attachment, LRU eviction, circuit breakers and GPU fallback. |
| 6 | [Inference: ASR, SER and deepfake detection](06-inference.md) | How each of the three predictions is actually computed, code level, including the forced-language fix and the checkpoint-selection story. |
| 7 | [Orchestration: the task fabric](07-orchestration.md) | Queues, workers, GPU locks, progress pub/sub, WebSockets, fan-out/fan-in. |
| 8 | [Caching and content addressing](08-caching.md) | Both key schemes, why they coexist, every key family and its payload contract. |
| 9 | [XAI implementation: the saliency service](09-xai-implementation.md) | The code that turns a model and a clip into a heatmap, all four methods, and every honesty guard in it. |
| 10 | [The acoustic profiler](10-acoustic-profiler.md) | Pitch, energy, spectrogram — pure signal processing, no model. |
| 11 | [The mutation engine: perturbation and counterfactuals](11-perturbation.md) | Every audio transformation, how each works mathematically, and the non-destructive contract. |
| 12 | [Latent projection: seeing a model's internal space](12-latent-projection.md) | Embeddings, PCA, t-SNE, UMAP, and what a scatter plot of audio means. |
| 13 | [Evaluation: accent bias and faithfulness auditing](13-evaluation.md) | Word Error Rate from scratch, cohort disparity, deletion scores, and the fabricated-metric incident. |
| 14 | [Datasets](14-datasets.md) | Seven corpora, the loader abstraction, streaming, integrity validation, licence handling. |
| 15 | [The frontend](15-frontend.md) | React composition, the canvas layering that draws the overlay, shared playback state, WebSocket consumption. |
| 16 | [Testing and verification](16-testing.md) | How this system is tested, why, and the classes of bug that testing caught. |
| 17 | [Building it from scratch](17-build-from-scratch.md) | A staged implementation plan: what to build first, what each stage should prove, and the traps in order of when you will hit them. |
| 18 | [Testing and evaluation in practice](18-testing-and-evaluation-in-practice.md) | The full record of the verification effort: every tool, every test file, how each layer was run, the measured results, the 23 defects, the two wrong metrics, what went well and what is still open. |

---

## A note on how this handbook treats the code

Every code excerpt here was read out of the repository while writing, not
recalled. Where the handbook explains *why* something is the way it is, the
reason comes from the comment at that line, the defect log, or a measurement
recorded in the repository — not from a plausible-sounding guess.

That matters more than it sounds. A large part of this codebase's shape is
scar tissue: a checkpoint that silently returned noise, a cache key that
served one model's answer under another's name, an "explanation" that was
actually an energy map wearing a Grad-CAM label. Those stories are in the
handbook because *the bug is the lesson*. If you rebuild this system without
knowing them, you will rebuild the bugs too.

Where the handbook says a thing is unfinished, wrong, or a known compromise,
that is deliberate. A document that presents a system as flawless teaches you
nothing about engineering it.

---

## Source documents

This handbook explains the system. It does not replace the governing
documents, which remain authoritative where they disagree with anything here:

- `docs/README.md` — conventions and errata (**including E4, the functional-requirement renumbering; read it before you cite any "FR" number**)
- `docs/SAD.md` — the architecture of record
- `docs/SRS.md` — the committed requirements
- `docs/ISSUE_PLAN.md` — every issue, dependency-ordered
- `docs/evaluation/DEFECT_LOG.md` — every defect found, with how it was found
- `docs/testing/AudioLIT_Test_Plan_Report_Merged.md` — the test and evaluation report
