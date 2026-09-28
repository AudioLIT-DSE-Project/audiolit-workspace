# Individual Contribution Report

**Project:** AudioLIT — An Interactive Multimodal Explainable-AI Workbench for Speech Recognition, Emotion Analytics, and Deepfake Detection (Group 19)
**Name:** Rahim M.I.
**Student ID:** 230506M

| Section | Details |
|---|---|
| **Documentation** | • Defined the core system baseline architecture and the infrastructure sizing matrix.<br>• Modelled the global component topology and the asynchronous system architecture.<br>• Documented the asynchronous task lifecycles and worker message-broker mappings, and the cryptographic tensor-retrieval dataflows including cache-bypass logic.<br>• Produced the network-latency pipeline profiling and deployment-constraint analysis, and the audio tensor storage / DSP computational sizing study. |
| **Project Management** | • Configured the Linear workspace automation and the live project Gantt chart. |
| **Backend Foundation** | • Scaffolded the asynchronous FastAPI server, application initialisation and CORS middleware.<br>• Built the multi-part file upload streaming route.<br>• Registered the inference and task-status route groups on the application. |
| **Asynchronous Task Infrastructure** | • Deployed the RQ task broker and parallel computing infrastructure: worker scaffolding, the Redis broker hook, and per-family queues.<br>• Implemented the concurrent multi-task execution orchestrator running speech recognition, emotion recognition and deepfake detection together on one clip. |
| **Caching System** | • Designed and built the deterministic cache-by-hash Redis architecture.<br>• Implemented the SHA-256 audio payload hashing middleware that computes a content hash before the request reaches the handler.<br>• Built the multi-dimensional tensor serialisation layer for Redis, adding msgpack encoding and lz4 compression so attribution matrices and attention tensors could be stored efficiently. |
| **Real-Time Task State** | • Implemented the WebSocket and HTTP long-polling handlers for task states.<br>• Built the frontend task-status hook, the global task progress component, and the WebSocket integration in the prediction display, so long-running jobs report progress instead of blocking. |
| **Explainability Core** | • Built the PyTorch Captum saliency core.<br>• Implemented the Integrated Gradients temporal attribution formulation and the 2D spatial Grad-CAM registration hooks.<br>• Wrote the LIME/SHAP explanation translators for spectrogram frameworks and the SHAP value visual matrix coordinate transformer.<br>• Implemented 2D saliency mapping for audio analysis and extended Captum saliency and attention attribution to the emotion model's outputs. |
| **Perturbation Engine** | • Coded the backend signal mutation engine and its array manipulation helpers.<br>• Developed the NumPy-driven time-frequency slice masking and pitch exporter.<br>• Implemented the 2D spectrogram patch segmentation and perturbation matrix generator. |
| **Model Ingestion** | • Implemented the model identifier resolver, safetensors validation, and the version-pinned model cache.<br>• Built the supported-family registry for dynamic Hugging Face model ingestion. |
| **Frontend Contributions** | • Built the XAI overlay canvas component and integrated it into the prediction panel.<br>• Added the waveform data handling and the unified task-result interfaces shared across panels. |
| **Testing & Profiling** | • Wrote unit tests for the Redis cache layer, the audio hashing utilities, the queue service, the saliency service and the perturbation service.<br>• Implemented the memory profiling and API stress-test suites, and a standalone performance profiling script for CPU and GPU runs. |

### Challenges & Solutions

- **Tests could not depend on a live Redis instance**, which made the cache layer effectively untestable in CI. → Introduced an in-memory Redis fake so the cache behaviour could be asserted deterministically.
- **Large attribution tensors were expensive to cache** as plain serialised data. → Adopted msgpack encoding with lz4 compression above a size threshold, keeping small payloads uncompressed.
- **CI installed only a subset of the required packages**, so backend jobs failed on imports that worked locally. → Corrected the CI dependency installation.
- **Long-running inference blocked the interface** with no feedback. → Combined the RQ broker with WebSocket progress events and a polling fallback, so the browser follows a job rather than waiting on a request.
- **A cache mock silently accepted calls it should have rejected**, hiding an expiry bug. → Tightened the mock to match the real client's signature, including the expiration parameter.
