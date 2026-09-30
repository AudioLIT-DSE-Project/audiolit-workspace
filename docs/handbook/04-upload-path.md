# Chapter 4 — An audio file's journey: the upload path

This chapter traces one clip from the moment a user picks it (or speaks it)
to the moment it is bytes on disk with a known identity, naming every library
and every check along the way.

It is the best chapter to read carefully if you want to understand this
codebase's style, because the upload route is short enough to hold in your head
and yet contains six distinct validations, three failure modes and two
retention mechanisms.

---

## 4.1 The route map

```
POST   /upload                     upload a clip
DELETE /upload/{file_id}           delete one
GET    /upload/file/{file_id}      serve it back for playback
HEAD   /upload/file/{file_id}      same, headers only
GET    /upload/metadata/{file_id}  duration, rate, size, channels
GET    /upload/list                everything currently held
GET    /upload/test                liveness probe
```

All in `Backend/app/api/routes/upload.py`, mounted without a prefix:

```python
app.include_router(upload_routes.router, tags=["Upload"])
```

---

## 4.2 The browser side: two ways in

### Path A — the file picker

```tsx
accept="audio/*,.wav,.mp3,.flac,.m4a,.ogg,.webm"
```

The `accept` attribute filters the OS dialog. It is a **convenience, not a
control** — a user can choose "all files" in most dialogs, and a scripted
client ignores it entirely. Client-side validation is for user experience;
server-side validation is for correctness. Both exist here, and they exist for
different reasons.

There is a client-side guard too:

```tsx
setErrorMessage("Please select a valid audio file (.wav, .mp3, .flac, .m4a, .ogg).");
```

This gives instant feedback without a round trip. It does not replace the
server check.

### Path B — recording in the browser

```tsx
const stream = await navigator.mediaDevices.getUserMedia({ audio: { ... } });
```

`getUserMedia` prompts for microphone permission and returns a live
`MediaStream`. Then `MediaRecorder` encodes it — but **which format you get
depends on the browser**:

```tsx
let mimeType = "audio/webm";
if (!MediaRecorder.isTypeSupported(mimeType)) {
  if (MediaRecorder.isTypeSupported("audio/mp4")) mimeType = "audio/mp4";
  else if (MediaRecorder.isTypeSupported("audio/ogg")) mimeType = "audio/ogg";
}
const recorder = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream);
```

Chrome and Firefox produce WebM/Opus. Safari produces MP4/AAC. Neither is a
format `soundfile` can read, and the sample rate is whatever the device
happened to use — 44,100 or 48,000 Hz, not 16,000.

So the frontend converts before uploading. This is the most interesting single
function in the frontend, because it hand-writes a WAV file:

```tsx
/**
 * Convert any browser recorded audio Blob into a standardized 16kHz PCM 16-bit WAV File.
 * Guarantees 100% backend compatibility with soundfile/librosa across Windows, Linux & macOS.
 */
async function blobToWavFile(rawBlob: Blob, filename: string): Promise<File> {
  try {
    const arrayBuffer = await rawBlob.arrayBuffer();
    const AudioCtx = window.AudioContext || (window as any).webkitAudioContext;
    const audioCtx = new AudioCtx();
    const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
```

**Step 1 — decode.** `decodeAudioData` uses the browser's own codecs to turn
WebM/Opus (or whatever arrived) into raw float samples. This is why the
frontend can handle formats it has no decoder for: the browser already has one.

```tsx
    const targetSampleRate = 16000;
    const numOfChannels = 1;
    const offlineCtx = new OfflineAudioContext(
      numOfChannels,
      Math.ceil(audioBuffer.duration * targetSampleRate),
      targetSampleRate
    );

    const source = offlineCtx.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(offlineCtx.destination);
    source.start(0);

    const renderedBuffer = await offlineCtx.startRendering();
```

**Step 2 — resample and downmix.** `OfflineAudioContext` renders audio as fast
as the CPU allows rather than in real time. Constructing it with
`numOfChannels = 1` and `sampleRate = 16000` makes the browser's own audio
engine do the channel downmix and the resampling — including the anti-alias
filtering that §1.1 said was mandatory. You get a correct resample for free
rather than writing one.

```tsx
    const pcmData = renderedBuffer.getChannelData(0);
    const wavBuffer = new ArrayBuffer(44 + pcmData.length * 2);
    const view = new DataView(wavBuffer);

    const writeString = (offset: number, str: string) => {
      for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
    };

    writeString(0, "RIFF");
    view.setUint32(4, 36 + pcmData.length * 2, true);
    writeString(8, "WAVE");
    writeString(12, "fmt ");
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);                      // format = 1 (PCM)
    view.setUint16(22, 1, true);                      // channels = 1
    view.setUint32(24, targetSampleRate, true);       // sample rate
    view.setUint32(28, targetSampleRate * 2, true);   // byte rate
    view.setUint16(32, 2, true);                      // block align
    view.setUint16(34, 16, true);                     // bits per sample
    writeString(36, "data");
    view.setUint32(40, pcmData.length * 2, true);
```

**Step 3 — the WAV header, by hand.** This is a complete 44-byte canonical WAV
header. Worth knowing the layout, because it is the simplest real audio format
and understanding it demystifies the rest:

| Offset | Size | Content |
|---|---|---|
| 0 | 4 | `"RIFF"` |
| 4 | 4 | file size − 8 |
| 8 | 4 | `"WAVE"` |
| 12 | 4 | `"fmt "` (note the trailing space) |
| 16 | 4 | length of the fmt block = 16 |
| 20 | 2 | audio format, 1 = uncompressed PCM |
| 22 | 2 | number of channels |
| 24 | 4 | sample rate |
| 28 | 4 | byte rate = rate × channels × bytes-per-sample |
| 32 | 2 | block align = channels × bytes-per-sample |
| 34 | 2 | bits per sample |
| 36 | 4 | `"data"` |
| 40 | 4 | size of the sample data |
| 44 | … | the samples |

The `true` in every `setUint32(..., true)` means **little-endian**, which RIFF
requires. Pass `false` and you produce a file nothing can read.

```tsx
    let offset = 44;
    for (let i = 0; i < pcmData.length; i++) {
      const s = Math.max(-1, Math.min(1, pcmData[i]));
      view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
      offset += 2;
    }
```

**Step 4 — quantise.** Float samples in [−1, 1] become signed 16-bit integers.
The clamp prevents overflow from any sample slightly outside the range. And the
asymmetry is correct, not a typo: signed 16-bit spans −32768 to +32767, so
negative values scale by `0x8000` (32768) and positive by `0x7fff` (32767).
Using 32768 for both would wrap the loudest positive sample around to the most
negative value — an audible click.

```tsx
  } catch (err) {
    console.warn("WAV encoding fallback to raw blob:", err);
    return new File([rawBlob], filename, { type: rawBlob.type || "audio/webm" });
  }
```

**Step 5 — the fallback.** If any of this fails, send the raw blob and let the
backend try. Better a chance of working than a guaranteed failure.

### The actual POST

```tsx
const formData = new FormData();
// ... append file and model ...
const response = await fetch(`${API_BASE}/upload`, {
  method: "POST",
  body: formData,
  // credentials included so the session cookie travels
});
```

`FormData` produces a `multipart/form-data` body, the encoding for file
uploads. `API_BASE` comes from the one shared place:

```ts
export const API_BASE: string = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
```

One module, one source of the backend address. Chapter 15 explains why
deriving it from `window.location` instead caused a real bug.

---

## 4.3 The server side, line by line

```python
@router.post("/upload")
async def upload_audio_file(file: UploadFile = File(...), model: str = Form("whisper-base")):
    """
    Upload an audio file and return the file path for processing
    """
    purge_expired_uploads()
```

`UploadFile` is FastAPI's streaming file type — it exposes `async read()` and
spools to a temporary file rather than holding everything in memory. `File(...)`
marks it required. `Form("whisper-base")` reads `model` from the multipart body
with a default.

The very first statement is the retention sweep (§4.6). Every upload cleans up
after previous ones.

### Check 1 — content type

```python
    if file.content_type and not (file.content_type.startswith('audio/')
                                  or file.content_type in ['video/webm', 'application/octet-stream']):
        raise HTTPException(status_code=400, detail="Invalid file type. Only audio files are allowed.")
```

Three things going on:

- `video/webm` is allowed because browser `MediaRecorder` output is frequently
  labelled that way even when it contains only audio. Rejecting it would break
  recording on Chrome.
- `application/octet-stream` is allowed because that is what a `Blob` with no
  declared type becomes.
- The whole check is skipped when `content_type` is falsy — `if file.content_type and ...`.

That last point is the honest reading: **this check is advisory.** The
`Content-Type` header is supplied by the client and can say anything. It filters
obvious mistakes; it is not a security boundary. The real gate is Check 4.

### Check 2 — extension

```python
    allowed_extensions = ['.wav', '.mp3', '.m4a', '.flac', '.webm', '.ogg', '.aac', '.opus']
    file_extension = Path(file.filename).suffix.lower() if file.filename else ''
    if not file_extension or file_extension not in allowed_extensions:
        # Default fallback for blob uploads without explicit extension
        file_extension = '.webm' if 'webm' in (file.content_type or '') else '.wav'
```

Note that an unrecognised extension does **not** reject — it *assigns* one.
That is a deliberate tolerance for blob uploads that legitimately have no
filename. Again: this is normalisation, not validation.

### Storage — a UUID name

```python
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = UPLOAD_DIR / unique_filename
```

The original filename is **never used as a path component**. This is the single
most important line in the function from a security standpoint.

A filename is attacker-controlled. `../../etc/passwd`, `..\\..\\windows\\system32\\x`,
a 4000-character name, an embedded NUL, a name that collides with an existing
file, a name differing only by case on a case-insensitive filesystem — all are
path-traversal or collision vectors. Generating the name yourself eliminates the
entire category. `uuid.uuid4()` is random enough that collisions are not a
practical concern.

The original name is still returned to the client for display:

```python
"filename": file.filename or unique_filename,
"file_id": unique_filename,
```

Two separate fields: one for humans, one for the system. They are never
conflated.

### Check 3 — size, enforced while streaming

```python
        bytes_written = 0
        with open(file_path, "wb") as buffer:
            while True:
                chunk = await file.read(UPLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                bytes_written += len(chunk)
                if bytes_written > MAX_UPLOAD_SIZE_BYTES:
                    buffer.close()
                    file_path.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds maximum upload size of {MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)}MB.",
                    )
                buffer.write(chunk)
```

The comment above the constant explains what this replaced:

```python
# previously unbounded - shutil.copyfileobj wrote a request body to
# disk in full before anything checked its size, so a multi-GB upload was
# fully buffered before being rejected (if ever). Enforced by counting bytes
# during the streamed write below rather than trusting a Content-Length
# header, which can be absent or spoofed.
```

Two separate lessons, and both generalise:

**Enforce a limit while consuming, not after.** `shutil.copyfileobj` copies
everything, then you check. A 10 GB upload is already 10 GB on disk by the time
you object. Counting as you go aborts at 100 MB and one chunk.

**Do not trust `Content-Length`.** It is a client-supplied header. It can be
absent (chunked transfer encoding) or simply wrong. The only reliable byte count
is the one you make yourself.

Also note the cleanup on rejection: `buffer.close()` then
`file_path.unlink(missing_ok=True)`. A rejected upload leaves nothing behind.
`missing_ok=True` avoids a second exception if the file is somehow already
gone — you do not want your error path to raise a different error.

Chunk size is 1 MiB:

```python
UPLOAD_CHUNK_SIZE = 1024 * 1024
```

Big enough that syscall overhead is negligible, small enough that memory stays
bounded regardless of upload size.

### Check 4 — decode, which is the real validation

```python
        try:
            try:
                audio_data, sample_rate = sf.read(file_path)
                duration = float(len(audio_data)) / float(sample_rate) if sample_rate > 0 else 0.0
            except Exception:
                audio_data, sample_rate = librosa.load(file_path, sr=None)
                duration = float(librosa.get_duration(y=audio_data, sr=sample_rate))
            file_size = file_path.stat().st_size
        except Exception as decode_err:
            logger.error(f"Audio decoding failure on {file_path}: {decode_err}")
            file_path.unlink(missing_ok=True)
            raise HTTPException(
                status_code=422,
                detail=f"File could not be decoded as audio. It may be corrupted or in an unsupported format.",
            )
```

**This is where a file is actually proved to be audio.** Not the header, not
the extension — the bytes either decode or they do not.

Two libraries, in order:

- **`soundfile`** (a binding to libsndfile) — fast, handles WAV, FLAC, OGG and
  most uncompressed and lossless formats. The project's sanctioned audio I/O
  library.
- **`librosa.load`** as fallback — slower, but reaches `audioread`/ffmpeg and
  therefore handles MP3, M4A, WebM and anything else installed on the system.

`sr=None` on the librosa call means "keep the file's native rate", because at
this point we want to *report* the true rate, not resample. Resampling happens
later, per model, in the inference path.

This is also the check that satisfies the "magic number and structural
validation" requirement — in a stronger form than the requirement asked for.
A magic-number check reads the first few bytes and confirms they look like a
WAV header. A full decode catches a valid header in front of a truncated or
corrupt body, which a magic-number check passes happily.

> **This exact handler had a live bug, found while writing this handbook.**
> `logger.error(...)` was called in a module where `logger` was never defined.
> The `NameError` propagated to the outer `except Exception`, which converted it
> into `500 "Failed to upload file: name 'logger' is not defined"`. So every
> undecodable upload reported an internal server error, the intended 422 branch
> was unreachable, and the real reason was hidden.
>
> It was found by sending a `.wav`-named file containing `b"not audio at all"`
> — something no existing test did, because the upload route had no tests at
> all. The fix is one import and one module-level `logger`. The test that now
> guards it is in `Backend/tests/test_upload_limits.py::TestDecodeRejection`.
>
> The lesson is not "define your loggers". It is that **an error path is code**,
> and code that is never executed is code that has never been tested. The happy
> path here was exercised constantly; the error path had never run once.

### Check 5 — duration

```python
        if MAX_UPLOAD_DURATION_SECONDS > 0 and duration > MAX_UPLOAD_DURATION_SECONDS:
            file_path.unlink(missing_ok=True)
            raise HTTPException(
                status_code=413,
                detail=(
                    f"Audio is {duration / 60:.1f} minutes long; the maximum is "
                    f"{MAX_UPLOAD_DURATION_SECONDS / 60:.0f} minutes."
                ),
            )
```

This check was **missing for most of the project's life**, and the reason it
went unnoticed is instructive. The duration was computed, returned in the
response, and displayed in the UI. It looked handled. Nothing ever compared it
to anything.

Why it matters: the size cap and the duration cap are not redundant. A 40-minute
8 kHz mono WAV is about 38 MB — comfortably under the 100 MB limit. It passed,
and was then fanned out to five worker families each holding the fully decoded
array. Two different resources, two different limits.

`> 0` makes `0` mean "no limit", so an operator running a long-clip demo can
disable it rather than having to pick an absurdly large number.

### Response

```python
        return JSONResponse(
            status_code=200,
            content={
                "message": "File uploaded successfully",
                "filename": file.filename or unique_filename,
                "file_path": str(file_path),
                "file_id": unique_filename,
                "duration": duration,
                "sample_rate": sample_rate,
                "size": file_size,
                "prediction": None
            }
        )
```

`prediction: None` is a placeholder for a shape the frontend expects; upload
does not predict.

### Exception ordering

```python
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")
```

`except HTTPException: raise` **must come first**. Without it, the deliberate
413s and 422s raised above would be caught by the generic handler and
re-wrapped as 500s, destroying both the status code and the message. Ordering
matters in Python's `except` chain: first match wins.

This is also exactly what made the `logger` bug so confusing — a real
`NameError` from the error path got wrapped into the generic 500, and the
symptom ("upload returns 500") pointed nowhere near the cause.

---

## 4.4 Serving audio back

```python
@router.get("/upload/file/{file_id}")
@router.head("/upload/file/{file_id}")
async def serve_audio_file(file_id: str):
    file_path = UPLOAD_DIR / file_id
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    file_extension = file_path.suffix.lower()
    media_type_map = {
        '.wav': 'audio/wav', '.mp3': 'audio/mpeg',
        '.m4a': 'audio/mp4', '.flac': 'audio/flac'
    }
    media_type = media_type_map.get(file_extension, 'audio/*')

    return FileResponse(
        path=file_path,
        media_type=media_type,
        headers={
            'Accept-Ranges': 'bytes',
            'Cache-Control': 'public, max-age=3600',
            'Content-Disposition': f'inline; filename="{file_id}"'
        }
    )
```

Three headers, three reasons:

- **`Accept-Ranges: bytes`** — tells the browser it may request byte ranges.
  Without it, seeking in an `<audio>` element requires downloading the whole
  file first. With it, dragging the playhead to 8 seconds fetches only what is
  needed. FastAPI's `FileResponse` handles the `Range` requests themselves.
- **`Cache-Control: public, max-age=3600`** — the browser may reuse it for an
  hour. Files are immutable (UUID names are never reused), so this is safe and
  removes repeated transfers during a session.
- **`Content-Disposition: inline`** — play it, do not download it. `attachment`
  would trigger a save dialog.

`HEAD` shares the handler. A HEAD request returns headers with no body, which is
how the frontend checks existence and size cheaply.

> **Both `GET` decorators stack on one function.** FastAPI allows this. It is
> how you register two methods without duplicating the handler.

---

## 4.5 Metadata and listing

```python
@router.get("/upload/metadata/{file_id}")
async def get_audio_metadata(file_id: str):
    file_path = UPLOAD_DIR / file_id
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    try:
        audio_data, sample_rate = librosa.load(file_path, sr=None)
        duration = librosa.get_duration(y=audio_data, sr=sample_rate)
        file_size = file_path.stat().st_size
        return JSONResponse(status_code=200, content={
            "file_id": file_id,
            "duration": duration,
            "sample_rate": sample_rate,
            "size": file_size,
            "channels": 1 if len(audio_data.shape) == 1 else audio_data.shape[0]
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read audio metadata: {str(e)}")
```

The channel count is inferred from the array's dimensionality: a 1-D array is
mono, a 2-D array's first axis is channels. That is librosa's convention
(`channels, samples`), which is the transpose of soundfile's. Getting these two
conventions backwards is a common and quiet bug — you end up treating 16,000
samples as 16,000 channels.

`/upload/list` walks the directory and reports `file_id`, size and creation
time.

---

## 4.6 Retention: how audio stops existing

The requirement is that uploaded audio is transient, purged on a configurable
TTL, with only analysis records kept permanently. That requirement was
documented, asserted in a code comment elsewhere, and **not implemented**. Only
an explicit `DELETE` removed anything, which the browser had to remember to
send — so every closed tab and failed request left a clip on disk forever.

```python
# SR4 / constraint C4 - uploaded audio is transient. Only an explicit
# DELETE /upload/{file_id} removed it before, so anything the browser never
# deleted (a closed tab, a failed request) stayed on disk indefinitely. The
# sweep below runs on upload and at startup; set to 0 to keep files.
UPLOAD_RETENTION_SECONDS = float(os.getenv("AUDIOLIT_UPLOAD_RETENTION_SECONDS", str(24 * 60 * 60)))


def purge_expired_uploads(retention_seconds: float | None = None) -> int:
    """Delete uploads older than the retention window. Returns the count.

    Best-effort by design: a file another request is mid-read on may fail to
    unlink on Windows, and that must not fail the upload that triggered the
    sweep.
    """
    window = UPLOAD_RETENTION_SECONDS if retention_seconds is None else retention_seconds
    if window <= 0:
        return 0
    cutoff = time.time() - window
    removed = 0
    try:
        entries = list(UPLOAD_DIR.iterdir())
    except OSError:
        return 0
    for entry in entries:
        try:
            if entry.is_file() and entry.stat().st_mtime < cutoff:
                entry.unlink()
                removed += 1
        except OSError:
            continue
    return removed
```

Design notes, each of which is a decision you would have to make yourself:

- **`window <= 0` disables it.** Zero means keep everything, not delete
  everything. Getting this polarity wrong would be catastrophic, so it is
  tested explicitly (`test_a_zero_window_keeps_everything`).
- **`entries = list(...)` before iterating.** You are deleting from the
  directory you are walking. Materialising the listing first avoids
  undefined behaviour.
- **Per-file `try/except OSError: continue`.** On Windows, unlinking a file
  another process has open raises. This sweep runs at the top of an upload
  request, and one locked file must not fail that upload. Best-effort is the
  correct semantics here, and the docstring says so rather than leaving a
  reader to guess whether the swallow is a bug.
- **`entry.is_file()`** — never touch directories.
- **Returns a count** so the caller can log something meaningful.

Two triggers:

```python
# in the route handler
purge_expired_uploads()
```

```python
# in main.py
@app.on_event("startup")
async def _purge_expired_uploads() -> None:
    """SR4 / constraint C4 - the per-upload sweep only runs when someone
    uploads, so a server that sat idle past the retention window would still be
    holding audio on the next boot. This clears it before serving a request."""
    try:
        removed = upload_routes.purge_expired_uploads()
        if removed:
            logger.info("Purged %d upload(s) past the retention window", removed)
    except Exception:
        logger.warning("Could not purge expired uploads at startup", exc_info=True)
```

Why both? The per-upload sweep is free (it runs during a request that is
already doing I/O) but only fires when someone uploads. A server that sits idle
for a week would still be holding week-old audio. The startup sweep covers
that. Together they approximate a scheduled job without needing a scheduler —
a deliberate simplification, and the ceiling is stated: a server that neither
restarts nor receives uploads will hold files past the window. For a
single-host academic deployment that is acceptable; for a long-running service
you would add a periodic task.

Note the startup hook never raises. A failing sweep must not prevent the
application from starting.

---

## 4.7 From reference to path

Later requests do not carry the file again. They carry a *reference*, and
something has to turn that into a path. Three forms are accepted:

```python
def resolve_audio_reference(file_path=None, dataset=None, dataset_file=None, session_id=None) -> Path:
    """Resolve an audio reference into an absolute Path on disk.

    Supports:
    1. explicit dataset + dataset_file
    2. relative file_path with embedded dataset directory (e.g. 'cv-valid-dev/sample-000775.mp3')
    3. direct file_path (absolute or relative to CWD)
    """
    if dataset and dataset_file:
        try:
            return resolve_file(dataset, dataset_file, session_id)
        except (FileNotFoundError, ValueError):
            pass

    if file_path:
        p = Path(file_path)
        if p.is_absolute() and p.exists():
            return p
        data_resolved = DATA_DIR / p
        if data_resolved.exists():
            return data_resolved.resolve()
        parts = p.parts
        # ... dataset-prefix handling ...
```

An uploaded clip and a dataset sample are both "audio the user selected", and
every downstream route — saliency, acoustic, inference — needs to accept
either without caring which. One resolver, used by all of them.

The `session_id` parameter exists because users can upload their own custom
datasets, which are scoped per session so two users' uploads do not collide.

---

## 4.8 Complete trace

A user records five seconds of speech in Chrome:

1. `getUserMedia` prompts for the microphone; the user allows it.
2. `MediaRecorder` records WebM/Opus at the device rate (say 48 kHz stereo).
3. `blobToWavFile` decodes it with `decodeAudioData`, renders it through an
   `OfflineAudioContext` at 1 channel / 16 kHz, and writes a 44-byte WAV header
   plus 16-bit PCM samples by hand. Result: a ~160 KB `.wav` File object.
4. `FormData` + `fetch` POST it to `/upload` with the session cookie.
5. The route sweeps expired uploads.
6. Content type `audio/wav` passes; extension `.wav` is on the allow-list.
7. A UUID filename is generated: `3f2a....wav`.
8. The body streams to `uploads/3f2a....wav` in 1 MiB chunks; 160 KB is well
   under the 100 MB cap.
9. `soundfile.read` decodes it; duration = 5.0 s, rate = 16000.
10. 5 s is under the 900 s cap.
11. A JSON response carries `file_id`, duration, rate and size.
12. The frontend stores `file_id`, renders the waveform, and uses it as the
    reference for every subsequent request — inference, saliency, acoustic
    profile, perturbation.

Nine validations and normalisations, in an endpoint that is about 90 lines of
real code.

---

## 4.9 Summary

- The browser converts anything it records to 16 kHz mono 16-bit WAV before
  uploading, using `decodeAudioData` + `OfflineAudioContext` for a correct
  resample and a hand-written RIFF header.
- Client-side checks are for user experience; server-side checks are for
  correctness. Both exist, neither replaces the other.
- Content type and extension are **advisory** — client-supplied and
  normalising. The decode is the real validation, and it is stronger than the
  magic-number check the requirement asked for.
- Filenames are always regenerated as UUIDs. The user's name is returned for
  display only, never used as a path component.
- Size is enforced *while* streaming, counting bytes rather than trusting
  `Content-Length`.
- Duration and size are independent limits; a long quiet clip passes the size
  cap easily.
- `except HTTPException: raise` must precede the generic handler, or deliberate
  4xx responses become 5xx.
- Error paths are code. The undecodable-upload branch here contained a
  `NameError` for a long time because nothing ever executed it.
- Retention is swept on every upload and at startup; `0` means keep. The sweep
  is best-effort per file so a locked file cannot fail an upload.
- One resolver turns uploads, dataset samples and session datasets into paths,
  so no downstream route needs to know which it got.

Next: [Chapter 5 — The model registry](05-model-registry.md).
