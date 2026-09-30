# Chapter 15 — The frontend

~17,600 lines of TypeScript and React. This chapter covers the architecture,
the canvas layering that draws the attribution overlay, shared playback state,
and the WebSocket handling — the parts you would have to get right to rebuild
it.

---

## 15.1 The stack

| Technology | Role |
|---|---|
| **React 18** | UI library — components, state, effects |
| **TypeScript** | static types over JavaScript |
| **Vite** | dev server and bundler |
| **Tailwind CSS** | utility classes for styling |
| **shadcn/ui** | accessible component primitives on Radix UI |
| **wavesurfer.js** | waveform rendering and audio playback |
| **Plotly** | the embedding scatter plot |
| **Recharts** | charts in the panels |
| **HTML5 Canvas** | the spectrogram and attribution overlays |
| **Jest** | unit tests |
| **Playwright** | end-to-end tests |

```bash
npm run dev    # Vite dev server on :8080, not Vite's 5173 default
```

Port 8080 is set in `vite.config.ts` and matters because the CORS allow-list and
the Docker Compose port mapping both assume it.

---

## 15.2 Composition

```tsx
// pages/Index.tsx
import { MainLayout } from "@/components/layout/MainLayout";

const Index = () => {
  return <MainLayout />;
};

export default Index;
```

Effectively a single page. `MainLayout` (823 lines) composes the workbench:

```tsx
import { Panel, PanelGroup, PanelResizeHandle } from "react-resizable-panels";
import { Toolbar, SelectedTasks } from "./Toolbar";
import { StatusBar } from "./StatusBar";
import { GlobalTaskProgress } from "./GlobalTaskProgress";
import { EmbeddingPanel } from "../panels/EmbeddingPanel";
import { AudioDatasetPanel } from "../panels/AudioDatasetPanel";
import { DatapointEditorPanel } from "../panels/DatapointEditorPanel";
import { PredictionPanel, UnifiedTaskResult } from "../panels/PredictionPanel";
import { EmbeddingProvider } from "../../contexts/EmbeddingContext";
```

`react-resizable-panels` gives draggable splitters. This is the standard shape
for an analysis workbench — several views of one object, all visible, resizable
to taste. It is also what makes the *shared time axis* (§15.4) matter: two
panels showing the same clip must agree on where "2.3 seconds" is on screen.

`@/` is a path alias for `src/`, configured in `vite.config.ts` and mirrored in
`jest.config.cjs`. Both must be kept in step or tests fail to resolve imports
that the app resolves fine.

### The panels

| Panel | Lines | Shows |
|---|---|---|
| `PerturbationTools` | 938 | mutation controls, canvas region selection |
| `EmbeddingPanel` | 922 | latent projection |
| `PredictionPanel` | 848 | ASR/SER/ADD results, XAI method tabs |
| `AudioDatasetPanel` | 788 | corpus browser |
| `CustomDatasetManager` | 693 | user dataset upload |
| `EmbeddingPlot` | 634 | Plotly scatter |
| `AudioDataTable` | 605 | per-file table with regenerate |
| `AttentionVisualization` | 556 | attention heatmaps |
| `WaveformViewer` | 451 | wavesurfer + region drag |
| `DatapointEditorPanel` | 423 | per-sample metadata |
| `SaliencyVisualization` | 266 | saliency series |
| `XAIOverlayCanvas` | 261 | **the layered canvas** |
| `DeepfakeForensicPanel` | 195 | confidence timeline |
| `FaithfulnessAuditPanel` | 190 | deletion scores |
| `AcousticProfilePanel` | 185 | pitch and energy |
| `AccentBiasPanel` | 148 | cohort disparity |

---

## 15.3 One place for the backend address

```ts
// src/lib/api.ts
export const API_BASE: string = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
```

One line, one module. `import.meta.env` is Vite's build-time environment
substitution — `VITE_API_BASE_URL` is baked into the bundle at build time,
which is why the Dockerfile passes it as a build argument:

```yaml
args:
  VITE_API_BASE_URL: http://127.0.0.1:8000
```

### The WebSocket origin bug

```ts
/**
 * WebSocket origin for the backend, derived from the same API_BASE every other
 * component uses. Deriving it from window.location instead pointed the socket at
 * the Vite dev server (port 8080, no proxy) rather than the API on 8000, and
 * forced ws:// on an https:// page, where the browser blocks it as mixed
 * content. http -> ws, https -> wss.
 */
const wsOrigin = (): string => {
  const base = new URL(API_BASE, window.location.origin);
  base.protocol = base.protocol === 'https:' ? 'wss:' : 'ws:';
  return base.origin;
};
```

Two bugs in one, from the same mistake.

`window.location` is where the *page* came from — the Vite dev server on 8080.
The API is on 8000. So the socket connected to a server that has no WebSocket
route.

And the protocol: hardcoding `ws://` on a page served over `https://` is
**mixed content**, which browsers block outright. The mapping must be
`http → ws`, `https → wss`.

Using `new URL(API_BASE, window.location.origin)` parses `API_BASE` properly
(handling a relative value by resolving it against the page origin) and then
swaps only the protocol. Deriving from the one source of truth rather than from
a second, plausible-looking one.

---

## 15.4 Shared playback state

```tsx
/**
 * FR10.2 — one playhead, shared by the player, the acoustic profiler and the
 * XAI overlay.
 *
 * The profiler's stated purpose is relating what a model attends to against the
 * physical signal. Without a shared time cursor that comparison is done by eye
 * across two independent axes, which is precisely the manual alignment the pane
 * exists to remove.
 *
 * `WaveformViewer` owns the wavesurfer instance and publishes here; every other
 * time-aligned view subscribes. Seeking is registered by the owner so a click on
 * any chart can drive the audio, not only the other way round.
 */
interface PlaybackState {
  currentTime: number;
  duration: number;
  /** Seek the audio. No-op until a player registers a handler. */
  seek: (seconds: number) => void;
  publish: (currentTime: number, duration: number) => void;
  registerSeek: (handler: ((seconds: number) => void) | null) => void;
}
```

The docstring argues for the feature rather than describing the code, and the
argument is right: a workbench whose panels each have their own independent
playhead has not removed the manual alignment work — it has multiplied it.

### The registration pattern

One component owns the audio; everyone else subscribes. But seeking has to work
in both directions — clicking the spectrogram should move the audio.

```tsx
export const PlaybackProvider = ({ children }: { children: ReactNode }) => {
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const seekRef = useRef<((seconds: number) => void) | null>(null);

  const publish = useCallback((t: number, d: number) => {
    setCurrentTime(t);
    if (d && Number.isFinite(d)) setDuration(d);
  }, []);
```

`seekRef` is a **ref**, not state. The seek handler is a function, not
displayed data — storing it in state would trigger a re-render of every
subscriber whenever the owner re-registered. A ref holds a mutable value across
renders without causing them.

`Number.isFinite(d)` guards against `NaN` duration, which wavesurfer reports
briefly before metadata loads. Writing `NaN` into `duration` would make every
consumer's `currentTime / duration` calculation produce `NaN` and every
percentage position vanish.

The owner registers on creation:

```tsx
wavesurferRef.current = wavesurfer;
registerSeek((seconds: number) => {
  const total = wavesurfer.getDuration();
  if (total > 0) wavesurfer.seekTo(Math.min(1, Math.max(0, seconds / total)));
});
```

`seekTo` takes a fraction in [0, 1], so the handler converts from seconds and
clamps. `if (total > 0)` avoids dividing by zero before the audio has loaded.

```tsx
// FR10.2: this component owns the wavesurfer instance, so it is the single
// source of playback time.
const { publish, registerSeek } = usePlayback();
```

**Single source of truth, explicitly stated.** With two components able to
publish time, they would fight, and the playhead would jitter between two
slightly different values.

---

## 15.5 The layered canvas

`XAIOverlayCanvas` is the visual centre of the product. Six stacked `<canvas>`
elements, absolutely positioned on top of each other:

```tsx
<div className="relative bg-black rounded-lg overflow-hidden border border-border"
     style={{ width, height }}>
  <canvas ref={baseCanvasRef} width={width} height={height}
          className="absolute top-0 left-0" />
  <canvas ref={waveCanvasRef} width={width} height={height}
          className="absolute top-0 left-0 pointer-events-none" />
  {(['gradcam', 'integrated_gradients', 'lime', 'shap'] as XAIMethod[]).map((method) => (
    <canvas key={method}
            ref={(el) => (overlayRefs.current[method] = el)}
            width={width} height={height}
            className="absolute top-0 left-0 transition-opacity duration-150 ease-out"
            style={{ opacity: activeMethod === method ? overlayOpacity : 0,
                     mixBlendMode: 'screen' }} />
  ))}
  <canvas ref={f0CanvasRef} width={width} height={height}
          className="absolute top-0 left-0 pointer-events-none" />
  ...
</div>
```

| Layer | Content | Notes |
|---|---|---|
| 1 | base spectrogram | greyscale |
| 2 | waveform envelope | `pointer-events-none` |
| 3–6 | one per XAI method | opacity 0 except the active one |
| 7 | F0 contour | `pointer-events-none` |
| 8 | playhead + click target | a `div`, not a canvas |

**Why one canvas per method rather than one that redraws?** Because switching
methods becomes a CSS opacity transition — instant, GPU-composited, animated —
instead of a full re-render of a 128×3000 matrix. All four are drawn once when
the results arrive; switching costs nothing.

The cost is memory: four canvases at 800×400 is about 5 MB of pixel buffers.
For a workbench, an unambiguously good trade.

**`pointer-events-none`** on the decorative layers lets clicks fall through to
the interactive `div` underneath. Without it, the topmost canvas would swallow
every click and seeking would not work.

**`mixBlendMode: 'screen'`** makes the heatmap brighten rather than obscure the
spectrogram, so you can see both the attribution and the signal it is
attributing to. `normal` blending would hide the thing you are trying to
compare against.

### Drawing a matrix efficiently

```tsx
const imgData = ctx.createImageData(timeFrames, melBins);
for (let y = 0; y < melBins; y++) {
  for (let x = 0; x < timeFrames; x++) {
    const val = baseSpectrogram[y][x];
    const idx = (y * timeFrames + x) * 4;
    const c = Math.floor(val * 255);
    imgData.data[idx] = c;
    imgData.data[idx + 1] = c;
    imgData.data[idx + 2] = c;
    imgData.data[idx + 3] = 255;
  }
}

const tempCanvas = document.createElement('canvas');
tempCanvas.width = timeFrames;
tempCanvas.height = melBins;
tempCanvas.getContext('2d')!.putImageData(imgData, 0, 0);
ctx.drawImage(tempCanvas, 0, 0, width, height);
```

This is the performance-critical pattern and it is worth understanding fully.

**`createImageData` + direct pixel writes.** `imgData.data` is a flat
`Uint8ClampedArray` of RGBA bytes. Index arithmetic is `(y * width + x) * 4`.
Writing bytes directly is orders of magnitude faster than 384,000 `fillRect`
calls.

**Draw at native resolution, then scale.** The image data is created at the
matrix's own dimensions (3000×128), painted onto an offscreen canvas of exactly
that size, and then `drawImage` scales it to the display size (800×400). The
browser's scaling is hardware-accelerated and does the interpolation for free.

The alternative — computing which matrix cell each of 320,000 display pixels
maps to — is both slower and worse-looking, because you would be
nearest-neighbour sampling where the browser would bilinearly interpolate.

**`Uint8ClampedArray` clamps automatically.** Assigning 300 stores 255;
assigning −5 stores 0. No manual clamping needed.

### The colour ramp

```ts
// Perceptually uniform colour ramp and its luminance, used by the attribution
// overlay (FR8.4). Kept out of the component module so the colour maths can be
// unit tested directly and the component file exports only a component.

const VIRIDIS: [number, number, number][] = [
  [68, 1, 84], [71, 13, 96], ... [253, 231, 37],
];

export const getHeatmapColor = (value: number): [number, number, number] => {
  const v = Math.max(0, Math.min(1, value));
  const pos = v * (VIRIDIS.length - 1);
  const i = Math.floor(pos);
  const j = Math.min(i + 1, VIRIDIS.length - 1);
  const t = pos - i;
  return [
    VIRIDIS[i][0] + (VIRIDIS[j][0] - VIRIDIS[i][0]) * t,
    VIRIDIS[i][1] + (VIRIDIS[j][1] - VIRIDIS[i][1]) * t,
    VIRIDIS[i][2] + (VIRIDIS[j][2] - VIRIDIS[i][2]) * t,
  ];
};
```

32 control points, linearly interpolated. And the choice is a requirement, not
a preference:

```tsx
// Viridis, 32 control points, linearly interpolated (FR8.4).
//
// The previous ramp was blue->cyan->green->yellow->red - the jet family. Jet's
// luminance is non-monotonic, so it invents banding that is not in the data,
// loses ordering in greyscale print, and is not colourblind-safe. The SRS names
// "perceptually uniform, accessible" explicitly, so this is a stated
// requirement rather than a preference.
```

**Jet is genuinely harmful for scientific visualisation.** Its brightness rises
and falls as it goes through the spectrum, so a smooth gradient in the data
appears to have edges in it. Viewers see structure that does not exist.
Viridis was designed so brightness increases monotonically — it survives
greyscale printing and is readable with the common forms of colour blindness.

The module is separate from the component *so the colour maths can be unit
tested*, including the monotonicity that the requirement actually demands:

```ts
/** Relative luminance, for the monotonicity test FR8.4 actually requires. */
export const heatmapLuminance = (v: number): number => { ... };
```

You cannot test a requirement like "perceptually uniform" by looking at it. You
can test that luminance increases monotonically.

### The colourbar

```tsx
{/* Colourbar. A heatmap without a scale is not readable (FR8.4). */}
<div className="absolute bottom-2 right-2 flex items-center gap-2 rounded bg-black/60 px-2 py-1">
  <span className="text-[10px] text-white/80">low</span>
  <div className="h-2 w-24 rounded-sm"
       style={{ background: `linear-gradient(to right, ${[0, 0.25, 0.5, 0.75, 1]
         .map((v) => { const [r, g, b] = getHeatmapColor(v);
                       return `rgb(${Math.round(r)},${Math.round(g)},${Math.round(b)})`; })
         .join(', ')})` }} />
  <span className="text-[10px] text-white/80">high</span>
</div>
```

The gradient is generated **from the same `getHeatmapColor` function** that
paints the heatmap. A hand-written CSS gradient would drift from the actual
colours on any change to the ramp, and the legend would then be lying.

*"A heatmap without a scale is not readable"* — correct. Without a legend, a
viewer cannot tell whether bright means high or low.

### The pitch contour, with gaps

```tsx
// Pen up on every unvoiced frame, so the contour shows where pitch was
// actually measured rather than a continuous line across silence.
let penDown = false;
f0Data.forEach((point) => {
  if (point.freq_hz === null || point.freq_hz === undefined || !isFinite(point.freq_hz)) {
    penDown = false;
    return;
  }
  const x = mapTimeToX(point.time_ms);
  const y = mapHzToY(point.freq_hz);
  if (!penDown) {
    ctx.moveTo(x, y);
    penDown = true;
  } else {
    ctx.lineTo(x, y);
  }
});
ctx.stroke();
```

The consumer side of the NaN decision from §10.3. `moveTo` starts a new
subpath (pen up, move, pen down); `lineTo` extends the current one. Tracking
`penDown` produces separate line segments with genuine gaps.

Three null checks — `null`, `undefined`, and `!isFinite` — because the value
could arrive as JSON `null`, be missing entirely, or be a `NaN` that slipped
through. Defensive, and cheap.

```tsx
export interface F0Point {
  time_ms: number;
  // null on unvoiced frames. pYIN reports no pitch for silence and noise, and
  // drawing through those gaps would invent a pitch track the model never saw.
  freq_hz: number | null;
}
```

The type *is* `number | null`. TypeScript forces every consumer to handle the
null case — the honesty requirement encoded in the type system rather than in a
comment.

### Mel-scaled placement

```tsx
const mapHzToY = (hz: number, maxFreq = maxFreqHz) => {
  const mel = 2595 * Math.log10(1 + hz / 500);
  const maxMel = 2595 * Math.log10(1 + maxFreq / 500);
  return height - (mel / maxMel) * height;
};
```

The spectrogram's y-axis is mel-scaled, so the pitch contour must be too. The
`height - ...` inverts the axis, because canvas y increases downward while
frequency should increase upward.

```tsx
// Upper bound of the spectrogram's mel axis. librosa defaults fmax to sr/2,
// so a fixed 8000 puts the F0 line at the wrong height on any other rate.
maxFreqHz = 8000,
```

A parameter with a documented default, because librosa's `fmax` follows the
sample rate. Hardcoding 8000 would misplace the contour for any clip not at
16 kHz.

### Click to seek

```tsx
<div className="absolute inset-0 cursor-crosshair"
     onClick={(e) => {
       const total = playDuration || audioDuration;
       if (!total) return;
       const rect = e.currentTarget.getBoundingClientRect();
       seek(((e.clientX - rect.left) / rect.width) * total);
     }}>
  {(playDuration || audioDuration) > 0 && (
    <div className="absolute top-0 bottom-0 w-px bg-white/90 pointer-events-none"
         style={{ left: `${Math.min(100, (currentTime / (playDuration || audioDuration)) * 100)}%` }} />
  )}
</div>
```

`getBoundingClientRect()` gives the element's position and size *as rendered*,
so `(clientX - rect.left) / rect.width` is the fraction across it — correct even
when the canvas is scaled by CSS to a different size than its pixel dimensions.

The playhead is a 1-pixel `div` positioned by percentage, not drawn on a canvas.
Moving it is a style update the browser composites on the GPU; redrawing a
canvas every animation frame would be far more expensive.

`playDuration || audioDuration` prefers the live playback duration and falls
back to the prop, because they can briefly differ while audio loads.

---

## 15.6 Consuming the WebSocket

```ts
export const useTaskStatus = (taskId: string | null): UseTaskStatusResult => {
  const [state, setState] = useState<TaskState>('QUEUED');
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout>();
  const pollIntervalRef = useRef<NodeJS.Timeout>();
  const retryCountRef = useRef<number>(0);
  const isManualClose = useRef(false);
```

Everything that is not rendered lives in a ref. A timer handle in state would
cause a re-render every time it changed, for no visual benefit.

```ts
  useEffect(() => {
    if (!taskId) return;

    isManualClose.current = false;
    retryCountRef.current = 0;
    setState('QUEUED');
    // Clear the previous task's outcome, or it renders briefly under the new id.
    setResult(null);
    setError(null);
```

**Clearing previous state is not housekeeping — it is a visible bug fix.**
Without it, switching to a new task shows the *old* task's result for one frame
before the new one arrives. The user sees a flash of the wrong answer.

### The retry ladder

```ts
      ws.onclose = () => {
        if (isManualClose.current) return;

        retryCountRef.current += 1;
        console.warn(`[WS] Disconnected. Retry attempt: ${retryCountRef.current}`);

        if (retryCountRef.current > 3) {
          console.warn(`[WS] Max retries reached. Falling back to HTTP polling.`);
          startPolling();
        } else {
          const delay = Math.pow(2, retryCountRef.current) * 1000;
          reconnectTimeoutRef.current = setTimeout(connectWs, delay);
        }
      };
```

**Exponential backoff:** 2 s, 4 s, 8 s. Then give up on WebSockets and poll.

**`isManualClose`** distinguishes an intentional close (task finished,
component unmounting) from a network failure. Without it, closing the socket
after `SUCCESS` would trigger the reconnect logic and reopen a socket for a
finished job.

```ts
    const startPolling = () => {
      if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);

      pollIntervalRef.current = setInterval(async () => {
        try {
          const res = await fetch(`${API_BASE}/api/tasks/${taskId}/status`);
          const data = await res.json();
          setState(data.state as TaskState);

          if (data.state === 'SUCCESS' || data.state === 'FAILURE') {
            if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
            if (data.error) setError(data.error);
            if (data.result) setResult(data.result);
          }
        } catch (e) {
          console.error('[Polling] Failed to fetch status', e);
        }
      }, 2000);
    };
```

2-second polling, stopping on a terminal state. `clearInterval` at the top
prevents two intervals if `startPolling` is somehow called twice — the classic
duplicate-timer bug.

The `try/catch` inside the interval means one failed request does not kill the
poller.

### Cleanup

```ts
    return () => {
      isManualClose.current = true;
      if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current);
      if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [taskId]);
```

React runs the effect's return function on unmount **and before re-running the
effect**. So changing `taskId` cleans up the old task's socket and timers before
opening new ones.

All four cleanups matter. `isManualClose = true` first, so the `close()` below
does not trigger a reconnect. Then both timers, then the socket. Miss any one
and you leak a timer that keeps firing against a stale task id.

### Reading both event shapes

```ts
          const data = JSON.parse(event.data);
          const currentState = data.state || data.stage;
```

The gateway's initial message uses `state`; relayed worker events use `stage`
(§7.4). Accepting both is pragmatic — though it is worth noting this is a
contract inconsistency being papered over in the client rather than fixed at
the source.

---

## 15.7 Contexts

Three, and they are in **two different directories**:

```
src/context/ModelRegistryContext.tsx
src/contexts/EmbeddingContext.tsx
src/contexts/PlaybackContext.tsx
```

`context` and `contexts`. An accident that has calcified — both are imported
throughout, so renaming means touching many files, and it has not been worth
the diff. When you rebuild this, pick one.

React Context is for state that many components at different depths need. The
alternative — threading props through every intermediate component — is "prop
drilling", and for playback time (needed by four unrelated panels) it would be
unmanageable.

The rule of thumb: Context for genuinely shared, cross-cutting state; local
`useState` for everything else. `MainLayout` holds a lot of local state
(~20 `useState` calls) precisely because most of it is only needed by it and
its direct children.

---

## 15.8 Accessibility

The target is WCAG 2.1 AA, verified by axe-core in Playwright. Two fixes worth
knowing because both are non-obvious.

### Icon-only buttons

An icon-only button has no accessible name, so a screen reader announces
"button". Eighteen of them was the single largest accessibility violation.

The fix derives each name from the control's own tooltip text rather than a
generic string — so the accessible name matches what a sighted user sees on
hover, instead of inventing a second vocabulary.

### The Radix slider trap

```tsx
// Radix puts `role="slider"` on the Thumb, not the Root, so an aria-label
// passed to the Root never reaches the element that needs it.
```

`aria-label` must be on the element carrying the ARIA role. Radix's `Slider`
renders `role="slider"` on the **Thumb**, so a label on the `Root` is attached
to a `div` with no role and the slider remains unnamed.

The shared wrapper forwards `aria-label` and `aria-labelledby` to the Thumb, so
every slider in the app is fixed at once rather than per use site.

### Contrast, and alpha blending

Colour contrast is computed from **relative luminance**, and the AA threshold
for normal text is 4.5:1.

A subtlety that cost a round trip: a first attempt at 27% lightness measured
4.45:1 — just under — because the element's background was semi-transparent and
the browser blended it with what was behind. Contrast must be measured on the
*composited* colour, not the declared one. 26% passed.

### The test that passed for the wrong reason

```
Every Playwright context is a fresh profile, so the first-run dialog opened and
its modal overlay hid the workbench from the accessibility tree. The scan was
examining the dialog.
```

The accessibility suite reported a clean pass. It was scanning a modal dialog
that covered the entire application. Dismissing the dialog first is what
revealed the eighteen unnamed buttons.

**A test that passes for the wrong reason is worse than a failing test**, and
this one actively certified an inaccessible page as accessible.

---

## 15.9 Summary

- Single page; `MainLayout` composes resizable panels. Resizable panels are the
  right shape for a workbench, and make the shared time axis essential.
- `API_BASE` in one module. Deriving the WebSocket origin from
  `window.location` instead pointed the socket at the dev server *and* forced
  `ws://` on an `https://` page.
- `PlaybackContext` gives one playhead across four panels. The seek handler is
  registered by the audio owner and stored in a **ref**, not state, so
  re-registering does not re-render subscribers.
- Guard `NaN` duration, or every percentage position in the UI becomes `NaN`.
- Six stacked canvases: one per XAI method so switching is a CSS opacity
  transition rather than a re-render. `pointer-events-none` on decorative
  layers; `mixBlendMode: screen` so the heatmap brightens rather than hides the
  signal.
- Draw matrices with `createImageData` and direct RGBA byte writes at native
  resolution, then `drawImage` to scale — the browser's interpolation is free
  and better than yours.
- Viridis, not jet. Jet's non-monotonic luminance invents banding that is not in
  the data. The ramp lives in its own module so luminance monotonicity can be
  unit tested.
- The colourbar gradient is generated from the same function that paints the
  heatmap, or the legend will eventually lie.
- Pitch gaps are drawn by tracking a pen-up flag; `freq_hz: number | null` puts
  the honesty requirement in the type system.
- Mel-scaled y-mapping with `maxFreqHz` as a parameter, because librosa's
  `fmax` follows the sample rate.
- The playhead is a positioned `div`, not a canvas redraw.
- `useTaskStatus`: everything non-rendered in refs; clear previous results or
  the old answer flashes; 2/4/8-second backoff then 2-second polling;
  `isManualClose` distinguishes intentional close from failure; clean up all
  four things on unmount and on `taskId` change.
- Accessibility: icon buttons need names derived from their own tooltips; Radix
  puts `role="slider"` on the Thumb so labels must go there; contrast must be
  measured on the composited colour; and a scan that runs with a modal open is
  scanning the modal.

Next: [Chapter 16 — Testing and verification](16-testing.md).
