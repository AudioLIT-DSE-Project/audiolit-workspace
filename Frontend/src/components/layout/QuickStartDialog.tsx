import { useState } from "react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "@/components/ui/dialog";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";

// SRS §3.7: "A quick-start walkthrough covering the deepfake-detection,
// canvas-mutation, bias, and faithfulness workflows." Gates the automatic
// first-visit open only - the toolbar's "Quick start" button always reopens
// this regardless of the dismissal state.
export const QUICKSTART_DISMISSED_KEY = "audiolit.quickstart.dismissed";

// localStorage throws in a private window with storage blocked - a failed
// read/write here must not crash the app, it should just behave as if
// nothing was ever dismissed/saved.
export function readQuickStartDismissed(): boolean {
  try {
    return window.localStorage.getItem(QUICKSTART_DISMISSED_KEY) === "true";
  } catch {
    return false;
  }
}

export function writeQuickStartDismissed(value: boolean): void {
  try {
    if (value) {
      window.localStorage.setItem(QUICKSTART_DISMISSED_KEY, "true");
    } else {
      window.localStorage.removeItem(QUICKSTART_DISMISSED_KEY);
    }
  } catch {
    // Private window / storage disabled - nothing we can persist.
  }
}

interface QuickStartDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

const TRACKS = [
  {
    value: "deepfake",
    label: "Deepfake Detection",
    steps: [
      <>Set the toolbar's <strong>Model</strong> dropdown to <strong>MelodyMachine (Deepfake)</strong> or <strong>Wav2Vec2 XLSR (Deepfake)</strong>.</>,
      <>Set <strong>Dataset</strong> to <strong>ASVspoof 2021</strong>, then pick a clip from the dataset table below.</>,
      <>Open the <strong>Saliency</strong> tab — the bona-fide/spoof verdict badge and the Deepfake Forensics timeline appear there.</>,
      <>Click <strong>GRADCAM</strong>, <strong>INTEGRATED GRADIENTS</strong>, <strong>LIME</strong>, or <strong>SHAP</strong> to generate an attribution overlay for the verdict.</>,
    ],
  },
  {
    value: "mutation",
    label: "Canvas Mutation",
    steps: [
      <>Select a clip with a loaded waveform, then open the <strong>Perturbation</strong> tab.</>,
      <>Drag on the <strong>Spectrogram Region Selector</strong> to mark a time-frequency region.</>,
      <>In <strong>Apply Mutation to Selected Region</strong>, pick the region chip and choose <strong>Localized Mute</strong>, <strong>Frequency Filter Band</strong>, or <strong>Gaussian White Noise</strong>.</>,
      <>Click <strong>Apply Mutation</strong>, then compare the Original and Perturbed waveforms shown above.</>,
    ],
  },
  {
    value: "bias",
    label: "Accent Bias",
    steps: [
      <>Set the toolbar's <strong>Model</strong> dropdown to <strong>Whisper Base (ASR)</strong>.</>,
      <>Click <strong>Advanced</strong> (next to the tab bar) to reveal the <strong>Accent Bias</strong> tab, then open it.</>,
      <>Click <strong>Run Accent-Bias Diagnostic (10 samples/cohort)</strong>.</>,
      <>Read the bar chart — accent cohorts are ranked worst-to-best by mean Word Error Rate over L2-ARCTIC.</>,
    ],
  },
  {
    value: "faithfulness",
    label: "Faithfulness",
    steps: [
      <>Set the toolbar's <strong>Model</strong> dropdown to <strong>Wav2Vec2 (SER)</strong>.</>,
      <>Click <strong>Advanced</strong>, then open the <strong>Faithfulness</strong> tab.</>,
      <>Choose an attribution method (Grad-CAM, Integrated Gradients, LIME, or SHAP) and click <strong>Run Audit</strong>.</>,
      <>Read the AUDC and verdict badges — a real, measured degradation curve, not a simulated estimate.</>,
    ],
  },
];

export const QuickStartDialog = ({ open, onOpenChange }: QuickStartDialogProps) => {
  const [dontShowAgain, setDontShowAgain] = useState(() => readQuickStartDismissed());

  const handleOpenChange = (next: boolean) => {
    if (!next) writeQuickStartDismissed(dontShowAgain);
    onOpenChange(next);
  };

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogContent className="sm:max-w-xl" data-testid="quickstart-dialog">
        <DialogHeader>
          <DialogTitle>Quick start</DialogTitle>
          <DialogDescription>
            Four short walkthroughs for AudioLIT's core workflows. Reopen this anytime from the toolbar.
          </DialogDescription>
        </DialogHeader>

        <Tabs defaultValue="deepfake">
          <TabsList className="grid grid-cols-4 w-full" data-testid="quickstart-tabs">
            {TRACKS.map((track) => (
              <TabsTrigger key={track.value} value={track.value} className="text-xs" data-testid={`quickstart-tab-${track.value}`}>
                {track.label}
              </TabsTrigger>
            ))}
          </TabsList>
          {TRACKS.map((track) => (
            <TabsContent key={track.value} value={track.value} data-testid={`quickstart-panel-${track.value}`}>
              <ol className="list-decimal list-inside space-y-2 text-sm py-2">
                {track.steps.map((step, i) => (
                  <li key={i}>{step}</li>
                ))}
              </ol>
            </TabsContent>
          ))}
        </Tabs>

        <DialogFooter className="items-center sm:justify-between">
          <div className="flex items-center gap-2">
            <Checkbox
              id="quickstart-dont-show-again"
              checked={dontShowAgain}
              onCheckedChange={(checked) => setDontShowAgain(checked === true)}
              data-testid="quickstart-dont-show-again"
            />
            <label htmlFor="quickstart-dont-show-again" className="text-xs text-muted-foreground">
              Don't show again
            </label>
          </div>
          <Button size="sm" onClick={() => handleOpenChange(false)} data-testid="quickstart-close">
            Got it
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};
