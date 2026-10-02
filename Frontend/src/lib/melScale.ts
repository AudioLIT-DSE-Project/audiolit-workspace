// The mel scale every spectrogram in the app is drawn on.
//
// The backend builds its spectrograms with librosa's defaults, which means the
// Slaney mel scale: linear below 1 kHz, logarithmic above. A canvas that maps
// pixels to hertz has to use the same curve, or a pitch line, a gridline or a
// drawn selection lands at a different frequency from the one the image shows.
// The canvases used to carry their own formula (`2595 * log10(1 + hz / 500)`),
// which is neither Slaney nor HTK, so a region drawn over a formant resolved to
// the wrong band before it was sent to the mutation endpoint.

const LINEAR_HZ_PER_MEL = 200 / 3;
const LOG_REGION_START_HZ = 1000;
const LOG_REGION_START_MEL = LOG_REGION_START_HZ / LINEAR_HZ_PER_MEL;
const LOG_STEP = Math.log(6.4) / 27;

export const hzToMel = (hz: number): number =>
  hz < LOG_REGION_START_HZ
    ? hz / LINEAR_HZ_PER_MEL
    : LOG_REGION_START_MEL + Math.log(hz / LOG_REGION_START_HZ) / LOG_STEP;

export const melToHz = (mel: number): number =>
  mel < LOG_REGION_START_MEL
    ? mel * LINEAR_HZ_PER_MEL
    : LOG_REGION_START_HZ * Math.exp(LOG_STEP * (mel - LOG_REGION_START_MEL));

const clamp01 = (value: number) => Math.min(Math.max(value, 0), 1);

/** Where `hz` sits on a mel axis running 0..maxHz: 0 is the bottom, 1 the top. */
export const hzToAxisFraction = (hz: number, maxHz: number): number => {
  const maxMel = hzToMel(maxHz);
  return maxMel > 0 ? clamp01(hzToMel(Math.max(0, hz)) / maxMel) : 0;
};

/** Inverse of hzToAxisFraction. */
export const axisFractionToHz = (fraction: number, maxHz: number): number =>
  melToHz(clamp01(fraction) * hzToMel(maxHz));
