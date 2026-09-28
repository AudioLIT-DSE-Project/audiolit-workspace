// Perceptually uniform colour ramp and its luminance, used by the attribution
// overlay (FR8.4). Kept out of the component module so the colour maths can be
// unit tested directly and the component file exports only a component.

const VIRIDIS: [number, number, number][] = [
  [68, 1, 84], [71, 13, 96], [72, 24, 106], [72, 35, 116], [71, 45, 123],
  [69, 55, 129], [66, 64, 134], [62, 73, 137], [59, 82, 139], [55, 91, 141],
  [51, 99, 141], [47, 107, 142], [44, 114, 142], [41, 122, 142], [38, 130, 142],
  [35, 137, 142], [33, 145, 140], [31, 152, 139], [31, 160, 136], [34, 167, 133],
  [40, 174, 128], [51, 182, 122], [64, 189, 114], [80, 196, 105], [99, 203, 95],
  [119, 209, 83], [141, 215, 68], [164, 220, 53], [187, 225, 39], [210, 229, 34],
  [232, 233, 39], [253, 231, 37],
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

/** Relative luminance, for the monotonicity test FR8.4 actually requires. */
export const heatmapLuminance = (v: number): number => {
  const [r, g, b] = getHeatmapColor(v);
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
};
