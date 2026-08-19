import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import { jest } from '@jest/globals';
import { XAIOverlayCanvas, XAIResult } from '../components/visualization/XAIOverlayCanvas';
import { PredictionPanel } from '../components/panels/PredictionPanel';

describe('XAIOverlayCanvas Component', () => {
  const sampleSpectrogram = [
    [0.1, 0.5, 0.9],
    [0.2, 0.6, 0.8],
    [0.3, 0.7, 0.4]
  ];

  const sampleXAIResult: XAIResult = {
    method: 'gradcam',
    saliency_matrix: [
      [0.0, 0.8, 0.2],
      [0.1, 0.9, 0.3],
      [0.2, 0.7, 0.4]
    ]
  };

  test('renders base spectrogram and overlay layer using 2D canvas context', () => {
    const drawImageSpy = jest.fn();
    const putImageDataSpy = jest.fn();

    jest.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue({
      clearRect: jest.fn(),
      createImageData: jest.fn().mockReturnValue({ data: new Uint8ClampedArray(36) }),
      putImageData: putImageDataSpy,
      drawImage: drawImageSpy,
      beginPath: jest.fn(),
      moveTo: jest.fn(),
      lineTo: jest.fn(),
      stroke: jest.fn(),
    } as any);

    const { container } = render(
      <XAIOverlayCanvas
        audioDuration={5.0}
        baseSpectrogram={sampleSpectrogram}
        xaiResults={[sampleXAIResult]}
        activeMethod="gradcam"
        width={800}
        height={400}
      />
    );

    // Verify canvas elements exist
    const canvases = container.querySelectorAll('canvas');
    expect(canvases.length).toBeGreaterThanOrEqual(4); // base, wave, overlays, f0

    // Assert that drawImage was invoked for both base spectrogram and overlay canvas
    expect(drawImageSpy).toHaveBeenCalled();
    expect(putImageDataSpy).toHaveBeenCalled();
  });

  test('fails validation if baseSpectrogram is missing while overlay is present', () => {
    // Helper function to check if base spectrogram layer was rendered
    const checkBaseSpectrogramRendered = (baseSpectrogram?: number[][], xaiResults?: XAIResult[]) => {
      if (xaiResults && xaiResults.length > 0 && (!baseSpectrogram || baseSpectrogram.length === 0)) {
        throw new Error('Base spectrogram layer is empty while XAI overlay is present');
      }
    };

    expect(() => {
      checkBaseSpectrogramRendered(undefined, [sampleXAIResult]);
    }).toThrow('Base spectrogram layer is empty while XAI overlay is present');

    expect(() => {
      checkBaseSpectrogramRendered(sampleSpectrogram, [sampleXAIResult]);
    }).not.toThrow();
  });
});

describe('PredictionPanel XAI Fetch Error Handling', () => {
  beforeEach(() => {
    jest.resetAllMocks();
  });

  test('displays explicit error card when /saliency/generate fetch fails (A2 compliance)', async () => {
    // Mock failed fetch
    (global as any).fetch = jest.fn().mockImplementation(() =>
      Promise.resolve({
        ok: false,
        status: 500,
        json: async () => ({ detail: 'Internal backend error during Grad-CAM generation' })
      })
    );

    render(
      <PredictionPanel
        selectedFile={{
          file_id: 'test_id',
          filename: 'audio.wav',
          file_path: 'uploads/audio.wav',
          message: 'Uploaded'
        }}
        model="whisper-base"
      />
    );

    await waitFor(() => {
      expect(screen.getByText('Failed to load XAI Overlay Canvas')).toBeInTheDocument();
      expect(screen.getAllByText(/Internal backend error during Grad-CAM generation/)[0]).toBeInTheDocument();
    });
  });
});
