import React from "react";
import { render, screen } from "@testing-library/react";
import "@testing-library/jest-dom";
import { AudioUploadRecorderModal } from "./AudioUploadRecorderModal";

describe("AudioUploadRecorderModal Component", () => {
  const mockOnClose = jest.fn();
  const mockOnFileUploaded = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it("renders modal header and tab choices when open", () => {
    render(
      <AudioUploadRecorderModal
        isOpen={true}
        onClose={mockOnClose}
        onFileUploaded={mockOnFileUploaded}
      />
    );

    expect(
      screen.getByText(/Audio Data Input & Live Voice Recorder/i)
    ).toBeTruthy();
    expect(screen.getByRole("tab", { name: /File Upload/i })).toBeTruthy();
    expect(screen.getByRole("tab", { name: /Live Voice Record/i })).toBeTruthy();
  });

  it("displays task selection checkboxes for ASR, SER, and ADD", () => {
    render(
      <AudioUploadRecorderModal
        isOpen={true}
        onClose={mockOnClose}
        onFileUploaded={mockOnFileUploaded}
      />
    );

    expect(screen.getByText(/ASR \(Transcription\)/i)).toBeTruthy();
    expect(screen.getByText(/SER \(Emotion\)/i)).toBeTruthy();
    expect(screen.getByText(/ADD \(Deepfake\)/i)).toBeTruthy();
  });
});
