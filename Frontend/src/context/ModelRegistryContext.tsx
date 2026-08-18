import React, { createContext, useContext, useState, useRef, useCallback } from "react";
import { API_BASE } from "@/lib/api";

export interface ResolvedModel {
  model_id: string;
  revision: string;
  family: string;
  weights_sha256: string;
  available_layers: string[];
}

export interface ResolveError {
  code: string;
  message: string;
}

export type DownloadStatus = "idle" | "downloading" | "resolved" | "error" | "cancelled";

interface ModelRegistryContextType {
  status: DownloadStatus;
  activeModelId: string | null;
  activeRevision: string;
  resolvedModel: ResolvedModel | null;
  error: ResolveError | null;
  resolvedCustomModels: string[];
  resolveModel: (modelId: string, revision?: string) => Promise<ResolvedModel | null>;
  cancelResolution: () => Promise<void>;
  clearState: () => void;
}

const ModelRegistryContext = createContext<ModelRegistryContextType | undefined>(undefined);

export const ModelRegistryProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [status, setStatus] = useState<DownloadStatus>("idle");
  const [activeModelId, setActiveModelId] = useState<string | null>(null);
  const [activeRevision, setActiveRevision] = useState<string>("main");
  const [resolvedModel, setResolvedModel] = useState<ResolvedModel | null>(null);
  const [error, setError] = useState<ResolveError | null>(null);
  const [resolvedCustomModels, setResolvedCustomModels] = useState<string[]>([]);
  
  const abortControllerRef = useRef<AbortController | null>(null);

  const resolveModel = useCallback(async (modelId: string, revision: string = "main"): Promise<ResolvedModel | null> => {
    const trimmedId = modelId.trim();
    if (!trimmedId) return null;

    // Abort previous download if any
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }

    const controller = new AbortController();
    abortControllerRef.current = controller;

    setStatus("downloading");
    setActiveModelId(trimmedId);
    setActiveRevision(revision.trim() || "main");
    setError(null);
    setResolvedModel(null);

    try {
      const response = await fetch(`${API_BASE}/models/resolve`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_id: trimmedId, revision: revision.trim() || "main" }),
        signal: controller.signal,
      });

      const body = await response.json();
      if (!response.ok) {
        const detail = body?.detail;
        if (detail && typeof detail === "object" && detail.code) {
          setError({ code: detail.code, message: detail.message || "Failed to resolve model" });
        } else {
          setError({ code: "ERROR", message: typeof detail === "string" ? detail : "Failed to resolve model" });
        }
        setStatus("error");
        return null;
      }

      const result = body as ResolvedModel;
      setResolvedModel(result);
      setStatus("resolved");

      setResolvedCustomModels((prev) => (prev.includes(result.model_id) ? prev : [...prev, result.model_id]));
      return result;
    } catch (err: any) {
      if (err.name === "AbortError") {
        setStatus("cancelled");
        setError({ code: "CANCELLED", message: `Download for '${trimmedId}' was cancelled.` });
      } else {
        setStatus("error");
        setError({ code: "ERROR", message: err instanceof Error ? err.message : "Failed to resolve model" });
      }
      return null;
    } finally {
      abortControllerRef.current = null;
    }
  }, []);

  const cancelResolution = useCallback(async () => {
    if (activeModelId) {
      const targetModel = activeModelId;
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
        abortControllerRef.current = null;
      }
      setStatus("cancelled");
      setError({ code: "CANCELLED", message: `Resolution for '${targetModel}' was cancelled.` });

      // Notify backend to clean PyTorch CUDA/CPU memory and abort weight download
      try {
        await fetch(`${API_BASE}/models/cancel`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ model_id: targetModel }),
        });
      } catch (e) {
        console.warn("Failed to notify backend cancel endpoint:", e);
      }
    }
  }, [activeModelId]);

  const clearState = useCallback(() => {
    setStatus("idle");
    setActiveModelId(null);
    setResolvedModel(null);
    setError(null);
  }, []);

  return (
    <ModelRegistryContext.Provider
      value={{
        status,
        activeModelId,
        activeRevision,
        resolvedModel,
        error,
        resolvedCustomModels,
        resolveModel,
        cancelResolution,
        clearState,
      }}
    >
      {children}
    </ModelRegistryContext.Provider>
  );
};

export const useModelRegistry = () => {
  const context = useContext(ModelRegistryContext);
  if (!context) {
    throw new Error("useModelRegistry must be used within a ModelRegistryProvider");
  }
  return context;
};
