import {
  useReactTable,
  getCoreRowModel,
  getPaginationRowModel,
  getSortedRowModel,
  getFilteredRowModel,
  flexRender,
  ColumnDef,
} from "@tanstack/react-table";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Play, ChevronLeft, ChevronRight, RotateCw, FolderPlus, Trash2 } from "lucide-react";
import { useMemo, useCallback, useEffect } from "react";

interface UploadedFile {
  file_id: string;
  filename: string;
  file_path: string;
  message: string;
  size?: number;
  duration?: number;
  sample_rate?: number;
  prediction?: string;
  ground_truth?: string;
}

interface AudioData {
  id: string;
  filename: string;
  prediction?: string;
  groundTruthLabel: string;
  confidence: number;
  duration: number;
  file_path?: string;
  size?: number;
}

interface AudioDataTableProps {
  selectedRow: string | null;
  onRowSelect: (id: string) => void;
  searchQuery: string;
  apiData?: unknown;
  model: string;
  dataset: string; // "custom" | "common-voice" | "ravdess"
  datasetMetadata?: Record<string, string | number>[];
  uploadedFiles?: UploadedFile[];
  onFilePlay?: (file: UploadedFile) => void;
  onDeleteLiveRecording?: (fileId: string) => void;
  onSaveLiveToCustom?: (file: UploadedFile) => void;
  predictionMap?: Record<string, string>;
  inferenceStatus?: Record<string, 'idle' | 'loading' | 'done' | 'error'>;
  onVisibleRowIdsChange?: (rowIds: string[]) => void;
  /** LIT-248: re-run inference for exactly this row, bypassing the cache. */
  onRegenerateRow?: (rowId: string) => void;
}

const ADD_MODEL_KEYS = ["melody-machine", "wav2vec2-add"];

// "Predicted"/"Ground Truth" column headers: Transcript for whisper, Label for
// the deepfake (ADD) models, Emotion for everything else (wav2vec2/custom SER).
const predictionColumnNoun = (model: string): string => {
  if (model.startsWith("whisper")) return "Transcript";
  if (ADD_MODEL_KEYS.includes(model)) return "Label";
  return "Emotion";
};

export const AudioDataTable = ({ selectedRow, onRowSelect, searchQuery, apiData, model, dataset, datasetMetadata, uploadedFiles, onFilePlay, onDeleteLiveRecording, onSaveLiveToCustom, predictionMap, inferenceStatus, onVisibleRowIdsChange, onRegenerateRow }: AudioDataTableProps) => {
  // Branch: dataset mode vs custom uploads
  const hasDatasetMetadata = (datasetMetadata?.length || 0) > 0;
  const hasUploadedFiles = uploadedFiles && uploadedFiles.length > 0;

  // Dataset metadata data and columns
  type DatasetRow = Record<string, string | number | null | undefined>;
  const datasetRows: DatasetRow[] = useMemo(() => datasetMetadata ?? [], [datasetMetadata]);

  const getFrom = useCallback((row: DatasetRow, keys: string[], fallback = ""): string => {
    for (const k of keys) {
      const v = row[k];
      if (v !== undefined && v !== null && String(v).length > 0) return String(v);
    }
    return fallback;
  }, []);

  // LIT-248: one "Regenerate" button cell, shared by all three column sets
  // below so they don't each re-implement the same loading/disabled logic.
  const regenerateCell = useCallback((rowId: string) => {
    const isLoading = inferenceStatus?.[rowId] === 'loading';
    return (
      <Button
        type="button"
        size="sm"
        variant="ghost"
        className="h-6 w-6 p-0"
        title="Regenerate: re-run the selected model on just this file"
        disabled={isLoading || !model || !onRegenerateRow}
        onClick={(e) => {
          e.stopPropagation();
          e.preventDefault();
          onRegenerateRow?.(rowId);
        }}
      >
        <RotateCw className={`h-3 w-3 ${isLoading ? "animate-spin" : ""}`} />
      </Button>
    );
  }, [inferenceStatus, model, onRegenerateRow]);

  // Custom uploads data and columns
  const customTableData = useMemo(() => (
    uploadedFiles?.map(file => ({
      file_id: file.file_id,
      id: file.file_id,
      filename: file.filename,
      prediction: file.prediction || "",
      groundTruthLabel: file.ground_truth || "",
      confidence: 0,
      duration: typeof file.duration === 'number' ? file.duration : 0,
      file_path: file.file_path,
      size: file.size
    })) || []
  ), [uploadedFiles]);

  const isUploadedRow = useCallback((rowOriginal: any): boolean => {
    if (!rowOriginal) return false;
    if ('file_id' in rowOriginal || rowOriginal.is_live || rowOriginal.is_uploaded) return true;
    
    const pathVal = getFrom(rowOriginal, ["path", "filepath", "file", "filename", "file_path", "id"], "");
    const cleanFilename = pathVal.split("/").pop()?.split("\\").pop() || pathVal;
    
    if (cleanFilename.startsWith("live_recording_") || pathVal.includes("uploads/")) return true;
    
    if (uploadedFiles && uploadedFiles.length > 0) {
      return uploadedFiles.some(f => 
        f.file_id === rowOriginal.id || 
        f.file_id === rowOriginal.file_id || 
        f.filename === cleanFilename ||
        f.filename === rowOriginal.filename ||
        f.file_path === pathVal ||
        (f.file_path && pathVal.endsWith(f.file_path.split("/").pop() || ""))
      );
    }
    return false;
  }, [uploadedFiles, getFrom]);

  const renderLiveFilename = useCallback((rowOriginal: any, fallbackRowId: string) => {
    const data = rowOriginal as any;
    const fileId = data.file_id || data.id || fallbackRowId;
    const pathVal = getFrom(data, ["path", "filepath", "file", "filename"], String(fileId));
    const filename = data.filename || pathVal.split("/").pop()?.split("\\").pop() || pathVal;
    const file = uploadedFiles?.find(f => f.file_id === fileId || f.filename === filename || f.file_id === fallbackRowId);
    return (
      <div className="flex items-center gap-1.5 min-w-0">
        <Badge variant="outline" className="text-[9px] px-1 py-0 bg-rose-500/10 text-rose-600 dark:text-rose-400 border-rose-500/30 shrink-0 font-semibold">
          LIVE
        </Badge>
        <Button
          type="button"
          size="sm"
          variant="ghost"
          className="h-6 w-6 p-0 shrink-0"
          onClick={(e) => {
            e.stopPropagation();
            e.preventDefault();
            if (file && onFilePlay) {
              onFilePlay(file);
            } else if (onFilePlay) {
              onFilePlay({
                file_id: String(fileId),
                filename: String(filename),
                file_path: data.file_path || String(filename),
                message: "Live recording"
              });
            }
          }}
        >
          <Play className="h-3 w-3" />
        </Button>
        <span className="font-mono text-xs truncate max-w-[130px] inline-block" title={filename}>
          {filename}
        </span>
      </div>
    );
  }, [uploadedFiles, getFrom, onFilePlay]);

  const renderLiveActions = useCallback((rowOriginal: any, fallbackRowId: string) => {
    const data = rowOriginal as any;
    const fileId = data.file_id || data.id || fallbackRowId;
    const filename = data.filename || getFrom(data, ["path", "filepath", "file", "filename"], String(fileId));
    const file = uploadedFiles?.find(f => f.file_id === fileId || f.filename === filename || f.file_id === fallbackRowId) || {
      file_id: String(fileId),
      filename: String(filename),
      file_path: data.file_path || String(filename),
      message: "Live recording"
    };
    return (
      <div className="flex items-center gap-1">
        {regenerateCell(String(fileId))}
        <Button
          type="button"
          size="sm"
          variant="ghost"
          className="h-6 w-6 p-0 text-muted-foreground hover:text-primary"
          title="Save live audio to custom dataset"
          onClick={(e) => {
            e.stopPropagation();
            e.preventDefault();
            if (onSaveLiveToCustom) {
              onSaveLiveToCustom(file);
            }
          }}
        >
          <FolderPlus className="h-3 w-3" />
        </Button>
        <Button
          type="button"
          size="sm"
          variant="ghost"
          className="h-6 w-6 p-0 text-muted-foreground hover:text-destructive"
          title="Delete live recording from session"
          onClick={(e) => {
            e.stopPropagation();
            e.preventDefault();
            if (onDeleteLiveRecording) {
              onDeleteLiveRecording(String(fileId));
            }
          }}
        >
          <Trash2 className="h-3 w-3 text-destructive" />
        </Button>
      </div>
    );
  }, [uploadedFiles, getFrom, regenerateCell, onSaveLiveToCustom, onDeleteLiveRecording]);

  const customColumns: ColumnDef<unknown, unknown>[] = useMemo(() => [
    {
      id: "filename",
      header: "Filename",
      cell: ({ row }) => {
        if (isUploadedRow(row.original)) {
          return renderLiveFilename(row.original, row.id as string);
        } else {
          const data = row.original as DatasetRow;
          const path = getFrom(data, ["path", "filepath", "file", "filename"], "");
          const filename = path.split("/").pop() || path;
          return <span className="font-mono text-xs">{filename}</span>;
        }
      },
    },
    {
      id: "prediction",
      header: `Predicted ${predictionColumnNoun(model)}`,
      cell: ({ row }) => {
        const rowId = row.id as string;
        const status = inferenceStatus?.[rowId];
        if (status === 'loading') {
          return "";
        }
        
        if (isUploadedRow(row.original)) {
          const data = row.original as any;
          const pred = predictionMap?.[rowId] || data.prediction || "";
          if (!pred) return "";
          
          const predictionText = typeof pred === 'string' ? pred : 
            (typeof pred === 'object' && pred !== null) ? 
              (pred as any).predicted_transcript || (pred as any).predicted_emotion || (pred as any).predicted_label || (pred as any).prediction || (pred as any).text || JSON.stringify(pred) : 
              String(pred);
          
          return <Badge variant="outline" className="text-xs">{predictionText}</Badge>;
        } else {
          const pred = predictionMap?.[rowId] ?? "";
          
          const predictionText = typeof pred === 'string' ? pred : 
            (typeof pred === 'object' && pred !== null) ? 
              (pred as any).predicted_transcript || (pred as any).predicted_emotion || (pred as any).predicted_label || (pred as any).prediction || (pred as any).text || JSON.stringify(pred) : 
              String(pred);
              
          return <span className="text-xs leading-snug line-clamp-3 break-words font-normal" title={predictionText}>{predictionText}</span>;
        }
      },
    },
    {
      id: "groundTruthLabel",
      header: "Ground Truth",
      cell: ({ row }) => {
        if (isUploadedRow(row.original)) {
          const data = row.original as any;
          return <span className="text-xs font-mono">{data.groundTruthLabel || data.ground_truth || "—"}</span>;
        } else {
          const data = row.original as DatasetRow;
          const gt = getFrom(data, ["sentence", "transcript", "text", "emotion", "label"], "");
          return <span className="text-xs leading-snug line-clamp-3 break-words text-slate-600 dark:text-slate-400 font-normal" title={gt}>{gt || "—"}</span>;
        }
      },
    },
    {
      id: "confidence",
      header: "Confidence",
      cell: ({ row }) => {
        if (isUploadedRow(row.original)) {
          const data = row.original as any;
          if (!data.confidence || data.confidence === 0) return null;
          return <span className="text-xs">{data.confidence}</span>;
        } else {
          return <span className="text-xs text-muted-foreground">N/A</span>;
        }
      },
    },
    {
      id: "duration",
      header: "Duration",
      cell: ({ row }) => {
        if (isUploadedRow(row.original)) {
          const data = row.original as any;
          const duration = typeof data.duration === 'number' ? data.duration : 0;
          return <span className="text-xs">{duration.toFixed(2)}s</span>;
        } else {
          const data = row.original as DatasetRow;
          const d = Number(getFrom(data, ["duration", "length"], "0"));
          if (d > 0) {
            return <span className="text-xs">{d.toFixed(2)}s</span>;
          }
          return <span className="text-xs text-muted-foreground">N/A</span>;
        }
      },
    },
    {
      id: "actions",
      header: "Actions",
      cell: ({ row }) => {
        if (isUploadedRow(row.original)) {
          return renderLiveActions(row.original, row.id as string);
        }
        return regenerateCell(row.id as string);
      },
    },
  ], [model, uploadedFiles, onFilePlay, predictionMap, inferenceStatus, getFrom, regenerateCell, isUploadedRow, renderLiveFilename, renderLiveActions]);

  const getDatasetRowId = useCallback((row: DatasetRow, fallback: string): string => {
    const v = row["filename"] ?? row["path"] ?? row["filepath"] ?? row["file"] ?? row["id"];
    return v !== undefined && v !== null && String(v).length > 0 ? String(v) : fallback;
  }, []);

  // Show ground truth for all datasets that contain ground truth metadata
  const shouldShowGroundTruth = true;

  const datasetColumnsCommonVoice: ColumnDef<unknown, unknown>[] = useMemo(() => {
    const baseColumns = [
      {
        id: "filename",
        header: "Filename",
        cell: ({ row }) => {
          if (isUploadedRow(row.original)) {
            return renderLiveFilename(row.original, row.id as string);
          }
          const data = row.original as DatasetRow;
          const path = getFrom(data, ["path", "filepath", "file", "filename"], "");
          const filename = path.split("/").pop() || path;
          return <span className="font-mono text-xs">{filename}</span>;
        },
      },
      {
        id: "prediction",
        header: `Predicted ${predictionColumnNoun(model)}`,
        cell: ({ row }) => {
          const rowId = row.id as string;
          const data = row.original as DatasetRow;
          const path = getFrom(data, ["path", "filepath", "file", "filename"], "");
          const filename = path.split("/").pop() || path;
          const fileId = String(data.id || path || filename);
          const status = inferenceStatus?.[rowId] || inferenceStatus?.[fileId] || inferenceStatus?.[filename];
          
          if (status === 'loading') {
            return <span className="text-xs text-blue-600">Loading...</span>;
          }
          
          const pred = predictionMap?.[rowId] ?? predictionMap?.[fileId] ?? predictionMap?.[filename] ?? (data.prediction as string) ?? "";
          
          if (!pred && status !== 'done') {
            return <span className="text-xs text-gray-400">-</span>;
          }
          
          // Handle object predictions (different models return different object structures)
          const predictionText = typeof pred === 'string' ? pred : 
            (typeof pred === 'object' && pred !== null) ? 
              (pred as any).predicted_transcript || (pred as any).predicted_emotion || (pred as any).predicted_label || (pred as any).prediction || (pred as any).text || JSON.stringify(pred) : 
              String(pred);
              
          return <span className="text-xs leading-snug line-clamp-3 break-words font-normal" title={predictionText}>{predictionText || <span className="text-gray-400">-</span>}</span>;
        },
      },
    ];

    // Add ground truth column only if applicable for this model-dataset combination
    if (shouldShowGroundTruth) {
      baseColumns.push({
        id: "ground_truth",
        header: `Ground Truth ${predictionColumnNoun(model)}`,
        cell: ({ row }) => {
          const data = row.original as DatasetRow;
          const groundTruthValue = getFrom(data, ["sentence", "transcript", "text", "statement", "emotion", "label", "ground_truth", "target"], "");
          return <span className="text-xs leading-snug line-clamp-3 break-words text-slate-600 dark:text-slate-400 font-normal" title={groundTruthValue}>{groundTruthValue || "—"}</span>;
        },
      });
    }

    baseColumns.push({
      id: "duration",
      header: "Duration",
      cell: ({ row }) => {
        const data = row.original as DatasetRow;
        const d = Number(getFrom(data, ["duration"], "0"));
        return <span className="text-xs">{isNaN(d) ? "" : `${d.toFixed(2)}s`}</span>;
      },
    });

    baseColumns.push({
      id: "actions",
      header: "Actions",
      cell: ({ row }) => {
        if (isUploadedRow(row.original)) {
          return renderLiveActions(row.original, row.id as string);
        }
        return regenerateCell(row.id as string);
      },
    });

    return baseColumns;
  }, [getFrom, model, predictionMap, inferenceStatus, shouldShowGroundTruth, regenerateCell, isUploadedRow, renderLiveFilename, renderLiveActions]);

  const datasetColumnsRavdess: ColumnDef<unknown, unknown>[] = useMemo(() => {
    const baseColumns = [
      {
        id: "filename",
        header: "Filename",
        cell: ({ row }) => {
          if (isUploadedRow(row.original)) {
            return renderLiveFilename(row.original, row.id as string);
          }
          const data = row.original as DatasetRow;
          const path = getFrom(data, ["path", "filepath", "file", "filename"], "");
          const filename = path.split("/").pop() || path;
          return <span className="font-mono text-xs">{filename}</span>;
        },
      },
      {
        id: "prediction",
        header: `Predicted ${predictionColumnNoun(model)}`,
        cell: ({ row }) => {
          const rowId = row.id as string;
          const status = inferenceStatus?.[rowId];
          
          if (status === 'loading') {
            return <span className="text-xs text-blue-600">Loading...</span>;
          }
          
          if (status !== 'done') {
            return <span className="text-xs text-gray-400">-</span>;
          }
          
          const pred = predictionMap?.[rowId] ?? "";
          
          // Handle object predictions (different models return different object structures)
          const predictionText = typeof pred === 'string' ? pred : 
            (typeof pred === 'object' && pred !== null) ? 
              (pred as any).predicted_transcript || (pred as any).predicted_emotion || (pred as any).predicted_label || (pred as any).prediction || (pred as any).text || JSON.stringify(pred) : 
              String(pred);
              
          return <span className="text-xs">{predictionText || <span className="text-gray-400">No prediction</span>}</span>;
        },
      },
    ];

    // Add ground truth column only if applicable for this model-dataset combination
    if (shouldShowGroundTruth) {
      baseColumns.push({
        id: "ground_truth",
        header: `Ground Truth ${predictionColumnNoun(model)}`,
        cell: ({ row }) => {
          const data = row.original as DatasetRow;
          const groundTruthValue = getFrom(data, ["sentence", "transcript", "text", "statement", "emotion", "label", "ground_truth", "target"], "");
          return <span className="text-xs">{groundTruthValue}</span>;
        },
      });
    }

    baseColumns.push({
      id: "duration",
      header: "Duration",
      cell: ({ row }) => {
        const data = row.original as DatasetRow;
        // Ravdess dataset doesn't have duration in metadata, so we'll show "N/A"
        const d = Number(getFrom(data, ["duration", "length"], "0"));
        if (d > 0) {
          return <span className="text-xs">{d.toFixed(2)}s</span>;
        }
        return <span className="text-xs text-muted-foreground">N/A</span>;
      },
    });

    baseColumns.push({
      id: "actions",
      header: "Actions",
      cell: ({ row }) => {
        if (isUploadedRow(row.original)) {
          return renderLiveActions(row.original, row.id as string);
        }
        return regenerateCell(row.id as string);
      },
    });

    return baseColumns;
  }, [getFrom, model, predictionMap, inferenceStatus, shouldShowGroundTruth, regenerateCell, isUploadedRow, renderLiveFilename, renderLiveActions]);

  // Build table config based on mode
  const data: unknown[] = useMemo(() => {
    if (hasUploadedFiles) {
      // When there are uploaded files, show uploaded files at top, then dataset files
      const combinedData: unknown[] = [...customTableData]; // Uploaded files first
      if (hasDatasetMetadata) {
        // Add dataset files after uploaded files
        combinedData.push(...datasetRows);
      }
      return combinedData;
    }
    // When no uploaded files, use original logic
    return hasDatasetMetadata ? datasetRows : customTableData;
  }, [hasDatasetMetadata, hasUploadedFiles, datasetRows, customTableData]);
  const columns: ColumnDef<unknown, unknown>[] = useMemo(
    () => {
      if (hasUploadedFiles) {
        // When showing combined data, use custom columns that can handle both types
        return customColumns;
      }
      // When no uploaded files, use original logic
      return hasDatasetMetadata ? (dataset === "ravdess" ? datasetColumnsRavdess : datasetColumnsCommonVoice) : customColumns;
    },
    [hasUploadedFiles, hasDatasetMetadata, dataset, datasetColumnsRavdess, datasetColumnsCommonVoice, customColumns]
  );

  const getRowId = useCallback((row: unknown, index?: number) => {
    if (hasUploadedFiles) {
      // When showing combined data, check if it's a dataset row or uploaded file
      if ('file_id' in (row as any)) {
        return (row as AudioData).id;
      } else {
        return getDatasetRowId(row as DatasetRow, String(index ?? ""));
      }
    }
    if (hasDatasetMetadata) {
      return getDatasetRowId(row as DatasetRow, String(index ?? ""));
    }
    return (row as AudioData).id;
  }, [hasUploadedFiles, hasDatasetMetadata, getDatasetRowId]);

  const table = useReactTable<unknown>({
    data,
    columns,
    getCoreRowModel: getCoreRowModel(),
    getPaginationRowModel: getPaginationRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
    state: {
      globalFilter: searchQuery,
    },
    onGlobalFilterChange: () => {},
    initialState: {
      pagination: {
        pageSize: 20,
      },
    },
    getRowId,
  });

  // Notify parent of currently visible row ids (for sequential per-page inference)
  useEffect(() => {
    if (!onVisibleRowIdsChange) return;
    const rows = table.getRowModel().rows;
    const ids = rows.map(r => String(r.id));
    onVisibleRowIdsChange(ids);
  }, [onVisibleRowIdsChange, searchQuery, dataset, model, hasDatasetMetadata]);

  const getColumnWidthClass = (columnId: string) => {
    switch (columnId) {
      case "filename":
        return "w-[24%] min-w-[140px]";
      case "prediction":
        return "w-[34%] min-w-[200px]";
      case "groundTruthLabel":
      case "ground_truth":
        return "w-[26%] min-w-[160px]";
      case "confidence":
        return "w-[6%] min-w-[55px] text-center";
      case "duration":
        return "w-[6%] min-w-[55px] text-right";
      case "actions":
        return "w-[95px] min-w-[95px] text-right shrink-0";
      default:
        return "";
    }
  };

  return (
    <div className="h-full flex flex-col">
      {/* A scrollable region must be reachable by keyboard, or someone who
          cannot use a pointer cannot scroll the table at all. tabIndex makes
          it focusable so the arrow keys work; the label tells a screen reader
          what the region holds once focus lands there. */}
      <div
        className="flex-1 overflow-auto"
        tabIndex={0}
        role="region"
        aria-label="Audio dataset table"
      >
        <Table className="w-full table-fixed">
          <TableHeader>
            {table.getHeaderGroups().map((headerGroup) => (
              <TableRow key={headerGroup.id}>
                {headerGroup.headers.map((header) => (
                  <TableHead key={header.id} className={`h-8 text-xs font-semibold ${getColumnWidthClass(header.id)}`}>
                    {header.isPlaceholder
                      ? null
                      : flexRender(
                          header.column.columnDef.header,
                          header.getContext()
                        )}
                  </TableHead>
                ))}
              </TableRow>
            ))}
          </TableHeader>
          <TableBody>
            {table.getRowModel().rows?.length ? (
              table.getRowModel().rows.map((row) => (
                <TableRow
                  key={row.id}
                  data-state={(() => {
                    let rowId: string;
                    if (hasUploadedFiles) {
                      // When showing combined data, check if it's a dataset row or uploaded file
                      if ('file_id' in (row.original as any)) {
                        rowId = (row.original as AudioData).id;
                      } else {
                        rowId = getDatasetRowId(row.original as DatasetRow, String(row.id));
                      }
                    } else if (hasDatasetMetadata) {
                      rowId = getDatasetRowId(row.original as DatasetRow, String(row.id));
                    } else {
                      rowId = (row.original as AudioData).id;
                    }
                    return selectedRow === rowId ? "selected" : undefined;
                  })()}
                  className="cursor-pointer hover:bg-muted/50 data-[state=selected]:bg-blue-50 data-[state=selected]:border-blue-200 data-[state=selected]:shadow-sm"
                  onClick={() => {
                    let rowId: string;
                    if (hasUploadedFiles) {
                      // When showing combined data, check if it's a dataset row or uploaded file
                      if ('file_id' in (row.original as any)) {
                        rowId = (row.original as AudioData).id;
                      } else {
                        rowId = getDatasetRowId(row.original as DatasetRow, String(row.id));
                      }
                    } else if (hasDatasetMetadata) {
                      rowId = getDatasetRowId(row.original as DatasetRow, String(row.id));
                    } else {
                      rowId = (row.original as AudioData).id;
                    }
                    onRowSelect(rowId);
                  }}
                >
                  {row.getVisibleCells().map((cell) => (
                    <TableCell key={cell.id} className={`py-2 align-middle ${getColumnWidthClass(cell.column.id)}`}>
                      {flexRender(cell.column.columnDef.cell, cell.getContext())}
                    </TableCell>
                  ))}
                </TableRow>
              ))
            ) : (
              <TableRow>
                <TableCell colSpan={Array.isArray(columns) ? columns.length : 0} className="h-24 text-center text-xs">
                  No results.
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </div>
      
      {/* Pagination */}
      <div className="border-t panel-border p-2 flex items-center justify-between">
        <div className="text-xs text-muted-foreground">
          Page {table.getState().pagination.pageIndex + 1} of {table.getPageCount()}
        </div>
        <div className="flex items-center gap-1">
          {/* The chevrons are the only content, so the name has to be
              supplied: a screen reader otherwise announces two buttons that
              differ only by an unreadable glyph. */}
          <Button
            variant="outline"
            size="sm"
            className="h-6 w-6 p-0"
            onClick={() => table.previousPage()}
            disabled={!table.getCanPreviousPage()}
            aria-label="Previous page"
          >
            <ChevronLeft className="h-3 w-3" aria-hidden="true" />
          </Button>
          <Button
            variant="outline"
            size="sm"
            className="h-6 w-6 p-0"
            onClick={() => table.nextPage()}
            disabled={!table.getCanNextPage()}
            aria-label="Next page"
          >
            <ChevronRight className="h-3 w-3" aria-hidden="true" />
          </Button>
        </div>
      </div>
    </div>
  );
};