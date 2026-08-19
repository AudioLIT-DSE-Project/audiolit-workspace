import React, { useState, useEffect } from 'react';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import {
  Upload,
  Plus,
  Trash2,
  FolderPlus,
  File as FileIcon,
  Database,
  AlertCircle,
  CheckCircle,
  X,
  ClipboardList,
  Info,
} from "lucide-react";
import { API_BASE } from '@/lib/api';

interface CustomDataset {
  dataset_name: string;
  formatted_name: string;
  created_at: string;
  session_id: string;
  files: Array<{
    filename: string;
    original_filename: string;
    duration: number;
    sample_rate: number;
    size: number;
    uploaded_at: string;
    ground_truth?: string;
  }>;
  total_files: number;
}

interface GroundTruthResult {
  matched_files: number;
  total_files: number;
  unmatched_csv_rows: string[];
}

interface CustomDatasetManagerProps {
  onDatasetCreated?: (datasetName: string) => void;
  onDatasetSelected?: (datasetName: string) => void;
}

export const CustomDatasetManager: React.FC<CustomDatasetManagerProps> = ({
  onDatasetCreated,
  onDatasetSelected
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const [activeTab, setActiveTab] = useState("list");
  const [datasets, setDatasets] = useState<CustomDataset[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  // Create dataset form
  const [newDatasetName, setNewDatasetName] = useState("");
  const [createLoading, setCreateLoading] = useState(false);
  
  // Upload files form
  const [selectedDataset, setSelectedDataset] = useState<string>("");
  const [selectedFiles, setSelectedFiles] = useState<FileList | null>(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadLoading, setUploadLoading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<Array<{file: string, status: 'pending' | 'uploading' | 'success' | 'error', error?: string}>>([]);

  // Ground truth CSV form (LIT-247)
  const [groundTruthDataset, setGroundTruthDataset] = useState<string>("");
  const [groundTruthFile, setGroundTruthFile] = useState<File | null>(null);
  const [groundTruthLoading, setGroundTruthLoading] = useState(false);
  const [groundTruthResult, setGroundTruthResult] = useState<GroundTruthResult | null>(null);
  const [groundTruthError, setGroundTruthError] = useState<string | null>(null);

  const fetchDatasets = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE}/upload/dataset/list`, {
        credentials: 'include'
      });
      
      if (!response.ok) {
        throw new Error(`Failed to fetch datasets: ${response.status}`);
      }
      
      const data = await response.json();
      setDatasets(data.datasets || []);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch datasets');
      console.error('Error fetching datasets:', err);
    } finally {
      setLoading(false);
    }
  };

  const createDataset = async () => {
    if (!newDatasetName.trim()) {
      setError("Dataset name is required");
      return;
    }

    setCreateLoading(true);
    setError(null);
    
    try {
      const formData = new FormData();
      formData.append('dataset_name', newDatasetName.trim());
      
      const response = await fetch(`${API_BASE}/upload/dataset/create`, {
        method: 'POST',
        credentials: 'include',
        body: formData
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `Failed to create dataset: ${response.status}`);
      }
      
      const data = await response.json();
      setNewDatasetName("");
      await fetchDatasets(); // Refresh the list
      setActiveTab("list"); // Switch to list tab
      
      if (onDatasetCreated) {
        onDatasetCreated(data.dataset_name);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create dataset');
      console.error('Error creating dataset:', err);
    } finally {
      setCreateLoading(false);
    }
  };

  const uploadFiles = async () => {
    if (!selectedDataset || !selectedFiles || selectedFiles.length === 0) {
      setError("Please select a dataset and files to upload");
      return;
    }

    setUploadLoading(true);
    setError(null);
    setUploadProgress(0);
    
    // Initialize upload status
    const initialStatus = Array.from(selectedFiles).map(file => ({
      file: file.name,
      status: 'pending' as const
    }));
    setUploadStatus(initialStatus);

    try {
      const formData = new FormData();
      Array.from(selectedFiles).forEach(file => {
        formData.append('files', file);
      });
      
      const response = await fetch(`${API_BASE}/upload/dataset/${selectedDataset}/files`, {
        method: 'POST',
        credentials: 'include',
        body: formData
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `Failed to upload files: ${response.status}`);
      }
      
      const data = await response.json();
      setUploadProgress(100);
      
      // Update upload status based on response
      const updatedStatus = initialStatus.map(item => {
        const uploadedFile = data.uploaded_files?.find((f: any) => f.original_filename === item.file);
        const hasError = data.errors?.some((error: string) => error.includes(item.file));
        
        return {
          ...item,
          status: hasError ? 'error' as const : uploadedFile ? 'success' as const : 'error' as const,
          error: hasError ? data.errors?.find((error: string) => error.includes(item.file)) : undefined
        };
      });
      setUploadStatus(updatedStatus);
      
      // Clear form
      setSelectedFiles(null);
      if (document.querySelector('input[type="file"]') as HTMLInputElement) {
        (document.querySelector('input[type="file"]') as HTMLInputElement).value = '';
      }
      
      await fetchDatasets(); // Refresh the list
      
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to upload files');
      console.error('Error uploading files:', err);
      
      // Mark all files as error
      setUploadStatus(prev => prev.map(item => ({
        ...item,
        status: 'error' as const,
        error: err instanceof Error ? err.message : 'Upload failed'
      })));
    } finally {
      setUploadLoading(false);
    }
  };

  const uploadGroundTruth = async () => {
    if (!groundTruthDataset || !groundTruthFile) {
      setGroundTruthError("Please select a dataset and a CSV file");
      return;
    }

    setGroundTruthLoading(true);
    setGroundTruthError(null);
    setGroundTruthResult(null);

    try {
      const formData = new FormData();
      formData.append('file', groundTruthFile);

      const response = await fetch(`${API_BASE}/upload/dataset/${groundTruthDataset}/ground-truth`, {
        method: 'POST',
        credentials: 'include',
        body: formData
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || `Failed to upload ground truth: ${response.status}`);
      }

      setGroundTruthResult({
        matched_files: data.matched_files,
        total_files: data.total_files,
        unmatched_csv_rows: data.unmatched_csv_rows || [],
      });
      setGroundTruthFile(null);
      if (document.querySelector('input#ground-truth-file') as HTMLInputElement) {
        (document.querySelector('input#ground-truth-file') as HTMLInputElement).value = '';
      }

      await fetchDatasets(); // Refresh so the coverage badge and any open table reflect the match
    } catch (err) {
      setGroundTruthError(err instanceof Error ? err.message : 'Failed to upload ground truth');
      console.error('Error uploading ground truth:', err);
    } finally {
      setGroundTruthLoading(false);
    }
  };

  const deleteDataset = async (datasetName: string) => {
    if (!confirm(`Are you sure you want to delete the dataset "${datasetName}"? This action cannot be undone.`)) {
      return;
    }

    try {
      const response = await fetch(`${API_BASE}/upload/dataset/${datasetName}`, {
        method: 'DELETE',
        credentials: 'include'
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `Failed to delete dataset: ${response.status}`);
      }
      
      await fetchDatasets(); // Refresh the list
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete dataset');
      console.error('Error deleting dataset:', err);
    }
  };

  const selectDataset = (formattedName: string) => {
    if (onDatasetSelected) {
      onDatasetSelected(formattedName);
    }
    setIsOpen(false);
  };

  useEffect(() => {
    if (isOpen) {
      fetchDatasets();
    }
  }, [isOpen]);

  const formatFileSize = (bytes: number) => {
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    if (bytes === 0) return '0 Bytes';
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return Math.round(bytes / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i];
  };

  return (
    <Dialog open={isOpen} onOpenChange={setIsOpen}>
      <DialogTrigger asChild>
        <Button variant="outline" size="sm" className="h-7 text-xs">
          <Database className="h-4 w-4 mr-2" />
          Manage Datasets
        </Button>
      </DialogTrigger>
      <DialogContent className="max-w-4xl max-h-[80vh] overflow-hidden">
        <DialogHeader>
          <DialogTitle>Custom Dataset Manager</DialogTitle>
          <DialogDescription>
            Create and manage custom audio datasets for analysis
          </DialogDescription>
        </DialogHeader>
        
        <Tabs value={activeTab} onValueChange={setActiveTab} className="flex-1">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="list" className="flex items-center gap-2">
              <Database className="h-4 w-4" />
              My Datasets
            </TabsTrigger>
            <TabsTrigger value="create" className="flex items-center gap-2">
              <FolderPlus className="h-4 w-4" />
              Create Dataset
            </TabsTrigger>
            <TabsTrigger value="upload" className="flex items-center gap-2">
              <Upload className="h-4 w-4" />
              Upload Files
            </TabsTrigger>
            <TabsTrigger value="ground-truth" className="flex items-center gap-2">
              <ClipboardList className="h-4 w-4" />
              Ground Truth CSV
            </TabsTrigger>
          </TabsList>
          
          <div className="mt-4 max-h-[50vh] overflow-y-auto">
            <TabsContent value="list" className="space-y-4">
              {loading && (
                <div className="text-center py-8">
                  <div className="w-6 h-6 border-2 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-2"></div>
                  Loading datasets...
                </div>
              )}
              
              {!loading && datasets.length === 0 && (
                <div className="text-center py-8 text-muted-foreground">
                  <Database className="h-12 w-12 mx-auto mb-4 opacity-50" />
                  <p>No custom datasets found</p>
                  <p className="text-sm">Create your first dataset to get started</p>
                </div>
              )}
              
              {!loading && datasets.map((dataset) => (
                <Card key={dataset.dataset_name} className="hover:shadow-md transition-shadow">
                  <CardHeader className="pb-3">
                    <div className="flex items-center justify-between">
                      <CardTitle className="text-lg">{dataset.dataset_name}</CardTitle>
                      <div className="flex items-center gap-2">
                        <Badge variant="secondary">
                          {dataset.total_files} files
                        </Badge>
                        {(() => {
                          const withGroundTruth = dataset.files.filter((f) => f.ground_truth).length;
                          if (withGroundTruth === 0) return null;
                          const complete = withGroundTruth === dataset.total_files;
                          return (
                            <Badge
                              variant="outline"
                              className={complete ? "bg-emerald-500/10 text-emerald-600 border-emerald-500/20" : "bg-amber-500/10 text-amber-600 border-amber-500/20"}
                            >
                              Ground truth: {withGroundTruth}/{dataset.total_files}
                            </Badge>
                          );
                        })()}
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => selectDataset(dataset.formatted_name)}
                        >
                          Select
                        </Button>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => deleteDataset(dataset.dataset_name)}
                          className="text-red-600 hover:text-red-700"
                        >
                          <Trash2 className="h-4 w-4" />
                        </Button>
                      </div>
                    </div>
                    <CardDescription>
                      Created {new Date(dataset.created_at).toLocaleDateString()}
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <span className="font-medium">Total Duration:</span>{" "}
                        {dataset.files.reduce((sum, file) => sum + file.duration, 0).toFixed(1)}s
                      </div>
                      <div>
                        <span className="font-medium">Total Size:</span>{" "}
                        {formatFileSize(dataset.files.reduce((sum, file) => sum + file.size, 0))}
                      </div>
                    </div>
                    
                    {dataset.files.length > 0 && (
                      <div className="mt-3">
                        <p className="text-sm font-medium mb-2">Recent Files:</p>
                        <div className="space-y-1">
                          {dataset.files.slice(0, 3).map((file) => (
                            <div key={file.filename} className="flex items-center gap-2 text-xs text-muted-foreground">
                              <FileIcon className="h-3 w-3" />
                              <span className="truncate">{file.original_filename}</span>
                              <span>({file.duration.toFixed(1)}s)</span>
                            </div>
                          ))}
                          {dataset.files.length > 3 && (
                            <div className="text-xs text-muted-foreground">
                              ... and {dataset.files.length - 3} more files
                            </div>
                          )}
                        </div>
                      </div>
                    )}
                  </CardContent>
                </Card>
              ))}
            </TabsContent>
            
            <TabsContent value="create" className="space-y-4">
              <div className="space-y-4">
                <div>
                  <Label htmlFor="dataset-name">Dataset Name</Label>
                  <Input
                    id="dataset-name"
                    value={newDatasetName}
                    onChange={(e) => setNewDatasetName(e.target.value)}
                    placeholder="Enter dataset name..."
                    className="mt-1"
                  />
                </div>
                
                <Button 
                  onClick={createDataset} 
                  disabled={createLoading || !newDatasetName.trim()}
                  className="w-full"
                >
                  {createLoading ? (
                    <>
                      <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin mr-2"></div>
                      Creating...
                    </>
                  ) : (
                    <>
                      <Plus className="h-4 w-4 mr-2" />
                      Create Dataset
                    </>
                  )}
                </Button>
              </div>
            </TabsContent>
            
            <TabsContent value="upload" className="space-y-4">
              <div className="space-y-4">
                <div>
                  <Label htmlFor="dataset-select">Select Dataset</Label>
                  <select
                    id="dataset-select"
                    value={selectedDataset}
                    onChange={(e) => setSelectedDataset(e.target.value)}
                    className="w-full mt-1 px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  >
                    <option value="">Choose a dataset...</option>
                    {datasets.map((dataset) => (
                      <option key={dataset.dataset_name} value={dataset.dataset_name}>
                        {dataset.dataset_name} ({dataset.total_files} files)
                      </option>
                    ))}
                  </select>
                </div>
                
                <div>
                  <Label htmlFor="files-input">Audio Files</Label>
                  <Input
                    id="files-input"
                    type="file"
                    multiple
                    accept="audio/*,.wav,.mp3,.m4a,.flac"
                    onChange={(e) => setSelectedFiles(e.target.files)}
                    className="mt-1"
                  />
                  {selectedFiles && (
                    <p className="text-sm text-muted-foreground mt-1">
                      {selectedFiles.length} file(s) selected
                    </p>
                  )}
                </div>
                
                {uploadLoading && (
                  <div className="space-y-2">
                    <Progress value={uploadProgress} className="w-full" />
                    <p className="text-sm text-center">Uploading files...</p>
                  </div>
                )}
                
                {uploadStatus.length > 0 && (
                  <div className="space-y-2 max-h-32 overflow-y-auto">
                    <p className="text-sm font-medium">Upload Status:</p>
                    {uploadStatus.map((item, index) => (
                      <div key={index} className="flex items-center gap-2 text-xs">
                        {item.status === 'success' && <CheckCircle className="h-4 w-4 text-green-500" />}
                        {item.status === 'error' && <AlertCircle className="h-4 w-4 text-red-500" />}
                        {item.status === 'pending' && <div className="w-4 h-4 border border-gray-300 rounded-full" />}
                        <span className="truncate">{item.file}</span>
                        {item.error && <span className="text-red-500">- {item.error}</span>}
                      </div>
                    ))}
                  </div>
                )}
                
                <Button 
                  onClick={uploadFiles} 
                  disabled={uploadLoading || !selectedDataset || !selectedFiles || selectedFiles.length === 0}
                  className="w-full"
                >
                  {uploadLoading ? (
                    <>
                      <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin mr-2"></div>
                      Uploading...
                    </>
                  ) : (
                    <>
                      <Upload className="h-4 w-4 mr-2" />
                      Upload Files
                    </>
                  )}
                </Button>
              </div>
            </TabsContent>

            <TabsContent value="ground-truth" className="space-y-4">
              <div className="space-y-4">
                <div className="p-3 bg-blue-50 border border-blue-200 rounded-md flex gap-2">
                  <Info className="h-4 w-4 text-blue-500 shrink-0 mt-0.5" />
                  <div className="text-xs text-blue-900 space-y-1.5">
                    <p className="font-medium">How ground truth matching works</p>
                    <ul className="list-disc list-inside space-y-1">
                      <li>
                        Upload a CSV with one column for the audio filename and one for the
                        ground truth text/label — e.g. <code className="bg-blue-100 px-1 rounded">filename,ground_truth</code>.
                        Column names are flexible: <code className="bg-blue-100 px-1 rounded">file</code>/<code className="bg-blue-100 px-1 rounded">audio_file</code> also
                        work for the filename column, and <code className="bg-blue-100 px-1 rounded">transcript</code>/<code className="bg-blue-100 px-1 rounded">label</code>/<code className="bg-blue-100 px-1 rounded">text</code> also work for the ground-truth column.
                      </li>
                      <li>
                        Matching is by filename only (extension and case don't matter — <code className="bg-blue-100 px-1 rounded">harvard.wav</code>,{" "}
                        <code className="bg-blue-100 px-1 rounded">Harvard</code>, and <code className="bg-blue-100 px-1 rounded">HARVARD.WAV</code> all match the same file).
                      </li>
                      <li>
                        <span className="font-medium">Order doesn't matter</span> — upload the CSV before or after the audio
                        files. Matching runs immediately for files already in the dataset, and automatically for any
                        audio file you add afterwards.
                      </li>
                      <li>
                        <span className="font-medium">Uploading a new CSV replaces the old one</span> for this dataset —
                        it does not merge. A file matched by an earlier CSV but missing from the new one loses its
                        ground truth.
                      </li>
                    </ul>
                  </div>
                </div>

                <div>
                  <Label htmlFor="ground-truth-dataset-select">Select Dataset</Label>
                  <select
                    id="ground-truth-dataset-select"
                    value={groundTruthDataset}
                    onChange={(e) => {
                      setGroundTruthDataset(e.target.value);
                      setGroundTruthResult(null);
                      setGroundTruthError(null);
                    }}
                    className="w-full mt-1 px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  >
                    <option value="">Choose a dataset...</option>
                    {datasets.map((dataset) => (
                      <option key={dataset.dataset_name} value={dataset.dataset_name}>
                        {dataset.dataset_name} ({dataset.total_files} files)
                      </option>
                    ))}
                  </select>
                </div>

                <div>
                  <Label htmlFor="ground-truth-file">Ground Truth CSV</Label>
                  <Input
                    id="ground-truth-file"
                    type="file"
                    accept=".csv,text/csv"
                    onChange={(e) => setGroundTruthFile(e.target.files?.[0] || null)}
                    className="mt-1"
                  />
                  {groundTruthFile && (
                    <p className="text-sm text-muted-foreground mt-1">{groundTruthFile.name} selected</p>
                  )}
                </div>

                {groundTruthResult && (
                  <div
                    className={`p-3 rounded-md border text-xs space-y-1 ${
                      groundTruthResult.matched_files === groundTruthResult.total_files && groundTruthResult.total_files > 0
                        ? "bg-emerald-50 border-emerald-200 text-emerald-800"
                        : "bg-amber-50 border-amber-200 text-amber-800"
                    }`}
                  >
                    <p className="font-medium flex items-center gap-1.5">
                      <CheckCircle className="h-3.5 w-3.5" />
                      Matched {groundTruthResult.matched_files} of {groundTruthResult.total_files} file(s) in the dataset.
                    </p>
                    {groundTruthResult.unmatched_csv_rows.length > 0 && (
                      <p>
                        {groundTruthResult.unmatched_csv_rows.length} row(s) in the CSV didn't match any file
                        (yet — they'll match automatically if you upload that audio file later):{" "}
                        <span className="font-mono">{groundTruthResult.unmatched_csv_rows.join(", ")}</span>
                      </p>
                    )}
                  </div>
                )}

                {groundTruthError && (
                  <div className="p-3 bg-red-50 border border-red-200 rounded-md flex items-center gap-2 text-xs text-red-700">
                    <AlertCircle className="h-4 w-4 text-red-500 shrink-0" />
                    {groundTruthError}
                  </div>
                )}

                <Button
                  onClick={uploadGroundTruth}
                  disabled={groundTruthLoading || !groundTruthDataset || !groundTruthFile}
                  className="w-full"
                >
                  {groundTruthLoading ? (
                    <>
                      <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin mr-2"></div>
                      Matching...
                    </>
                  ) : (
                    <>
                      <ClipboardList className="h-4 w-4 mr-2" />
                      Upload Ground Truth CSV
                    </>
                  )}
                </Button>
              </div>
            </TabsContent>
          </div>
        </Tabs>
        
        {error && (
          <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-md flex items-center gap-2">
            <AlertCircle className="h-4 w-4 text-red-500" />
            <span className="text-sm text-red-700">{error}</span>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setError(null)}
              className="ml-auto"
            >
              <X className="h-4 w-4" />
            </Button>
          </div>
        )}
        
        <DialogFooter>
          <Button variant="outline" onClick={() => setIsOpen(false)}>
            Close
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};