import { useState, useRef, useCallback } from "react";
import {
  Upload,
  FileText,
  X,
  AlertTriangle,
  CheckCircle2,
  Loader2,
} from "lucide-react";
import { Button } from "../ui/button";
import { Progress } from "../ui/progress";
import { Badge } from "../ui/badge";
import type { UploadedFile } from "./types";

interface DocumentUploaderProps {
  onProcess: (files: File[]) => Promise<void> | void;
  isCompact?: boolean;
}

const MAX_FILE_SIZE = 2 * 1024 * 1024; // 2MB

export default function DocumentUploader({
  onProcess,
  isCompact = false,
}: DocumentUploaderProps) {
  const [files, setFiles] = useState<UploadedFile[]>([]);
  const [dragActive, setDragActive] = useState(false);
  const [processing, setProcessing] = useState(false);
  const [processProgress, setProcessProgress] = useState(0);
  const [currentFile, setCurrentFile] = useState("");
  const [processStage, setProcessStage] = useState("");
  const [duplicateWarning, setDuplicateWarning] = useState<string | null>(null);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const addFiles = useCallback(
    (newFiles: FileList | File[]) => {
      const fileArr = Array.from(newFiles);
      const uploaded: UploadedFile[] = [];

      fileArr.forEach((file) => {
        // Check PDF
        if (!file.name.toLowerCase().endsWith(".pdf")) {
          return;
        }
        // Check duplicate name
        const existing = files.find((f) => f.file.name === file.name);
        if (existing) {
          setDuplicateWarning(`"${file.name}" has already been added.`);
          setTimeout(() => setDuplicateWarning(null), 4000);
          return;
        }
        // Check size
        const hasError = file.size > MAX_FILE_SIZE;
        uploaded.push({
          id: crypto.randomUUID(),
          file,
          status: hasError ? "error" : "pending",
          error: hasError ? "File exceeds 2MB limit" : undefined,
        });
      });
      setFiles((prev) => [...prev, ...uploaded]);
    },
    [files]
  );

  const removeFile = (id: string) => {
    setFiles((prev) => prev.filter((f) => f.id !== id));
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragActive(false);
    addFiles(e.dataTransfer.files);
  };

  const validFiles = files.filter((f) => f.status !== "error");

  const simulateProcessing = async () => {
    setProcessing(true);
    setSubmitError(null);
    const stages = [
      "Validating file...",
      "Scanning for injection threats...",
      "Classifying document type...",
      "Extracting structured data...",
      "Building semantic index...",
    ];

    for (let i = 0; i < validFiles.length; i++) {
      const file = validFiles[i];
      setCurrentFile(file.file.name);

      for (let s = 0; s < stages.length; s++) {
        setProcessStage(stages[s]);
        const progress =
          ((i * stages.length + s + 1) / (validFiles.length * stages.length)) *
          100;
        setProcessProgress(progress);
        await new Promise((r) => setTimeout(r, 500 + Math.random() * 400));
      }
    }

    try {
      await onProcess(validFiles.map((f) => f.file));
      setFiles([]);
      setProcessProgress(0);
      setCurrentFile("");
      setProcessStage("");
    } catch (err) {
      setSubmitError(err instanceof Error ? err.message : "Processing failed");
    } finally {
      setProcessing(false);
    }
  };

  if (isCompact) {
    return (
      <div className="space-y-3">
        <div
          className={`border-2 border-dashed rounded-lg p-4 text-center cursor-pointer transition-colors ${
            dragActive
              ? "border-[#1e293b] bg-slate-50"
              : "border-border hover:border-muted-foreground/40"
          }`}
          onDragOver={(e) => {
            e.preventDefault();
            setDragActive(true);
          }}
          onDragLeave={() => setDragActive(false)}
          onDrop={handleDrop}
          onClick={() => inputRef.current?.click()}
        >
          <Upload className="size-5 text-muted-foreground mx-auto mb-2" />
          <p className="text-sm text-muted-foreground">
            Drop PDFs here or click to browse
          </p>
          <input
            ref={inputRef}
            type="file"
            className="hidden"
            accept=".pdf"
            multiple
            onChange={(e) => e.target.files && addFiles(e.target.files)}
          />
        </div>
        {files.length > 0 && (
          <div className="space-y-1.5">
            {files.map((f) => (
              <div
                key={f.id}
                className="flex items-center gap-2 p-2 rounded-md bg-muted/40 text-sm"
              >
                <FileText className="size-3.5 text-muted-foreground shrink-0" />
                <span className="truncate flex-1 text-xs">{f.file.name}</span>
                {f.error && (
                  <span className="text-[10px] text-red-500">{f.error}</span>
                )}
                <button
                  onClick={() => removeFile(f.id)}
                  className="p-0.5 hover:bg-muted rounded"
                >
                  <X className="size-3 text-muted-foreground" />
                </button>
              </div>
            ))}
          </div>
        )}
        {validFiles.length > 0 && (
          <Button
            size="sm"
            className="w-full bg-[#1e293b] hover:bg-[#334155]"
            onClick={simulateProcessing}
            disabled={processing}
          >
            {processing ? (
              <>
                <Loader2 className="size-3.5 animate-spin" />
                Processing...
              </>
            ) : (
              `Add ${validFiles.length} Document${validFiles.length > 1 ? "s" : ""}`
            )}
          </Button>
        )}
        {submitError && (
          <div className="text-[11px] text-red-500 leading-relaxed">
            {submitError}
          </div>
        )}
      </div>
    );
  }

  // Processing state
  if (processing) {
    return (
      <div className="flex-1 flex items-center justify-center p-6">
        <div className="w-full max-w-md">
          <div className="text-center mb-8">
            <div className="w-16 h-16 rounded-2xl bg-[#f1f5f9] flex items-center justify-center mx-auto mb-4">
              <Loader2 className="size-7 text-[#475569] animate-spin" />
            </div>
            <h2 style={{ fontWeight: 600 }}>Processing Documents</h2>
            <p className="text-sm text-muted-foreground mt-1">
              Analyzing and indexing your files securely
            </p>
          </div>

          <div className="space-y-4">
            <Progress value={processProgress} className="h-2" />
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>{Math.round(processProgress)}% complete</span>
              <span>
                {validFiles.findIndex((f) => f.file.name === currentFile) + 1} of{" "}
                {validFiles.length} files
              </span>
            </div>

            <div className="bg-muted/40 rounded-lg p-4 space-y-2">
              <div className="flex items-center gap-2">
                <FileText className="size-4 text-muted-foreground" />
                <span className="text-sm truncate" style={{ fontWeight: 500 }}>
                  {currentFile}
                </span>
              </div>
              <p className="text-xs text-muted-foreground pl-6">
                {processStage}
              </p>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Upload state
  return (
    <div className="flex-1 flex items-center justify-center p-6">
      <div className="w-full max-w-lg">
        <div className="text-center mb-8">
          <div className="w-16 h-16 rounded-2xl bg-[#f1f5f9] flex items-center justify-center mx-auto mb-4">
            <Upload className="size-7 text-[#475569]" />
          </div>
          <h2 style={{ fontWeight: 600 }}>Upload Candidate Documents</h2>
          <p className="text-sm text-muted-foreground mt-1">
            Add PDFs to build the candidate's knowledge base. Each document will
            be validated, scanned for security, and semantically indexed.
          </p>
        </div>

        {/* Drop zone */}
        <div
          className={`border-2 border-dashed rounded-xl p-10 text-center cursor-pointer transition-all ${
            dragActive
              ? "border-[#1e293b] bg-slate-50 scale-[1.01]"
              : "border-border hover:border-muted-foreground/40 hover:bg-muted/20"
          }`}
          onDragOver={(e) => {
            e.preventDefault();
            setDragActive(true);
          }}
          onDragLeave={() => setDragActive(false)}
          onDrop={handleDrop}
          onClick={() => inputRef.current?.click()}
        >
          <div className="w-12 h-12 rounded-xl bg-muted/60 flex items-center justify-center mx-auto mb-3">
            <FileText className="size-5 text-muted-foreground" />
          </div>
          <p className="text-sm" style={{ fontWeight: 500 }}>
            Drop PDF files here or click to browse
          </p>
          <p className="text-xs text-muted-foreground mt-1">
            PDF only, up to 2MB per file
          </p>
          <input
            ref={inputRef}
            type="file"
            className="hidden"
            accept=".pdf"
            multiple
            onChange={(e) => e.target.files && addFiles(e.target.files)}
          />
        </div>

        {/* Duplicate warning */}
        {duplicateWarning && (
          <div className="mt-4 flex items-center gap-2 p-3 rounded-lg bg-amber-50 border border-amber-200 text-xs text-amber-700">
            <AlertTriangle className="size-3.5 shrink-0" />
            {duplicateWarning}
          </div>
        )}

        {/* File list */}
        {files.length > 0 && (
          <div className="mt-5 space-y-2">
            <p className="text-xs text-muted-foreground" style={{ fontWeight: 500 }}>
              {files.length} file{files.length > 1 ? "s" : ""} selected
            </p>
            <div className="space-y-1.5 max-h-48 overflow-y-auto">
              {files.map((f) => (
                <div
                  key={f.id}
                  className={`flex items-center gap-3 p-3 rounded-lg border ${
                    f.error
                      ? "bg-red-50/60 border-red-200"
                      : "bg-muted/30 border-transparent"
                  }`}
                >
                  <FileText
                    className={`size-4 shrink-0 ${
                      f.error ? "text-red-400" : "text-muted-foreground"
                    }`}
                  />
                  <div className="flex-1 min-w-0">
                    <p className="text-sm truncate" style={{ fontWeight: 500 }}>
                      {f.file.name}
                    </p>
                    <p className="text-[11px] text-muted-foreground">
                      {(f.file.size / 1024).toFixed(0)} KB
                    </p>
                  </div>
                  {f.error ? (
                    <Badge className="text-[10px] py-0 h-5 bg-red-100 text-red-600 border border-red-200">
                      {f.error}
                    </Badge>
                  ) : (
                    <CheckCircle2 className="size-4 text-emerald-500 shrink-0" />
                  )}
                  <button
                    onClick={() => removeFile(f.id)}
                    className="p-1 hover:bg-muted rounded-md transition-colors"
                  >
                    <X className="size-3.5 text-muted-foreground" />
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}

        {submitError && (
          <div className="mt-4 flex items-center gap-2 p-3 rounded-lg bg-red-50 border border-red-200 text-xs text-red-700">
            <AlertTriangle className="size-3.5 shrink-0" />
            {submitError}
          </div>
        )}

        {/* Process button */}
        {validFiles.length > 0 && (
          <Button
            className="w-full mt-5 h-11 bg-[#1e293b] hover:bg-[#334155]"
            onClick={simulateProcessing}
            disabled={processing}
          >
            Process {validFiles.length} Document
            {validFiles.length > 1 ? "s" : ""}
          </Button>
        )}
      </div>
    </div>
  );
}
