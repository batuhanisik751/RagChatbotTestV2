import { useState } from "react";
import {
  FileText,
  CheckCircle2,
  AlertTriangle,
  XCircle,
  Shield,
  MessageCircle,
  Plus,
  ChevronDown,
  ChevronUp,
} from "lucide-react";
import { Button } from "../ui/button";
import { Badge } from "../ui/badge";
import type { ProcessedDocument } from "./types";
import DocumentUploader from "./DocumentUploader";

function getRiskColor(score: number) {
  if (score < 0.2) return "emerald";
  if (score < 0.5) return "amber";
  return "red";
}

function RiskBadge({ score }: { score: number }) {
  const color = getRiskColor(score);
  const label = score < 0.2 ? "Low" : score < 0.5 ? "Medium" : "High";
  const Icon = score < 0.2 ? CheckCircle2 : score < 0.5 ? AlertTriangle : XCircle;
  const bgClass =
    color === "emerald"
      ? "bg-emerald-50 text-emerald-700 border-emerald-200"
      : color === "amber"
      ? "bg-amber-50 text-amber-700 border-amber-200"
      : "bg-red-50 text-red-700 border-red-200";

  return (
    <span
      className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] border ${bgClass}`}
      style={{ fontWeight: 500 }}
    >
      <Icon className="size-3" />
      {label} ({Math.round(score * 100)}%)
    </span>
  );
}

interface DocumentDashboardProps {
  documents: ProcessedDocument[];
  onStartChat: () => void;
  onAddMore: (files: File[]) => void;
}

export default function DocumentDashboard({
  documents,
  onStartChat,
  onAddMore,
}: DocumentDashboardProps) {
  const [addMoreOpen, setAddMoreOpen] = useState(false);
  const hasElevatedRisk = documents.some((d) => d.riskScore >= 0.2);

  return (
    <div className="flex-1 flex items-start justify-center p-6 overflow-y-auto">
      <div className="w-full max-w-2xl py-4">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="w-16 h-16 rounded-2xl bg-emerald-50 flex items-center justify-center mx-auto mb-4">
            <CheckCircle2 className="size-7 text-emerald-500" />
          </div>
          <h2 style={{ fontWeight: 600 }}>
            {documents.length} Document{documents.length > 1 ? "s" : ""} Ready
          </h2>
          <p className="text-sm text-muted-foreground mt-1">
            All documents have been processed and indexed. You can start
            chatting or add more files.
          </p>
        </div>

        {/* Global warning */}
        {hasElevatedRisk && (
          <div className="flex items-start gap-3 p-4 rounded-lg bg-amber-50 border border-amber-200 mb-6">
            <AlertTriangle className="size-4 text-amber-500 mt-0.5 shrink-0" />
            <div>
              <p className="text-sm text-amber-800" style={{ fontWeight: 500 }}>
                Security Notice
              </p>
              <p className="text-xs text-amber-700 mt-0.5">
                One or more documents have elevated risk scores. Review the
                indicators below. The AI will still use these documents but with
                additional caution.
              </p>
            </div>
          </div>
        )}

        {/* Document Cards */}
        <div className="grid gap-3">
          {documents.map((doc) => (
            <div
              key={doc.id}
              className="border rounded-xl p-4 bg-white hover:shadow-sm transition-shadow"
            >
              <div className="flex items-start gap-3">
                <div className="w-10 h-10 rounded-lg bg-[#f1f5f9] flex items-center justify-center shrink-0">
                  <FileText className="size-5 text-[#475569]" />
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-start justify-between gap-3">
                    <div className="min-w-0">
                      <p className="text-sm truncate" style={{ fontWeight: 600 }}>
                        {doc.name}
                      </p>
                      <div className="flex items-center gap-2 mt-1">
                        <Badge
                          variant="secondary"
                          className="text-[10px] py-0 h-5"
                        >
                          {doc.detectedType}
                        </Badge>
                        <span className="text-[11px] text-muted-foreground">
                          {Math.round(doc.confidence * 100)}% confidence
                        </span>
                      </div>
                    </div>
                    <RiskBadge score={doc.riskScore} />
                  </div>
                  <p className="text-xs text-muted-foreground mt-2">
                    {doc.metadataSummary}
                  </p>
                  {doc.extractedHighlights && (
                    <div className="mt-2.5 flex flex-wrap gap-1.5">
                      {doc.extractedHighlights.map((h, i) => (
                        <span
                          key={i}
                          className="text-[10px] px-2 py-0.5 rounded-full bg-muted/60 text-muted-foreground"
                        >
                          {h}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Add More Documents */}
        <div className="mt-6 border rounded-xl overflow-hidden">
          <button
            className="w-full flex items-center gap-2 p-4 text-sm text-muted-foreground hover:bg-muted/30 transition-colors"
            style={{ fontWeight: 500 }}
            onClick={() => setAddMoreOpen(!addMoreOpen)}
          >
            <Plus className="size-4" />
            Add More Documents
            <span className="ml-auto">
              {addMoreOpen ? (
                <ChevronUp className="size-4" />
              ) : (
                <ChevronDown className="size-4" />
              )}
            </span>
          </button>
          {addMoreOpen && (
            <div className="p-4 border-t bg-muted/10">
              <DocumentUploader onProcess={onAddMore} isCompact />
            </div>
          )}
        </div>

        {/* Start Chat CTA */}
        <Button
          className="w-full mt-6 h-12 bg-[#1e293b] hover:bg-[#334155] text-[15px]"
          onClick={onStartChat}
        >
          <MessageCircle className="size-4 mr-2" />
          Start Conversation
        </Button>
      </div>
    </div>
  );
}
