import {
  CheckCircle2,
  AlertTriangle,
  XCircle,
  FileText,
  Shield,
  Settings,
  ChevronDown,
  ChevronRight,
  Power,
  RotateCcw,
  Zap,
} from "lucide-react";
import { Button } from "../ui/button";
import { Badge } from "../ui/badge";
import { ScrollArea } from "../ui/scroll-area";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "../ui/dialog";
import { useState } from "react";
import type {
  PersonaConfig,
  ProcessedDocument,
  SecurityAlert,
  AppPhase,
  AvailableTool,
  ModelConfig,
} from "./types";

function getRiskColor(score: number) {
  if (score < 0.2) return "text-emerald-600";
  if (score < 0.5) return "text-amber-500";
  return "text-red-500";
}

function getRiskBg(score: number) {
  if (score < 0.2) return "bg-emerald-50 border-emerald-200";
  if (score < 0.5) return "bg-amber-50 border-amber-200";
  return "bg-red-50 border-red-200";
}

function getRiskLabel(score: number) {
  if (score < 0.2) return "Low";
  if (score < 0.5) return "Medium";
  return "High";
}

function RiskIcon({ score, className }: { score: number; className?: string }) {
  if (score < 0.2)
    return <CheckCircle2 className={`size-3.5 text-emerald-500 ${className}`} />;
  if (score < 0.5)
    return <AlertTriangle className={`size-3.5 text-amber-500 ${className}`} />;
  return <XCircle className={`size-3.5 text-red-500 ${className}`} />;
}

interface AppSidebarProps {
  persona: PersonaConfig;
  documents: ProcessedDocument[];
  alerts: SecurityAlert[];
  phase: AppPhase;
  apiKeyPresent: boolean;
  modelConfig: ModelConfig;
  availableTools: AvailableTool[];
  personaError?: string | null;
  onReset: () => void;
  onQuit: () => void;
  collapsed: boolean;
}

export default function AppSidebar({
  persona,
  documents,
  alerts,
  phase,
  apiKeyPresent,
  modelConfig,
  availableTools,
  personaError,
  onReset,
  onQuit,
  collapsed,
}: AppSidebarProps) {
  const [configOpen, setConfigOpen] = useState(true);
  const [docsOpen, setDocsOpen] = useState(true);
  const [alertsOpen, setAlertsOpen] = useState(true);
  const [quitDialogOpen, setQuitDialogOpen] = useState(false);

  if (collapsed) return null;

  const initials = persona.name
    .split(" ")
    .map((p) => p[0])
    .join("")
    .slice(0, 2)
    .toUpperCase();

  return (
    <>
      <aside className="w-72 xl:w-80 border-r bg-[#fafbfc] flex flex-col h-full shrink-0">
        {/* Persona Header */}
        <div className="p-4 border-b">
          <div className="flex items-center gap-3">
            {persona.avatarUrl ? (
              <img
                src={persona.avatarUrl}
                alt={persona.name}
                className="w-10 h-10 rounded-full object-cover ring-2 ring-white shadow-sm"
              />
            ) : (
              <div className="w-10 h-10 rounded-full bg-[#1e293b] text-white flex items-center justify-center ring-2 ring-white shadow-sm text-xs" style={{ fontWeight: 600 }}>
                {initials}
              </div>
            )}
            <div className="min-w-0">
              <p className="text-sm truncate" style={{ fontWeight: 600 }}>
                {persona.name}
              </p>
              <p className="text-xs text-muted-foreground truncate">
                {persona.role}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-1.5 mt-2.5">
            {personaError ? (
              <>
                <XCircle className="size-3 text-red-500" />
                <span className="text-[11px] text-red-600" style={{ fontWeight: 500 }}>
                  Persona config error
                </span>
              </>
            ) : (
              <>
                <CheckCircle2 className="size-3 text-emerald-500" />
                <span className="text-[11px] text-emerald-600" style={{ fontWeight: 500 }}>
                  Persona loaded
                </span>
              </>
            )}
          </div>
          {personaError && (
            <p className="text-[10px] text-red-500 mt-1 leading-relaxed">
              {personaError}
            </p>
          )}
        </div>

        <ScrollArea className="flex-1">
          <div className="p-3 space-y-1">
            {/* Config Section */}
            <button
              onClick={() => setConfigOpen(!configOpen)}
              className="w-full flex items-center gap-2 px-2 py-1.5 rounded-md text-xs text-muted-foreground hover:bg-muted/60 transition-colors"
              style={{ fontWeight: 500 }}
            >
              {configOpen ? (
                <ChevronDown className="size-3.5" />
              ) : (
                <ChevronRight className="size-3.5" />
              )}
              <Settings className="size-3.5" />
              Configuration
            </button>
            {configOpen && (
              <div className="ml-4 pl-3 border-l border-border/60 space-y-2.5 py-2">
                {/* API Key */}
                <div className="flex items-center justify-between">
                  <span className="text-[11px] text-muted-foreground">API Key</span>
                  {apiKeyPresent ? (
                    <Badge
                      variant="secondary"
                      className="text-[10px] py-0 h-5 bg-emerald-50 text-emerald-700 border border-emerald-200"
                    >
                      <CheckCircle2 className="size-2.5 mr-0.5" />
                      Connected
                    </Badge>
                  ) : (
                    <Badge
                      variant="secondary"
                      className="text-[10px] py-0 h-5 bg-red-50 text-red-700 border border-red-200"
                    >
                      <XCircle className="size-2.5 mr-0.5" />
                      Missing
                    </Badge>
                  )}
                </div>

                {/* Models */}
                <div>
                  <span className="text-[11px] text-muted-foreground block mb-1">
                    Models
                  </span>
                  <div className="space-y-1">
                    {Object.entries(modelConfig).map(([key, val]) => (
                      <div key={key} className="flex items-center justify-between">
                        <span className="text-[10px] text-muted-foreground capitalize">
                          {key}
                        </span>
                        <span className="text-[10px] bg-muted px-1.5 py-0.5 rounded font-mono">
                          {val}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Tools */}
                <div>
                  <span className="text-[11px] text-muted-foreground block mb-1">
                    Agent Tools
                  </span>
                  <div className="flex flex-wrap gap-1">
                    {availableTools.map((tool) => (
                      <Badge
                        key={tool.name}
                        variant="outline"
                        className="text-[10px] py-0 h-5"
                      >
                        <Zap className="size-2.5 text-emerald-500" />
                        {tool.name}
                      </Badge>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {/* Documents Section */}
            {documents.length > 0 && (
              <>
                <button
                  onClick={() => setDocsOpen(!docsOpen)}
                  className="w-full flex items-center gap-2 px-2 py-1.5 rounded-md text-xs text-muted-foreground hover:bg-muted/60 transition-colors"
                  style={{ fontWeight: 500 }}
                >
                  {docsOpen ? (
                    <ChevronDown className="size-3.5" />
                  ) : (
                    <ChevronRight className="size-3.5" />
                  )}
                  <FileText className="size-3.5" />
                  Documents
                  <Badge
                    variant="secondary"
                    className="text-[10px] py-0 h-4 ml-auto"
                  >
                    {documents.length}
                  </Badge>
                </button>
                {docsOpen && (
                  <div className="ml-4 pl-3 border-l border-border/60 space-y-1.5 py-2">
                    {documents.map((doc) => (
                      <div
                        key={doc.id}
                        className="flex items-center gap-2 p-1.5 rounded hover:bg-muted/40 transition-colors"
                      >
                        <RiskIcon score={doc.riskScore} />
                        <div className="min-w-0 flex-1">
                          <p
                            className="text-[11px] truncate"
                            style={{ fontWeight: 500 }}
                          >
                            {doc.name}
                          </p>
                          <p className="text-[10px] text-muted-foreground">
                            {doc.detectedType}
                          </p>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </>
            )}

            {/* Security Alerts */}
            {alerts.length > 0 && (
              <>
                <button
                  onClick={() => setAlertsOpen(!alertsOpen)}
                  className="w-full flex items-center gap-2 px-2 py-1.5 rounded-md text-xs text-muted-foreground hover:bg-muted/60 transition-colors"
                  style={{ fontWeight: 500 }}
                >
                  {alertsOpen ? (
                    <ChevronDown className="size-3.5" />
                  ) : (
                    <ChevronRight className="size-3.5" />
                  )}
                  <Shield className="size-3.5" />
                  Security Alerts
                  <Badge
                    variant="secondary"
                    className="text-[10px] py-0 h-4 ml-auto bg-amber-50 text-amber-700 border border-amber-200"
                  >
                    {alerts.length}
                  </Badge>
                </button>
                {alertsOpen && (
                  <div className="ml-4 pl-3 border-l border-border/60 space-y-1.5 py-2">
                    {alerts.map((alert) => (
                      <div
                        key={alert.id}
                        className={`p-2 rounded-md border text-[11px] ${getRiskBg(
                          alert.riskScore
                        )}`}
                      >
                        <div className="flex items-center gap-1.5 mb-1">
                          <RiskIcon score={alert.riskScore} />
                          <span
                            className={`${getRiskColor(alert.riskScore)}`}
                            style={{ fontWeight: 500 }}
                          >
                            {getRiskLabel(alert.riskScore)} Risk
                          </span>
                        </div>
                        <p className="text-muted-foreground leading-relaxed">
                          {alert.message}
                        </p>
                      </div>
                    ))}
                  </div>
                )}
              </>
            )}
          </div>
        </ScrollArea>

        {/* Bottom actions */}
        <div className="p-3 border-t space-y-1.5">
          {(phase === "dashboard" || phase === "chat") && (
            <Button
              variant="ghost"
              size="sm"
              className="w-full justify-start text-xs text-muted-foreground h-8"
              onClick={onReset}
            >
              <RotateCcw className="size-3.5 mr-2" />
              Reset All
            </Button>
          )}
          <Button
            variant="ghost"
            size="sm"
            className="w-full justify-start text-xs text-red-500 hover:text-red-600 hover:bg-red-50 h-8"
            onClick={() => setQuitDialogOpen(true)}
          >
            <Power className="size-3.5 mr-2" />
            Quit App
          </Button>
        </div>
      </aside>

      {/* Quit Confirmation Dialog */}
      <Dialog open={quitDialogOpen} onOpenChange={setQuitDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Quit PersonaChat?</DialogTitle>
            <DialogDescription>
              This will end your session and clear all uploaded documents and
              chat history. This action cannot be undone.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setQuitDialogOpen(false)}
            >
              Cancel
            </Button>
            <Button variant="destructive" onClick={onQuit}>
              Quit App
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}
