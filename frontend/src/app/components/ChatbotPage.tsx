import { useState, useCallback, useEffect } from "react";
import { useNavigate } from "react-router";
import {
  MessageCircle,
  PanelLeftClose,
  PanelLeft,
  ArrowLeft,
  Loader2,
  AlertTriangle,
} from "lucide-react";
import { Button } from "./ui/button";
import AppSidebar from "./chatbot/AppSidebar";
import DocumentUploader from "./chatbot/DocumentUploader";
import DocumentDashboard from "./chatbot/DocumentDashboard";
import ChatInterface from "./chatbot/ChatInterface";
import type {
  AppPhase,
  BackendConfig,
  BackendState,
  ChatMessage,
  PersonaConfig,
} from "./chatbot/types";
import {
  addDocuments,
  bootstrapApp,
  processDocuments,
  resetSession,
  sendChatMessage,
} from "../lib/api";

const FALLBACK_PERSONA: PersonaConfig = {
  name: "Candidate",
  role: "Persona unavailable",
  location: "",
  tone: "",
  bio: "",
  avatarUrl: "",
  contactLinks: [],
};

function derivePhase(state: BackendState): AppPhase {
  if (!state.processed) return "upload";
  if (state.chatHistory.length > 0) return "chat";
  return "dashboard";
}

export default function ChatbotPage() {
  const navigate = useNavigate();
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [loading, setLoading] = useState(true);
  const [bootstrapError, setBootstrapError] = useState<string | null>(null);
  const [operationError, setOperationError] = useState<string | null>(null);
  const [chatThinking, setChatThinking] = useState(false);
  const [config, setConfig] = useState<BackendConfig | null>(null);
  const [appState, setAppState] = useState<BackendState | null>(null);
  const [phase, setPhase] = useState<AppPhase>("upload");

  const loadBootstrap = useCallback(async () => {
    setLoading(true);
    setBootstrapError(null);
    setOperationError(null);
    try {
      const data = await bootstrapApp();
      setConfig(data.config);
      setAppState(data.state);
      setPhase(derivePhase(data.state));
    } catch (err) {
      setBootstrapError(err instanceof Error ? err.message : "Failed to load app");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadBootstrap();
  }, [loadBootstrap]);

  const handleProcess = useCallback(
    async (files: File[]) => {
      if (!appState) throw new Error("Session not initialized");
      setOperationError(null);
      setPhase("processing");
      try {
        const result = await processDocuments(appState.sessionId, files);
        setAppState(result.state);
        setPhase("dashboard");
      } catch (err) {
        setPhase(appState.processed ? "dashboard" : "upload");
        throw err;
      }
    },
    [appState]
  );

  const handleAddMore = useCallback(
    async (files: File[]) => {
      if (!appState) throw new Error("Session not initialized");
      setOperationError(null);
      const result = await addDocuments(appState.sessionId, files);
      setAppState(result.state);
    },
    [appState]
  );

  const handleStartChat = useCallback(() => {
    setPhase("chat");
  }, []);

  const handleReset = useCallback(async () => {
    if (!appState) return;
    try {
      setOperationError(null);
      const result = await resetSession(appState.sessionId);
      setAppState(result.state);
      setPhase("upload");
      setChatThinking(false);
    } catch (err) {
      setOperationError(err instanceof Error ? err.message : "Reset failed");
    }
  }, [appState]);

  const handleQuit = useCallback(() => {
    navigate("/");
  }, [navigate]);

  const handleSendMessage = useCallback(
    async (text: string) => {
      if (!appState) return;
      setOperationError(null);
      setChatThinking(true);

      const optimisticUser: ChatMessage = {
        id: crypto.randomUUID(),
        role: "user",
        content: text,
        timestamp: new Date(),
      };

      setAppState((prev) =>
        prev
          ? {
              ...prev,
              chatHistory: [...prev.chatHistory, optimisticUser],
            }
          : prev
      );

      try {
        const result = await sendChatMessage(appState.sessionId, text);
        setAppState(result.state);
        setPhase("chat");
      } catch (err) {
        const message = err instanceof Error ? err.message : "Failed to send message";
        setOperationError(message);
        setAppState((prev) =>
          prev
            ? {
                ...prev,
                chatHistory: [
                  ...prev.chatHistory,
                  {
                    id: crypto.randomUUID(),
                    role: "assistant",
                    content: `Error: ${message}`,
                    timestamp: new Date(),
                  },
                ],
              }
            : prev
        );
      } finally {
        setChatThinking(false);
      }
    },
    [appState]
  );

  const persona = config?.persona ?? FALLBACK_PERSONA;
  const documents = appState?.documents ?? [];
  const alerts = appState?.alerts ?? [];
  const messages = appState?.chatHistory ?? [];

  if (loading) {
    return (
      <div className="h-screen bg-background flex items-center justify-center">
        <div className="text-center">
          <div className="w-14 h-14 rounded-2xl bg-[#f1f5f9] flex items-center justify-center mx-auto mb-4">
            <Loader2 className="size-6 text-[#475569] animate-spin" />
          </div>
          <p className="text-sm text-muted-foreground">Loading PersonaChat...</p>
        </div>
      </div>
    );
  }

  if (bootstrapError || !appState || !config) {
    return (
      <div className="h-screen bg-background flex items-center justify-center p-6">
        <div className="w-full max-w-md border rounded-2xl bg-white p-6 text-center shadow-sm">
          <AlertTriangle className="size-6 text-amber-500 mx-auto mb-3" />
          <h2 style={{ fontWeight: 600 }}>Backend connection issue</h2>
          <p className="text-sm text-muted-foreground mt-2">
            {bootstrapError ??
              "The app could not initialize. Make sure the Python API server is running."}
          </p>
          <Button className="mt-5 bg-[#1e293b] hover:bg-[#334155]" onClick={() => void loadBootstrap()}>
            Retry
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="h-screen flex flex-col bg-background">
      <header className="h-12 border-b bg-white flex items-center px-4 gap-3 shrink-0 z-10">
        <Button
          variant="ghost"
          size="icon"
          className="size-8"
          onClick={() => navigate("/")}
        >
          <ArrowLeft className="size-4" />
        </Button>
        <div className="w-px h-5 bg-border" />
        <Button
          variant="ghost"
          size="icon"
          className="size-8"
          onClick={() => setSidebarOpen(!sidebarOpen)}
        >
          {sidebarOpen ? (
            <PanelLeftClose className="size-4" />
          ) : (
            <PanelLeft className="size-4" />
          )}
        </Button>
        <div className="flex items-center gap-2">
          <div className="w-6 h-6 rounded-md bg-[#1e293b] flex items-center justify-center">
            <MessageCircle className="size-3 text-white" />
          </div>
          <span className="text-sm" style={{ fontWeight: 600 }}>
            PersonaChat
          </span>
        </div>

        <div className="ml-auto flex items-center gap-1">
          {(["upload", "processing", "dashboard", "chat"] as AppPhase[]).map(
            (p, i) => {
              const labels = ["Upload", "Process", "Review", "Chat"];
              const phaseOrder: AppPhase[] = ["upload", "processing", "dashboard", "chat"];
              const currentIdx = phaseOrder.indexOf(phase);
              const isActive = i <= currentIdx;
              const isCurrent = p === phase;

              return (
                <div key={p} className="flex items-center gap-1">
                  {i > 0 && (
                    <div
                      className={`w-6 h-px ${
                        isActive ? "bg-[#1e293b]" : "bg-border"
                      }`}
                    />
                  )}
                  <div
                    className={`flex items-center gap-1.5 px-2 py-1 rounded-full text-[11px] transition-colors ${
                      isCurrent
                        ? "bg-[#1e293b] text-white"
                        : isActive
                        ? "text-foreground"
                        : "text-muted-foreground"
                    }`}
                    style={{ fontWeight: isCurrent ? 600 : 400 }}
                  >
                    <span
                      className={`w-4 h-4 rounded-full flex items-center justify-center text-[10px] ${
                        isCurrent
                          ? "bg-white text-[#1e293b]"
                          : isActive
                          ? "bg-[#1e293b] text-white"
                          : "bg-muted text-muted-foreground"
                      }`}
                      style={{ fontWeight: 600 }}
                    >
                      {i + 1}
                    </span>
                    <span className="hidden sm:inline">{labels[i]}</span>
                  </div>
                </div>
              );
            }
          )}
        </div>
      </header>

      <div className="flex-1 flex overflow-hidden">
        <AppSidebar
          persona={persona}
          documents={documents}
          alerts={alerts}
          phase={phase}
          apiKeyPresent={config.apiKeyPresent}
          modelConfig={config.modelConfig}
          availableTools={config.availableTools}
          personaError={config.personaError}
          onReset={() => void handleReset()}
          onQuit={handleQuit}
          collapsed={!sidebarOpen}
        />

        <main className="flex-1 flex flex-col overflow-hidden bg-white">
          {operationError && (
            <div className="mx-6 mt-4 rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
              {operationError}
            </div>
          )}

          {phase === "upload" || phase === "processing" ? (
            <DocumentUploader onProcess={handleProcess} />
          ) : null}

          {phase === "dashboard" && (
            <DocumentDashboard
              documents={documents}
              onStartChat={handleStartChat}
              onAddMore={handleAddMore}
            />
          )}

          {phase === "chat" && (
            <ChatInterface
              persona={persona}
              messages={messages}
              isThinking={chatThinking}
              onSendMessage={handleSendMessage}
              disabled={!config.apiKeyPresent}
            />
          )}
        </main>
      </div>
    </div>
  );
}
