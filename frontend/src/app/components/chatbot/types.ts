export interface PersonaConfig {
  name: string;
  role: string;
  location: string;
  tone: string;
  bio: string;
  avatarUrl: string;
  contactLinks: { label: string; url: string }[];
}

export interface ProcessedDocument {
  id: string;
  name: string;
  fileType: "resume" | "non_resume";
  detectedType: string;
  riskScore: number;
  confidence: number;
  metadataSummary: string;
  extractedHighlights?: string[];
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
  suggestions?: string[];
  debug?: DebugInfo;
  isThinking?: boolean;
}

export interface DebugInfo {
  agentRounds: number;
  toolCalls: ToolCall[];
  finishReason: string;
  errors?: string[];
}

export interface ToolCall {
  tool: string;
  args: string;
  resultCount: number;
  error?: string;
}

export interface UploadedFile {
  id: string;
  file: File;
  status: "pending" | "validating" | "processing" | "done" | "error";
  error?: string;
  warning?: string;
}

export type AppPhase = "upload" | "processing" | "dashboard" | "chat";

export interface SecurityAlert {
  id: string;
  documentName: string;
  riskScore: number;
  message: string;
  timestamp: Date;
}

export interface ModelConfig {
  classifier: string;
  extraction: string;
  answer: string;
  chat?: string;
}

export interface AvailableTool {
  name: string;
  status: "active";
}

export interface BackendState {
  sessionId: string;
  phase: "upload" | "dashboard" | "chat";
  processed: boolean;
  documents: ProcessedDocument[];
  alerts: SecurityAlert[];
  chatHistory: ChatMessage[];
  suggestedQuestions: string[];
}

export interface BackendConfig {
  persona: PersonaConfig;
  personaError?: string | null;
  apiKeyPresent: boolean;
  modelConfig: ModelConfig;
  availableTools: AvailableTool[];
  maxFileSizeMB: number;
}

export interface BootstrapResponse {
  config: BackendConfig;
  state: BackendState;
}
