import type {
  BackendState,
  BootstrapResponse,
  ChatMessage,
  ProcessedDocument,
  SecurityAlert,
} from "../components/chatbot/types";

const API_BASE = (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? "http://localhost:8000";
const SESSION_KEY = "persona_chat_session_id";

type RawChatMessage = Omit<ChatMessage, "timestamp"> & { timestamp: string | Date };
type RawSecurityAlert = Omit<SecurityAlert, "timestamp"> & { timestamp: string | Date };
type RawBackendState = Omit<BackendState, "chatHistory" | "alerts"> & {
  chatHistory: RawChatMessage[];
  alerts: RawSecurityAlert[];
  documents: ProcessedDocument[];
};

function reviveMessage(msg: RawChatMessage): ChatMessage {
  return {
    ...msg,
    timestamp: msg.timestamp instanceof Date ? msg.timestamp : new Date(msg.timestamp),
  };
}

function reviveAlert(alert: RawSecurityAlert): SecurityAlert {
  return {
    ...alert,
    timestamp: alert.timestamp instanceof Date ? alert.timestamp : new Date(alert.timestamp),
  };
}

function reviveState(state: RawBackendState): BackendState {
  return {
    ...state,
    chatHistory: state.chatHistory.map(reviveMessage),
    alerts: state.alerts.map(reviveAlert),
  };
}

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      ...(init?.headers ?? {}),
    },
  });

  const text = await response.text();
  let data: unknown = {};
  if (text) {
    try {
      data = JSON.parse(text);
    } catch {
      data = { detail: text };
    }
  }

  if (!response.ok) {
    const message =
      (data as { detail?: string })?.detail ||
      `Request failed (${response.status})`;
    throw new Error(message);
  }

  return data as T;
}

export function getStoredSessionId(): string | null {
  try {
    return localStorage.getItem(SESSION_KEY);
  } catch {
    return null;
  }
}

function storeSessionId(sessionId: string) {
  try {
    localStorage.setItem(SESSION_KEY, sessionId);
  } catch {
    // no-op (SSR/private mode)
  }
}

export async function bootstrapApp(): Promise<BootstrapResponse> {
  const sessionId = getStoredSessionId();
  const query = sessionId ? `?session_id=${encodeURIComponent(sessionId)}` : "";
  const raw = await apiFetch<{ config: BootstrapResponse["config"]; state: RawBackendState }>(
    `/api/bootstrap${query}`
  );
  storeSessionId(raw.state.sessionId);
  return {
    config: raw.config,
    state: reviveState(raw.state),
  };
}

export async function resetSession(sessionId: string): Promise<{ state: BackendState }> {
  const raw = await apiFetch<{ state: RawBackendState }>("/api/reset", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sessionId }),
  });
  return { state: reviveState(raw.state) };
}

function appendFiles(form: FormData, files: File[]) {
  files.forEach((file) => form.append("files", file));
}

export async function processDocuments(sessionId: string, files: File[]) {
  const form = new FormData();
  form.append("sessionId", sessionId);
  appendFiles(form, files);

  const raw = await apiFetch<{ state: RawBackendState; result: unknown }>("/api/process", {
    method: "POST",
    body: form,
  });
  return { state: reviveState(raw.state), result: raw.result };
}

export async function addDocuments(sessionId: string, files: File[]) {
  const form = new FormData();
  form.append("sessionId", sessionId);
  appendFiles(form, files);

  const raw = await apiFetch<{ state: RawBackendState; result: unknown }>("/api/add-documents", {
    method: "POST",
    body: form,
  });
  return { state: reviveState(raw.state), result: raw.result };
}

export async function sendChatMessage(sessionId: string, question: string) {
  const raw = await apiFetch<{
    assistantMessage: RawChatMessage;
    state: RawBackendState;
  }>("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sessionId, question }),
  });

  return {
    assistantMessage: reviveMessage(raw.assistantMessage),
    state: reviveState(raw.state),
  };
}

