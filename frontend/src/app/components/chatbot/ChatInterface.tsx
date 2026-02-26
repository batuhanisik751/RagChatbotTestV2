import { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  Send,
  ChevronDown,
  ChevronRight,
  Search,
  Globe,
  Github,
  Linkedin,
  Cloud,
  AlertCircle,
  Sparkles,
} from "lucide-react";
import { Button } from "../ui/button";
import type { ChatMessage, PersonaConfig, ToolCall } from "./types";

const toolIcons: Record<string, React.ElementType> = {
  semantic_search: Search,
  web_search: Globe,
  github_search: Github,
  linkedin_search: Linkedin,
  weather_lookup: Cloud,
};

function PersonaAvatar({
  persona,
  sizeClass,
  textClass,
}: {
  persona: PersonaConfig;
  sizeClass: string;
  textClass?: string;
}) {
  const initials = persona.name
    .split(" ")
    .map((p) => p[0])
    .join("")
    .slice(0, 2)
    .toUpperCase();

  if (persona.avatarUrl) {
    return (
      <img
        src={persona.avatarUrl}
        alt={persona.name}
        className={`${sizeClass} rounded-full object-cover`}
      />
    );
  }

  return (
    <div
      className={`${sizeClass} rounded-full bg-[#1e293b] text-white flex items-center justify-center ${textClass ?? "text-xs"}`}
      style={{ fontWeight: 600 }}
      aria-label={persona.name}
    >
      {initials}
    </div>
  );
}

function ToolCallItem({ call }: { call: ToolCall }) {
  const Icon = toolIcons[call.tool] || Search;
  return (
    <div className="flex items-center gap-2 py-1.5">
      <Icon className="size-3 text-muted-foreground shrink-0" />
      <span className="text-[11px]" style={{ fontWeight: 500 }}>
        {call.tool.replace(/_/g, " ")}
      </span>
      <span className="text-[10px] text-muted-foreground truncate max-w-[180px]">
        "{call.args}"
      </span>
      <span className="ml-auto text-[10px] text-muted-foreground whitespace-nowrap">
        {call.error ? (
          <span className="text-red-500">{call.error}</span>
        ) : (
          `${call.resultCount} results`
        )}
      </span>
    </div>
  );
}

function DebugPanel({ debug }: { debug: ChatMessage["debug"] }) {
  const [open, setOpen] = useState(false);
  if (!debug) return null;

  return (
    <div className="mt-2">
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-1.5 text-[11px] text-muted-foreground hover:text-foreground transition-colors"
      >
        {open ? (
          <ChevronDown className="size-3" />
        ) : (
          <ChevronRight className="size-3" />
        )}
        <span style={{ fontWeight: 500 }}>Debug</span>
        <span className="text-[10px]">
          {debug.agentRounds} round{debug.agentRounds > 1 ? "s" : ""},{" "}
          {debug.toolCalls.length} tool call{debug.toolCalls.length > 1 ? "s" : ""}
        </span>
      </button>
      {open && (
        <div className="mt-2 p-3 rounded-lg bg-muted/40 border text-xs space-y-1">
          <div className="flex items-center justify-between mb-2">
            <span className="text-muted-foreground">Finish reason:</span>
            <span className="font-mono text-[11px]">{debug.finishReason}</span>
          </div>
          <div className="border-t pt-2">
            <span
              className="text-muted-foreground text-[11px] block mb-1"
              style={{ fontWeight: 500 }}
            >
              Tool Calls
            </span>
            {debug.toolCalls.map((call, i) => (
              <ToolCallItem key={i} call={call} />
            ))}
          </div>
          {debug.errors && debug.errors.length > 0 && (
            <div className="border-t pt-2">
              {debug.errors.map((err, i) => (
                <div
                  key={i}
                  className="flex items-center gap-1.5 text-red-500 text-[11px]"
                >
                  <AlertCircle className="size-3" />
                  {err}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

interface ChatInterfaceProps {
  persona: PersonaConfig;
  messages: ChatMessage[];
  isThinking: boolean;
  onSendMessage: (text: string) => Promise<void> | void;
  disabled?: boolean;
}

export default function ChatInterface({
  persona,
  messages,
  isThinking,
  onSendMessage,
  disabled = false,
}: ChatInterfaceProps) {
  const [input, setInput] = useState("");
  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, isThinking]);

  const sendMessage = async (text: string) => {
    if (!text.trim() || isThinking || disabled) return;
    setInput("");
    await onSendMessage(text.trim());
    inputRef.current?.focus();
  };

  const handleSuggestionClick = (suggestion: string) => {
    void sendMessage(suggestion);
  };

  const lastAssistantMsg = [...messages]
    .reverse()
    .find((m) => m.role === "assistant" && m.suggestions?.length);

  return (
    <div className="flex-1 flex flex-col h-full">
      <div className="border-b px-6 py-3 flex items-center gap-3 bg-white shrink-0">
        <PersonaAvatar persona={persona} sizeClass="w-8 h-8" />
        <div>
          <p className="text-sm" style={{ fontWeight: 600 }}>
            {persona.name}
          </p>
          <p className="text-[11px] text-muted-foreground">{persona.role}</p>
        </div>
        <div className="ml-auto flex items-center gap-1.5">
          <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
          <span className="text-[11px] text-emerald-600" style={{ fontWeight: 500 }}>
            Active
          </span>
        </div>
      </div>

      <div ref={scrollRef} className="flex-1 overflow-y-auto">
        <div className="max-w-3xl mx-auto px-6 py-6 space-y-6">
          {messages.length === 0 && (
            <div className="text-center py-12">
              <PersonaAvatar
                persona={persona}
                sizeClass="w-16 h-16 mx-auto mb-4"
                textClass="text-base ring-4 ring-muted/40"
              />
              <h3 style={{ fontWeight: 600 }}>Chat with {persona.name}</h3>
              <p className="text-sm text-muted-foreground mt-1 max-w-md mx-auto">
                Ask questions as if you're interviewing {persona.name}. Answers
                are grounded in uploaded documents and persona configuration.
              </p>
              <div className="flex flex-wrap justify-center gap-2 mt-6">
                {[
                  "Tell me about your background.",
                  "What are your strongest technical skills?",
                  "Why are you looking for a new role?",
                ].map((q) => (
                  <button
                    key={q}
                    onClick={() => handleSuggestionClick(q)}
                    className="px-3 py-1.5 rounded-full border text-xs text-muted-foreground hover:bg-muted/40 hover:text-foreground transition-colors"
                    disabled={isThinking || disabled}
                  >
                    {q}
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.map((msg) => (
            <div
              key={msg.id}
              className={`flex gap-3 ${
                msg.role === "user" ? "justify-end" : "justify-start"
              }`}
            >
              {msg.role === "assistant" && (
                <PersonaAvatar persona={persona} sizeClass="w-7 h-7 mt-1 shrink-0" />
              )}
              <div className={`max-w-[80%] ${msg.role === "user" ? "order-1" : ""}`}>
                <div
                  className={`rounded-2xl px-4 py-3 text-sm ${
                    msg.role === "user"
                      ? "bg-[#1e293b] text-white rounded-br-md"
                      : "bg-[#f8fafc] border rounded-bl-md"
                  }`}
                  style={{ lineHeight: 1.6 }}
                >
                  {msg.role === "assistant" ? (
                    <div className="chat-markdown">
                      <ReactMarkdown
                        remarkPlugins={[remarkGfm]}
                        components={{
                          p: ({ children }) => <p className="my-1.5 first:mt-0 last:mb-0">{children}</p>,
                          ul: ({ children }) => <ul className="list-disc my-1.5 pl-5 space-y-0.5">{children}</ul>,
                          ol: ({ children }) => <ol className="list-decimal my-1.5 pl-5 space-y-0.5">{children}</ol>,
                          li: ({ children }) => <li className="my-0.5">{children}</li>,
                          strong: ({ children }) => <strong className="font-semibold">{children}</strong>,
                        }}
                      >
                        {msg.content}
                      </ReactMarkdown>
                    </div>
                  ) : (
                    msg.content
                  )}
                </div>
                {msg.role === "assistant" && msg.debug && <DebugPanel debug={msg.debug} />}
                <p className="text-[10px] text-muted-foreground mt-1.5 px-1">
                  {msg.timestamp.toLocaleTimeString([], {
                    hour: "2-digit",
                    minute: "2-digit",
                  })}
                </p>
              </div>
            </div>
          ))}

          {isThinking && (
            <div className="flex gap-3">
              <PersonaAvatar persona={persona} sizeClass="w-7 h-7 mt-1 shrink-0" />
              <div className="rounded-2xl rounded-bl-md px-4 py-3 bg-[#f8fafc] border">
                <div className="flex items-center gap-2 text-sm text-muted-foreground">
                  <Sparkles className="size-3.5 animate-pulse" />
                  <span>Researching and composing response...</span>
                </div>
                <div className="flex gap-1 mt-2">
                  <span
                    className="w-1.5 h-1.5 rounded-full bg-muted-foreground/40 animate-bounce"
                    style={{ animationDelay: "0ms" }}
                  />
                  <span
                    className="w-1.5 h-1.5 rounded-full bg-muted-foreground/40 animate-bounce"
                    style={{ animationDelay: "150ms" }}
                  />
                  <span
                    className="w-1.5 h-1.5 rounded-full bg-muted-foreground/40 animate-bounce"
                    style={{ animationDelay: "300ms" }}
                  />
                </div>
              </div>
            </div>
          )}

          {!isThinking && lastAssistantMsg?.suggestions && (
            <div className="flex flex-wrap gap-2 pl-10">
              {lastAssistantMsg.suggestions.map((s) => (
                <button
                  key={s}
                  onClick={() => handleSuggestionClick(s)}
                  className="px-3 py-1.5 rounded-full border text-xs text-muted-foreground hover:bg-muted/50 hover:text-foreground transition-colors hover:border-muted-foreground/30"
                  disabled={disabled}
                >
                  {s}
                </button>
              ))}
            </div>
          )}
        </div>
      </div>

      <div className="border-t bg-white px-6 py-4 shrink-0">
        <div className="max-w-3xl mx-auto">
          <form
            onSubmit={(e) => {
              e.preventDefault();
              void sendMessage(input);
            }}
            className="flex gap-2"
          >
            <input
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder={`Ask ${persona.name} a question...`}
              className="flex-1 h-11 px-4 rounded-xl border bg-[#f8fafc] focus:outline-none focus:ring-2 focus:ring-[#1e293b]/20 focus:border-[#1e293b]/30 transition-all text-sm"
              disabled={isThinking || disabled}
            />
            <Button
              type="submit"
              className="h-11 w-11 rounded-xl bg-[#1e293b] hover:bg-[#334155] shrink-0"
              disabled={!input.trim() || isThinking || disabled}
            >
              <Send className="size-4" />
            </Button>
          </form>
          <p className="text-[10px] text-muted-foreground text-center mt-2">
            Responses are grounded in uploaded documents. The agent may use
            tools for retrieval and external lookups when needed.
          </p>
        </div>
      </div>
    </div>
  );
}

