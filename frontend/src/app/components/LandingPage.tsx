import { useNavigate } from "react-router";
import { useState, useEffect, useRef } from "react";
import { Button } from "./ui/button";
import {
  Shield,
  FileText,
  MessageCircle,
  ArrowRight,
  Search,
  Github,
  Linkedin,
  Cloud,
  Upload,
  ScanSearch,
  Database,
  Lock,
  Eye,
  Fingerprint,
  ChevronRight,
} from "lucide-react";

const MOCK_CONVERSATION = [
  {
    role: "recruiter" as const,
    text: "Walk me through your experience leading distributed systems projects.",
  },
  {
    role: "candidate" as const,
    text: "At Stripe, I led the redesign of our payment pipeline to handle 3x throughput using event-driven architecture. Before that at Notion, I migrated our monolith to microservices, cutting p99 latency by 40%.",
    tools: ["semantic_search"],
  },
  {
    role: "recruiter" as const,
    text: "How do you approach mentoring junior engineers?",
  },
  {
    role: "candidate" as const,
    text: "I run weekly 1-on-1 architecture reviews where juniors present design decisions. I also created an internal 'system design book club' at Stripe that grew to 30+ engineers across teams.",
    tools: ["semantic_search", "web_search"],
  },
];

function TypewriterText({
  text,
  speed = 20,
  onDone,
}: {
  text: string;
  speed?: number;
  onDone?: () => void;
}) {
  const [displayed, setDisplayed] = useState("");
  const idx = useRef(0);

  useEffect(() => {
    idx.current = 0;
    setDisplayed("");
    const interval = setInterval(() => {
      idx.current++;
      setDisplayed(text.slice(0, idx.current));
      if (idx.current >= text.length) {
        clearInterval(interval);
        onDone?.();
      }
    }, speed);
    return () => clearInterval(interval);
  }, [text, speed]);

  return <>{displayed}</>;
}

function MockChatPreview() {
  const [visibleMessages, setVisibleMessages] = useState<number[]>([]);
  const [typingIdx, setTypingIdx] = useState<number | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let timeout: ReturnType<typeof setTimeout>;
    let current = 0;
    let cancelled = false;

    function showNext() {
      if (cancelled) return;
      if (current >= MOCK_CONVERSATION.length) {
        // Restart loop
        timeout = setTimeout(() => {
          if (cancelled) return;
          setVisibleMessages([]);
          setTypingIdx(null);
          current = 0;
          timeout = setTimeout(showNext, 1200);
        }, 4000);
        return;
      }

      const msg = MOCK_CONVERSATION[current];
      if (msg.role === "candidate") {
        // Show typing indicator first
        setTypingIdx(current);
        timeout = setTimeout(() => {
          if (cancelled) return;
          setTypingIdx(null);
          const idx = current;
          setVisibleMessages((prev) => [...prev, idx]);
          current++;
          timeout = setTimeout(showNext, 1800);
        }, 1200);
      } else {
        const idx = current;
        setVisibleMessages((prev) => [...prev, idx]);
        current++;
        timeout = setTimeout(showNext, 1400);
      }
    }

    timeout = setTimeout(showNext, 800);
    return () => {
      cancelled = true;
      clearTimeout(timeout);
    };
  }, []);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [visibleMessages, typingIdx]);

  return (
    <div className="bg-white rounded-2xl border shadow-xl shadow-black/5 overflow-hidden w-full max-w-md">
      {/* Chat header */}
      <div className="px-4 py-3 border-b bg-[#fafbfc] flex items-center gap-3">
        <div className="w-8 h-8 rounded-full bg-gradient-to-br from-[#1e293b] to-[#475569] flex items-center justify-center text-white text-[11px]" style={{ fontWeight: 600 }}>
          JC
        </div>
        <div>
          <p className="text-[13px]" style={{ fontWeight: 600 }}>
            Jordan Chen
          </p>
          <p className="text-[10px] text-muted-foreground">
            Sr. Full-Stack Engineer
          </p>
        </div>
        <div className="ml-auto flex items-center gap-1.5">
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
          <span className="text-[10px] text-emerald-600" style={{ fontWeight: 500 }}>
            Live
          </span>
        </div>
      </div>

      {/* Messages */}
      <div
        ref={scrollRef}
        className="p-4 space-y-3 h-[280px] overflow-y-auto"
        style={{ scrollBehavior: "smooth" }}
      >
        {visibleMessages.map((msgIdx) => {
          const msg = MOCK_CONVERSATION[msgIdx];
          if (!msg) return null;
          const isRecruiter = msg.role === "recruiter";
          return (
            <div
              key={msgIdx}
              className={`flex ${isRecruiter ? "justify-end" : "justify-start"}`}
            >
              <div
                className={`max-w-[85%] rounded-2xl px-3.5 py-2.5 text-[12px] ${
                  isRecruiter
                    ? "bg-[#1e293b] text-white rounded-br-md"
                    : "bg-[#f1f5f9] rounded-bl-md"
                }`}
                style={{ lineHeight: 1.55 }}
              >
                {msg.text}
                {!isRecruiter && msg.tools && (
                  <div className="flex gap-1 mt-1.5 pt-1.5 border-t border-black/5">
                    {msg.tools.map((t) => (
                      <span
                        key={t}
                        className="text-[9px] px-1.5 py-0.5 rounded bg-white/80 text-muted-foreground"
                      >
                        {t.replace("_", " ")}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            </div>
          );
        })}

        {/* Typing indicator */}
        {typingIdx !== null && (
          <div className="flex justify-start">
            <div className="bg-[#f1f5f9] rounded-2xl rounded-bl-md px-4 py-3 flex items-center gap-1.5">
              <span
                className="w-1.5 h-1.5 rounded-full bg-[#94a3b8] animate-bounce"
                style={{ animationDelay: "0ms" }}
              />
              <span
                className="w-1.5 h-1.5 rounded-full bg-[#94a3b8] animate-bounce"
                style={{ animationDelay: "150ms" }}
              />
              <span
                className="w-1.5 h-1.5 rounded-full bg-[#94a3b8] animate-bounce"
                style={{ animationDelay: "300ms" }}
              />
            </div>
          </div>
        )}
      </div>

      {/* Input bar */}
      <div className="px-4 py-3 border-t bg-[#fafbfc]">
        <div className="flex items-center gap-2 bg-white border rounded-xl px-3 py-2">
          <span className="text-[11px] text-muted-foreground flex-1">
            Ask Jordan a question...
          </span>
          <div className="w-6 h-6 rounded-lg bg-[#1e293b] flex items-center justify-center">
            <ArrowRight className="size-3 text-white" />
          </div>
        </div>
      </div>
    </div>
  );
}

const PIPELINE_STEPS = [
  {
    icon: Upload,
    label: "Upload",
    detail: "PDF validation & dedup",
    color: "bg-blue-50 text-blue-600 border-blue-200",
  },
  {
    icon: Shield,
    label: "Scan",
    detail: "Injection detection & risk scoring",
    color: "bg-amber-50 text-amber-600 border-amber-200",
  },
  {
    icon: ScanSearch,
    label: "Classify",
    detail: "Resume vs. transcript vs. letter",
    color: "bg-purple-50 text-purple-600 border-purple-200",
  },
  {
    icon: Database,
    label: "Index",
    detail: "Chunking & semantic embeddings",
    color: "bg-emerald-50 text-emerald-600 border-emerald-200",
  },
  {
    icon: MessageCircle,
    label: "Converse",
    detail: "First-person RAG agent",
    color: "bg-[#f0f4ff] text-[#3b5bdb] border-[#c5d2f6]",
  },
];

export default function LandingPage() {
  const navigate = useNavigate();
  const [hoveredStep, setHoveredStep] = useState<number | null>(null);

  return (
    <div className="min-h-screen bg-background flex flex-col">
      {/* Minimal nav */}
      <nav className="sticky top-0 z-50 bg-white/80 backdrop-blur-md border-b">
        <div className="max-w-7xl mx-auto px-6 h-14 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="w-7 h-7 rounded-lg bg-[#1e293b] flex items-center justify-center">
              <MessageCircle className="size-3.5 text-white" />
            </div>
            <span className="text-[15px]" style={{ fontWeight: 600 }}>
              PersonaChat
            </span>
          </div>
          <Button
            size="sm"
            className="bg-[#1e293b] hover:bg-[#334155]"
            onClick={() => navigate("/chat")}
          >
            Open App
            <ArrowRight className="size-3.5" />
          </Button>
        </div>
      </nav>

      <main className="flex-1">
        {/* ─── HERO: Split layout with live preview ─── */}
        <section className="max-w-7xl mx-auto px-6 py-16 md:py-24">
          <div className="flex flex-col lg:flex-row items-center gap-12 lg:gap-16">
            {/* Left: Copy */}
            <div className="flex-1 max-w-xl">
              <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-[#f1f5f9] text-[11px] text-[#475569] mb-5" style={{ fontWeight: 500 }}>
                <Lock className="size-3" />
                Recruiter-only, single-tenant persona
              </div>
              <h1
                className="mb-4"
                style={{ fontSize: "2.5rem", lineHeight: 1.12, fontWeight: 700, letterSpacing: "-0.02em" }}
              >
                Interview the resume,
                <br />
                before the person.
              </h1>
              <p
                className="text-[#64748b] mb-8 text-[16px]"
                style={{ lineHeight: 1.7 }}
              >
                Upload a candidate's documents — resume, cover letter,
                transcripts — and talk directly to an AI that{" "}
                <span className="text-foreground" style={{ fontWeight: 500 }}>
                  answers as the candidate
                </span>
                . Every response is grounded in their actual materials, with
                full security scanning and source transparency.
              </p>
              <div className="flex flex-col sm:flex-row gap-3">
                <Button
                  size="lg"
                  className="bg-[#1e293b] hover:bg-[#334155] h-12 px-7"
                  onClick={() => navigate("/chat")}
                >
                  Upload Documents & Chat
                  <ArrowRight className="size-4" />
                </Button>
              </div>
              {/* Quick stats */}
              <div className="flex gap-8 mt-10 pt-8 border-t">
                {[
                  { value: "5", label: "Agent tools" },
                  { value: "6", label: "Max reasoning rounds" },
                  { value: "<2s", label: "Avg. response" },
                ].map((stat) => (
                  <div key={stat.label}>
                    <p
                      className="text-[1.25rem]"
                      style={{ fontWeight: 700, letterSpacing: "-0.01em" }}
                    >
                      {stat.value}
                    </p>
                    <p className="text-[11px] text-muted-foreground">
                      {stat.label}
                    </p>
                  </div>
                ))}
              </div>
            </div>

            {/* Right: Live chat preview */}
            <div className="flex-shrink-0 relative">
              {/* Decorative elements */}
              <div className="absolute -top-4 -right-4 w-32 h-32 bg-[#f1f5f9] rounded-full blur-3xl opacity-60" />
              <div className="absolute -bottom-6 -left-6 w-24 h-24 bg-emerald-50 rounded-full blur-2xl opacity-80" />
              <div className="relative">
                <MockChatPreview />
                {/* Floating tool badges */}
                <div className="absolute -left-12 top-16 hidden xl:flex flex-col gap-2">
                  {[
                    { icon: Search, label: "Docs" },
                    { icon: Github, label: "GitHub" },
                    { icon: Linkedin, label: "LinkedIn" },
                  ].map((t) => (
                    <div
                      key={t.label}
                      className="flex items-center gap-1.5 bg-white border rounded-lg px-2.5 py-1.5 shadow-sm text-[10px] text-muted-foreground"
                    >
                      <t.icon className="size-3" />
                      {t.label}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* ─── PIPELINE: Horizontal flow ─── */}
        <section className="border-t bg-[#fafbfc]">
          <div className="max-w-6xl mx-auto px-6 py-16 md:py-20">
            <div className="max-w-md mb-10">
              <p
                className="text-[11px] text-[#94a3b8] uppercase tracking-wider mb-2"
                style={{ fontWeight: 600 }}
              >
                Document Pipeline
              </p>
              <h2
                style={{ fontSize: "1.5rem", fontWeight: 700, letterSpacing: "-0.01em" }}
              >
                From PDF to first-person answers
              </h2>
              <p className="text-[#64748b] text-sm mt-2" style={{ lineHeight: 1.6 }}>
                Every document passes through a five-stage pipeline before the
                AI can reference it. Nothing is taken at face value.
              </p>
            </div>

            {/* Pipeline visualization */}
            <div className="flex flex-col md:flex-row items-stretch gap-3">
              {PIPELINE_STEPS.map((step, i) => (
                <div key={step.label} className="flex items-center flex-1 min-w-0">
                  <div
                    className={`flex-1 border rounded-xl p-4 bg-white transition-all cursor-default ${
                      hoveredStep === i ? "shadow-md scale-[1.02]" : "shadow-sm"
                    }`}
                    onMouseEnter={() => setHoveredStep(i)}
                    onMouseLeave={() => setHoveredStep(null)}
                  >
                    <div
                      className={`w-9 h-9 rounded-lg border flex items-center justify-center mb-3 ${step.color}`}
                    >
                      <step.icon className="size-4" />
                    </div>
                    <p className="text-sm" style={{ fontWeight: 600 }}>
                      {step.label}
                    </p>
                    <p
                      className="text-[11px] text-muted-foreground mt-0.5"
                      style={{ lineHeight: 1.5 }}
                    >
                      {step.detail}
                    </p>
                  </div>
                  {i < PIPELINE_STEPS.length - 1 && (
                    <ChevronRight className="size-4 text-[#cbd5e1] mx-1 shrink-0 hidden md:block" />
                  )}
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* ─── TRUST: Security-forward section ─── */}
        <section className="border-t">
          <div className="max-w-6xl mx-auto px-6 py-16 md:py-20">
            <div className="flex flex-col lg:flex-row gap-12 lg:gap-16">
              {/* Left: Security narrative */}
              <div className="flex-1">
                <p
                  className="text-[11px] text-[#94a3b8] uppercase tracking-wider mb-2"
                  style={{ fontWeight: 600 }}
                >
                  Security & Trust
                </p>
                <h2
                  className="mb-4"
                  style={{ fontSize: "1.5rem", fontWeight: 700, letterSpacing: "-0.01em" }}
                >
                  Documents are scanned,
                  <br />
                  not just stored.
                </h2>
                <p
                  className="text-[#64748b] mb-8 max-w-md"
                  style={{ lineHeight: 1.7 }}
                >
                  Candidates sometimes embed prompt injections, hidden Unicode,
                  or adversarial formatting in PDFs. PersonaChat detects these
                  automatically and shows you exactly what it found — so you
                  can trust the conversation.
                </p>

                {/* Risk level preview */}
                <div className="space-y-2.5 max-w-sm">
                  {[
                    {
                      level: "Low Risk",
                      score: "0.05",
                      color: "bg-emerald-500",
                      barW: "w-[5%]",
                      bgColor: "bg-emerald-50 border-emerald-100",
                      textColor: "text-emerald-700",
                      desc: "Clean document, no anomalies detected",
                    },
                    {
                      level: "Medium Risk",
                      score: "0.35",
                      color: "bg-amber-400",
                      barW: "w-[35%]",
                      bgColor: "bg-amber-50 border-amber-100",
                      textColor: "text-amber-700",
                      desc: "Minor formatting irregularities found",
                    },
                    {
                      level: "High Risk",
                      score: "0.82",
                      color: "bg-red-500",
                      barW: "w-[82%]",
                      bgColor: "bg-red-50 border-red-100",
                      textColor: "text-red-600",
                      desc: "Suspected injection patterns detected",
                    },
                  ].map((risk) => (
                    <div
                      key={risk.level}
                      className={`border rounded-lg p-3 ${risk.bgColor}`}
                    >
                      <div className="flex items-center justify-between mb-1.5">
                        <span
                          className={`text-[12px] ${risk.textColor}`}
                          style={{ fontWeight: 600 }}
                        >
                          {risk.level}
                        </span>
                        <span className="text-[11px] text-muted-foreground font-mono">
                          {risk.score}
                        </span>
                      </div>
                      <div className="w-full h-1.5 bg-black/5 rounded-full overflow-hidden mb-1.5">
                        <div
                          className={`h-full rounded-full ${risk.color}`}
                          style={{
                            width: `${parseFloat(risk.score) * 100}%`,
                          }}
                        />
                      </div>
                      <p className="text-[10px] text-muted-foreground">
                        {risk.desc}
                      </p>
                    </div>
                  ))}
                </div>
              </div>

              {/* Right: Feature list */}
              <div className="flex-1 lg:pt-12">
                <div className="space-y-6">
                  {[
                    {
                      icon: Fingerprint,
                      title: "Prompt Injection Detection",
                      desc: "Every document is scanned for adversarial prompts, jailbreak attempts, and instruction-override patterns before entering the knowledge base.",
                    },
                    {
                      icon: Eye,
                      title: "Full Response Transparency",
                      desc: "Expand any AI response to see which tools were called, what documents were searched, and how many reasoning rounds the agent used.",
                    },
                    {
                      icon: Shield,
                      title: "Ephemeral Sessions",
                      desc: "Documents are processed in-session only. Nothing persists between sessions — your candidate data stays under your control.",
                    },
                    {
                      icon: FileText,
                      title: "Strict Input Validation",
                      desc: "PDF-only, 2MB max per file, duplicate detection by content hash and candidate name matching. Bad files are rejected before processing.",
                    },
                  ].map((feature) => (
                    <div key={feature.title} className="flex gap-4">
                      <div className="w-10 h-10 rounded-xl bg-[#f1f5f9] flex items-center justify-center shrink-0">
                        <feature.icon className="size-[18px] text-[#475569]" />
                      </div>
                      <div>
                        <p className="text-[14px] mb-1" style={{ fontWeight: 600 }}>
                          {feature.title}
                        </p>
                        <p
                          className="text-[13px] text-[#64748b]"
                          style={{ lineHeight: 1.6 }}
                        >
                          {feature.desc}
                        </p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* ─── CTA: Minimal, not salesy ─── */}
        <section className="border-t bg-[#1e293b]">
          <div className="max-w-3xl mx-auto px-6 py-14 flex flex-col md:flex-row items-center gap-6 md:gap-12">
            <div className="flex-1">
              <h2
                className="text-white mb-1.5"
                style={{ fontSize: "1.25rem", fontWeight: 600 }}
              >
                Ready to prep for your next interview?
              </h2>
              <p className="text-[#94a3b8] text-sm">
                Upload the candidate's docs. Ask your questions. Get grounded
                answers.
              </p>
            </div>
            <Button
              size="lg"
              className="bg-white text-[#1e293b] hover:bg-slate-100 h-11 px-7 shrink-0"
              onClick={() => navigate("/chat")}
            >
              Get Started
              <ArrowRight className="size-4" />
            </Button>
          </div>
        </section>
      </main>
    </div>
  );
}