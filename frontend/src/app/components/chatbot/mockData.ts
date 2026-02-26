import type { PersonaConfig, ProcessedDocument, ChatMessage, SecurityAlert } from "./types";

export const mockPersona: PersonaConfig = {
  name: "Jordan Chen",
  role: "Senior Full-Stack Engineer",
  location: "San Francisco, CA",
  tone: "Professional, thoughtful, concise",
  bio: "Experienced engineer with 7+ years building scalable web applications. Passionate about developer experience, distributed systems, and mentoring junior engineers.",
  avatarUrl: "https://images.unsplash.com/photo-1689600944138-da3b150d9cb8?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxwcm9mZXNzaW9uYWwlMjBoZWFkc2hvdCUyMHBvcnRyYWl0JTIwYnVzaW5lc3N8ZW58MXx8fHwxNzcyMTEwNTUzfDA&ixlib=rb-4.1.0&q=80&w=1080&utm_source=figma&utm_medium=referral",
  contactLinks: [
    { label: "GitHub", url: "https://github.com/jordanchen" },
    { label: "LinkedIn", url: "https://linkedin.com/in/jordanchen" },
    { label: "Email", url: "mailto:jordan@example.com" },
  ],
};

export const mockProcessedDocs: ProcessedDocument[] = [
  {
    id: "doc-1",
    name: "Jordan_Chen_Resume_2026.pdf",
    fileType: "resume",
    detectedType: "Resume",
    riskScore: 0.05,
    confidence: 0.97,
    metadataSummary: "7 years experience, 4 companies, MS Computer Science",
    extractedHighlights: [
      "Senior Engineer at Stripe (2022-Present)",
      "Full-Stack Developer at Notion (2019-2022)",
      "MS CS Stanford University",
    ],
  },
  {
    id: "doc-2",
    name: "Cover_Letter_Acme_Corp.pdf",
    fileType: "non_resume",
    detectedType: "Cover Letter",
    riskScore: 0.02,
    confidence: 0.94,
    metadataSummary: "Addressed to Acme Corp engineering team lead position",
  },
  {
    id: "doc-3",
    name: "Stanford_Transcript.pdf",
    fileType: "non_resume",
    detectedType: "Transcript",
    riskScore: 0.12,
    confidence: 0.91,
    metadataSummary: "GPA 3.8, focus on distributed systems and ML",
  },
];

export const mockChatHistory: ChatMessage[] = [
  {
    id: "msg-1",
    role: "user",
    content: "Tell me about your experience with distributed systems.",
    timestamp: new Date(Date.now() - 300000),
  },
  {
    id: "msg-2",
    role: "assistant",
    content:
      "I've worked extensively with distributed systems throughout my career. At Stripe, I led the redesign of our payment processing pipeline to handle 3x throughput using event-driven architecture with Kafka and gRPC microservices. Before that at Notion, I helped migrate our monolithic backend to a service-oriented architecture, reducing p99 latency by 40%. My graduate research at Stanford focused on consensus algorithms and fault-tolerant distributed databases, which gave me a strong theoretical foundation I apply every day in production systems.",
    timestamp: new Date(Date.now() - 290000),
    suggestions: [
      "What specific Kafka patterns did you use at Stripe?",
      "How did you handle data consistency across services?",
      "Tell me about your leadership experience on the Stripe project.",
    ],
    debug: {
      agentRounds: 2,
      toolCalls: [
        {
          tool: "semantic_search",
          args: "distributed systems experience",
          resultCount: 4,
        },
        {
          tool: "semantic_search",
          args: "Stripe architecture work",
          resultCount: 2,
        },
      ],
      finishReason: "stop",
    },
  },
];

export const mockSecurityAlerts: SecurityAlert[] = [
  {
    id: "alert-1",
    documentName: "Stanford_Transcript.pdf",
    riskScore: 0.12,
    message: "Minor formatting anomalies detected. Content verified as safe.",
    timestamp: new Date(Date.now() - 600000),
  },
];

export const modelConfig = {
  classifier: "gpt-4o-mini",
  extraction: "gpt-4o",
  answer: "gpt-4o",
};

export const availableTools = [
  { name: "Semantic Search", status: "active" as const },
  { name: "Web Search", status: "active" as const },
  { name: "GitHub", status: "active" as const },
  { name: "LinkedIn", status: "active" as const },
  { name: "Weather", status: "active" as const },
];
