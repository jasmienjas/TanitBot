"use client"

import { useEffect, useRef, useState } from "react"
import Image from "next/image"
import {
  Database,
  Lock,
  Megaphone,
  RotateCcw,
  Send,
  Shield,
  ShieldCheck,
} from "lucide-react"
import { t, type Lang, type Topic } from "@/lib/i18n"
import { cn } from "@/lib/utils"

type Message = { id: string; role: "user" | "assistant"; content: string }

const topicIcons = {
  shield: Shield,
  lock: Lock,
  database: Database,
  megaphone: Megaphone,
}

export function ChatInterface({ lang }: { lang: Lang }) {
  const d = t[lang]
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState("")
  const [isStreaming, setIsStreaming] = useState(false)
  const scrollRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" })
  }, [messages, isStreaming])

  async function sendMessage(text: string) {
    const content = text.trim()
    if (!content || isStreaming) return

    const userMsg: Message = { id: crypto.randomUUID(), role: "user", content }
    const history = [...messages, userMsg]
    setMessages(history)
    setInput("")
    setIsStreaming(true)

    const assistantId = crypto.randomUUID()
    setMessages((prev) => [...prev, { id: assistantId, role: "assistant", content: "" }])

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          lang,
          messages: history.map((m) => ({ role: m.role, content: m.content })),
        }),
      })

      if (!res.body) throw new Error("no body")
      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let acc = ""

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        acc += decoder.decode(value, { stream: true })
        setMessages((prev) =>
          prev.map((m) => (m.id === assistantId ? { ...m, content: acc } : m)),
        )
      }
    } catch {
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantId
            ? { ...m, content: lang === "ar" ? "صار خطأ، عاود من فضلك." : "Something went wrong. Please try again." }
            : m,
        ),
      )
    } finally {
      setIsStreaming(false)
      textareaRef.current?.focus()
    }
  }

  const hasMessages = messages.length > 0

  return (
    <div className="flex h-[640px] max-h-[80vh] flex-col overflow-hidden rounded-3xl border border-border bg-card shadow-sm">
      {/* Header */}
      <header className="flex items-center justify-between gap-3 border-b border-border bg-primary px-5 py-4 text-primary-foreground">
        <div className="flex items-center gap-3">
          <div className="relative h-11 w-11 overflow-hidden rounded-xl bg-card">
            <Image src="/tanitbot-logo.png" alt="TanitBot" fill className="object-contain p-0.5" />
          </div>
          <div className="leading-tight">
            <p className="font-heading text-base font-bold">{d.chatTitle}</p>
            <span className="flex items-center gap-1.5 text-xs text-primary-foreground/80">
              <span className="inline-block h-2 w-2 rounded-full bg-lime" aria-hidden />
              {d.chatStatus}
            </span>
          </div>
        </div>
        {hasMessages && (
          <button
            onClick={() => setMessages([])}
            className="flex items-center gap-1.5 rounded-full bg-primary-foreground/15 px-3 py-1.5 text-xs font-medium transition hover:bg-primary-foreground/25"
          >
            <RotateCcw className="h-3.5 w-3.5" />
            {d.newChat}
          </button>
        )}
      </header>

      {/* Messages */}
      <div ref={scrollRef} className="flex-1 space-y-4 overflow-y-auto px-4 py-5 sm:px-6">
        {/* Greeting + topics shown when empty */}
        {!hasMessages && (
          <div className="space-y-5">
            <Bubble role="assistant" lang={lang}>
              {d.greeting}
            </Bubble>

            <div>
              <p className="mb-2 px-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                {d.topicsLabel}
              </p>
              <div className="grid grid-cols-1 gap-2.5 sm:grid-cols-2">
                {d.topics.map((topic: Topic) => {
                  const Icon = topicIcons[topic.icon]
                  return (
                    <button
                      key={topic.id}
                      onClick={() => sendMessage(topic.prompt)}
                      className="group flex items-start gap-3 rounded-2xl border border-border bg-background p-3 text-start transition hover:border-primary hover:shadow-sm"
                    >
                      <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-primary/10 text-primary">
                        <Icon className="h-5 w-5" />
                      </span>
                      <span className="leading-tight">
                        <span className="block text-sm font-bold text-foreground">{topic.title}</span>
                        <span className="block text-xs text-muted-foreground">{topic.desc}</span>
                      </span>
                    </button>
                  )
                })}
              </div>
            </div>
          </div>
        )}

        {messages.map((m) => (
          <Bubble key={m.id} role={m.role} lang={lang}>
            {m.content || (isStreaming ? <TypingDots /> : "")}
          </Bubble>
        ))}
      </div>

      {/* Composer */}
      <div className="border-t border-border bg-card px-4 py-3 sm:px-5">
        <form
          onSubmit={(e) => {
            e.preventDefault()
            sendMessage(input)
          }}
          className="flex items-end gap-2"
        >
          <textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault()
                sendMessage(input)
              }
            }}
            rows={1}
            placeholder={d.inputPlaceholder}
            className="max-h-32 min-h-[44px] flex-1 resize-none rounded-2xl border border-input bg-background px-4 py-2.5 text-sm text-foreground outline-none transition focus:border-primary focus:ring-2 focus:ring-primary/20"
          />
          <button
            type="submit"
            disabled={!input.trim() || isStreaming}
            aria-label={d.send}
            className="flex h-11 w-11 shrink-0 items-center justify-center rounded-2xl bg-primary text-primary-foreground transition hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-40"
          >
            <Send className={cn("h-5 w-5", lang === "ar" && "-scale-x-100")} />
          </button>
        </form>
        <p className="mt-2 flex items-center justify-center gap-1.5 text-center text-[11px] text-muted-foreground">
          <ShieldCheck className="h-3.5 w-3.5 text-primary" />
          {d.disclaimer}
        </p>
      </div>
    </div>
  )
}

function Bubble({
  role,
  lang,
  children,
}: {
  role: "user" | "assistant"
  lang: Lang
  children: React.ReactNode
}) {
  const isUser = role === "user"
  return (
    <div className={cn("flex", isUser ? "justify-end" : "justify-start")}>
      {!isUser && (
        <div className="relative mt-0.5 me-2 h-8 w-8 shrink-0 self-start overflow-hidden rounded-lg bg-card ring-1 ring-border">
          <Image src="/tanitbot-logo.png" alt="" fill className="object-contain p-0.5" />
        </div>
      )}
      <div
        className={cn(
          "max-w-[80%] whitespace-pre-wrap rounded-2xl px-4 py-2.5 text-sm leading-relaxed",
          isUser
            ? "rounded-ee-sm bg-primary text-primary-foreground"
            : "rounded-es-sm border border-border bg-background text-foreground",
        )}
      >
        {children}
      </div>
    </div>
  )
}

function TypingDots() {
  return (
    <span className="flex items-center gap-1 py-1">
      <span className="h-2 w-2 animate-bounce rounded-full bg-muted-foreground/50 [animation-delay:-0.3s]" />
      <span className="h-2 w-2 animate-bounce rounded-full bg-muted-foreground/50 [animation-delay:-0.15s]" />
      <span className="h-2 w-2 animate-bounce rounded-full bg-muted-foreground/50" />
    </span>
  )
}
