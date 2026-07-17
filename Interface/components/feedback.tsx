"use client"

import { useState } from "react"
import { Star } from "lucide-react"
import { t, type Lang } from "@/lib/i18n"
import { cn } from "@/lib/utils"

function StarRow({
  value,
  onSelect,
  size = "h-5 w-5",
}: {
  value: number
  onSelect: (n: number) => void
  size?: string
}) {
  const [hovered, setHovered] = useState(0)
  return (
    <div className="flex items-center gap-0.5" onMouseLeave={() => setHovered(0)}>
      {[1, 2, 3, 4, 5].map((n) => (
        <button
          key={n}
          type="button"
          onClick={() => onSelect(n)}
          onMouseEnter={() => setHovered(n)}
          aria-label={`${n}/5`}
          className="p-0.5 text-amber-400 transition hover:scale-110"
        >
          <Star className={cn(size, n <= (hovered || value) ? "fill-current" : "fill-transparent")} />
        </button>
      ))}
    </div>
  )
}

function YesNo({
  value,
  onChange,
  yes,
  no,
}: {
  value: boolean | null
  onChange: (v: boolean) => void
  yes: string
  no: string
}) {
  return (
    <div className="flex gap-2">
      {[
        { label: yes, v: true },
        { label: no, v: false },
      ].map(({ label, v }) => (
        <button
          key={label}
          type="button"
          onClick={() => onChange(v)}
          className={cn(
            "rounded-full border px-3 py-1 text-xs font-medium transition",
            value === v
              ? "border-primary bg-primary text-primary-foreground"
              : "border-border bg-background text-foreground hover:border-primary",
          )}
        >
          {label}
        </button>
      ))}
    </div>
  )
}

export function FeedbackPrompt({ lang, qa }: { lang: Lang; qa: string }) {
  const d = t[lang].feedback
  const [rating, setRating] = useState(0)
  const [open, setOpen] = useState(false)
  const [sent, setSent] = useState(false)
  const [q1, setQ1] = useState<boolean | null>(null)
  const [q2, setQ2] = useState<boolean | null>(null)
  const [comment, setComment] = useState("")
  const [includeQa, setIncludeQa] = useState(false)

  if (sent) return <p className="ps-10 text-xs text-muted-foreground">{d.thanks}</p>

  function submit() {
    setSent(true)
    setOpen(false)
    // Fire-and-forget: feedback storage must never disturb the chat
    fetch("/api/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ rating, comment, q1, q2, lang, qa: includeQa ? qa : "" }),
    }).catch(() => {})
  }

  return (
    <>
      <div className="flex items-center gap-2 ps-10">
        <span className="text-xs text-muted-foreground">{d.rateLabel}</span>
        <StarRow
          value={0}
          size="h-4 w-4"
          onSelect={(n) => {
            setRating(n)
            setOpen(true)
          }}
        />
      </div>

      {open && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-foreground/40 p-4"
          onClick={() => setOpen(false)}
        >
          <div
            role="dialog"
            aria-label={d.title}
            onClick={(e) => e.stopPropagation()}
            className="w-full max-w-sm space-y-4 rounded-3xl border border-border bg-card p-5 shadow-lg"
          >
            <p className="font-heading text-base font-bold text-foreground">{d.title}</p>
            <StarRow value={rating} onSelect={setRating} />
            <div className="space-y-1.5">
              <p className="text-sm text-foreground">{d.q1}</p>
              <YesNo value={q1} onChange={setQ1} yes={d.yes} no={d.no} />
            </div>
            <div className="space-y-1.5">
              <p className="text-sm text-foreground">{d.q2}</p>
              <YesNo value={q2} onChange={setQ2} yes={d.yes} no={d.no} />
            </div>
            <textarea
              value={comment}
              onChange={(e) => setComment(e.target.value)}
              rows={3}
              maxLength={2000}
              placeholder={d.commentPlaceholder}
              className="w-full resize-none rounded-2xl border border-input bg-background px-3 py-2 text-sm text-foreground outline-none transition focus:border-primary focus:ring-2 focus:ring-primary/20"
            />
            <label className="flex items-start gap-2 text-xs text-muted-foreground">
              <input
                type="checkbox"
                checked={includeQa}
                onChange={(e) => setIncludeQa(e.target.checked)}
                className="mt-0.5 accent-primary"
              />
              {d.includeQa}
            </label>
            <div className="flex justify-end gap-2">
              <button
                type="button"
                onClick={() => setOpen(false)}
                className="rounded-full px-4 py-2 text-sm font-medium text-muted-foreground transition hover:text-foreground"
              >
                {d.skip}
              </button>
              <button
                type="button"
                onClick={submit}
                className="rounded-full bg-primary px-4 py-2 text-sm font-medium text-primary-foreground transition hover:opacity-90"
              >
                {d.submit}
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  )
}
