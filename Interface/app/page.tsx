"use client"

import { useEffect, useState } from "react"
import Image from "next/image"
import { Globe, Lock, ShieldCheck, Sparkles } from "lucide-react"
import { ChatInterface } from "@/components/chat-interface"
import { t, type Lang } from "@/lib/i18n"

export default function Page() {
  const [lang, setLang] = useState<Lang>("ar")
  const d = t[lang]

  // Keep the document direction in sync with the selected language
  useEffect(() => {
    document.documentElement.lang = lang
    document.documentElement.dir = d.dir
  }, [lang, d.dir])

  return (
    <main className="min-h-screen bg-background text-foreground">
      <SiteHeader lang={lang} onToggle={() => setLang((l) => (l === "ar" ? "en" : "ar"))} />

      <section className="mx-auto grid max-w-6xl items-center gap-10 px-4 py-10 sm:px-6 lg:grid-cols-2 lg:gap-12 lg:py-16">
        {/* Hero copy */}
        <div className="text-center lg:text-start">
          <span className="inline-flex items-center gap-2 rounded-full border border-lime/40 bg-lime/15 px-4 py-1.5 text-xs font-bold uppercase tracking-wide text-foreground">
            <Sparkles className="h-3.5 w-3.5 text-lime" />
            {d.badge}
          </span>

          <h1 className="mt-5 text-pretty font-heading text-4xl font-extrabold leading-tight sm:text-5xl">
            {d.heroTitle}
            <span className="text-accent">{d.heroHighlight}</span>
            {d.heroTitleEnd}
          </h1>

          <p className="mx-auto mt-5 max-w-xl text-pretty text-base leading-relaxed text-muted-foreground lg:mx-0">
            {d.heroSubtitle}
          </p>

          <div className="mt-7 flex flex-wrap items-center justify-center gap-3 lg:justify-start">
            <a
              href="#chat"
              className="rounded-full bg-primary px-6 py-3 text-sm font-bold text-primary-foreground transition hover:opacity-90"
            >
              {d.startChat}
            </a>
            <a
              href="#about"
              className="rounded-full border border-border bg-card px-6 py-3 text-sm font-bold text-foreground transition hover:border-primary"
            >
              {d.learnMore}
            </a>
          </div>

          <div className="mt-8 flex flex-wrap items-center justify-center gap-x-6 gap-y-3 lg:justify-start">
            <Feature icon={<Lock className="h-4 w-4" />} text={d.anonymous} />
            <Feature icon={<ShieldCheck className="h-4 w-4" />} text={d.nav.safety} />
          </div>
        </div>

        {/* Chat */}
        <div id="chat" className="scroll-mt-24">
          <ChatInterface key={lang} lang={lang} />
        </div>
      </section>

      {/* Anonymous reassurance band */}
      <section id="about" className="border-t border-border bg-primary text-primary-foreground">
        <div className="mx-auto flex max-w-4xl flex-col items-center gap-4 px-4 py-12 text-center sm:px-6">
          <span className="flex h-14 w-14 items-center justify-center rounded-2xl bg-primary-foreground/10">
            <Lock className="h-7 w-7" />
          </span>
          <h2 className="font-heading text-2xl font-bold">{d.anonymous}</h2>
          <p className="max-w-2xl text-pretty leading-relaxed text-primary-foreground/85">
            {d.anonymousDesc}
          </p>
        </div>
      </section>

      <SiteFooter lang={lang} />
    </main>
  )
}

function Feature({ icon, text }: { icon: React.ReactNode; text: string }) {
  return (
    <span className="flex items-center gap-2 text-sm font-medium text-foreground">
      <span className="flex h-7 w-7 items-center justify-center rounded-full bg-primary/10 text-primary">
        {icon}
      </span>
      {text}
    </span>
  )
}

function SiteHeader({ lang, onToggle }: { lang: Lang; onToggle: () => void }) {
  const d = t[lang]
  return (
    <header className="sticky top-0 z-30 border-b border-border bg-background/85 backdrop-blur">
      <div className="mx-auto flex max-w-6xl items-center justify-between gap-4 px-4 py-3 sm:px-6">
        <a href="#" className="flex items-center gap-2.5">
          <div className="relative h-10 w-10 overflow-hidden rounded-xl">
            <Image src="/elmedia-logo.png" alt="El Media Lab" fill className="object-contain" />
          </div>
          <span className="font-heading text-lg font-extrabold leading-none">
            <span className="text-primary">Tanit</span>
            <span className="text-lime">Bot</span>
          </span>
        </a>

        <nav className="hidden items-center gap-7 text-sm font-medium text-foreground md:flex">
          <a href="#" className="transition hover:text-primary">{d.nav.home}</a>
          <a href="#about" className="transition hover:text-primary">{d.nav.about}</a>
          <a href="#chat" className="transition hover:text-primary">{d.nav.safety}</a>
        </nav>

        <button
          onClick={onToggle}
          className="flex items-center gap-1.5 rounded-full border border-border bg-card px-3.5 py-2 text-sm font-bold text-foreground transition hover:border-primary"
          aria-label="Switch language"
        >
          <Globe className="h-4 w-4 text-primary" />
          {d.langLabel}
        </button>
      </div>
    </header>
  )
}

function SiteFooter({ lang }: { lang: Lang }) {
  const isAr = lang === "ar"
  return (
    <footer className="bg-foreground text-background">
      <div className="mx-auto flex max-w-6xl flex-col items-center justify-between gap-3 px-4 py-6 text-center sm:flex-row sm:px-6 sm:text-start">
        <p className="text-sm">
          {isAr ? "تانيت بوت — مشروع من " : "TanitBot — a project by "}
          <span className="font-bold">El Media Lab</span>
        </p>
        <p className="text-xs text-background/60">
          {isAr ? "© 2026 جميع الحقوق محفوظة" : "© 2026 All rights reserved"}
        </p>
      </div>
    </footer>
  )
}
