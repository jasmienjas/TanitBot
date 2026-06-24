import type { NextRequest } from "next/server"

export const runtime = "nodejs"
export const maxDuration = 60

type ChatMessage = { role: "user" | "assistant" | "system"; content: string }

const SYSTEM_PROMPT_AR = `أنت "تانيت بوت"، مساعد ذكي متخصص في الأمان الرقمي والخصوصية وحماية البيانات وحرية التعبير، موجّه للمستخدمين في تونس.
- أجب بالعربية التونسية الواضحة والبسيطة عندما يكتب المستخدم بالعربية.
- اعتمد فقط على المعلومات الموجودة في "السياق" أدناه عند توفّرها، ولا تختلق معلومات قانونية.
- قدّم خطوات عملية وواضحة، واذكر الحقوق حسب الإطار القانوني التونسي عند الاقتضاء.
- ذكّر المستخدم بأن محادثته مجهولة، وكن متعاطفاً وغير حكمي.
- في الحالات الخطيرة انصح باستشارة مختص أو منظمة حقوقية.`

const SYSTEM_PROMPT_EN = `You are "TanitBot", an AI assistant specialized in digital safety, privacy, data protection, and freedom of expression for users in Tunisia.
- Reply in clear, simple English when the user writes in English.
- Ground your answers in the provided "Context" when available, and never fabricate legal information.
- Give practical, step-by-step guidance and reference the Tunisian legal framework where relevant.
- Remind the user the conversation is anonymous; be empathetic and non-judgmental.
- For serious situations, advise consulting a professional or a digital rights organization.`

/**
 * RAG retrieval step.
 * TODO: Connect to your vector store (e.g. embeddings index of Tunisian digital
 * rights / data protection documents). Return the most relevant passages.
 */
async function retrieveContext(_query: string, _lang: "ar" | "en"): Promise<string[]> {
  // Example shape — replace with a real similarity search against your knowledge base:
  // const results = await vectorStore.query({ text: _query, topK: 4 })
  // return results.map((r) => r.text)
  return []
}

/**
 * Calls the RunPod-hosted Command-R model.
 * TODO: Set RUNPOD_ENDPOINT_URL and RUNPOD_API_KEY in Project Settings → Vars.
 * Adjust the request/response shape to match your RunPod handler.
 */
async function callCommandR(prompt: string): Promise<Response | null> {
  const endpoint = process.env.RUNPOD_ENDPOINT_URL
  const apiKey = process.env.RUNPOD_API_KEY
  if (!endpoint || !apiKey) return null

  const upstream = await fetch(endpoint, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      input: {
        prompt,
        max_new_tokens: 700,
        temperature: 0.3,
        stream: true,
      },
    }),
  })

  return upstream
}

function buildPrompt(messages: ChatMessage[], context: string[], lang: "ar" | "en") {
  const system = lang === "ar" ? SYSTEM_PROMPT_AR : SYSTEM_PROMPT_EN
  const ctxLabel = lang === "ar" ? "السياق المسترجع" : "Retrieved context"
  const ctxBlock = context.length ? `\n\n${ctxLabel}:\n${context.join("\n---\n")}` : ""
  const convo = messages
    .map((m) => `${m.role === "user" ? "User" : "Assistant"}: ${m.content}`)
    .join("\n")
  return `${system}${ctxBlock}\n\n${convo}\nAssistant:`
}

function fallbackReply(query: string, lang: "ar" | "en"): string {
  if (lang === "ar") {
    return `شكراً على سؤالك حول "${query.slice(0, 80)}".

في الوقت الحالي نموذج الذكاء الاصطناعي (Command-R على RunPod) ما زال غير مربوط بعد. إليك بعض النصائح العامة في انتظار التفعيل:

• استعمل كلمات سر قوية ومختلفة لكل حساب، وفعّل التحقق بخطوتين.
• لا تشارك معلوماتك الشخصية أو موقعك مع جهات مجهولة.
• راجع إعدادات الخصوصية في حساباتك بانتظام.
• في حالة تهديد أو ابتزاز، احتفظ بالأدلة ولا تحذفها.

لتفعيل الإجابات الذكية الكاملة، يجب ربط نقطة RunPod (RUNPOD_ENDPOINT_URL و RUNPOD_API_KEY) وقاعدة المعرفة الخاصة بالحقوق الرقمية.`
  }
  return `Thanks for your question about "${query.slice(0, 80)}".

The AI model (Command-R on RunPod) is not connected yet. Here is some general guidance in the meantime:

• Use strong, unique passwords for each account and enable two-factor authentication.
• Don't share personal details or your location with unknown parties.
• Review your privacy settings regularly.
• If you face a threat or blackmail, preserve evidence and do not delete it.

To enable full AI answers, connect the RunPod endpoint (RUNPOD_ENDPOINT_URL and RUNPOD_API_KEY) and your digital-rights knowledge base.`
}

export async function POST(req: NextRequest) {
  try {
    const rawBody = await req.json()
    const { messages, lang = "ar" } = rawBody as {
      messages: ChatMessage[]
      lang?: "ar" | "en"
    }

    const endpoint = process.env.RUNPOD_ENDPOINT_URL
    const apiKey = process.env.RUNPOD_API_KEY

    // If endpoint is set, forward the payload to the Python FastAPI backend
    if (endpoint) {
      console.log(`[Next.js] Forwarding chat request to: ${endpoint}`)
      try {
        const upstream = await fetch(endpoint, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            ...(apiKey ? { Authorization: `Bearer ${apiKey}` } : {})
          },
          body: JSON.stringify({ messages, lang })
        })

        if (upstream.ok && upstream.body) {
          return new Response(upstream.body, {
            headers: {
              "Content-Type": "text/plain; charset=utf-8",
              "Cache-Control": "no-cache",
            },
          })
        }
      } catch (fetchErr) {
        console.error("[Next.js] Failed to fetch from RunPod backend:", fetchErr)
      }
    }

    const lastUser = [...messages].reverse().find((m) => m.role === "user")
    const query = lastUser?.content ?? ""

    // 1) Retrieval-Augmented Generation: fetch relevant context locally
    const context = await retrieveContext(query, lang)

    // 2) Build the Command-R prompt
    const prompt = buildPrompt(messages, context, lang)

    // 3) Try the RunPod-hosted model
    const upstream = await callCommandR(prompt)

    if (upstream && upstream.ok && upstream.body) {
      // Stream the model output straight through to the client.
      return new Response(upstream.body, {
        headers: {
          "Content-Type": "text/plain; charset=utf-8",
          "Cache-Control": "no-cache",
        },
      })
    }

    // 4) Fallback: stream a helpful placeholder so the chat stays interactive
    const text = fallbackReply(query, lang)
    const encoder = new TextEncoder()
    const stream = new ReadableStream({
      async start(controller) {
        for (const chunk of text.split(/(\s+)/)) {
          controller.enqueue(encoder.encode(chunk))
          await new Promise((r) => setTimeout(r, 18))
        }
        controller.close()
      },
    })

    return new Response(stream, {
      headers: {
        "Content-Type": "text/plain; charset=utf-8",
        "Cache-Control": "no-cache",
      },
    })
  } catch (err) {
    console.log("[v0] chat route error:", (err as Error).message)
    return new Response("error", { status: 500 })
  }
}
