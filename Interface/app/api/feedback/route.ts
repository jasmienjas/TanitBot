import type { NextRequest } from "next/server"
import { JWT } from "google-auth-library"

export const runtime = "nodejs"

// Accepts the key with literal \n escapes, real newlines, or newlines stripped
// entirely (common when pasting the JSON key value into .env) and rebuilds valid PEM.
function normalizePrivateKey(raw: string): string {
  let key = raw.trim().replace(/^["']|["']$/g, "").replace(/\\n/g, "\n")
  if (!key.includes("\n")) {
    const body = key
      .replace("-----BEGIN PRIVATE KEY-----", "")
      .replace("-----END PRIVATE KEY-----", "")
      .replace(/\s+/g, "")
    key = `-----BEGIN PRIVATE KEY-----\n${body.replace(/(.{64})/g, "$1\n").trim()}\n-----END PRIVATE KEY-----\n`
  }
  return key
}

type FeedbackBody = {
  rating: number
  comment?: string
  q1?: boolean | null
  q2?: boolean | null
  lang?: "ar" | "en"
  qa?: string
}

export async function POST(req: NextRequest) {
  try {
    const body = (await req.json()) as FeedbackBody

    const rating = Number(body.rating)
    if (!Number.isInteger(rating) || rating < 1 || rating > 5) {
      return new Response("invalid rating", { status: 400 })
    }
    const comment = (body.comment ?? "").slice(0, 2000)
    const qa = (body.qa ?? "").slice(0, 2000)
    const lang = body.lang === "en" ? "en" : "ar"
    const toCell = (v: boolean | null | undefined) => (v === true ? "yes" : v === false ? "no" : "")

    const sheetId = process.env.FEEDBACK_SHEET_ID
    const email = process.env.GOOGLE_SERVICE_ACCOUNT_EMAIL
    const privateKey = process.env.GOOGLE_PRIVATE_KEY

    // Never surface a feedback-storage problem to the chat user
    if (!sheetId || !email || !privateKey) {
      console.error("[feedback] Missing FEEDBACK_SHEET_ID / GOOGLE_SERVICE_ACCOUNT_EMAIL / GOOGLE_PRIVATE_KEY")
      return new Response(null, { status: 204 })
    }

    const jwt = new JWT({
      email,
      key: normalizePrivateKey(privateKey),
      scopes: ["https://www.googleapis.com/auth/spreadsheets"],
    })
    const { token } = await jwt.getAccessToken()
    if (!token) throw new Error("failed to obtain access token")

    const upstream = await fetch(
      `https://sheets.googleapis.com/v4/spreadsheets/${sheetId}/values/A1:append?valueInputOption=RAW`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          values: [
            [new Date().toISOString(), rating, comment, toCell(body.q1), toCell(body.q2), lang, qa],
          ],
        }),
      },
    )

    if (!upstream.ok) {
      console.error("[feedback] Sheets append failed:", upstream.status, await upstream.text())
      return new Response("upstream error", { status: 500 })
    }
    return new Response(null, { status: 204 })
  } catch (err) {
    console.error("[feedback] error:", (err as Error).message)
    return new Response("error", { status: 500 })
  }
}
