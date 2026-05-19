export interface ChatResponse {
  answer: string
  used_context: boolean
  project: string | null
}

export interface ProjectsResponse {
  projects: string[]
}

const PAT_HEADER = 'X-Azure-Devops-Pat'

async function parseError(res: Response): Promise<string> {
  const text = await res.text().catch(() => '')
  if (!text) return `Request failed (${res.status})`
  try {
    const data = JSON.parse(text)
    if (typeof data?.detail === 'string') return data.detail
    if (Array.isArray(data?.detail)) return data.detail.map((d: { msg?: string }) => d.msg).filter(Boolean).join(', ')
  } catch {
    // fall through
  }
  return text
}

export async function fetchProjects(pat: string): Promise<string[]> {
  const res = await fetch('/api/projects', {
    headers: { [PAT_HEADER]: pat },
  })
  if (!res.ok) throw new Error(await parseError(res))
  const data = (await res.json()) as ProjectsResponse
  return data.projects ?? []
}

export async function sendChat(args: {
  pat: string
  question: string
  project: string
}): Promise<ChatResponse> {
  const res = await fetch('/api/chat', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      [PAT_HEADER]: args.pat,
    },
    body: JSON.stringify({ question: args.question, project: args.project }),
  })
  if (!res.ok) throw new Error(await parseError(res))
  return res.json()
}
