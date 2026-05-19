import { useState, type FormEvent } from 'react'
import { useMutation } from '@tanstack/react-query'
import { fetchProjects } from './api'

interface Props {
  onAuthenticated: (pat: string, projects: string[]) => void
}

export function PatGate({ onAuthenticated }: Props) {
  const [pat, setPat] = useState('')
  const [show, setShow] = useState(false)

  const mutation = useMutation({
    mutationFn: (token: string) => fetchProjects(token),
    onSuccess: (projects, token) => onAuthenticated(token, projects),
  })

  const submit = (e: FormEvent) => {
    e.preventDefault()
    const token = pat.trim()
    if (!token || mutation.isPending) return
    mutation.mutate(token)
  }

  return (
    <div className="gate">
      <form className="gate-card" onSubmit={submit}>
        <div className="gate-logo">◆</div>
        <h1>Connect to Docs Chatbot</h1>
        <p className="gate-sub">
          Enter your Azure DevOps Personal Access Token to load the projects you have access to.
        </p>

        <label className="gate-label" htmlFor="pat">Personal Access Token</label>
        <div className="gate-input-wrap">
          <input
            id="pat"
            type={show ? 'text' : 'password'}
            value={pat}
            onChange={(e) => setPat(e.target.value)}
            placeholder="Paste your PAT"
            autoComplete="off"
            spellCheck={false}
            disabled={mutation.isPending}
          />
          <button
            type="button"
            className="gate-toggle"
            onClick={() => setShow((s) => !s)}
            tabIndex={-1}
          >
            {show ? 'Hide' : 'Show'}
          </button>
        </div>

        {mutation.isError && (
          <div className="gate-error">{mutation.error?.message || 'Authentication failed.'}</div>
        )}

        <button type="submit" className="gate-submit" disabled={!pat.trim() || mutation.isPending}>
          {mutation.isPending ? 'Verifying…' : 'Continue'}
        </button>

        <p className="gate-foot">
          Your PAT is kept only in this browser tab (sessionStorage) and sent on each API request.
        </p>
      </form>
    </div>
  )
}
