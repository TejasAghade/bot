import { useEffect, useRef, useState, type FormEvent, type KeyboardEvent } from 'react'
import { useMutation } from '@tanstack/react-query'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { sendChat, type ChatResponse } from './api'

export type Role = 'user' | 'assistant'

export interface Message {
  id: string
  role: Role
  content: string
  usedContext?: boolean
  error?: boolean
}

const newId = () =>
  typeof crypto !== 'undefined' && 'randomUUID' in crypto
    ? crypto.randomUUID()
    : Math.random().toString(36).slice(2)

interface Props {
  pat: string
  project: string
  messages: Message[]
  onChange: (updater: (prev: Message[]) => Message[]) => void
}

export function ChatPanel({ pat, project, messages, onChange }: Props) {
  const [input, setInput] = useState('')
  const scrollRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  const chat = useMutation<ChatResponse, Error, { id: string; question: string }>({
    mutationFn: ({ question }) => sendChat({ pat, project, question }),
    onSuccess: (data, vars) => {
      onChange((prev) =>
        prev.map((m) =>
          m.id === vars.id
            ? { ...m, content: data.answer, usedContext: data.used_context }
            : m,
        ),
      )
    },
    onError: (err, vars) => {
      onChange((prev) =>
        prev.map((m) =>
          m.id === vars.id
            ? { ...m, content: err.message || 'Something went wrong.', error: true }
            : m,
        ),
      )
    },
  })

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' })
  }, [messages, chat.isPending])

  useEffect(() => {
    const ta = textareaRef.current
    if (!ta) return
    ta.style.height = 'auto'
    ta.style.height = Math.min(ta.scrollHeight, 200) + 'px'
  }, [input])

  useEffect(() => {
    setInput('')
    chat.reset()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [project])

  const submit = (e?: FormEvent) => {
    e?.preventDefault()
    const question = input.trim()
    if (!question || chat.isPending) return

    const userMsg: Message = { id: newId(), role: 'user', content: question }
    const placeholderId = newId()
    const placeholder: Message = { id: placeholderId, role: 'assistant', content: '' }
    onChange((prev) => [...prev, userMsg, placeholder])
    setInput('')
    chat.mutate({ id: placeholderId, question })
  }

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      submit()
    }
  }

  const isEmpty = messages.length === 0

  return (
    <div className="chat-panel">
      <header className="panel-header">
        <div className="panel-title">
          <span className="panel-eyebrow">Project</span>
          <span className="panel-project">{project}</span>
        </div>
      </header>

      <div className="chat" ref={scrollRef}>
        {isEmpty ? (
          <div className="empty">
            <h1>Ask about {project}</h1>
            <p>Questions are answered only from documents indexed for this project.</p>
          </div>
        ) : (
          <div className="messages">
            {messages.map((m) => (
              <MessageBubble
                key={m.id}
                message={m}
                pending={chat.isPending && m.role === 'assistant' && m.content === ''}
              />
            ))}
          </div>
        )}
      </div>

      <form className="composer" onSubmit={submit}>
        <div className="composer-inner">
          <textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder={`Message about ${project}…`}
            rows={1}
            disabled={chat.isPending}
          />
          <button
            type="submit"
            className="send"
            disabled={!input.trim() || chat.isPending}
            aria-label="Send"
          >
            <SendIcon />
          </button>
        </div>
        <p className="hint">Press Enter to send · Shift+Enter for newline</p>
      </form>
    </div>
  )
}

function MessageBubble({ message, pending }: { message: Message; pending: boolean }) {
  const isUser = message.role === 'user'
  return (
    <div className={`row ${isUser ? 'row-user' : 'row-assistant'}`}>
      <div className="avatar" aria-hidden>
        {isUser ? 'You' : 'AI'}
      </div>
      <div className="bubble-wrap">
        <div className={`bubble ${message.error ? 'bubble-error' : ''}`}>
          {pending ? (
            <span className="typing">
              <span></span>
              <span></span>
              <span></span>
            </span>
          ) : isUser || message.error ? (
            <div className="content">{message.content}</div>
          ) : (
            <div className="markdown">
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                components={{
                  a: ({ node: _node, ...props }) => (
                    <a {...props} target="_blank" rel="noopener noreferrer" />
                  ),
                }}
              >
                {message.content}
              </ReactMarkdown>
            </div>
          )}
        </div>
        {!isUser && message.usedContext === false && !pending && !message.error && (
          <div className="meta">No matching context — answered without retrieval.</div>
        )}
      </div>
    </div>
  )
}

function SendIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
      <path
        d="M4 12L20 4L12 20L10 13L4 12Z"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinejoin="round"
        strokeLinecap="round"
      />
    </svg>
  )
}
