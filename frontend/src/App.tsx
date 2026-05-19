import { useCallback, useEffect, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { PatGate } from './PatGate'
import { ProjectSidebar } from './ProjectSidebar'
import { ChatPanel, type Message } from './ChatPanel'
import { fetchProjects } from './api'
import { clearPat, loadPat, savePat } from './auth'
import './App.css'

function App() {
  const [pat, setPat] = useState<string>(() => loadPat())
  const [selected, setSelected] = useState<string | null>(null)
  const [history, setHistory] = useState<Record<string, Message[]>>({})

  const projectsQuery = useQuery({
    queryKey: ['projects', pat],
    queryFn: () => fetchProjects(pat),
    enabled: !!pat,
    staleTime: 60_000,
  })

  useEffect(() => {
    if (!projectsQuery.data) return
    if (selected && projectsQuery.data.includes(selected)) return
    setSelected(projectsQuery.data[0] ?? null)
  }, [projectsQuery.data, selected])

  useEffect(() => {
    if (projectsQuery.isError) {
      // PAT became invalid — drop it and bounce to gate.
      clearPat()
      setPat('')
      setSelected(null)
      setHistory({})
    }
  }, [projectsQuery.isError])

  const handleAuthenticated = useCallback((token: string, projects: string[]) => {
    savePat(token)
    setPat(token)
    setSelected(projects[0] ?? null)
  }, [])

  const signOut = useCallback(() => {
    clearPat()
    setPat('')
    setSelected(null)
    setHistory({})
  }, [])

  const updateHistory = useCallback(
    (project: string, updater: (prev: Message[]) => Message[]) => {
      setHistory((prev) => ({ ...prev, [project]: updater(prev[project] ?? []) }))
    },
    [],
  )

  if (!pat) {
    return <PatGate onAuthenticated={handleAuthenticated} />
  }

  const projects = projectsQuery.data ?? []

  return (
    <div className="app">
      <ProjectSidebar
        projects={projects}
        selected={selected}
        onSelect={setSelected}
        onSignOut={signOut}
        refetching={projectsQuery.isFetching}
        onRefresh={() => projectsQuery.refetch()}
      />

      <main className="main">
        {projectsQuery.isLoading ? (
          <CenteredNote>Loading projects…</CenteredNote>
        ) : projects.length === 0 ? (
          <CenteredNote>
            <strong>No accessible projects.</strong>
            <span>This PAT has no read access to any ingested project.</span>
          </CenteredNote>
        ) : !selected ? (
          <CenteredNote>Select a project to start chatting.</CenteredNote>
        ) : (
          <ChatPanel
            pat={pat}
            project={selected}
            messages={history[selected] ?? []}
            onChange={(updater) => updateHistory(selected, updater)}
          />
        )}
      </main>
    </div>
  )
}

function CenteredNote({ children }: { children: React.ReactNode }) {
  return (
    <div className="centered-note">
      <div>{children}</div>
    </div>
  )
}

export default App
