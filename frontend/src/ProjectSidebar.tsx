interface Props {
  projects: string[]
  selected: string | null
  onSelect: (project: string) => void
  onSignOut: () => void
  refetching: boolean
  onRefresh: () => void
}

export function ProjectSidebar({
  projects,
  selected,
  onSelect,
  onSignOut,
  refetching,
  onRefresh,
}: Props) {
  return (
    <aside className="sidebar">
      <div className="sidebar-head">
        <div className="brand">
          <span className="logo">◆</span>
          <span>Docs Chatbot</span>
        </div>
      </div>

      <div className="sidebar-sub">
        <span>Projects</span>
        <button
          className="icon-btn"
          onClick={onRefresh}
          disabled={refetching}
          title="Refresh projects"
        >
          {refetching ? '…' : '↻'}
        </button>
      </div>

      <nav className="project-list">
        {projects.length === 0 ? (
          <div className="empty-projects">No accessible projects.</div>
        ) : (
          projects.map((p) => (
            <button
              key={p}
              className={`project-item ${p === selected ? 'active' : ''}`}
              onClick={() => onSelect(p)}
            >
              <span className="dot" aria-hidden />
              <span className="project-name">{p}</span>
            </button>
          ))
        )}
      </nav>

      <div className="sidebar-foot">
        <button className="sign-out" onClick={onSignOut}>
          Change PAT
        </button>
      </div>
    </aside>
  )
}
