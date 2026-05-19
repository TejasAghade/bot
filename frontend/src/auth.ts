const STORAGE_KEY = 'azdo_pat'

export function loadPat(): string {
  try {
    return sessionStorage.getItem(STORAGE_KEY) ?? ''
  } catch {
    return ''
  }
}

export function savePat(pat: string): void {
  try {
    sessionStorage.setItem(STORAGE_KEY, pat)
  } catch {
    // ignore
  }
}

export function clearPat(): void {
  try {
    sessionStorage.removeItem(STORAGE_KEY)
  } catch {
    // ignore
  }
}
