import { useEffect, useRef, useState } from 'react'

export function usePersistentState<T>(
  key: string,
  initial: T,
  options?: { sanitize?: (value: T) => T },
): [T, React.Dispatch<React.SetStateAction<T>>] {
  const sanitize = options?.sanitize
  const [value, setValue] = useState<T>(() => {
    try {
      const raw = localStorage.getItem(key)
      if (raw == null) return initial
      const parsed = JSON.parse(raw) as T
      return sanitize ? sanitize(parsed) : parsed
    } catch {
      return initial
    }
  })

  const sanitizeRef = useRef(sanitize)
  sanitizeRef.current = sanitize

  useEffect(() => {
    try {
      const toStore = sanitizeRef.current ? sanitizeRef.current(value) : value
      localStorage.setItem(key, JSON.stringify(toStore))
    } catch {
      // ignore quota / serialization errors
    }
  }, [key, value])

  return [value, setValue]
}

export function clearPersistentState(key: string): void {
  try {
    localStorage.removeItem(key)
  } catch {
    // ignore
  }
}
