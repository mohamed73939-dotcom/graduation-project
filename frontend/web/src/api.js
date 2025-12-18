// API client with configurable base URL via env
// Set VITE_API_URL="http://localhost:8000" in .env if needed
import axios from 'axios'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export async function summarizeVideo(file, language = 'auto') {
  const form = new FormData()
  form.append('video', file)
  form.append('language', language)
  const res = await axios.post(`${API_URL}/api/summarize`, form, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 1000 * 60 * 30 // 30 min for large videos
  })
  return res.data
}