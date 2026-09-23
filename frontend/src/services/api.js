const API = import.meta.env.VITE_API_URL || 'http://localhost:5001/api'
export async function getPlaces() { const r = await fetch(`${API}/places`); return r.json() }
export async function predict(payload) { const r = await fetch(`${API}/predict`, { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(payload) }); const data = await r.json(); if (!r.ok) throw new Error(data.detail || 'Prediction failed'); return data }
export async function getDashboard() { const r = await fetch(`${API}/dashboard`); return r.json() }
export { API }
