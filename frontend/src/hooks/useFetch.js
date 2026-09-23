import { useEffect, useState } from 'react'
export function useFetch(loader) { const [data, setData] = useState(null); const [loading, setLoading] = useState(true); const [error, setError] = useState(''); useEffect(() => { loader().then(setData).catch(e => setError(e.message)).finally(() => setLoading(false)) }, []); return {data, loading, error} }
