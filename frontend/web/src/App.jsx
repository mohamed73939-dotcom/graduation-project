import React, { useState } from 'react'
import { summarizeVideo } from './api'
import { Upload, FileVideo, Languages, Loader2, Download, FileText, Sparkles, CheckCircle, AlertCircle, ClipboardCopy } from 'lucide-react'

export default function App() {
  const [file, setFile] = useState(null)
  const [language, setLanguage] = useState('auto')
  const [processing, setProcessing] = useState(false)
  const [progressText, setProgressText] = useState('')
  const [summary, setSummary] = useState('')
  const [transcription, setTranscription] = useState('')
  const [detectedLanguage, setDetectedLanguage] = useState('unknown')
  const [confidence, setConfidence] = useState(0)
  const [latency, setLatency] = useState(0)
  const [chunkMeta, setChunkMeta] = useState([])
  const [error, setError] = useState('')

  const handleFileUpload = (e) => {
    const uploaded = e.target.files?.[0]
    if (uploaded && uploaded.type.startsWith('video/')) {
      setFile(uploaded)
      setError('')
      setSummary('')
      setTranscription('')
      setDetectedLanguage('unknown')
    } else {
      setError('الرجاء اختيار ملف فيديو صالح')
    }
  }

  const startSummarize = async () => {
    if (!file) return
    setProcessing(true)
    setError('')
    setProgressText('رفع الملف وبدء المعالجة...')
    try {
      const res = await summarizeVideo(file, language)
      if (res.status === 'success') {
        const data = res.data || {}
        setSummary(data.summary || '')
        setTranscription(data.transcription || '')
        setDetectedLanguage(data.detected_language || 'unknown')
        setConfidence(data.confidence?.avg_logprob ?? 0)
        setLatency(data.metrics?.latency_seconds ?? 0)
        setChunkMeta(data.chunk_metadata ?? [])
        setProgressText('اكتمل التلخيص')
      } else {
        setError(res.message || 'تعذر إكمال العملية')
      }
    } catch (e) {
      setError(e?.response?.data?.message || e.message || 'خطأ غير متوقع')
    } finally {
      setProcessing(false)
    }
  }

  const downloadText = (txt, filename) => {
    const blob = new Blob([txt], { type: 'text/plain' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = filename
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  const copyToClipboard = (txt) => {
    navigator.clipboard?.writeText(txt)
  }

  const rtl = (detectedLanguage || language) === 'ar'
  const dirCls = rtl ? 'rtl' : 'ltr'

  return (
    <div className="min-h-screen">
      {/* Header */}
      <div className="bg-white shadow-sm border-b">
        <div className="max-w-6xl mx-auto px-4 py-4">
          <div className="flex items-center gap-3">
            <div className="bg-gradient-to-r from-purple-600 to-blue-600 p-2 rounded-xl">
              <Sparkles className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-bold bg-gradient-to-r from-purple-600 to-blue-600 bg-clip-text text-transparent">
                Sidecut
              </h1>
              <p className="text-sm text-gray-600">نظام تلخيص المحاضرات بالذكاء الاصطناعي</p>
            </div>
          </div>
        </div>
      </div>

      {/* Body */}
      <div className="max-w-5xl mx-auto px-4 py-8">
        {!summary && (
          <div className="bg-white rounded-2xl shadow-lg p-8 mb-6">
            <div className="text-center mb-6">
              <h2 className="text-2xl font-bold text-gray-800 mb-2">ارفع محاضرتك</h2>
              <p className="text-gray-600">سنقوم بتحليل الفيديو وإنشاء ملخص شامل تلقائياً</p>
            </div>

            {/* File Upload */}
            <div className="mb-6">
              <label className="flex flex-col items-center justify-center w-full h-48 border-2 border-dashed border-purple-300 rounded-xl cursor-pointer hover:bg-purple-50 transition-all bg-gradient-to-br from-purple-50/50 to-blue-50/50">
                <div className="flex flex-col items-center justify-center pt-5 pb-6">
                  {file ? (
                    <>
                      <CheckCircle className="w-12 h-12 text-green-500 mb-3" />
                      <p className="text-sm font-medium text-gray-700">{file.name}</p>
                      <p className="text-xs text-gray-500 mt-1">{(file.size / (1024 * 1024)).toFixed(2)} MB</p>
                    </>
                  ) : (
                    <>
                      <FileVideo className="w-12 h-12 text-purple-400 mb-3" />
                      <p className="text-sm font-medium text-gray-700">اضغط لرفع الفيديو</p>
                      <p className="text-xs text-gray-500 mt-1">MP4, AVI, MOV, MKV (حتى 500MB)</p>
                    </>
                  )}
                </div>
                <input type="file" className="hidden" accept="video/*" onChange={handleFileUpload} />
              </label>
            </div>

            {/* Language */}
            <div className="mb-6">
              <label className="block text-sm font-medium text-gray-700 mb-2 flex items-center gap-2">
                <Languages className="w-4 h-4" />
                لغة المحاضرة
              </label>
              <div className="grid grid-cols-3 gap-3">
                <button
                  onClick={() => setLanguage('auto')}
                  className={`p-3 rounded-lg border-2 transition-all ${language === 'auto' ? 'border-purple-600 bg-purple-50 text-purple-700' : 'border-gray-200 hover:border-purple-300'}`}
                >
                  🌍 تلقائي
                </button>
                <button
                  onClick={() => setLanguage('ar')}
                  className={`p-3 rounded-lg border-2 transition-all ${language === 'ar' ? 'border-purple-600 bg-purple-50 text-purple-700' : 'border-gray-200 hover:border-purple-300'}`}
                >
                  🇸🇦 العربية
                </button>
                <button
                  onClick={() => setLanguage('en')}
                  className={`p-3 rounded-lg border-2 transition-all ${language === 'en' ? 'border-purple-600 bg-purple-50 text-purple-700' : 'border-gray-200 hover:border-purple-300'}`}
                >
                  🇬🇧 English
                </button>
              </div>
            </div>

            {/* Error */}
            {error && (
              <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg flex items-start gap-2">
                <AlertCircle className="w-5 h-5 text-red-500 flex-shrink-0 mt-0.5" />
                <p className="text-sm text-red-700">{error}</p>
              </div>
            )}

            {/* Start */}
            <button
              onClick={startSummarize}
              disabled={!file || processing}
              className="w-full bg-gradient-to-r from-purple-600 to-blue-600 text-white py-4 rounded-xl font-semibold hover:shadow-lg transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
            >
              {processing ? (<><Loader2 className="w-5 h-5 animate-spin" /> جاري المعالجة...</>) : (<><Sparkles className="w-5 h-5" /> ابدأ التلخيص</>)}
            </button>

            {/* Progress */}
            {processing && (
              <div className="mt-4 text-center text-sm text-gray-600">{progressText}</div>
            )}
          </div>
        )}

        {/* Results */}
        {summary && (
          <div className="space-y-6">
            <div className="bg-white rounded-2xl shadow-lg p-8">
              <div className="flex items-center justify-between mb-6">
                <h3 className="text-2xl font-bold text-gray-800 flex items-center gap-2">
                  <CheckCircle className="w-6 h-6 text-green-500" />
                  الملخص جاهز
                </h3>
                <div className="flex gap-2">
                  <button
                    onClick={() => downloadText(summary, 'summary.txt')}
                    className="flex items-center gap-2 px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors"
                  >
                    <Download className="w-4 h-4" />
                    TXT
                  </button>
                  <button
                    onClick={() => copyToClipboard(summary)}
                    className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                  >
                    <ClipboardCopy className="w-4 h-4" />
                    نسخ
                  </button>
                </div>
              </div>

              {/* Direction-aware summary */}
              <div className={`bg-gray-50 rounded-xl p-6 whitespace-pre-line leading-relaxed text-gray-800 ${dirCls}`}>
                {summary}
              </div>

              {/* Info cards */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-6">
                <div className="bg-purple-50 p-4 rounded-lg">
                  <div className="text-sm text-gray-600">اللغة المكتشفة</div>
                  <div className="text-xl font-semibold">{(detectedLanguage || 'unknown').toUpperCase()}</div>
                </div>
                <div className="bg-blue-50 p-4 rounded-lg">
                  <div className="text-sm text-gray-600">متوسط الثقة</div>
                  <div className="text-xl font-semibold">{confidence.toFixed(3)}</div>
                </div>
                <div className="bg-green-50 p-4 rounded-lg">
                  <div className="text-sm text-gray-600">المدة (ث)</div>
                  <div className="text-xl font-semibold">{latency.toFixed(2)}</div>
                </div>
              </div>

              {/* Transcript */}
              <div className="mt-6">
                <details className="bg-white border rounded-lg">
                  <summary className="cursor-pointer px-4 py-3 font-medium">📄 النص الكامل (معاينة)</summary>
                  <pre className={`px-4 py-3 overflow-x-auto text-sm ${dirCls}`}>{transcription}</pre>
                </details>
              </div>

              {/* Chunks */}
              <div className="mt-4">
                <details className="bg-white border rounded-lg">
                  <summary className="cursor-pointer px-4 py-3 font-medium">🔍 تفاصيل المقاطع (Chunks)</summary>
                  {chunkMeta && chunkMeta.length > 0 ? (
                    <div className="px-4 py-3 text-sm overflow-auto">
                      <table className="min-w-full text-left">
                        <thead>
                          <tr>
                            <th className="px-2 py-1">ID</th>
                            <th className="px-2 py-1">Start</th>
                            <th className="px-2 py-1">End</th>
                            <th className="px-2 py-1">Conf</th>
                            <th className="px-2 py-1">Lang</th>
                            <th className="px-2 py-1">Tokens</th>
                          </tr>
                        </thead>
                        <tbody>
                          {chunkMeta.map((c) => (
                            <tr key={c.chunk_id}>
                              <td className="px-2 py-1">{String(c.chunk_id).slice(0,8)}…</td>
                              <td className="px-2 py-1">{c.start?.toFixed?.(1) ?? c.start}</td>
                              <td className="px-2 py-1">{c.end?.toFixed?.(1) ?? c.end}</td>
                              <td className="px-2 py-1">{(c.conf ?? 0).toFixed(3)}</td>
                              <td className="px-2 py-1">{(c.language || '').toUpperCase()}</td>
                              <td className="px-2 py-1">{c.tokens ?? c.token_count ?? 0}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  ) : (
                    <div className="px-4 py-3 text-sm text-gray-600">لا توجد بيانات مقاطع متاحة.</div>
                  )}
                </details>
              </div>
            </div>

            {/* New Upload */}
            <button
              onClick={() => { setFile(null); setSummary(''); setTranscription(''); setError(''); }}
              className="w-full bg-white text-purple-600 border-2 border-purple-600 py-4 rounded-xl font-semibold hover:bg-purple-50 transition-all flex items-center justify-center gap-2"
            >
              <Upload className="w-5 h-5" />
              تلخيص محاضرة جديدة
            </button>
          </div>
        )}

        {/* Features */}
        {!processing && !summary && (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-8">
            <div className="bg-white p-6 rounded-xl shadow-md text-center">
              <div className="w-12 h-12 bg-purple-100 rounded-full flex items-center justify-center mx-auto mb-3">
                <Sparkles className="w-6 h-6 text-purple-600" />
              </div>
              <h4 className="font-semibold text-gray-800 mb-2">ذكاء اصطناعي متقدم</h4>
              <p className="text-sm text-gray-600">استخدام نماذج Whisper و mT5 للدقة العالية</p>
            </div>
            <div className="bg-white p-6 rounded-xl shadow-md text-center">
              <div className="w-12 h-12 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-3">
                <Languages className="w-6 h-6 text-blue-600" />
              </div>
              <h4 className="font-semibold text-gray-800 mb-2">دعم متعدد اللغات</h4>
              <p className="text-sm text-gray-600">يدعم العربية والإنجليزية بدقة عالية</p>
            </div>
            <div className="bg-white p-6 rounded-xl shadow-md text-center">
              <div className="w-12 h-12 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-3">
                <CheckCircle className="w-6 h-6 text-green-600" />
              </div>
              <h4 className="font-semibold text-gray-800 mb-2">مفتوح المصدر</h4>
              <p className="text-sm text-gray-600">مرن وقابل للتطوير لاحتياجاتك</p>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}