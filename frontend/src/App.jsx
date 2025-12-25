import React, { useState } from 'react';

const API_URL = 'https://photo-analyzer-aa3j.onrender.com';

export default function PhotoAnalyzer() {
  const [uploadedImage, setUploadedImage] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [dragActive, setDragActive] = useState(false);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    const files = e.dataTransfer.files;
    if (files && files[0]) {
      handleImageUpload(files[0]);
    }
  };

  const handleImageUpload = (file) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      setUploadedImage(e.target.result);
      setAnalysisResult(null);
      analyzeImage(e.target.result);
    };
    reader.readAsDataURL(file);
  };

  const analyzeImage = async (imageData) => {
    setIsAnalyzing(true);
    try {
      const response = await fetch(`${API_URL}/api/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: imageData })
      });

      const data = await response.json();
      setAnalysisResult(data);
    } catch (error) {
      console.error('Erreur:', error);
      alert('Erreur lors de l\'analyse: ' + error.message);
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
      {/* Header */}
      <header className="border-b border-slate-700 bg-slate-900/50 backdrop-blur">
        <div className="max-w-6xl mx-auto px-6 py-6">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-cyan-500 rounded-lg flex items-center justify-center">
              ✨
            </div>
            <h1 className="text-3xl font-bold text-white">PhotoAI Analyzer</h1>
          </div>
          <p className="text-slate-400 mt-2">Analyse ta photo • Génère un prompt IA</p>
        </div>
      </header>

      {/* Main */}
      <main className="max-w-6xl mx-auto px-6 py-12">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          
          {/* Upload Zone */}
          <div className="lg:col-span-1">
            <div
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
              className={`border-2 border-dashed rounded-2xl p-8 text-center cursor-pointer transition-all ${
                dragActive
                  ? 'border-blue-500 bg-blue-500/10'
                  : 'border-slate-600 bg-slate-800/50 hover:bg-slate-800'
              }`}
            >
              <input
                type="file"
                accept="image/*"
                onChange={(e) => e.target.files && handleImageUpload(e.target.files[0])}
                className="hidden"
                id="image-input"
              />
              
              <label htmlFor="image-input" className="flex flex-col items-center gap-4 cursor-pointer">
                <div className="w-16 h-16 bg-gradient-to-br from-blue-500 to-cyan-500 rounded-xl flex items-center justify-center text-2xl">
                  📸
                </div>
                <div>
                  <p className="font-semibold text-white">Dépose ta photo ici</p>
                  <p className="text-sm text-slate-400">ou clique pour sélectionner</p>
                </div>
              </label>
            </div>

            {uploadedImage && (
              <div className="mt-6">
                <p className="text-sm font-semibold text-slate-300 mb-3">Aperçu</p>
                <img
                  src={uploadedImage}
                  alt="Preview"
                  className="w-full rounded-xl border border-slate-700 object-cover max-h-96"
                />
              </div>
            )}
          </div>

          {/* Results */}
          <div className="lg:col-span-2">
            {isAnalyzing ? (
              <div className="bg-slate-800/50 border border-slate-700 rounded-2xl p-12 flex items-center justify-center min-h-96">
                <div className="text-center">
                  <div className="inline-block">
                    <div className="w-12 h-12 border-4 border-slate-600 border-t-blue-500 rounded-full animate-spin"></div>
                  </div>
                  <p className="mt-4 text-lg font-semibold text-white">Analyse en cours...</p>
                  <p className="text-slate-400 text-sm mt-1">Cela peut prendre 10-15 secondes</p>
                </div>
              </div>
            ) : analysisResult ? (
              <div className="space-y-6">
                {/* Metrics */}
                <div className="bg-slate-800/50 border border-slate-700 rounded-2xl p-6">
                  <h2 className="text-xl font-bold text-white mb-4">📊 Analyse Technique</h2>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="bg-slate-900/50 rounded-xl p-4 border border-slate-700">
                      <p className="text-xs text-slate-400 mb-2">Netteté</p>
                      <p className="text-2xl font-bold text-white">
                        {analysisResult.sharpness?.toFixed(1) || 'N/A'}
                      </p>
                    </div>
                    <div className="bg-slate-900/50 rounded-xl p-4 border border-slate-700">
                      <p className="text-xs text-slate-400 mb-2">Luminosité</p>
                      <p className="text-2xl font-bold text-white">
                        {analysisResult.brightness?.toFixed(1) || 'N/A'}
                      </p>
                    </div>
                    <div className="bg-slate-900/50 rounded-xl p-4 border border-slate-700">
                      <p className="text-xs text-slate-400 mb-2">Contraste</p>
                      <p className="text-2xl font-bold text-white">
                        {analysisResult.contrast?.toFixed(1) || 'N/A'}
                      </p>
                    </div>
                    <div className="bg-slate-900/50 rounded-xl p-4 border border-slate-700">
                      <p className="text-xs text-slate-400 mb-2">Bruit</p>
                      <p className="text-2xl font-bold text-white">
                        {analysisResult.noise?.toFixed(1) || 'N/A'}
                      </p>
                    </div>
                  </div>
                </div>

                {/* Quality Score */}
                <div className="bg-gradient-to-br from-blue-500/20 to-cyan-500/20 border border-blue-500/30 rounded-2xl p-6">
                  <h2 className="text-xl font-bold text-white mb-4">⚡ Score Qualité</h2>
                  <div className="flex items-end gap-4">
                    <div>
                      <div className="text-5xl font-bold text-blue-400">
                        {analysisResult.quality_score || 0}
                      </div>
                      <p className="text-slate-300 mt-1">/ 100</p>
                    </div>
                  </div>
                </div>

                {/* Advice */}
                {analysisResult.advice && (
                  <div className="bg-slate-800/50 border border-slate-700 rounded-2xl p-6">
                    <h2 className="text-xl font-bold text-white mb-4">💡 Conseils</h2>
                    <ul className="space-y-2">
                      {analysisResult.advice.slice(0, 5).map((tip, idx) => (
                        <li key={idx} className="flex gap-3 text-slate-300 text-sm">
                          <span className="text-blue-400 font-bold">→</span>
                          {tip}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {/* New Analysis Button */}
                <button
                  onClick={() => {
                    setUploadedImage(null);
                    setAnalysisResult(null);
                  }}
                  className="w-full bg-gradient-to-r from-slate-700 to-slate-600 text-white font-semibold py-3 px-6 rounded-xl hover:from-slate-600 hover:to-slate-500 transition-all"
                >
                  Analyser une autre photo
                </button>
              </div>
            ) : uploadedImage && !isAnalyzing ? (
              <div className="bg-slate-800/50 border border-slate-700 rounded-2xl p-12 flex items-center justify-center min-h-96">
                <button
                  onClick={() => analyzeImage(uploadedImage)}
                  className="bg-gradient-to-r from-blue-500 to-cyan-500 text-white font-bold py-4 px-8 rounded-xl text-lg hover:shadow-lg hover:shadow-blue-500/50 transition-all"
                >
                  🚀 Lancer l'analyse
                </button>
              </div>
            ) : (
              <div className="bg-slate-800/50 border border-slate-700 rounded-2xl p-12 flex items-center justify-center min-h-96">
                <p className="text-slate-400 text-lg">Dépose une photo pour commencer</p>
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}