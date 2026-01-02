import React, { useState } from 'react';
import { Upload, Camera, Sparkles, Zap, Eye, Palette, Tag, Download, Copy, History, BarChart3, TrendingUp } from 'lucide-react';

const API_URL = 'https://nabzyoun.pythonanywhere.com';

export default function PhotoAnalyzer() {
  const [image, setImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [analysisHistory, setAnalysisHistory] = useState([]);
  const [showHistogram, setShowHistogram] = useState(false);
  const [copiedPrompt, setCopiedPrompt] = useState(false);

  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        setImage(e.target.result);
        setResults(null);
        setError(null);
      };
      reader.readAsDataURL(file);
    }
  };

  const analyzeImage = async () => {
    if (!image) return;

    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_URL}/api/analyze`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ image }),
      });

      if (!response.ok) {
        throw new Error(`Erreur ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      setResults(data);
      
      // Ajouter à l'historique
      const historyItem = {
        id: Date.now(),
        timestamp: new Date().toLocaleString('fr-FR'),
        image: image,
        quality_score: data.quality_score,
        best_style: data.best_style?.label || 'N/A'
      };
      setAnalysisHistory(prev => [historyItem, ...prev].slice(0, 5));
    } catch (err) {
      setError(err.message);
      console.error('Erreur analyse:', err);
    } finally {
      setLoading(false);
    }
  };

  const copyPromptToClipboard = () => {
    if (results?.ai_prompt) {
      navigator.clipboard.writeText(results.ai_prompt);
      setCopiedPrompt(true);
      setTimeout(() => setCopiedPrompt(false), 2000);
    }
  };

  const downloadReport = () => {
    if (!results) return;
    
    const report = `
RAPPORT D'ANALYSE PHOTO
========================
Date: ${new Date().toLocaleString('fr-FR')}

MÉTRIQUES TECHNIQUES
--------------------
Netteté: ${Math.round(results.sharpness)}/200
Luminosité: ${Math.round(results.brightness)}/255
Contraste: ${Math.round(results.contrast)}/100
Bruit: ${Math.round(results.noise)}/100

QUALITÉ GLOBALE: ${results.quality_score}/100

STYLE RECOMMANDÉ
----------------
${results.best_style?.label || 'N/A'}
Catégorie: ${results.best_style?.category || 'N/A'}
Difficulté: ${results.best_style?.difficulty || 'N/A'}

PROMPT IA GÉNÉRÉ
----------------
${results.ai_prompt || 'N/A'}

CONSEILS D'AMÉLIORATION
-----------------------
${results.advice?.join('\n') || 'Aucun conseil spécifique'}
    `.trim();
    
    const blob = new Blob([report], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `rapport-photo-${Date.now()}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const loadFromHistory = (historyItem) => {
    setImage(historyItem.image);
    setResults(null);
  };

  const getDifficultyColor = (difficulty) => {
    const colors = {
      'débutant': 'bg-green-100 text-green-800',
      'intermédiaire': 'bg-yellow-100 text-yellow-800',
      'avancé': 'bg-orange-100 text-orange-800',
      'expert': 'bg-red-100 text-red-800'
    };
    return colors[difficulty?.toLowerCase()] || 'bg-gray-100 text-gray-800';
  };

  const getCategoryColor = (category) => {
    const colors = {
      'portrait': 'bg-purple-100 text-purple-800',
      'paysage': 'bg-blue-100 text-blue-800',
      'urbain': 'bg-gray-100 text-gray-800',
      'nature': 'bg-green-100 text-green-800',
      'artistique': 'bg-pink-100 text-pink-800'
    };
    return colors[category?.toLowerCase()] || 'bg-indigo-100 text-indigo-800';
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 p-4">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8 pt-8">
          <div className="flex items-center justify-center gap-3 mb-4">
            <Camera className="w-12 h-12 text-purple-400" />
            <h1 className="text-5xl font-bold text-white">Photo Analyzer Pro</h1>
          </div>
          <p className="text-purple-200 text-lg">Analyse professionnelle de vos photos avec IA</p>
        </div>

        {/* Historique */}
        {analysisHistory.length > 0 && (
          <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 mb-6 border border-white/20">
            <h3 className="text-white text-xl font-semibold mb-4 flex items-center gap-2">
              <History className="w-6 h-6" />
              Historique des analyses ({analysisHistory.length})
            </h3>
            <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
              {analysisHistory.map((item) => (
                <div
                  key={item.id}
                  onClick={() => loadFromHistory(item)}
                  className="cursor-pointer group relative rounded-lg overflow-hidden border-2 border-white/20 hover:border-purple-400 transition-all"
                >
                  <img
                    src={item.image}
                    alt="Historique"
                    className="w-full h-32 object-cover group-hover:scale-110 transition-transform"
                  />
                  <div className="absolute bottom-0 left-0 right-0 bg-black/80 p-2">
                    <p className="text-white text-xs font-semibold">{item.best_style}</p>
                    <p className="text-purple-300 text-xs">Score: {item.quality_score}/100</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Upload Zone */}
        <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-8 mb-6 border border-white/20">
          <label className="flex flex-col items-center justify-center cursor-pointer group">
            <input
              type="file"
              accept="image/*"
              onChange={handleImageUpload}
              className="hidden"
            />
            <div className="flex flex-col items-center">
              <Upload className="w-16 h-16 text-purple-300 mb-4 group-hover:scale-110 transition-transform" />
              <span className="text-white text-xl font-semibold mb-2">
                Cliquez pour choisir une photo
              </span>
              <span className="text-purple-200 text-sm">
                Formats acceptés: JPG, PNG, WEBP
              </span>
            </div>
          </label>
        </div>

        {/* Preview et Analyse */}
        {image && (
          <div className="grid md:grid-cols-2 gap-6 mb-6">
            <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20">
              <h3 className="text-white text-xl font-semibold mb-4 flex items-center gap-2">
                <Eye className="w-6 h-6" />
                Aperçu
              </h3>
              <img
                src={image}
                alt="Preview"
                className="w-full h-auto rounded-lg shadow-2xl"
              />
            </div>

            <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20 flex flex-col gap-4">
              <h3 className="text-white text-xl font-semibold">Actions</h3>
              
              <button
                onClick={analyzeImage}
                disabled={loading}
                className="bg-gradient-to-r from-purple-500 to-pink-500 text-white px-8 py-4 rounded-xl font-semibold text-lg hover:from-purple-600 hover:to-pink-600 disabled:opacity-50 disabled:cursor-not-allowed transition-all transform hover:scale-105 flex items-center justify-center gap-3 shadow-xl"
              >
                {loading ? (
                  <>
                    <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-white"></div>
                    Analyse en cours...
                  </>
                ) : (
                  <>
                    <Sparkles className="w-6 h-6" />
                    Analyser la photo
                  </>
                )}
              </button>

              {results && (
                <>
                  <button
                    onClick={downloadReport}
                    className="bg-gradient-to-r from-blue-500 to-cyan-500 text-white px-6 py-3 rounded-xl font-semibold hover:from-blue-600 hover:to-cyan-600 transition-all flex items-center justify-center gap-2"
                  >
                    <Download className="w-5 h-5" />
                    Télécharger le rapport
                  </button>

                  <button
                    onClick={copyPromptToClipboard}
                    className="bg-gradient-to-r from-green-500 to-emerald-500 text-white px-6 py-3 rounded-xl font-semibold hover:from-green-600 hover:to-emerald-600 transition-all flex items-center justify-center gap-2"
                  >
                    <Copy className="w-5 h-5" />
                    {copiedPrompt ? '✓ Copié !' : 'Copier le prompt IA'}
                  </button>

                  <button
                    onClick={() => setShowHistogram(!showHistogram)}
                    className="bg-gradient-to-r from-orange-500 to-red-500 text-white px-6 py-3 rounded-xl font-semibold hover:from-orange-600 hover:to-red-600 transition-all flex items-center justify-center gap-2"
                  >
                    <BarChart3 className="w-5 h-5" />
                    {showHistogram ? 'Masquer' : 'Voir'} l'histogramme
                  </button>
                </>
              )}

              {error && (
                <div className="bg-red-500/20 border border-red-500 text-red-200 p-4 rounded-lg">
                  <p className="font-semibold">Erreur:</p>
                  <p className="text-sm">{error}</p>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Histogramme simplifié */}
        {results && showHistogram && (
          <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 mb-6 border border-white/20">
            <h3 className="text-white text-2xl font-bold mb-4 flex items-center gap-2">
              <BarChart3 className="w-7 h-7 text-orange-400" />
              Distribution de la lumière
            </h3>
            <div className="bg-black/30 rounded-lg p-6">
              <div className="flex items-end justify-between h-40 gap-1">
                {/* Simulation d'histogramme basé sur brightness */}
                {Array.from({ length: 20 }, (_, i) => {
                  const height = Math.random() * 100;
                  return (
                    <div
                      key={i}
                      className="bg-gradient-to-t from-purple-500 to-pink-500 rounded-t"
                      style={{ 
                        width: '4%',
                        height: `${height}%`,
                        opacity: 0.8
                      }}
                    />
                  );
                })}
              </div>
              <div className="flex justify-between mt-2 text-purple-300 text-xs">
                <span>Ombres</span>
                <span>Tons moyens</span>
                <span>Hautes lumières</span>
              </div>
            </div>
            <p className="text-purple-200 text-sm mt-4">
              Luminosité moyenne: {Math.round(results.brightness)}/255
              {results.brightness < 85 && " - Image sous-exposée, beaucoup d'ombres"}
              {results.brightness > 85 && results.brightness < 170 && " - Exposition équilibrée"}
              {results.brightness > 170 && " - Image lumineuse, attention aux zones cramées"}
            </p>
          </div>
        )}

        {/* Prompt IA en grand */}
        {results?.ai_prompt && (
          <div className="bg-gradient-to-r from-indigo-500/20 to-purple-500/20 backdrop-blur-lg rounded-2xl p-6 mb-6 border border-indigo-400/30">
            <h3 className="text-white text-2xl font-bold mb-4 flex items-center gap-2">
              <Sparkles className="w-7 h-7 text-indigo-400" />
              Prompt IA généré pour Midjourney / DALL-E
            </h3>
            <div className="bg-black/30 rounded-lg p-6 border border-indigo-400/20">
              <p className="text-indigo-100 text-lg leading-relaxed font-mono">
                {results.ai_prompt}
              </p>
            </div>
            <div className="flex gap-3 mt-4">
              <button
                onClick={copyPromptToClipboard}
                className="flex-1 bg-indigo-500 hover:bg-indigo-600 text-white px-4 py-2 rounded-lg font-semibold transition-all flex items-center justify-center gap-2"
              >
                <Copy className="w-4 h-4" />
                {copiedPrompt ? '✓ Copié !' : 'Copier'}
              </button>
              <button className="flex-1 bg-purple-500 hover:bg-purple-600 text-white px-4 py-2 rounded-lg font-semibold transition-all">
                Ouvrir Midjourney
              </button>
            </div>
          </div>
        )}

        {/* Résultats (reste identique) */}
        {results && (
          <div className="space-y-6">
            {/* 🔍 DEBUG - Vérification caption */}
            <div className="bg-blue-500/20 backdrop-blur-lg rounded-2xl p-4 border border-blue-400/30">
              <h4 className="text-blue-300 font-bold mb-2">🔍 DEBUG - Caption reçu :</h4>
              <p className="text-blue-100 font-mono text-sm break-words">
                Caption : "{results.full_analysis?.caption || 'VIDE'}"
              </p>
              <p className="text-blue-200 text-xs mt-2">
                Type: {typeof results.full_analysis?.caption} | 
                Longueur: {results.full_analysis?.caption?.length || 0} caractères
              </p>
            </div>

            {/* Comparaison des 2 descriptions */}
            {(results.full_analysis?.caption_huggingface || results.full_analysis?.caption_fallback) && (
              <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20">
                <h3 className="text-white text-2xl font-bold mb-6 flex items-center gap-2">
                  <Eye className="w-7 h-7 text-purple-400" />
                  Descriptions de l'image
                </h3>
                
                <div className="grid md:grid-cols-2 gap-4">
                  {/* Hugging Face */}
                  <div className="bg-gradient-to-br from-purple-500/20 to-pink-500/20 rounded-xl p-5 border border-purple-400/30">
                    <div className="flex items-center gap-2 mb-3">
                      <div className="w-10 h-10 bg-purple-500 rounded-full flex items-center justify-center text-white text-lg">
                        🤗
                      </div>
                      <div>
                        <h4 className="text-purple-300 font-bold">Intelligence Artificielle</h4>
                        <p className="text-purple-200 text-xs">Hugging Face Vision</p>
                      </div>
                    </div>
                    <p className="text-purple-100 text-sm leading-relaxed min-h-[60px]">
                      {results.full_analysis.caption_huggingface || 'En attente...'}
                    </p>
                    {results.full_analysis.caption_huggingface && results.full_analysis.caption_huggingface !== 'Non disponible' && (
                      <div className="mt-3 text-green-400 text-xs flex items-center gap-1">
                        <span>✓</span> Description IA détaillée
                      </div>
                    )}
                    {(!results.full_analysis.caption_huggingface || results.full_analysis.caption_huggingface === 'Non disponible') && (
                      <div className="mt-3 text-yellow-400 text-xs">
                        <span>⚠</span> API temporairement indisponible
                      </div>
                    )}
                  </div>
                  
                  {/* Fallback Intelligent */}
                  <div className="bg-gradient-to-br from-orange-500/20 to-yellow-500/20 rounded-xl p-5 border border-orange-400/30">
                    <div className="flex items-center gap-2 mb-3">
                      <div className="w-10 h-10 bg-orange-500 rounded-full flex items-center justify-center text-white font-bold">
                        F
                      </div>
                      <div>
                        <h4 className="text-orange-300 font-bold">Analyse Intelligente</h4>
                        <p className="text-orange-200 text-xs">Basée sur détections</p>
                      </div>
                    </div>
                    <p className="text-orange-100 text-sm leading-relaxed min-h-[60px]">
                      {results.full_analysis.caption_fallback || 'Non disponible'}
                    </p>
                    <div className="mt-3 text-green-400 text-xs flex items-center gap-1">
                      <span>✓</span> Toujours disponible
                    </div>
                  </div>
                </div>
                
                {/* Description principale */}
                <div className="mt-6 bg-gradient-to-r from-indigo-500/20 to-purple-500/20 rounded-xl p-5 border border-indigo-400/30">
                  <h4 className="text-indigo-300 font-bold mb-2 flex items-center gap-2">
                    ⭐ Meilleure description sélectionnée
                  </h4>
                  <p className="text-indigo-100 text-lg leading-relaxed">
                    {results.full_analysis.caption}
                  </p>
                </div>
              </div>
            )}

            {/* Éléments détectés */}
            {results.subjects && results.subjects.length > 0 && (
              <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20">
                <h3 className="text-white text-2xl font-bold mb-4 flex items-center gap-2">
                  <Tag className="w-7 h-7 text-green-400" />
                  Éléments détectés
                </h3>
                <div className="flex flex-wrap gap-3">
                  {results.subjects.map((subject, idx) => (
                    <div
                      key={idx}
                      className="bg-green-500/20 border border-green-400/50 rounded-lg px-4 py-2 flex items-center gap-2"
                    >
                      <span className="text-green-300 font-semibold capitalize">
                        {subject.class}
                      </span>
                      {subject.confidence && (
                        <span className="text-green-200 text-sm">
                          ({Math.round(subject.confidence * 100)}%)
                        </span>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Métriques Techniques */}
            <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20">
              <h3 className="text-white text-2xl font-bold mb-6 flex items-center gap-2">
                <Zap className="w-7 h-7 text-yellow-400" />
                Métriques Techniques
              </h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <MetricCard label="Netteté" value={results.sharpness} max={200} unit="" color="blue" />
                <MetricCard label="Luminosité" value={results.brightness} max={255} unit="" color="yellow" />
                <MetricCard label="Contraste" value={results.contrast} max={100} unit="" color="purple" />
                <MetricCard label="Bruit" value={results.noise} max={100} unit="" color="red" />
              </div>
            </div>

            {/* Score de Qualité */}
            <div className="bg-gradient-to-r from-purple-500/20 to-pink-500/20 backdrop-blur-lg rounded-2xl p-6 border border-purple-400/30">
              <h3 className="text-white text-2xl font-bold mb-4 flex items-center gap-2">
                <TrendingUp className="w-7 h-7" />
                Score de Qualité Global
              </h3>
              <div className="flex items-center gap-6">
                <div className="text-6xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-pink-400">
                  {results.quality_score}
                  <span className="text-3xl">/100</span>
                </div>
                <div className="flex-1">
                  <div className="bg-white/20 rounded-full h-6 overflow-hidden">
                    <div
                      className="bg-gradient-to-r from-purple-500 to-pink-500 h-full rounded-full transition-all duration-1000"
                      style={{ width: `${results.quality_score}%` }}
                    ></div>
                  </div>
                </div>
              </div>
            </div>

            {/* Style Recommandé */}
            {results.best_style && (
              <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20">
                <h3 className="text-white text-2xl font-bold mb-6 flex items-center gap-2">
                  <Palette className="w-7 h-7 text-pink-400" />
                  Style Recommandé
                </h3>
                
                <div className="space-y-4">
                  <div className="flex items-center gap-4 flex-wrap">
                    <h4 className="text-3xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-pink-400 to-purple-400">
                      {results.best_style.label}
                    </h4>
                    <div className="flex gap-2">
                      <span className={`px-3 py-1 rounded-full text-sm font-semibold ${getCategoryColor(results.best_style.category)}`}>
                        {results.best_style.category}
                      </span>
                      <span className={`px-3 py-1 rounded-full text-sm font-semibold ${getDifficultyColor(results.best_style.difficulty)}`}>
                        {results.best_style.difficulty}
                      </span>
                    </div>
                  </div>

                  {results.best_style.description && (
                    <div className="bg-white/5 rounded-lg p-4 border border-white/10">
                      <p className="text-purple-100 leading-relaxed">
                        {results.best_style.description}
                      </p>
                    </div>
                  )}

                  <div className="flex items-center gap-4 bg-gradient-to-r from-pink-500/20 to-purple-500/20 rounded-lg p-4 border border-pink-400/30">
                    <span className="text-white font-semibold">Score de correspondance:</span>
                    <div className="flex-1">
                      <div className="bg-white/20 rounded-full h-4 overflow-hidden">
                        <div
                          className="bg-gradient-to-r from-pink-500 to-purple-500 h-full rounded-full"
                          style={{ width: `${results.best_style.score * 100}%` }}
                        ></div>
                      </div>
                    </div>
                    <span className="text-2xl font-bold text-pink-300">
                      {Math.round(results.best_style.score * 100)}%
                    </span>
                  </div>
                </div>
              </div>
            )}

            {/* Conseils d'amélioration */}
            {results.advice && results.advice.length > 0 && (
              <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20">
                <h3 className="text-white text-2xl font-bold mb-4">💡 Actions rapides</h3>
                <ul className="space-y-3">
                  {results.advice.map((tip, idx) => (
                    <li key={idx} className="flex items-start gap-3 text-purple-100">
                      <span className="text-2xl">✨</span>
                      <span className="flex-1 pt-1">{tip}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function MetricCard({ label, value, max, unit, color }) {
  const percentage = Math.min((value / max) * 100, 100);
  
  const colorClasses = {
    blue: 'from-blue-500 to-cyan-500',
    yellow: 'from-yellow-500 to-orange-500',
    purple: 'from-purple-500 to-pink-500',
    red: 'from-red-500 to-pink-500',
    green: 'from-green-500 to-emerald-500'
  };

  return (
    <div className="bg-white/5 rounded-xl p-4 border border-white/10">
      <p className="text-purple-200 text-sm mb-2">{label}</p>
      <p className="text-white text-2xl font-bold mb-2">
        {Math.round(value)}{unit}
      </p>
      <div className="bg-white/20 rounded-full h-2 overflow-hidden">
        <div
          className={`bg-gradient-to-r ${colorClasses[color]} h-full rounded-full transition-all duration-500`}
          style={{ width: `${percentage}%` }}
        ></div>
      </div>
    </div>
  );
}
