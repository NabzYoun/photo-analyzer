import React, { useState } from 'react';
import { Upload, Camera, Sparkles, Zap, Eye, Palette, Tag } from 'lucide-react';

const API_URL = 'https://nabzyoun.pythonanywhere.com';

export default function PhotoAnalyzer() {
  const [image, setImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

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
    } catch (err) {
      setError(err.message);
      console.error('Erreur analyse:', err);
    } finally {
      setLoading(false);
    }
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
            <h1 className="text-5xl font-bold text-white">Photo Analyzer</h1>
          </div>
          <p className="text-purple-200 text-lg">Analyse professionnelle de vos photos avec IA</p>
        </div>

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

            <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20 flex flex-col">
              <h3 className="text-white text-xl font-semibold mb-4">Actions</h3>
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

              {error && (
                <div className="mt-4 bg-red-500/20 border border-red-500 text-red-200 p-4 rounded-lg">
                  <p className="font-semibold">Erreur:</p>
                  <p className="text-sm">{error}</p>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Résultats */}
        {results && (
          <div className="space-y-6">
            {/* Description de la photo */}
            {results.full_analysis?.caption && (
              <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20">
                <h3 className="text-white text-2xl font-bold mb-4 flex items-center gap-2">
                  <Eye className="w-7 h-7 text-purple-400" />
                  Description de la photo
                </h3>
                <p className="text-purple-100 text-lg leading-relaxed">
                  {results.full_analysis.caption}
                </p>
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
              <h3 className="text-white text-2xl font-bold mb-4">Score de Qualité Global</h3>
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

            {/* Conseils */}
            {results.advice && results.advice.length > 0 && (
              <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20">
                <h3 className="text-white text-2xl font-bold mb-4">Conseils d'Amélioration</h3>
                <ul className="space-y-3">
                  {results.advice.map((tip, idx) => (
                    <li key={idx} className="flex items-start gap-3 text-purple-100">
                      <span className="text-2xl">💡</span>
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