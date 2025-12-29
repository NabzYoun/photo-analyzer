import React, { useState } from 'react';
import { Upload, Camera, Sparkles, Zap, Eye, Palette, Tag } from 'lucide-react';

const API_URL = 'https://nabzyoun.pythonanywhere.com';
//const API_URL = 'http://localhost:10000';

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
            <h1 className="text-5xl font-bold text-white"> TEST Photo Analyzer</h1>
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

            {/* 📘 Section Pédagogique : Lecture de l'image */}
            <div className="bg-gradient-to-r from-indigo-900/40 to-purple-900/40 backdrop-blur-lg rounded-2xl p-6 border border-indigo-500/30">
              <h3 className="text-white text-2xl font-bold mb-4 flex items-center gap-2">
                📘 Lecture de l'image
              </h3>
              <div className="space-y-3 text-purple-100">
                <p>
                  <strong className="text-purple-300">Luminosité :</strong> Cette image présente une dominante lumineuse de {results.brightness}/255, 
                  ce qui crée une ambiance plutôt {results.brightness > 150 ? "claire et aérée" : results.brightness > 100 ? "équilibrée" : "sombre et intimiste"}.
                  {results.brightness < 80 && " ⚠️ L'image est sous-exposée, envisagez d'augmenter l'exposition."}
                  {results.brightness > 200 && " ⚠️ L'image est surexposée, attention aux zones cramées."}
                </p>
                
                <p>
                  <strong className="text-purple-300">Contraste :</strong> Le contraste de {Math.round(results.contrast)} 
                  {results.contrast > 70 ? " est élevé, créant une image dynamique et percutante avec des noirs profonds et des blancs lumineux." : 
                   results.contrast > 40 ? " est modéré, offrant un bon équilibre entre ombres et lumières." :
                   " est faible, donnant une image douce mais potentiellement plate. Augmentez le contraste pour plus d'impact."}
                </p>
                
                <p>
                  <strong className="text-purple-300">Netteté :</strong> Avec un score de {Math.round(results.sharpness)}/200, 
                  {results.sharpness > 150 ? " votre image est très nette, idéale pour les impressions grand format." :
                   results.sharpness > 100 ? " la netteté est correcte pour un usage web et réseaux sociaux." :
                   " l'image manque de netteté. Utilisez un trépied ou augmentez la vitesse d'obturation."}
                </p>
                
                <p>
                  <strong className="text-purple-300">Bruit numérique :</strong> Niveau de bruit : {Math.round(results.noise)}
                  {results.noise < 20 ? " (excellent - image très propre)" :
                   results.noise < 40 ? " (acceptable - bruit modéré)" :
                   " (élevé - réduisez les ISO ou utilisez la réduction de bruit en post-production)"}
                </p>
              </div>
            </div>

            {/* 🎨 Section Pédagogique : Composition */}
            {results.composition_score && (
              <div className="bg-gradient-to-r from-pink-900/40 to-orange-900/40 backdrop-blur-lg rounded-2xl p-6 border border-pink-500/30">
                <h3 className="text-white text-2xl font-bold mb-4 flex items-center gap-2">
                  🎨 Analyse de la composition
                </h3>
                <div className="space-y-3 text-orange-100">
                  <p>
                    <strong className="text-orange-300">Score de composition :</strong> {results.composition_score}/100
                  </p>
                  <p>
                    {results.composition_score > 70 ? 
                      "✨ Excellente composition ! Vos sujets sont bien placés selon les règles photographiques classiques." :
                     results.composition_score > 50 ?
                      "👍 Bonne composition générale. Quelques ajustements pourraient améliorer l'équilibre visuel." :
                      "💡 La composition peut être améliorée. Pensez à la règle des tiers, aux lignes directrices et à l'équilibre des masses."}
                  </p>
                  
                  {results.subjects && results.subjects.length > 0 && (
                    <p>
                      <strong className="text-orange-300">Sujets principaux :</strong> {results.subjects.length} élément(s) détecté(s).
                      {results.subjects.length === 1 && " Un sujet unique permet une composition minimaliste et forte."}
                      {results.subjects.length > 3 && " ⚠️ Attention à ne pas surcharger l'image, privilégiez la simplicité."}
                    </p>
                  )}
                </div>
              </div>
            )}

            {/* 🎯 Section Pédagogique : Conseils personnalisés */}
            <div className="bg-gradient-to-r from-green-900/40 to-teal-900/40 backdrop-blur-lg rounded-2xl p-6 border border-green-500/30">
              <h3 className="text-white text-2xl font-bold mb-4 flex items-center gap-2">
                🎯 Vos axes d'amélioration
              </h3>
              
              <div className="space-y-4">
                {/* Analyse technique */}
                <div className="bg-black/20 rounded-lg p-4 border border-green-500/20">
                  <h4 className="text-green-300 font-semibold mb-2">📐 Technique</h4>
                  <ul className="space-y-2 text-teal-100 text-sm">
                    {results.sharpness < 100 && (
                      <li>• Améliorez la netteté : utilisez un trépied, augmentez la vitesse d'obturation ou activez la stabilisation</li>
                    )}
                    {results.noise > 40 && (
                      <li>• Réduisez le bruit : diminuez les ISO, shootez avec plus de lumière ou utilisez la réduction de bruit</li>
                    )}
                    {(results.brightness < 80 || results.brightness > 200) && (
                      <li>• Corrigez l'exposition : {results.brightness < 80 ? "augmentez" : "diminuez"} la luminosité pour un meilleur équilibre</li>
                    )}
                    {results.contrast < 30 && (
                      <li>• Augmentez le contraste pour donner plus de profondeur et d'impact à votre image</li>
                    )}
                    {!results.sharpness && !results.noise && results.brightness > 80 && results.brightness < 200 && results.contrast > 30 && (
                      <li>✅ Excellente maîtrise technique ! Continuez comme ça.</li>
                    )}
                  </ul>
                </div>

                {/* Analyse créative */}
                <div className="bg-black/20 rounded-lg p-4 border border-green-500/20">
                  <h4 className="text-green-300 font-semibold mb-2">🎨 Créativité</h4>
                  <ul className="space-y-2 text-teal-100 text-sm">
                    {results.best_style && (
                      <li>• Explorez le style {results.best_style.label} pour développer votre signature artistique</li>
                    )}
                    {results.composition_score < 60 && (
                      <li>• Travaillez votre composition : règle des tiers, lignes directrices, cadrage</li>
                    )}
                    {results.subjects && results.subjects.length === 0 && (
                      <li>• Ajoutez un sujet principal pour donner un point focal à votre image</li>
                    )}
                    <li>• Expérimentez avec différentes heures de la journée pour varier les ambiances lumineuses</li>
                  </ul>
                </div>
              </div>
            </div>

            {/* 📚 Section Pédagogique : Concepts photographiques */}
            <div className="bg-gradient-to-r from-blue-900/40 to-cyan-900/40 backdrop-blur-lg rounded-2xl p-6 border border-blue-500/30">
              <h3 className="text-white text-2xl font-bold mb-4 flex items-center gap-2">
                📚 Le saviez-vous ?
              </h3>
              
              <div className="grid md:grid-cols-2 gap-4">
                {/* Carte concept 1 */}
                <div className="bg-black/30 rounded-lg p-4 border border-blue-400/20">
                  <h4 className="text-blue-300 font-semibold mb-2 flex items-center gap-2">
                    💡 Triangle d'exposition
                  </h4>
                  <p className="text-cyan-100 text-sm">
                    L'exposition parfaite est un équilibre entre <strong>ISO</strong> (sensibilité), 
                    <strong> vitesse d'obturation</strong> (temps) et <strong>ouverture</strong> (diaphragme). 
                    Votre image à {results.brightness} de luminosité suggère 
                    {results.brightness > 150 ? " une bonne exposition ou une scène lumineuse." : " un manque de lumière ou une sous-exposition."}
                  </p>
                </div>

                {/* Carte concept 2 */}
                <div className="bg-black/30 rounded-lg p-4 border border-blue-400/20">
                  <h4 className="text-blue-300 font-semibold mb-2 flex items-center gap-2">
                    🎯 Règle des tiers
                  </h4>
                  <p className="text-cyan-100 text-sm">
                    Divisez mentalement l'image en 9 parties égales avec 2 lignes horizontales et 2 verticales. 
                    Placez vos sujets importants aux intersections pour une composition dynamique et équilibrée.
                  </p>
                </div>

                {/* Carte concept 3 */}
                <div className="bg-black/30 rounded-lg p-4 border border-blue-400/20">
                  <h4 className="text-blue-300 font-semibold mb-2 flex items-center gap-2">
                    🌅 Golden Hour
                  </h4>
                  <p className="text-cyan-100 text-sm">
                    Les meilleures lumières naturelles apparaissent 1h après le lever et 1h avant le coucher du soleil. 
                    {results.brightness > 150 ? " Votre image semble bénéficier d'une belle lumière naturelle !" : 
                     " Essayez de shooter pendant ces heures magiques pour des couleurs chaudes et des ombres douces."}
                  </p>
                </div>

                {/* Carte concept 4 */}
                <div className="bg-black/30 rounded-lg p-4 border border-blue-400/20">
                  <h4 className="text-blue-300 font-semibold mb-2 flex items-center gap-2">
                    📏 Profondeur de champ
                  </h4>
                  <p className="text-cyan-100 text-sm">
                    Une grande ouverture (f/1.8-f/2.8) crée un arrière-plan flou (bokeh), idéal pour les portraits. 
                    Une petite ouverture (f/8-f/16) garde tout net, parfait pour les paysages.
                  </p>
                </div>
              </div>
            </div>

            {/* Conseils d'amélioration (conservé) */}
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