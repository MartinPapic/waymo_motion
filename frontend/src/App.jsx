import React, { useState, useEffect, useCallback } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, Legend, ResponsiveContainer, Cell, Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis } from 'recharts';
import Plot from 'react-plotly.js';
import waymoData from './waymo_frontend.json';
import './App.css';

function App() {
  const [currentSlide, setCurrentSlide] = useState(0);

  // Procesar los datos
  const vehicles = waymoData.filter(d => d.object_type === 'vehicle');
  const pedestrians = waymoData.filter(d => d.object_type === 'pedestrian');
  const cyclists = waymoData.filter(d => d.object_type === 'cyclist');

  const calcAvg = (arr, key) => arr.length > 0 ? (arr.reduce((sum, item) => sum + item[key], 0) / arr.length).toFixed(2) : 0;

  const barData = [
    { name: 'Vehículos', Cantidad: vehicles.length, fill: '#4ade80' },
    { name: 'Peatones', Cantidad: pedestrians.length, fill: '#f87171' },
    { name: 'Ciclistas', Cantidad: cyclists.length, fill: '#60a5fa' }
  ];

  const scatter2DDataVehicles = vehicles.map(d => ({ x: d.box_length, y: d.box_width, type: 'Vehículo' }));
  const scatter2DDataPedestrians = pedestrians.map(d => ({ x: d.box_length, y: d.box_width, type: 'Peatón' }));

  const boxPlotData = [
    { y: vehicles.map(d => d.box_length), type: 'box', name: 'Vehículos', marker: { color: '#4ade80' } },
    { y: pedestrians.map(d => d.box_length), type: 'box', name: 'Peatones', marker: { color: '#f87171' } }
  ];

  const scatter3DData = [
    // Agregamos el "Ego-Vehicle" (Nosotros) en el centro para dar contexto
    { x: [0], y: [0], z: [0], mode: 'markers', type: 'scatter3d', name: '📍 Nuestro Auto (Waymo)', marker: { color: '#3b82f6', size: 8, symbol: 'diamond' } },
    { x: vehicles.map(d => d.box_center_x), y: vehicles.map(d => d.box_center_y), z: vehicles.map(d => d.box_center_z), mode: 'markers', type: 'scatter3d', name: 'Otros Vehículos', marker: { color: '#4ade80', size: 3, opacity: 0.8 } },
    { x: pedestrians.map(d => d.box_center_x), y: pedestrians.map(d => d.box_center_y), z: pedestrians.map(d => d.box_center_z), mode: 'markers', type: 'scatter3d', name: 'Peatones', marker: { color: '#f87171', size: 4, opacity: 0.8 } }
  ];

  const slides = [
    {
      title: "1. Volumen y Sesgo de Clases",
      description: "Este gráfico de barras evidencia el Desbalance de Clases capturado por los sensores LiDAR en los entornos reales. La sobre-representación de vehículos frente a peatones genera un Sesgo Ético: si entrenamos un modelo de Inteligencia Artificial con estos datos crudos, el vehículo autónomo tendrá una tasa peligrosa de falsos negativos al detectar personas. Esto justifica usar técnicas como SMOTE en el futuro.",
      chart: (
        <ResponsiveContainer width="100%" height={400}>
          <BarChart data={barData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#444" />
            <XAxis dataKey="name" stroke="#ccc" />
            <YAxis stroke="#ccc" />
            <RechartsTooltip contentStyle={{ backgroundColor: '#333', border: 'none', borderRadius: '8px' }} />
            <Bar dataKey="Cantidad" fill="#8884d8">
              {barData.map((entry, index) => <Cell key={`cell-${index}`} fill={entry.fill} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      )
    },
    {
      title: "2. Distribución de Tamaño y Anomalías (IQR)",
      description: "El BoxPlot (Diagrama de Caja y Bigotes) analiza la varianza en la Longitud de los objetos. La 'caja' representa el 50% central de los datos. Esta visualización es clave porque demuestra matemáticamente cómo funciona nuestro filtro de Kedro: los puntos sueltos por encima de los bigotes son Anomalías Estadísticas (Outliers) que escapan del Rango Intercuartílico (IQR).",
      chart: (
        <Plot
          data={boxPlotData}
          layout={{ width: 800, height: 400, paper_bgcolor: '#2a2a2a', plot_bgcolor: '#2a2a2a', font: { color: '#ccc' }, margin: { t: 20, b: 30, l: 40, r: 20 } }}
        />
      )
    },
    {
      title: "3. Clústers y Separabilidad (Largo vs Ancho)",
      description: "Este Gráfico de Dispersión 2D es el favorito de los algoritmos de clasificación. Compara el Largo (X) vs el Ancho (Y) de cada bounding box. Revela algo fundamental: los Vehículos (verde) y los Peatones (rojo) forman dos 'clústers' completamente separados y distintos. Esto significa que las clases son linealmente separables por sus dimensiones.",
      chart: (
        <Plot
          data={[
            { x: scatter2DDataVehicles.map(d => d.x), y: scatter2DDataVehicles.map(d => d.y), mode: 'markers', type: 'scatter', name: 'Vehículos', marker: { color: '#4ade80', size: 6, opacity: 0.7 } },
            { x: scatter2DDataPedestrians.map(d => d.x), y: scatter2DDataPedestrians.map(d => d.y), mode: 'markers', type: 'scatter', name: 'Peatones', marker: { color: '#f87171', size: 6, opacity: 0.9 } }
          ]}
          layout={{ 
            width: 800, height: 400, paper_bgcolor: '#2a2a2a', plot_bgcolor: '#2a2a2a', font: { color: '#ccc' }, margin: { t: 20, b: 40, l: 50, r: 20 },
            xaxis: { title: 'Largo de la Caja (m)' }, yaxis: { title: 'Ancho de la Caja (m)' }
          }}
        />
      )
    },
    {
      title: "4. Nube de Puntos Espacial (La Visión del Vehículo)",
      description: "Imagina que eres el vehículo autónomo. El rombo azul en la coordenada (0, 0, 0) es nuestro propio auto. El resto de puntos verdes y rojos son los vehículos y peatones reales que el sensor láser detectó a nuestro alrededor (en metros). Eje X: Adelante/Atrás, Eje Y: Izquierda/Derecha, Eje Z: Altura desde el suelo. (¡Rota el gráfico con tu mouse!)",
      chart: (
        <Plot
          data={scatter3DData}
          layout={{ 
            width: 800, height: 400, paper_bgcolor: '#2a2a2a', font: { color: '#ccc' }, margin: { t: 0, b: 0, l: 0, r: 0 }, 
            scene: { 
              xaxis: { title: 'X (Adelante/Atrás m)' }, 
              yaxis: { title: 'Y (Izquierda/Derecha m)' }, 
              zaxis: { title: 'Z (Altura)' } 
            } 
          }}
        />
      )
    }
  ];

  const nextSlide = useCallback(() => {
    setCurrentSlide(prev => (prev === slides.length - 1 ? prev : prev + 1));
  }, [slides.length]);

  const prevSlide = useCallback(() => {
    setCurrentSlide(prev => (prev === 0 ? prev : prev - 1));
  }, []);

  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'ArrowRight') nextSlide();
      if (e.key === 'ArrowLeft') prevSlide();
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [nextSlide, prevSlide]);

  return (
    <div className="App" style={{ padding: '2rem', backgroundColor: '#1a1a1a', color: 'white', minHeight: '100vh', fontFamily: 'sans-serif', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
      <h1 style={{ textAlign: 'center', marginBottom: '0.5rem' }}>Presentación EDA - Waymo Motion</h1>
      <p style={{ textAlign: 'center', color: '#9ca3af', marginBottom: '2rem' }}>Usa las flechas del teclado ⬅️ ➡️ o los botones para navegar.</p>
      
      <div style={{ backgroundColor: '#2a2a2a', padding: '2rem', borderRadius: '16px', width: '100%', maxWidth: '900px', boxShadow: '0 10px 25px rgba(0,0,0,0.5)' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '2rem' }}>
          <button onClick={prevSlide} disabled={currentSlide === 0} style={{ padding: '10px 20px', cursor: currentSlide === 0 ? 'not-allowed' : 'pointer', backgroundColor: currentSlide === 0 ? '#444' : '#3b82f6', color: 'white', border: 'none', borderRadius: '8px', fontWeight: 'bold' }}>
            ⬅️ Anterior
          </button>
          <span style={{ fontWeight: 'bold', color: '#9ca3af' }}>{currentSlide + 1} / {slides.length}</span>
          <button onClick={nextSlide} disabled={currentSlide === slides.length - 1} style={{ padding: '10px 20px', cursor: currentSlide === slides.length - 1 ? 'not-allowed' : 'pointer', backgroundColor: currentSlide === slides.length - 1 ? '#444' : '#3b82f6', color: 'white', border: 'none', borderRadius: '8px', fontWeight: 'bold' }}>
            Siguiente ➡️
          </button>
        </div>

        <h2 style={{ textAlign: 'center', color: '#60a5fa', marginBottom: '1rem' }}>{slides[currentSlide].title}</h2>
        
        <div style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '400px', marginBottom: '2rem' }}>
          {slides[currentSlide].chart}
        </div>

        <div style={{ backgroundColor: '#1a1a1a', padding: '1.5rem', borderRadius: '8px', borderLeft: '4px solid #3b82f6' }}>
          <p style={{ fontSize: '1.1rem', lineHeight: '1.6', margin: 0 }}>
            {slides[currentSlide].description}
          </p>
        </div>
      </div>
    </div>
  );
}

export default App;
