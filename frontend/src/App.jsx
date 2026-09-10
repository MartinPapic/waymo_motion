import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, Legend, ResponsiveContainer, Cell, Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis } from 'recharts';
import Plot from 'react-plotly.js';
import waymoData from './waymo_frontend.json';
import './App.css';

function App() {
  // Procesar los datos
  const vehicles = waymoData.filter(d => d.object_type === 'vehicle');
  const pedestrians = waymoData.filter(d => d.object_type === 'pedestrian');
  const cyclists = waymoData.filter(d => d.object_type === 'cyclist');

  const calcAvg = (arr, key) => arr.length > 0 ? (arr.reduce((sum, item) => sum + item[key], 0) / arr.length).toFixed(2) : 0;

  // 1. Datos para BarChart
  const barData = [
    { name: 'Vehículos', Cantidad: vehicles.length, fill: '#4ade80' },
    { name: 'Peatones', Cantidad: pedestrians.length, fill: '#f87171' },
    { name: 'Ciclistas', Cantidad: cyclists.length, fill: '#60a5fa' }
  ];

  // 2. Datos para RadarChart (Perfil de Clases)
  const radarData = [
    {
      subject: 'Velocidad Promedio',
      Vehículos: parseFloat(calcAvg(vehicles, 'velocity_abs')),
      Peatones: parseFloat(calcAvg(pedestrians, 'velocity_abs')) * 5, // Escalonado visualmente
      fullMark: 15,
    },
    {
      subject: 'Largo Promedio (m)',
      Vehículos: parseFloat(calcAvg(vehicles, 'box_length')),
      Peatones: parseFloat(calcAvg(pedestrians, 'box_length')) * 5,
      fullMark: 5,
    },
    {
      subject: 'Ancho Promedio (m)',
      Vehículos: parseFloat(calcAvg(vehicles, 'box_width')),
      Peatones: parseFloat(calcAvg(pedestrians, 'box_width')) * 5,
      fullMark: 3,
    }
  ];

  // 3. Datos para BoxPlot (Plotly) - Regla IQR
  const boxPlotData = [
    {
      y: vehicles.map(d => d.velocity_abs),
      type: 'box',
      name: 'Vehículos',
      marker: { color: '#4ade80' }
    },
    {
      y: pedestrians.map(d => d.velocity_abs),
      type: 'box',
      name: 'Peatones',
      marker: { color: '#f87171' }
    }
  ];

  // 4. Datos para Scatter 3D (Plotly)
  const scatter3DData = [
    {
      x: vehicles.map(d => d.box_center_x),
      y: vehicles.map(d => d.box_center_y),
      z: vehicles.map(d => d.box_center_z),
      mode: 'markers',
      type: 'scatter3d',
      name: 'Vehículos',
      marker: { color: '#4ade80', size: 3, opacity: 0.8 }
    },
    {
      x: pedestrians.map(d => d.box_center_x),
      y: pedestrians.map(d => d.box_center_y),
      z: pedestrians.map(d => d.box_center_z),
      mode: 'markers',
      type: 'scatter3d',
      name: 'Peatones',
      marker: { color: '#f87171', size: 4, opacity: 0.8 }
    }
  ];

  return (
    <div className="App" style={{ padding: '2rem', backgroundColor: '#1a1a1a', color: 'white', minHeight: '100vh', fontFamily: 'sans-serif' }}>
      <h1 style={{ textAlign: 'center' }}>Dashboard Waymo Advanced EDA</h1>
      <p style={{ textAlign: 'center', color: '#9ca3af', marginBottom: '3rem' }}>
        Las 4 Visualizaciones Clave para tu Auditoría de Datos
      </p>
      
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(500px, 1fr))', gap: '2rem', justifyItems: 'center' }}>
        
        {/* 1. BarChart */}
        <div style={{ width: '100%', backgroundColor: '#2a2a2a', padding: '1rem', borderRadius: '12px' }}>
          <h3 style={{ textAlign: 'center', marginBottom: '1rem' }}>1. Volumen de Clases (Sesgo)</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={barData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#444" />
              <XAxis dataKey="name" stroke="#ccc" />
              <YAxis stroke="#ccc" />
              <RechartsTooltip contentStyle={{ backgroundColor: '#333', border: 'none', borderRadius: '8px' }} />
              <Bar dataKey="Cantidad" fill="#8884d8">
                {barData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.fill} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* 2. BoxPlot */}
        <div style={{ width: '100%', backgroundColor: '#2a2a2a', padding: '1rem', borderRadius: '12px', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
          <h3 style={{ textAlign: 'center', marginBottom: '1rem' }}>2. BoxPlot Velocidad (Outliers / IQR)</h3>
          <Plot
            data={boxPlotData}
            layout={{ 
              width: 500, height: 300, 
              paper_bgcolor: '#2a2a2a', plot_bgcolor: '#2a2a2a',
              font: { color: '#ccc' }, margin: { t: 20, b: 30, l: 40, r: 20 }
            }}
            config={{ responsive: true }}
          />
        </div>

        {/* 3. Radar Chart */}
        <div style={{ width: '100%', backgroundColor: '#2a2a2a', padding: '1rem', borderRadius: '12px' }}>
          <h3 style={{ textAlign: 'center', marginBottom: '1rem' }}>3. Perfil Multidimensional (Radar)</h3>
          <ResponsiveContainer width="100%" height={300}>
            <RadarChart cx="50%" cy="50%" outerRadius="80%" data={radarData}>
              <PolarGrid stroke="#555" />
              <PolarAngleAxis dataKey="subject" stroke="#ccc" />
              <PolarRadiusAxis angle={30} domain={[0, 15]} tick={false} axisLine={false} />
              <Radar name="Vehículos" dataKey="Vehículos" stroke="#4ade80" fill="#4ade80" fillOpacity={0.5} />
              <Radar name="Peatones (Escalado x5)" dataKey="Peatones" stroke="#f87171" fill="#f87171" fillOpacity={0.5} />
              <Legend />
              <RechartsTooltip contentStyle={{ backgroundColor: '#333', border: 'none' }} />
            </RadarChart>
          </ResponsiveContainer>
        </div>

        {/* 4. Scatter 3D */}
        <div style={{ width: '100%', backgroundColor: '#2a2a2a', padding: '1rem', borderRadius: '12px', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
          <h3 style={{ textAlign: 'center', marginBottom: '1rem' }}>4. Nube de Puntos Espacial (3D LiDAR)</h3>
          <Plot
            data={scatter3DData}
            layout={{ 
              width: 500, height: 300, 
              paper_bgcolor: '#2a2a2a', 
              font: { color: '#ccc' },
              margin: { t: 0, b: 0, l: 0, r: 0 },
              scene: {
                xaxis: { title: 'X', showgrid: false },
                yaxis: { title: 'Y', showgrid: false },
                zaxis: { title: 'Z', showgrid: false }
              }
            }}
          />
        </div>

      </div>
    </div>
  );
}

export default App;
