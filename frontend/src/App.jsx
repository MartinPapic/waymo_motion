import React, { useState, useEffect, useCallback } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer, Cell, LabelList } from 'recharts';
import Plot from 'react-plotly.js';
import waymoData from './waymo_frontend.json';
import './App.css';

/* Cifras de la ultima corrida de `kedro run` sobre el dataset completo.
   El JSON que consume este dashboard es la muestra de 1.000 filas que
   genera la capa 03_primary; estas cifras describen el universo del que sale. */
const PIPELINE = {
  crudos: 2414127,
  limpios: 2405906,
  descartados: 8221,
  segmentos: 199,
  completitud: '99,66%'
};

/* Un color estable por clase: el mismo criterio que la memoria en Jupyter */
const CLASES = [
  { id: 'vehicle', nombre: 'Vehículos', color: '#4ade80' },
  { id: 'sign', nombre: 'Señales', color: '#fbbf24' },
  { id: 'pedestrian', nombre: 'Peatones', color: '#f87171' },
  { id: 'cyclist', nombre: 'Ciclistas', color: '#60a5fa' }
];

const miles = (n) => n.toLocaleString('es-CL');

/* Estilo compartido de los graficos de Plotly, para que se integren al fondo */
const layoutBase = {
  autosize: true,
  paper_bgcolor: 'rgba(0,0,0,0)',
  plot_bgcolor: 'rgba(0,0,0,0)',
  font: { color: '#cfdaeb', family: 'Segoe UI, system-ui, sans-serif', size: 12 },
  margin: { t: 20, b: 50, l: 60, r: 20 },
  legend: { orientation: 'h', y: -0.18 },
  /* type linear explicito: con varias series Plotly puede inferir un eje
     categorico y apilar todos los valores como etiquetas ilegibles */
  xaxis: { type: 'linear', gridcolor: 'rgba(255,255,255,0.08)', zerolinecolor: 'rgba(255,255,255,0.15)' },
  yaxis: { type: 'linear', gridcolor: 'rgba(255,255,255,0.08)', zerolinecolor: 'rgba(255,255,255,0.15)' }
};

const configPlotly = { displayModeBar: false, responsive: true };
const estiloPlot = { width: '100%', height: '400px' };

function App() {
  const [slideActual, setSlideActual] = useState(0);

  // Un arreglo por clase, calculado una sola vez desde el JSON del pipeline
  const porClase = CLASES.map((clase) => ({
    ...clase,
    filas: waymoData.filter((d) => d.object_type === clase.id)
  }));

  const datosBarras = porClase.map((c) => ({ nombre: c.nombre, Detecciones: c.filas.length, color: c.color }));
  const claseMayor = [...porClase].sort((a, b) => b.filas.length - a.filas.length)[0];
  const claseMenor = [...porClase].sort((a, b) => a.filas.length - b.filas.length)[0];
  const razonDesbalance = claseMenor.filas.length > 0
    ? Math.round(claseMayor.filas.length / claseMenor.filas.length)
    : 0;

  const cajasPorClase = porClase.map((c) => ({
    y: c.filas.map((d) => d.box_length),
    type: 'box',
    name: c.nombre,
    marker: { color: c.color, size: 3 },
    boxpoints: 'outliers'
  }));

  const dispersion2D = porClase.map((c) => ({
    x: c.filas.map((d) => d.box_length),
    y: c.filas.map((d) => d.box_width),
    mode: 'markers',
    type: 'scatter',
    name: c.nombre,
    marker: { color: c.color, size: 6, opacity: 0.75 }
  }));

  const nube3D = [
    {
      x: [0], y: [0], z: [0],
      mode: 'markers',
      type: 'scatter3d',
      name: 'Nuestro auto (ego)',
      marker: { color: '#38bdf8', size: 9, symbol: 'diamond' }
    },
    ...porClase.map((c) => ({
      x: c.filas.map((d) => d.box_center_x),
      y: c.filas.map((d) => d.box_center_y),
      z: c.filas.map((d) => d.box_center_z),
      mode: 'markers',
      type: 'scatter3d',
      name: c.nombre,
      marker: { color: c.color, size: 3, opacity: 0.8 }
    }))
  ];

  const slides = [
    {
      titulo: 'Volumen y sesgo de clases',
      tesis: `El sensor ve ${razonDesbalance} veces más ${claseMayor.nombre.toLowerCase()} que ${claseMenor.nombre.toLowerCase()}.`,
      lectura: (
        <p>
          Las cuatro clases detectadas por el LiDAR están lejos del equilibrio. Los vehículos y las señales
          dominan la escena urbana, mientras los ciclistas son una <strong>rareza estadística</strong>.
          Esto no es un detalle: un modelo entrenado con esta distribución aprende a reconocer autos con
          precisión y a <strong>fallar justamente con las clases más vulnerables</strong>. Es el sesgo de
          representación que la memoria cuantifica y mitiga con SMOTE.
        </p>
      ),
      grafico: (
        <ResponsiveContainer width="100%" height={400}>
          <BarChart data={datosBarras} margin={{ top: 24, right: 20, left: 10, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
            <XAxis dataKey="nombre" stroke="#93a4bd" tickLine={false} />
            <YAxis stroke="#93a4bd" tickLine={false} />
            <RechartsTooltip
              cursor={{ fill: 'rgba(255,255,255,0.05)' }}
              contentStyle={{ backgroundColor: '#0e1626', border: '1px solid rgba(255,255,255,0.12)', borderRadius: '10px', color: '#e6edf7' }}
              itemStyle={{ color: '#38bdf8', fontWeight: 600 }}
              labelStyle={{ color: '#93a4bd' }}
            />
            <Bar dataKey="Detecciones" radius={[8, 8, 0, 0]}>
              {datosBarras.map((fila) => <Cell key={fila.nombre} fill={fila.color} />)}
              <LabelList dataKey="Detecciones" position="top" fill="#cfdaeb" fontSize={13} />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      )
    },
    {
      titulo: 'Distribución de tamaño y filtro de anomalías (IQR)',
      tesis: 'Cada clase tiene una firma dimensional propia, y eso legitima el filtro.',
      lectura: (
        <p>
          El BoxPlot muestra la varianza del largo de cada caja detectada: la caja central concentra el 50%
          de los datos y los bigotes marcan el rango intercuartílico. Un peatón nunca mide lo que mide un
          camión, y esa separación es la que justifica filtrar por tamaño.{' '}
          <strong>Una precisión importante:</strong> estos datos ya pasaron por el filtro del pipeline, que
          se aplica sobre la <strong>distribución global</strong> de <code>box_length</code>. Como este
          gráfico separa <strong>por clase</strong>, siguen apareciendo puntos fuera de los bigotes: son
          valores atípicos respecto de su propia clase, pero perfectamente plausibles dentro del conjunto.
          No son errores que se hayan escapado.
        </p>
      ),
      grafico: (
        <Plot
          data={cajasPorClase}
          layout={{
            ...layoutBase,
            /* en un BoxPlot el eje X sí es categórico: una caja por clase */
            xaxis: { ...layoutBase.xaxis, type: 'category' },
            yaxis: { ...layoutBase.yaxis, title: { text: 'Largo de la caja (m)' } }
          }}
          config={configPlotly}
          style={estiloPlot}
          useResizeHandler
        />
      )
    },
    {
      titulo: 'Clústers y separabilidad (largo vs ancho)',
      tesis: 'Las clases se separan solas: no hace falta un modelo complejo para distinguirlas.',
      lectura: (
        <p>
          Este gráfico compara el largo contra el ancho de cada caja. Revela algo clave para la etapa de
          modelamiento: las clases forman <strong>clústers visiblemente distintos</strong>. Los peatones se
          agrupan en la esquina de dimensiones pequeñas y los vehículos ocupan la zona alta en ambos ejes.
          Que sean separables por dos variables simples anticipa que un clasificador básico ya obtendría
          buen desempeño, y confirma que las dimensiones físicas son las variables con mayor poder predictivo.
        </p>
      ),
      grafico: (
        <Plot
          data={dispersion2D}
          layout={{
            ...layoutBase,
            xaxis: { ...layoutBase.xaxis, title: { text: 'Largo de la caja (m)' } },
            yaxis: { ...layoutBase.yaxis, title: { text: 'Ancho de la caja (m)' } }
          }}
          config={configPlotly}
          style={estiloPlot}
          useResizeHandler
        />
      )
    },
    {
      titulo: 'Nube de puntos espacial: la visión del vehículo',
      tesis: 'Así ve el mundo el auto autónomo, en metros y en tiempo real.',
      lectura: (
        <p>
          Ponte en el lugar del vehículo: el rombo celeste en la coordenada (0, 0, 0) somos nosotros, y cada
          punto es un objeto que el láser detectó alrededor. El eje X es adelante y atrás, el eje Y
          izquierda y derecha, el eje Z la altura desde el suelo. <strong>Puedes rotar el gráfico con el
          mouse.</strong> Esta es la materia prima con la que el sistema decide frenar, esquivar o avanzar,
          y por eso la calidad de estos datos no es un tema estético sino de seguridad.
        </p>
      ),
      grafico: (
        <Plot
          data={nube3D}
          layout={{
            ...layoutBase,
            margin: { t: 0, b: 0, l: 0, r: 0 },
            scene: {
              xaxis: { title: { text: 'X · adelante/atrás (m)' }, gridcolor: 'rgba(255,255,255,0.12)', backgroundcolor: 'rgba(0,0,0,0)' },
              yaxis: { title: { text: 'Y · izquierda/derecha (m)' }, gridcolor: 'rgba(255,255,255,0.12)', backgroundcolor: 'rgba(0,0,0,0)' },
              zaxis: { title: { text: 'Z · altura (m)' }, gridcolor: 'rgba(255,255,255,0.12)', backgroundcolor: 'rgba(0,0,0,0)' }
            }
          }}
          config={configPlotly}
          style={{ width: '100%', height: '520px' }}
          useResizeHandler
        />
      )
    }
  ];

  const siguiente = useCallback(() => {
    setSlideActual((previo) => (previo === slides.length - 1 ? previo : previo + 1));
  }, [slides.length]);

  const anterior = useCallback(() => {
    setSlideActual((previo) => (previo === 0 ? previo : previo - 1));
  }, []);

  useEffect(() => {
    const alPresionar = (evento) => {
      if (evento.key === 'ArrowRight') siguiente();
      if (evento.key === 'ArrowLeft') anterior();
    };
    window.addEventListener('keydown', alPresionar);
    return () => window.removeEventListener('keydown', alPresionar);
  }, [siguiente, anterior]);

  const slide = slides[slideActual];

  return (
    <div className="app">
      <header className="cabecera">
        <div className="cabecera__sello">Evaluación Práctica 1 · Machine Learning y Arquitectura de Datos</div>
        <h1 className="cabecera__titulo">Waymo Open Dataset · Análisis Exploratorio</h1>
        <p className="cabecera__bajada">
          Datos reales de sensores LiDAR, procesados con un pipeline reproducible en Kedro
        </p>
      </header>

      <section className="kpis">
        <div className="kpi">
          <div className="kpi__valor">{miles(PIPELINE.crudos)}</div>
          <div className="kpi__etiqueta">Detecciones crudas</div>
        </div>
        <div className="kpi">
          <div className="kpi__valor">{PIPELINE.segmentos}</div>
          <div className="kpi__etiqueta">Segmentos de conducción</div>
        </div>
        <div className="kpi">
          <div className="kpi__valor">{miles(PIPELINE.descartados)}</div>
          <div className="kpi__etiqueta">Anomalías descartadas</div>
        </div>
        <div className="kpi">
          <div className="kpi__valor">{PIPELINE.completitud}</div>
          <div className="kpi__etiqueta">Tasa de completitud</div>
        </div>
      </section>

      <main className="escenario">
        <div className="barra">
          <button className="boton" onClick={anterior} disabled={slideActual === 0}>← Anterior</button>
          <div className="puntos">
            {slides.map((s, indice) => (
              <button
                key={s.titulo}
                className={indice === slideActual ? 'punto punto--activo' : 'punto'}
                onClick={() => setSlideActual(indice)}
                aria-label={`Ir al slide ${indice + 1}: ${s.titulo}`}
              />
            ))}
          </div>
          <button className="boton" onClick={siguiente} disabled={slideActual === slides.length - 1}>Siguiente →</button>
        </div>

        <div className="slide__numero">Slide {slideActual + 1} de {slides.length}</div>
        <h2 className="slide__titulo">{slide.titulo}</h2>
        <p className="slide__tesis">{slide.tesis}</p>

        <div className="lienzo">{slide.grafico}</div>

        <div className="lectura">
          <div className="lectura__rotulo">Cómo leer este gráfico</div>
          {slide.lectura}
        </div>
      </main>

      <footer className="pie">
        Navega con <span className="tecla">←</span> <span className="tecla">→</span> o con los botones.<br />
        Los gráficos muestran una muestra aleatoria de {miles(waymoData.length)} detecciones,
        extraída por el pipeline desde {miles(PIPELINE.limpios)} registros validados.
      </footer>
    </div>
  );
}

export default App;
