import React, { useState, useEffect, useCallback, useRef } from 'react';
import { useDropzone } from 'react-dropzone';
import axios from 'axios';
import { FiUploadCloud, FiZap, FiStar, FiImage, FiCode, FiTarget, FiVideo } from 'react-icons/fi';
import './App.css';

const API_BASE = 'http://localhost:8000/api';

const SplitSlider = ({ beforeImage, afterImage }) => {
  const [sliderPos, setSliderPos] = useState(50);
  
  const handleMove = (e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    let pos = ((e.clientX - rect.left) / rect.width) * 100;
    pos = Math.max(0, Math.min(100, pos));
    setSliderPos(pos);
  };

  return (
    <div className="split-slider-container" onMouseMove={handleMove}>
      <img src={beforeImage} alt="Before" className="img-background" />
      <img 
        src={afterImage} 
        alt="After" 
        className="img-foreground" 
        style={{ '--pos': `${sliderPos}%` }} 
      />
      <div className="slider-divider" style={{ left: `${sliderPos}%` }}>
         <div className="slider-handle">
           <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M15 18l-6-6 6-6"/></svg>
           <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M9 18l6-6-6-6"/></svg>
         </div>
      </div>
    </div>
  );
};

function App() {
  const [file, setFile] = useState(null);
  const [uploadedFilename, setUploadedFilename] = useState('');
  const [previewUrl, setPreviewUrl] = useState('');
  const [resultUrl, setResultUrl] = useState('');
  const [isVideo, setIsVideo] = useState(false);
  
  const [models, setModels] = useState([]);
  const [preset, setPreset] = useState('');
  const [fp16, setFp16] = useState(true);
  const [tileSize, setTileSize] = useState(512);
  const [tilePad, setTilePad] = useState(32);
  
  const [isProcessing, setIsProcessing] = useState(false);
  const [logs, setLogs] = useState([]);
  
  const [sharpnessLr, setSharpnessLr] = useState(0);
  const [sharpnessSr, setSharpnessSr] = useState(0);
  
  const logsEndRef = useRef(null);

  useEffect(() => {
    // Fetch models
    axios.get(`${API_BASE}/models`).then(res => {
      setModels(res.data.models);
      if (res.data.models.length > 0) setPreset(res.data.models[0]);
    }).catch(err => console.error(err));

    // WebSocket for logs
    const ws = new WebSocket("ws://localhost:8000/api/ws/logs");
    ws.onmessage = (event) => {
      setLogs(prev => [...prev, event.data]);
    };
    return () => ws.close();
  }, []);

  useEffect(() => {
    if (logsEndRef.current) {
      logsEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [logs]);

  const onDrop = useCallback(async (acceptedFiles) => {
    const selectedFile = acceptedFiles[0];
    if (!selectedFile) return;

    setFile(selectedFile);
    setPreviewUrl(URL.createObjectURL(selectedFile));
    setResultUrl('');
    setSharpnessSr(0);
    setLogs([]);
    
    const isVid = selectedFile.type.startsWith('video/');
    setIsVideo(isVid);
    
    const formData = new FormData();
    formData.append('file', selectedFile);
    
    try {
      const res = await axios.post(`${API_BASE}/upload`, formData);
      setUploadedFilename(res.data.filename);
      setSharpnessLr(res.data.sharpness);
    } catch (err) {
      console.error("Upload failed", err);
      alert("Failed to upload image.");
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({ 
    onDrop,
    accept: {'image/*': [], 'video/*': []},
    multiple: false
  });

  const handleInference = async () => {
    if (!uploadedFilename) return alert("Please upload an image first.");
    setIsProcessing(true);
    setLogs(prev => [...prev, "--- Starting Inference Request ---"]);
    try {
      const res = await axios.post(`${API_BASE}/run_inference`, {
        filename: uploadedFilename,
        preset, fp16, tile_size: tileSize, tile_pad: tilePad
      });
      if (res.data.status === 'success') {
        setResultUrl(`${API_BASE}/output/infer/${res.data.output_file}`);
        setSharpnessSr(res.data.sharpness);
      } else {
        alert(res.data.message || "Inference failed.");
      }
    } catch (err) {
      alert(err.response?.data?.message || "Error contacting backend.");
    } finally {
      setIsProcessing(false);
    }
  };

  const handleVideoInference = async () => {
    if (!uploadedFilename) return alert("Please upload a video first.");
    setIsProcessing(true);
    setLogs(prev => [...prev, "--- Starting Video Enhancement Pipeline ---"]);
    try {
      const res = await axios.post(`${API_BASE}/run_video`, {
        filename: uploadedFilename,
        preset, fp16, tile_size: tileSize, tile_pad: tilePad
      });
      if (res.data.status === 'success') {
        setResultUrl(`${API_BASE}/output/videos/${res.data.output_file}`);
      } else {
        alert(res.data.message || "Video processing failed.");
      }
    } catch (err) {
      alert(err.response?.data?.message || "Error contacting backend.");
    } finally {
      setIsProcessing(false);
    }
  };

  const handleMicrocosm = async () => {
    if (!uploadedFilename) return alert("Please upload an image first.");
    setIsProcessing(true);
    setLogs(prev => [...prev, "--- Starting Microcosm Request ---"]);
    try {
      const res = await axios.post(`${API_BASE}/run_microcosm`, {
        filename: uploadedFilename, preset, levels: 3, frames: 60
      });
    } catch (err) {
      alert("Error contacting backend.");
    } finally {
      setIsProcessing(false);
    }
  };
  
  const handleFingerprint = async () => {
    setIsProcessing(true);
    setLogs(prev => [...prev, "--- Starting Fingerprint Demo ---"]);
    try {
      await axios.post(`${API_BASE}/run_fingerprint`);
    } catch (err) {
      alert("Error contacting backend.");
    } finally {
      setIsProcessing(false);
    }
  };

  const improvement = sharpnessLr > 0 && sharpnessSr > 0 
    ? (((sharpnessSr - sharpnessLr) / sharpnessLr) * 100).toFixed(1) 
    : 0;

  return (
    <div className="app-container">
      <header className="header">
        <h1>Vision Studio</h1>
        <p>Next-Gen Neural Super Resolution & Artificial Forensics</p>
      </header>

      <main className="main-content">
        <div className="left-column">
          <section className="glass-card">
            <h2 className="card-title"><FiUploadCloud /> Input Source</h2>
            <div {...getRootProps()} className={`dropzone ${isDragActive ? 'active' : ''}`}>
              <input {...getInputProps()} />
              <FiUploadCloud className="drop-icon" />
              {isDragActive ? 
                <p>Drop the image here...</p> : 
                <p>Drag & drop an image, or click to select</p>
              }
              {file && <p style={{marginTop: '1rem', color: 'var(--accent-pink)'}}>{file.name}</p>}
            </div>

            <div className="settings-group">
              <div className="setting-item">
                <label>Model Preset</label>
                <select value={preset} onChange={e => setPreset(e.target.value)}>
                  {models.map(m => <option key={m} value={m}>{m}</option>)}
                </select>
              </div>
              
              <div className="setting-row">
                <div className="setting-item">
                  <label>Tile Size</label>
                  <input type="number" value={tileSize} onChange={e => setTileSize(Number(e.target.value))} />
                </div>
                <div className="setting-item">
                  <label>Tile Overlap</label>
                  <input type="number" value={tilePad} onChange={e => setTilePad(Number(e.target.value))} />
                </div>
              </div>
              
              <div className="checkbox-item">
                <input type="checkbox" id="fp16" checked={fp16} onChange={e => setFp16(e.target.checked)} />
                <label htmlFor="fp16">Half-Precision (FP16 GPU)</label>
              </div>
            </div>

            <div className="actions">
              {isVideo ? (
                <button className="btn btn-primary" onClick={handleVideoInference} disabled={!uploadedFilename || isProcessing}>
                  <FiVideo /> Enhance Video (4K)
                </button>
              ) : (
                <button className="btn btn-primary" onClick={handleInference} disabled={!uploadedFilename || isProcessing}>
                  <FiZap /> Enhance Pipeline
                </button>
              )}
            </div>
            
            {!isVideo && (
              <div className="actions secondary-actions">
                <button className="btn btn-magic" onClick={handleMicrocosm} disabled={!uploadedFilename || isProcessing}>
                  <FiStar /> Microcosm Trip
                </button>
                <button className="btn btn-dark" onClick={handleFingerprint} disabled={isProcessing}>
                  <FiTarget /> Fingerprint Demo
                </button>
              </div>
            )}
          </section>

          <section className="glass-card terminal-card">
            <h2 className="card-title" style={{fontSize: '1.2rem'}}><FiCode /> Execution Logs</h2>
            <div className="terminal">
              {logs.map((log, i) => <div key={i} className="log-line">{log}</div>)}
              <div ref={logsEndRef} />
            </div>
          </section>
        </div>

        <div className="right-column">
          <section className="glass-card full-height">
            <h2 className="card-title"><FiImage /> Visual Inspector</h2>
            
            {sharpnessSr > 0 && (
              <div className="metrics-bar">
                <div>Input Sharpness: <strong>{sharpnessLr.toFixed(1)}</strong></div>
                <div>ESRGAN Sharpness: <strong>{sharpnessSr.toFixed(1)}</strong></div>
                <div className="improvement">+{improvement}% Details</div>
              </div>
            )}
            
            <div className="preview-container">
              {isVideo ? (
                resultUrl ? (
                  <video src={resultUrl} controls autoPlay loop className="img-background" />
                ) : previewUrl ? (
                  <video src={previewUrl} controls className="img-background" style={{opacity: 0.5}} />
                ) : (
                  <div className="empty-state">
                    <FiVideo size={48} />
                    <p>Awaiting video...</p>
                  </div>
                )
              ) : (
                resultUrl && previewUrl ? (
                  <SplitSlider beforeImage={previewUrl} afterImage={resultUrl} />
                ) : previewUrl ? (
                  <div className="image-wrapper single-preview">
                    <img src={previewUrl} alt="Input Original" />
                  </div>
                ) : (
                  <div className="empty-state">
                    <FiImage size={48} />
                    <p>Awaiting processing...</p>
                  </div>
                )
              )}
            </div>
          </section>
        </div>
      </main>
    </div>
  );
}

export default App;
