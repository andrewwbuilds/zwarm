import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import { BrowserRouter, Routes, Route } from "react-router-dom";
import TestPage from "./pages/TestPage";
import FleetPage from "./pages/FleetPage";
import SwarmPage from "./pages/SwarmPage";

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<App />} />
        <Route path="/test" element={<TestPage />} />
        <Route path="/fleet" element={<FleetPage />} />
      </Routes>
    </BrowserRouter>
  </StrictMode>,
)
