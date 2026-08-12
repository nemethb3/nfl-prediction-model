import React, { useState } from 'react';
import { DASHBOARD_SECTIONS } from '../constants/sections';
import '../styles/Navigation.css';

export default function Navigation({ activeSection, onSectionChange }) {
  const [menuOpen, setMenuOpen] = useState(false);

  const handleSectionChange = (sectionId) => {
    onSectionChange(sectionId);
    setMenuOpen(false);
  };

  return (
    <nav className="dashboard-nav">
      <div className="nav-brand">NFL Predictions</div>

      <button
        className="hamburger"
        onClick={() => setMenuOpen(!menuOpen)}
        aria-label="Toggle menu"
        aria-expanded={menuOpen}
      >
        ☰
      </button>

      <ul className={`nav-sections ${menuOpen ? 'open' : ''}`}>
        {DASHBOARD_SECTIONS.map((section) => (
          <li key={section.id}>
            <button
              className={activeSection === section.id ? 'active' : ''}
              onClick={() => handleSectionChange(section.id)}
            >
              {section.label}
            </button>
          </li>
        ))}
      </ul>
    </nav>
  );
}
