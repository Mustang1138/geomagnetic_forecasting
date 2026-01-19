# Project Progress Notes
## Week 1 (19–25 January 2026)

### 🎯 Objectives
- Set up development environment
- Create project structure
- Research data sources
- Begin data acquisition pipeline

---

### ✅ Completed
#### Environment Setup
- ✓ Installed PyCharm (via Toolbox)
- ✓ Created project with Python 3.13 virtual environment
- ✓ Initialized Git repository
- ✓ Installed core dependencies:
  - NumPy, Pandas, SciPy
  - scikit-learn
  - PyTorch (CPU-only)
  - Matplotlib, Seaborn
  - Jupyter

#### Project Structure
- ✓ Created directory hierarchy
- ✓ Configured `.gitignore` for Python/PyCharm
- ✓ Created `config.yaml` for parameters
- ✓ Wrote comprehensive `README.md`
- ✓ Implemented utility functions (`utils.py`)

#### Documentation
- ✓ Project structure documented
- ✓ Configuration system in place
- ✓ Progress tracking started

---

### 🔄 In Progress
- Data acquisition scripts (`data_loader.py`)
- API endpoint research and testing
- Initial data exploration

---

### 📋 Next Steps
#### Week 1 Remaining
1. Research NOAA DSCOVR API documentation
2. Test API endpoints with sample requests
3. Implement basic data loader for solar wind data
4. Download 1-month sample dataset for testing

#### Week 2 Planning
1. Complete data acquisition for all sources (Dst, Kp)
2. Implement preprocessing pipeline
3. Begin exploratory data analysis in Jupyter
4. Design baseline model architecture

---

### 🚧 Blockers/Issues
- None currently

---

### ❓ Questions for Supervisor (Meeting: 28-01-2026)
1. **Data Handling:**
   - Preferred approach for missing/incomplete data points?
   - Should I interpolate gaps or use forward-fill?

2. **Train/Test Split:**
   - Recommended split strategy for time-series data?
   - Use chronological split or random?

3. **Preprocessing:**
   - Any domain-specific normalization requirements for space weather data?
   - Should features be standardised or min-max scaled?

4. **Timeline:**
   - Target date for baseline model completion?
   - When should LSTM implementation begin?

5. **Evaluation:**
   - Any specific performance benchmarks to aim for?
   - Should I compare against published results?

---

### ⏱️ Time Tracking

| Activity | Hours |
|----------|-------|
| Environment setup & tooling | 2.0 |
| Project structure & config | 1.5 |
| Research & planning | 1.5 |
| Documentation | 1.0 |
| Utility code implementation | 0.5 |
| **Week 1 Total** | **6.5** |

---

### 📝 Technical Notes
#### Data Sources Research
- **NOAA DSCOVR:**
  - Provides real-time solar wind measurements
  - JSON API available
  - 1-minute cadence data
  - Parameters: Bz, velocity, density, temperature

- **Kyoto Dst Index:**
  - Available via text file format
  - Need to implement parser
  - Hourly resolution

- **GFZ Kp Index:**
  - 3-hour resolution
  - May need interpolation for alignment

#### Technology Decisions
- **Python 3.13:** Latest stable release, good performance
- **PyTorch (CPU):** More flexible than Keras, sufficient for project scale
- **YAML config:** Easy to modify parameters without code changes
- **Jupyter:** Essential for exploratory analysis

#### Risk Assessment
- **Low:** Environment setup - COMPLETED
- **Medium:** Data availability/API reliability - MONITORING
- **Low:** Model implementation - standard architectures
- **Low:** Timeline - sufficient buffer built in

---

### 🎓 Learning & Insights
- Git workflow refresher valuable
- PyTorch CPU-only installation much cleaner than CUDA version
- Configuration-driven approach will make experimentation easier
- Good project structure upfront saves time later

---
## Week 2 (26 January – 1 February 2026)
### 🎯 Objectives
*To be filled in*

---

*Last Updated: 19-01-2026*
*Next Review: 28-01-2026*