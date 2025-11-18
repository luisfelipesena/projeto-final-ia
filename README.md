# YouBot Autonomous Collection System

**Course:** MATA64 - Inteligência Artificial
**Institution:** UFBA (Universidade Federal da Bahia)
**Student:** Luis Felipe Cordeiro Sena
**Deadline:** January 6, 2026, 23:59

## 🎯 Project Overview

Autonomous system for KUKA YouBot mobile manipulator that collects 15 randomly distributed colored cubes (green, blue, red) and deposits them in corresponding color-coded boxes. The system uses:

- **Neural Networks (MLP/CNN)** for LIDAR-based obstacle detection and RGB camera-based cube identification
- **Fuzzy Logic** for control and decision-making
- **No GPS** - navigation based solely on LIDAR and camera sensors

## 🚀 Quick Start

### Prerequisites

- **Webots R2023b** - Robot simulator
- **Python 3.8+** - Required for Webots controller compatibility
- **8GB RAM minimum** - For simulation performance
- **macOS** (Intel/Apple Silicon) or **Linux Ubuntu 22.04+**

### Installation Steps

**Detailed instructions:** See [`specs/001-webots-setup/quickstart.md`](specs/001-webots-setup/quickstart.md)

1. **Install Webots R2023b**
   ```bash
   # macOS: Download DMG from GitHub releases
   open https://github.com/cyberbotics/webots/releases/tag/R2023b

   # Linux Ubuntu 22.04+: Download .deb package
   wget https://github.com/cyberbotics/webots/releases/download/R2023b/webots_2023b_amd64.deb
   sudo apt install ./webots_2023b_amd64.deb
   ```

2. **Verify Python Version**
   ```bash
   python3 --version  # Should be 3.8 or higher
   ```

3. **Create Virtual Environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Validate Setup**
   ```bash
   pytest tests/test_webots_setup.py -v
   ```

5. **Open World File**
   ```bash
   webots IA_20252/worlds/IA_20252.wbt
   ```

## 📂 Project Structure

```
projeto-final-ia/
├── IA_20252/                        # Webots world and controllers (DO NOT MODIFY supervisor/)
│   ├── controllers/
│   │   ├── youbot/                  # Robot controller (Python)
│   │   └── supervisor/              # Cube spawning logic (IMMUTABLE - constitution constraint)
│   ├── worlds/
│   │   └── IA_20252.wbt             # Simulation world file
│   └── libraries/                   # Webots libraries
├── src/                             # Source code (implementation in Phase 2+)
│   ├── perception/                  # Neural networks for LIDAR & camera
│   ├── control/                     # Fuzzy logic controller
│   ├── navigation/                  # Path planning and odometry
│   └── manipulation/                # Grasping and depositing sequences
├── tests/                           # Automated validation tests
│   └── test_webots_setup.py         # Environment setup validation (Phase 1)
├── specs/001-webots-setup/          # Feature specifications and planning artifacts
│   ├── spec.md                      # Feature specification
│   ├── plan.md                      # Implementation plan
│   ├── quickstart.md                # Setup guide (5-step process)
│   ├── data-model.md                # Entity definitions
│   ├── research.md                  # Research findings
│   └── contracts/                   # Test specifications
├── docs/                            # Documentation
│   ├── setup/                       # Setup guides and troubleshooting
│   └── environment.md               # Environment configuration details
├── logs/                            # Execution and test logs
├── models/                          # Trained neural network models (.pth files)
├── CLAUDE.md                        # Agent context and project background
├── DECISIONS.md                     # Technical decision log with scientific justification
├── REFERENCIAS.md                   # Scientific references (Top 10)
├── TODO.md                          # Phase-based execution plan
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## 🧪 Testing

Run validation tests to verify environment setup:

```bash
# Run all tests
pytest tests/test_webots_setup.py -v

# Run fast tests only (skip Webots version check)
pytest tests/test_webots_setup.py -v -m "not slow"

# Run specific test category
pytest tests/test_webots_setup.py::TestPythonEnvironment -v
```

**Expected output:** 4/4 tests pass when environment is correctly configured

## 📋 Development Phases

- **Phase 0:** ✅ Setup and Documentation (COMPLETE)
- **Phase 1.1:** 🚧 Webots Environment Setup (IN PROGRESS)
  - ✅ Planning artifacts complete
  - ✅ Validation tests created
  - ⏳ Manual Webots installation pending
  - ⏳ World file testing pending
- **Phase 1.2-1.4:** Sensor exploration and arena mapping
- **Phase 2:** Neural networks for perception (LIDAR + camera)
- **Phase 3:** Fuzzy logic controller
- **Phase 4:** Navigation and path planning
- **Phase 5:** Manipulation and grasping
- **Phase 6:** System integration
- **Phase 7:** Optimization and refinement
- **Phase 8:** Documentation and 15-minute video presentation

**Full roadmap:** See [TODO.md](TODO.md)

## 🏛️ Project Constitution

This project follows strict principles documented in `.specify/memory/constitution.md`:

1. **Scientific Justification (Principle I):** All decisions backed by academic references
2. **Complete Traceability (Principle II):** Every decision documented in DECISIONS.md
3. **Incremental Development (Principle III):** 8-phase approach with clear gates
4. **Senior Quality (Principle IV):** >80% test coverage, comprehensive documentation
5. **Disciplinary Constraints (Principle V):**
   - ❌ **NO modification to supervisor.py** (point deduction)
   - ❌ **NO code in presentation video** (3-10 point deduction)
   - ❌ **NO GPS sensor usage** (navigation must use LIDAR + camera only)

## 📚 Key Technologies

- **Webots R2023b:** Professional robot simulator (Michel, 2004)
- **KUKA YouBot:** Mobile manipulator platform (Bischoff et al., 2011)
- **PyTorch:** Neural network framework for perception
- **scikit-fuzzy:** Fuzzy logic controller implementation
- **pytest:** Automated testing framework
- **NumPy/SciPy:** Sensor data processing

## 📖 Documentation

- **Setup Guide:** [`specs/001-webots-setup/quickstart.md`](specs/001-webots-setup/quickstart.md) - 5-step setup process (~25 minutes)
- **Technical Decisions:** [`DECISIONS.md`](DECISIONS.md) - Scientific justification for all choices
- **References:** [`REFERENCIAS.md`](REFERENCIAS.md) - Top 10 academic papers
- **Phase Plan:** [`TODO.md`](TODO.md) - 8-phase development roadmap
- **Agent Context:** [`CLAUDE.md`](CLAUDE.md) - Project background and constraints

## 🤝 Contributing

This is an academic project for MATA64 course evaluation. External contributions are not accepted.

## 📝 License

Academic project - UFBA MATA64 2025.1

## ⚠️ Important Notes

1. **Supervisor Immutability:** The `IA_20252/controllers/supervisor/supervisor.py` file MUST NOT be modified (constitution constraint)
2. **No GPS Navigation:** All navigation must be based on LIDAR and camera sensors only
3. **Presentation Format:** 15-minute video with diagrams, graphs, and demonstrations - NO CODE allowed
4. **Deadline:** January 6, 2026, 23:59 (strict deadline)

## 🔗 References

- Michel, O. (2004). "Webots: Professional Mobile Robot Simulation"
- Bischoff et al. (2011). "KUKA youBot - a mobile manipulator for research and education"

**Full reference list:** See [REFERENCIAS.md](REFERENCIAS.md)

---

**Last Updated:** 2025-11-18
**Current Phase:** 1.1 - Webots Environment Setup
**Status:** Planning Complete, Implementation Pending Manual Steps
