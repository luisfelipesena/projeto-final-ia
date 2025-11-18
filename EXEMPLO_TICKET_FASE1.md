# Fase 1 - Validation Checklist

## Branch: `001-webots-setup`

### ✅ Quick Validation (2 min)

```bash
# 1. Verify structure
find specs/001-webots-setup -type f | wc -l  # Expected: 8 files
ls tests/test_webots_setup.py README.md docs/environment.md

# 2. Check decisions documented
grep -c "DECISÃO 00[5-8]" DECISIONS.md  # Expected: 4

# 3. Verify git state
git status  # Expected: clean working tree
git log --oneline -1  # Expected: feat(webots-setup): complete Phase 1.1...
```

**Pass criteria:** All commands return expected values ✅

---

### 📋 Content Validation (5 min)

#### Artifacts Complete?
- [ ] `specs/001-webots-setup/spec.md` - 4 user stories exist
- [ ] `specs/001-webots-setup/quickstart.md` - 5-step guide readable
- [ ] `specs/001-webots-setup/tasks.md` - 44 tasks listed
- [ ] `DECISIONS.md` - DECISÃO 005-008 with scientific refs
- [ ] `README.md` - Project overview clear
- [ ] `tests/test_webots_setup.py` - Test suite structure looks good

#### Key Decisions Documented?
```bash
grep "DECISÃO 005" DECISIONS.md  # Webots installation method
grep "DECISÃO 006" DECISIONS.md  # Python integration strategy
grep "DECISÃO 007" DECISIONS.md  # Testing framework
grep "DECISÃO 008" DECISIONS.md  # Sensor validation
```

#### TODO.md Updated?
```bash
grep -A 8 "1.1 Setup do Webots" TODO.md
```
**Expected:** Documentation tasks marked [x], manual tasks marked [ ]

---

### 🔬 SpecKit Quality Check (optional)

```bash
/speckit.analyze
```

**Validates:**
- spec.md ↔ plan.md ↔ tasks.md consistency
- All requirements have corresponding tasks
- No contradictions between artifacts

---

### ⏳ Cannot Validate Yet (requires Webots installed)

- [ ] Webots R2023b installs successfully
- [ ] World file `IA_20252.wbt` loads <30s
- [ ] 15 cubes spawn correctly
- [ ] `pytest tests/test_webots_setup.py -v` passes 4/4 tests

**Action required:** Follow `specs/001-webots-setup/quickstart.md` to install Webots

---

### 🚀 Ready to Push?

```bash
# If validation passes:
git push origin 001-webots-setup

# Then: Install Webots manually following quickstart.md
```

---

## 📝 Next Task: Phase 1.2 - Sensor Exploration

**Branch:** `002-sensor-exploration`

### Setup New Feature

```bash
# 1. Ensure Phase 1.1 is complete
git checkout main
git merge 001-webots-setup  # Or create PR and merge via GitHub

# 2. Create new feature branch
git checkout -b 002-sensor-exploration

# 3. Start SpecKit workflow
/speckit.specify
```

**Prompt for `/speckit.specify`:**
```
Create specification for Phase 1.2-1.3: Sensor Exploration

Scope:
- Explore YouBot LIDAR sensor (read data, understand format, visualize)
- Explore RGB camera (capture frames, test color detection)
- Test basic robot controls (movement, arm, gripper)
- Map arena dimensions and obstacle positions

Input from TODO.md:
- Phase 1.2: Basic controls (base, arm, gripper)
- Phase 1.3: Sensor analysis (LIDAR 512 points, RGB camera)
- Phase 1.4: Arena mapping

Deliverables:
- Test script: tests/test_basic_controls.py
- Notebook: notebooks/01_sensor_exploration.ipynb
- Documentation: docs/arena_map.md

Requirements: Must have Webots installed and Phase 1.1 complete
```

### SpecKit Workflow for Phase 1.2

```bash
/speckit.specify   # Create spec for sensor exploration
/speckit.clarify   # Ask clarifying questions
/speckit.plan      # Generate implementation plan
/speckit.tasks     # Break down into tasks
/speckit.implement # Execute implementation
```

**Expected artifacts:**
- `specs/002-sensor-exploration/spec.md`
- `specs/002-sensor-exploration/plan.md`
- `specs/002-sensor-exploration/tasks.md`
- Implementation: notebooks, test scripts, documentation

---

## 🎯 Summary

**Phase 1.1 Status:** Planning + automation complete, manual steps pending
**Next Action:** Validate branch → Push → Install Webots → Start Phase 1.2
**Branch Strategy:** One feature branch per phase (001, 002, 003...)
**Workflow:** SpecKit cycle for each phase (specify → clarify → plan → tasks → implement)
