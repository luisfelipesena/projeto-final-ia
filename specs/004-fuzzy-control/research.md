# Fuzzy Logic Controller Design Research - Mobile Robot Navigation

**Date:** 2025-11-21
**Phase:** Fase 3 - Controle Fuzzy
**Purpose:** Research best practices for fuzzy logic controller design to inform DECISÃO XXX

---

## 1. Membership Function Design

### 1.1 Optimal Shapes for Distance/Angle Variables

**Decision: Triangular membership functions with 50% overlap as baseline**

**Rationale:**
- **Triangular functions** are the most common starting point for mobile robot navigation
- **Computational efficiency:** Simpler calculation than Gaussian (piecewise linear)
- **Interpretability:** Clearer boundaries for linguistic terms
- **50% overlap:** Standard recommendation for smooth control transitions

**Literature Evidence:**

**Omrane et al. (2016)** - "Fuzzy Logic Based Control for Autonomous Mobile Robot Navigation":
- Used **5 triangular membership functions** for distance inputs (VS, S, M, B, VB)
- Universe of discourse: [0, 500mm] for proximity sensors
- **7 membership functions** for angle inputs (-180° to 180°): NB, NM, NS, Z, PS, PM, PB
- Centroid defuzzification method

**Faisal et al. (2013)** - "Fuzzy Logic Navigation and Obstacle Avoidance":
- **Triangular and trapezoidal** membership functions for fuzzification
- Gaussian membership functions found **most efficient** for navigation accuracy
- Trade-off: Gaussian > accuracy, Triangular > speed

**Key Finding:** Start with triangular (50% overlap), switch to Gaussian if accuracy <90%

### 1.2 Overlap Percentage Between Adjacent Functions

**Decision: 50% overlap as baseline, adjustable to 30-70% based on smoothness requirements**

**Rationale:**
- **50% overlap** is industry standard for symmetric triangular functions
- Overlapping membership functions ensure **smooth control signals** near boundaries
- Too little overlap (<30%): Discontinuous control, chattering
- Too much overlap (>70%): Excessive rule firing, computational overhead

**Research Findings:**
- IntechOpen (2020): "50% overlap recommended for symmetric triangular membership functions"
- ResearchGate discussion: "Overlapping denotes uncertainty in participation of members to a set"
- When membership functions overlap, system obtains **smooth and continuous control signal**

**Arena-Specific Tuning (7m×4m, LIDAR 0.3-5m):**
- Distance MFs: 40-60% overlap (medium smoothness for 5m max range)
- Angle MFs: 50% overlap (standard for angular control)
- Velocity output MFs: 30-50% overlap (crisp action changes acceptable)

### 1.3 Number of Membership Functions Per Variable

**Decision: 5 MFs for distance, 7 MFs for angle, 5 MFs for velocity outputs**

**Rationale:**

**Distance Inputs (Obstacle/Cube Distance):**
- **5 MFs** (Very Small, Small, Medium, Big, Very Big) - Omrane et al. (2016)
- Adequate granularity for 5m LIDAR range
- Fewer than 5: insufficient precision for safety
- More than 7: diminishing returns, exponential rule growth

**Angle Inputs (Bearing to Obstacle/Cube):**
- **7 MFs** (NB, NM, NS, Z, PS, PM, PB) - Standard in mobile robotics literature
- Covers full ±180° range with adequate angular resolution
- Allows nuanced turning decisions (slight left vs strong left)

**Velocity Outputs (Linear/Angular):**
- **5 MFs** (Stop, Slow, Medium, Fast, VeryFast) or equivalent
- Sufficient for smooth acceleration profiles
- 3 MFs: too coarse (abrupt speed changes)
- 7 MFs: overkill for typical actuator resolution

**Comparative Study (ResearchGate):**
- "FLC with **5 membership functions** provides **better performance** than 7 MF controller"
- Sweet spot: 5-7 MFs balances precision vs rule explosion

### 1.4 Range Tuning for 7m×4m Arena (LIDAR 0.3-5m)

**Decision: Domain-specific ranges for YouBot arena**

**Distance Membership Functions (LIDAR: 0.3-5.0m):**
```
Universe of Discourse: [0.0, 5.0] meters

Very Close (VC):  [0.0, 0.3, 0.6]     # Triangular, peak at 0.3m
Close (C):        [0.4, 0.8, 1.2]     # Triangular, peak at 0.8m
Medium (M):       [1.0, 1.8, 2.6]     # Triangular, peak at 1.8m
Far (F):          [2.2, 3.5, 4.3]     # Triangular, peak at 3.5m
Very Far (VF):    [4.0, 5.0, 5.0]     # Trapezoidal (saturation at max range)
```

**Angle Membership Functions (Bearing: -180° to 180°):**
```
Universe of Discourse: [-180, 180] degrees

Negative Big (NB):      [-180, -180, -90, -45]  # Trapezoidal left
Negative Medium (NM):   [-90, -60, -30]         # Triangular
Negative Small (NS):    [-45, -15, 0]           # Triangular
Zero (Z):               [-15, 0, 15]            # Triangular
Positive Small (PS):    [0, 15, 45]             # Triangular
Positive Medium (PM):   [30, 60, 90]            # Triangular
Positive Big (PB):      [45, 90, 180, 180]      # Trapezoidal right
```

**Velocity Output Functions (Linear: 0-0.3 m/s, Angular: -0.5 to 0.5 rad/s):**
```
Linear Velocity: [0.0, 0.3] m/s
Stop (S):       [0.00, 0.00, 0.05]    # Triangular
Slow (SL):      [0.03, 0.08, 0.13]    # Triangular
Medium (M):     [0.10, 0.18, 0.25]    # Triangular
Fast (F):       [0.20, 0.30, 0.30]    # Trapezoidal (max speed)

Angular Velocity: [-0.5, 0.5] rad/s
Strong Left (SL):   [-0.5, -0.5, -0.3, -0.15]  # Trapezoidal
Left (L):           [-0.3, -0.15, 0.0]         # Triangular
Straight (ST):      [-0.1, 0.0, 0.1]           # Triangular
Right (R):          [0.0, 0.15, 0.3]           # Triangular
Strong Right (SR):  [0.15, 0.3, 0.5, 0.5]      # Trapezoidal
```

**Justification:**
- Arena diagonal = √(7² + 4²) ≈ 8.1m → 5m LIDAR covers >60% of arena
- Cube size 0.05m → "Very Close" threshold at 0.3m (6× cube size) for safe grasping
- Obstacle clearance: "Close" at 0.8m allows reaction time at 0.2 m/s
- Angular resolution: ±15° = "Straight" acceptable for alignment tolerance

**Alternatives Considered:**
1. **Gaussian membership functions:**
   - ✅ Higher accuracy (±2-5% improvement in tracking error)
   - ❌ 2-3× slower defuzzification (exp() computation)
   - **Verdict:** Use if triangular accuracy <88% (target: >90%)

2. **7 MFs for distance:**
   - ✅ Finer granularity
   - ❌ Rule explosion: 7 (distance) × 7 (angle) = 49 rules vs 5×7=35
   - **Verdict:** Rejected, 5 MFs sufficient per literature

3. **Trapezoidal for all MFs:**
   - ✅ Flat top = stable output in saturation regions
   - ❌ 4 parameters vs 3 (triangular) = 33% more tuning complexity
   - **Verdict:** Use only for boundary MFs (VF, NB, PB)

---

## 2. Rule Base Construction

### 2.1 Minimum Rule Coverage for Safety (Obstacle Avoidance)

**Decision: 100% rule coverage for obstacle avoidance scenarios, minimum 15-25 safety-critical rules**

**Rationale:**
- **Safety-first principle:** Every possible obstacle configuration must have defined behavior
- Incomplete rule base → undefined behavior → potential collisions
- Literature shows 20-35 rules typical for obstacle avoidance alone

**Literature Evidence:**

**Khairudin et al. (2020)** - "Mobile Robot Control in Obstacle Avoidance":
- **25 rules** for covering robot's interactions with various obstacles
- Minimum viable rule set for comprehensive obstacle handling

**Omrane et al. (2016):**
- **35 IF-THEN rules** for autonomous navigation + obstacle avoidance
- Rule priority: Safety (obstacle) > Task (goal) > Exploration

**Rule Coverage Matrix (5 distance × 7 angle = 35 combinations):**
```
Safety-Critical Subset (Must Define):
- Very Close × Any Angle (7 rules) → STOP or EMERGENCY_TURN
- Close × Center angles (Z, NS, PS) (3 rules) → SLOW + TURN_AWAY
- Close × Side angles (NM, PM, NB, PB) (4 rules) → TURN_STRONG

Total Safety Rules: ~15 minimum (covers critical collision scenarios)
```

**Testing Approach:**
- Simulate obstacle at all MF combinations (5×7=35 scenarios)
- Verify no undefined output (NaN or zero control)
- Validate safe behavior (collision-free in 100% of cases)

### 2.2 Rule Prioritization Strategies (Safety > Task > Exploration)

**Decision: Hierarchical rule structure with explicit safety override**

**Rationale:**
- **Safety rules** must fire unconditionally when obstacles detected
- **Task rules** (approach cube, navigate to box) active only when obstacle-free
- **Exploration rules** (search pattern) lowest priority, fill idle states

**Hierarchical Architecture:**

**Layer 1: Safety (Reactive - Highest Priority)**
```python
IF distance_to_obstacle IS VeryClose THEN
    linear_velocity = Stop
    angular_velocity = StrongLeft  # or away from obstacle
    action = AVOIDING
    [OVERRIDE ALL OTHER RULES]
```
- 15 rules covering all critical obstacle scenarios
- Fire immediately, suppress task/exploration rules
- Implementation: Separate fuzzy system or rule weights (w=10.0)

**Layer 2: Task Execution (Goal-Oriented - Medium Priority)**
```python
IF cube_detected AND distance_to_cube IS Close AND obstacle_clear THEN
    action = GRASPING
    linear_velocity = Slow
    angular_velocity = AlignToCube
```
- 20-30 rules for: approach cube, grasp, navigate to box, deposit
- Conditional on `obstacle_clear` flag from Layer 1
- Rule weights: w=5.0

**Layer 3: Exploration (Idle Behavior - Lowest Priority)**
```python
IF NOT cube_detected AND obstacle_clear THEN
    action = SEARCHING
    linear_velocity = Medium
    angular_velocity = SlowRotate  # scanning pattern
```
- 5-10 rules for search patterns (spiral, wall-following, random walk)
- Only active when no obstacles and no cubes detected
- Rule weights: w=1.0

**Implementation Strategy:**
- **Weighted rule consequents:** Safety rules get 10× weight multiplier
- **Conditional rule activation:** Task rules disabled if `safety_flag=True`
- **Mamdani system with priority layers** (scikit-fuzzy: manual rule weighting)

**Literature Support:**
- **Omrane et al. (2016):** "Rule priority: Safety (obstacle) > Task (goal) > Exploration"
- **Saffiotti (1997):** Behavior-based architecture with fuzzy arbitration
- **Brooks (1986):** Subsumption architecture (safety subsumes task)

### 2.3 Handling Conflicting Rules (Multiple Antecedents Match)

**Decision: Max-Min inference with Centroid defuzzification (Mamdani standard)**

**Rationale:**
- **Max-Min** is default Mamdani inference method (robust, well-tested)
- When multiple rules fire, **aggregation** combines their outputs
- **Centroid defuzzification** provides smooth weighted average

**Conflict Resolution Mechanisms:**

**1. Aggregation Method (Multiple Rules Fire Simultaneously):**
```python
# Example: Two rules fire at same time
Rule 1: IF distance IS Close AND angle IS Left THEN turn_right (μ=0.7)
Rule 2: IF distance IS Medium AND angle IS Center THEN go_straight (μ=0.4)

# Max aggregation: Take maximum membership across overlapping outputs
# Result: Blended output weighted by firing strengths (0.7 and 0.4)
# Centroid defuzzification: Final output ≈ 60% turn_right + 40% straight
```

**2. Rule Weight Prioritization (Safety Override):**
```python
# Safety rule fires with high weight
Safety Rule: IF distance IS VeryClose THEN stop (w=10.0, μ=0.9)
Task Rule: IF cube_detected THEN approach (w=1.0, μ=0.6)

# Weighted aggregation: 10.0×0.9 = 9.0 (safety) vs 1.0×0.6 = 0.6 (task)
# Result: Safety rule dominates output (93% contribution)
```

**3. Conditional Firing (Mutually Exclusive States):**
```python
# State machine guards prevent conflicting rules
IF state == AVOIDING:
    # Only safety rules active
ELIF state == GRASPING:
    # Only manipulation rules active
```

**Implementation (scikit-fuzzy):**
- `ctrl.ControlSystemSimulation` handles aggregation automatically
- Default: Max-Min inference + Centroid defuzzification
- Custom weights via `consequent.accumulation_method = np.fmax` (element-wise max)

**Literature Support:**
- **Mamdani & Assilian (1975):** Original Max-Min inference method
- **Ross (2010), "Fuzzy Logic with Engineering Applications":** Centroid defuzzification most common (>70% of controllers)
- **Omrane et al. (2016):** Used Mamdani with centroid for mobile robot

### 2.4 Typical Rule Counts for Mobile Robot Navigation

**Decision: 35-50 total rules (15 safety + 25 task + 5-10 exploration)**

**Rationale:**
- Literature survey shows **20-62 rules** typical for mobile robot fuzzy controllers
- Rule count scales with: (# inputs) × (# MFs per input) × task complexity
- Trade-off: Coverage vs maintainability

**Literature Survey Results:**

| Study | Application | Rule Count | Notes |
|-------|-------------|------------|-------|
| Omrane et al. (2016) | Navigation + Obstacle Avoidance | **35 rules** | 2 inputs (distance, angle), 5×7 MFs |
| Faisal et al. (2013) | Navigation + Avoidance | **62 rules** | Sugeno-type, trajectory tracking |
| Khairudin et al. (2020) | Obstacle Avoidance Only | **25 rules** | Simplified, reactive only |
| Type-2 Fuzzy (MDPI 2022) | Obstacle Avoidance | **63 rules** | Type-2 fuzzy (interval MFs) |
| Simple Navigation | Wall Following | **20 rules** | Minimalist approach |

**Calculation for YouBot System:**

**Inputs:**
- `distance_to_obstacle` (5 MFs)
- `angle_to_obstacle` (7 MFs)
- `distance_to_cube` (5 MFs)
- `angle_to_cube` (7 MFs)
- `cube_detected` (2 crisp: Yes/No)
- `holding_cube` (2 crisp: Yes/No)

**Full Combinatorial:** 5×7×5×7×2×2 = **9,800 rules** (INFEASIBLE!)

**Practical Rule Design (Conditional Partitioning):**

**Scenario 1: Obstacle Nearby (Safety Layer)**
- Inputs: `distance_to_obstacle`, `angle_to_obstacle`
- Rules: 5×7 = 35, but only ~15 critical (VeryClose + Close scenarios)

**Scenario 2: Cube Detected, No Obstacles (Task Layer)**
- Inputs: `distance_to_cube`, `angle_to_cube`
- Rules: 5×7 = 35, but only ~20 used (approaching behavior)

**Scenario 3: Searching, No Obstacles, No Cube (Exploration Layer)**
- Inputs: minimal (random walk or wall-following)
- Rules: 5-10 (simple scanning patterns)

**Total Estimated:** 15 (safety) + 20 (task) + 10 (exploration) = **45 rules**

**Rule Reduction Techniques (if >50 rules):**
1. **Ordinal structure fuzzy logic** (Huang et al., 2010): Reduces high-dimensional rules while preserving interpretability
2. **Genetic Algorithm optimization** (Homaifar et al., 1995): Prunes redundant rules
3. **Rule base simplification:** Merge similar consequents (e.g., "Slow" and "VerySlow" → "Slow")

**Verdict:** Target **35-50 rules**, document in `src/control/fuzzy_rules.txt`

**Alternatives Considered:**
1. **Minimal rule set (20 rules):**
   - ✅ Fast inference (<10ms)
   - ❌ Incomplete coverage, undefined behavior in edge cases
   - **Verdict:** Risky for safety-critical navigation

2. **Comprehensive rule set (100+ rules):**
   - ✅ Handles all scenarios explicitly
   - ❌ Maintainability nightmare, slow inference (>100ms)
   - ❌ Violates literature best practice (20-62 typical)
   - **Verdict:** Over-engineered

3. **Adaptive fuzzy (ANFIS):**
   - ✅ Auto-tunes membership functions from data
   - ❌ Requires training dataset (not available Phase 3)
   - ❌ Black-box nature reduces interpretability
   - **Verdict:** Defer to Phase 7 (optimization) if needed

---

## 3. Defuzzification Methods

### 3.1 Centroid vs Bisector vs Mean of Maximum

**Decision: Centroid defuzzification as primary method**

**Rationale:**
- **Centroid** is **most commonly used** defuzzification method (>70% of literature)
- Returns center of gravity of fuzzy output set → smooth, continuous control
- Well-tested in mobile robotics, predictable behavior
- Default in scikit-fuzzy and MATLAB Fuzzy Toolbox

**Comparison of Methods:**

**Centroid (Center of Gravity - CoG):**
```
Formula: x* = ∫ x·μ(x) dx / ∫ μ(x) dx
Properties:
- Weighted average of all activated output MFs
- Smooth transitions between rules
- Most intuitive for control applications
```

**Bisector (Equal Area Method):**
```
Definition: Vertical line that divides fuzzy set into two equal areas
Properties:
- Sometimes coincides with centroid, sometimes not
- Less intuitive than centroid
- Comparable computational cost to centroid
- Rarely used in robotics (no clear advantage)
```

**Mean of Maximum (MOM):**
```
Definition: Average of x-values where μ(x) is maximum
Properties:
- Selects most dominant fuzzy set
- Discontinuous transitions (jumps between MFs)
- Faster than centroid (no integration)
- Poor for smooth control (causes chattering)
```

**Largest of Maximum (LOM) / Smallest of Maximum (SOM):**
```
Similar to MOM but biased to right/left maximum
- Even more discontinuous than MOM
- Rarely used except for discrete decision-making
```

**Literature Evidence:**

**MATLAB Documentation (2024):**
- "In general, using the default **centroid method is good enough** for most applications"
- Centroid returns center of area under the curve (most popular)

**IIT Kharagpur Tutorial (Samanta):**
- "**Centroid defuzzification** is most commonly used"
- Takagi-Sugeno-Kang fuzzy inference + centroid = standard approach

**Comparison Study (IEEE 2007) - Non-Interval Data:**
- "Choice of defuzzification method has **no significant influence** on output for typical data"
- Centroid, bisector, MOM produce similar results in most scenarios
- Centroid preferred for **smooth control signals**

**Omrane et al. (2016) - Mobile Robot Navigation:**
- Used **Mamdani inference with centroid defuzzification**
- Achieved smooth trajectory tracking with <5% error

**When to Use Alternatives:**

**Use Bisector if:**
- Centroid produces asymmetric output bias (rare)
- Geometric center more meaningful than weighted average
- Computational cost identical, so no downside to try

**Use MOM if:**
- Discrete actions required (e.g., SELECT_BOX_1 vs SELECT_BOX_2)
- Speed critical and discontinuities acceptable
- NOT recommended for velocity control (causes jerky motion)

**YouBot Application:**
- **Linear/Angular Velocity Outputs:** Centroid (smooth acceleration)
- **Action Selection Output:** MOM acceptable (discrete states: SEARCH, GRASP, DEPOSIT)

### 3.2 Computational Cost Comparison

**Decision: Centroid acceptable for <50ms inference target, optimize if >100ms**

**Computational Complexity Analysis:**

**Centroid (CoG):**
```
Time Complexity: O(n) where n = resolution of output universe
Process:
1. Aggregate all fired rules (Max operation): O(r) where r = number of rules
2. Compute membership for each output point: O(n)
3. Integrate numerator ∫ x·μ(x) dx: O(n)
4. Integrate denominator ∫ μ(x) dx: O(n)
5. Divide: O(1)
Total: O(r + 3n) ≈ O(n) for n >> r

Typical: n=100 points, r=35 rules → ~300-500 operations
Benchmark: 5-20ms on modern CPU (Python scikit-fuzzy)
```

**Bisector:**
```
Time Complexity: O(n) [same as centroid]
Process:
1. Compute total area: ∫ μ(x) dx: O(n)
2. Binary search for equal-area line: O(log n)
3. Iterative refinement: O(k) iterations, each O(n)
Total: O(n·k) where k=5-10 iterations typical

Benchmark: 5-20ms (comparable to centroid)
Note: No significant speed advantage over centroid
```

**Mean of Maximum (MOM):**
```
Time Complexity: O(n)
Process:
1. Find maximum membership: O(n)
2. Find all x where μ(x) = max: O(n)
3. Compute average: O(m) where m = number of maxima
Total: O(2n + m) ≈ O(n)

Benchmark: 2-10ms (slightly faster than centroid)
Speedup: ~2× faster but not dramatic
Trade-off: Discontinuous output NOT worth 2× speedup for velocity control
```

**Literature Benchmarks:**

**Fast Defuzzification Study (Academia.edu):**
- Centroid: ~15ms for 100-point resolution
- Fast centroid estimation (approximation): ~5ms
- Bisector: ~15ms (no speed advantage)
- MOM: ~8ms (2× faster but discontinuous)

**Practical Recommendation:**
- **Target inference time:** <50ms for real-time control (20 Hz)
- **Centroid baseline:** 5-20ms typical (Python scikit-fuzzy)
- **Optimization trigger:** Only if total inference >100ms (unlikely with 35-50 rules)

**Optimization Strategies (if needed):**
1. **Reduce output universe resolution:** 100 → 50 points (2× speedup)
2. **Lookup table caching:** Pre-compute common rule combinations
3. **Sugeno inference:** Linear consequents (no defuzzification step)
4. **C++ implementation:** 10× speedup over Python (use Cython)

**Verdict:** Centroid sufficient, defer optimization to Phase 7

### 3.3 When to Use Each Method

**Decision Matrix:**

| Use Case | Recommended Method | Justification |
|----------|-------------------|---------------|
| **Linear Velocity Control** | Centroid | Smooth acceleration, no chattering |
| **Angular Velocity Control** | Centroid | Smooth turning, predictable trajectories |
| **Discrete Action Selection** (SEARCH/GRASP/DEPOSIT) | MOM or crisp logic | Discrete states, speed > smoothness |
| **Arm Position Control** | Centroid | Smooth motion planning |
| **Gripper Control** (OPEN/CLOSE) | MOM or crisp | Binary action, speed critical |
| **Emergency Stop** | MOM (fastest) | Milliseconds matter |

**Implementation Plan:**
```python
# Primary velocity controller (Mamdani + Centroid)
velocity_controller = ctrl.ControlSystem([rule1, rule2, ...])
velocity_sim = ctrl.ControlSystemSimulation(velocity_controller)
velocity_sim.defuzzify_method = 'centroid'  # Default

# Discrete action selector (optional: use MOM for speed)
action_controller = ctrl.ControlSystem([action_rules])
action_sim = ctrl.ControlSystemSimulation(action_controller)
action_sim.defuzzify_method = 'mom'  # Faster for discrete outputs
```

**Alternatives Considered:**
1. **Bisector for all outputs:**
   - ❌ No clear advantage over centroid
   - ❌ Less tested in robotics literature
   - **Verdict:** Not justified

2. **MOM for velocity control:**
   - ❌ Discontinuous output causes chattering
   - ❌ Jerky motion (step changes in velocity)
   - **Verdict:** Only for discrete actions

3. **Sugeno inference (no defuzzification):**
   - ✅ Faster (no integration step)
   - ✅ Linear consequents (e.g., output = 0.5·x + 0.3·y)
   - ❌ Less interpretable than Mamdani
   - ❌ Harder to design rules (need to specify linear equations)
   - **Verdict:** Consider if centroid inference >100ms (unlikely)

---

## 4. Performance Optimization

### 4.1 Techniques to Reduce Inference Time Below 50ms

**Decision: Implement 3-tier optimization strategy (baseline → caching → Sugeno fallback)**

**Target:** <50ms total inference time for 20 Hz control loop (50ms = 0.050s)

**Baseline Performance Estimate (scikit-fuzzy, Python):**
```
Rule evaluation: 35 rules × 0.5ms = 17.5ms
Aggregation: 5ms
Centroid defuzzification: 10-15ms (100-point resolution)
Total: ~35-40ms (within target)
```

**Optimization Tier 1: Algorithmic (No Code Changes)**

**1.1 Reduce Output Universe Resolution:**
```python
# Before optimization
linear_velocity = ctrl.Consequent(np.linspace(0, 0.3, 100), 'linear_velocity')  # 100 points

# After optimization
linear_velocity = ctrl.Consequent(np.linspace(0, 0.3, 50), 'linear_velocity')   # 50 points

# Speedup: 2× faster defuzzification (50 points vs 100)
# Trade-off: Output resolution 0.006 m/s vs 0.003 m/s (negligible for 0.3 m/s max)
```

**1.2 Simplify Membership Function Shapes:**
```python
# Triangular: 3 parameters, 2 line segments
mf_triangle = fuzz.trimf(x, [a, b, c])  # Fast

# Gaussian: exp() computation, slower
mf_gaussian = fuzz.gaussmf(x, mean, sigma)  # 2-3× slower

# Recommendation: Use triangular for all MFs unless accuracy requires Gaussian
```

**1.3 Prune Redundant Rules:**
```python
# Identify rules with identical consequents
Rule 1: IF distance IS Close AND angle IS Left THEN turn_right[Medium]
Rule 2: IF distance IS Medium AND angle IS Left THEN turn_right[Medium]

# Merge into single rule with broader antecedent
Rule Combined: IF (distance IS Close OR Medium) AND angle IS Left THEN turn_right[Medium]

# Speedup: 2 rules → 1 rule (2× reduction in evaluation time)
```

**Optimization Tier 2: Caching Strategies**

**2.1 Lookup Table for Common Scenarios:**
```python
# Pre-compute fuzzy outputs for grid of inputs
lookup_table = {}
for dist in [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]:  # 8 distances
    for angle in [-135, -90, -45, 0, 45, 90, 135]:     # 7 angles
        velocity_sim.input['distance'] = dist
        velocity_sim.input['angle'] = angle
        velocity_sim.compute()
        lookup_table[(dist, angle)] = (velocity_sim.output['linear_velocity'],
                                       velocity_sim.output['angular_velocity'])

# Runtime: Interpolate between cached values (10× faster than full inference)
# Coverage: 8×7 = 56 cached points cover 90% of typical scenarios
```

**2.2 Membership Function Value Caching:**
```python
# Cache MF evaluations for repeated inputs
mf_cache = {}

def cached_mf_eval(mf, x):
    if (mf, x) in mf_cache:
        return mf_cache[(mf, x)]
    value = fuzz.interp_membership(mf.universe, mf.mf, x)
    mf_cache[(mf, x)] = value
    return value

# Speedup: ~30-50% for repeated sensor readings (stable robot position)
# Memory: 1000 cache entries × 8 bytes = 8KB (negligible)
```

**2.3 Temporal Caching (Sensor Stability):**
```python
# Don't recompute if inputs unchanged
prev_inputs = None
prev_outputs = None

def fuzzy_control_with_cache(distance, angle):
    if (distance, angle) == prev_inputs:
        return prev_outputs  # Skip inference entirely

    # Compute fuzzy inference
    outputs = velocity_sim.compute(distance, angle)
    prev_inputs = (distance, angle)
    prev_outputs = outputs
    return outputs

# Speedup: 100% (zero inference time) when robot stationary
# Usefulness: High during GRASPING state (stable position for 2-3s)
```

**Optimization Tier 3: Sugeno Fallback (If >100ms)**

**3.1 Replace Mamdani with Sugeno-Type Inference:**
```python
# Mamdani consequent (fuzzy set)
rule1 = ctrl.Rule(distance['close'] & angle['left'], linear_velocity['slow'])
# Defuzzification required (10-15ms)

# Sugeno consequent (linear function)
rule1_sugeno = ctrl.Rule(distance['close'] & angle['left'],
                        linear_output = 0.1 + 0.05*distance - 0.02*angle)
# No defuzzification (direct output, 2-3ms)

# Speedup: 5-7× faster inference
# Trade-off: Rules harder to design (need to specify coefficients)
```

**Literature Support:**
- **Takagi & Sugeno (1985):** Sugeno-type controllers 5-10× faster than Mamdani
- **Omrane et al. (2016):** Used Mamdani for interpretability, Sugeno for real-time critical systems
- **Industry practice:** Sugeno preferred for embedded systems (automotive ECUs)

**When to Use Sugeno:**
- Inference time >100ms with 35-50 rules (unlikely)
- Embedded deployment (microcontroller with limited CPU)
- Rule consequents can be expressed as linear functions

**Verdict for YouBot:**
- **Phase 3:** Mamdani + Centroid (interpretability priority)
- **Phase 7:** Profile performance, switch to Sugeno if >100ms

**Expected Performance After Optimization:**

| Optimization Level | Inference Time | Implementation Effort |
|-------------------|----------------|----------------------|
| Baseline (35 rules, 100 pts) | 35-40ms | None (default) |
| Tier 1 (50 pts, pruned rules) | 20-25ms | Low (2 hours) |
| Tier 2 (lookup table cache) | 5-10ms (cached), 20ms (miss) | Medium (1 day) |
| Tier 3 (Sugeno inference) | 5-8ms | High (3 days, rule redesign) |

**Target:** Tier 1 sufficient (<25ms < 50ms target)

### 4.2 Rule Evaluation Shortcuts

**Decision: Implement early termination for safety rules**

**Technique 1: Priority-Based Short-Circuit Evaluation**
```python
# Evaluate safety rules first
safety_rules = [rule1, rule2, ..., rule15]  # Obstacle avoidance
task_rules = [rule16, rule17, ..., rule45]   # Cube manipulation

# Check if any safety rule fires above threshold
for rule in safety_rules:
    firing_strength = rule.evaluate(inputs)
    if firing_strength > 0.8:  # High confidence obstacle detected
        # Skip task rules, return emergency output immediately
        return emergency_action(rule.consequent)

# Only evaluate task rules if no high-priority safety rule fired
for rule in task_rules:
    # Normal inference
```

**Speedup:** 40-60% when obstacles nearby (skip 20-30 task rules)
**Safety:** Ensures fastest response to collision threats

**Technique 2: Rule Activation Filtering (Conditional Rules)**
```python
# State-based rule partitioning
if robot_state == 'AVOIDING':
    active_rules = safety_rules  # Only 15 rules
elif robot_state == 'GRASPING':
    active_rules = manipulation_rules  # Only 10 rules
elif robot_state == 'SEARCHING':
    active_rules = exploration_rules  # Only 10 rules

# Evaluate only active subset
velocity_sim.rules = active_rules
velocity_sim.compute()
```

**Speedup:** 60-70% (evaluate 10-15 rules instead of 45)
**Trade-off:** Requires explicit state machine (already planned in Phase 3)

**Technique 3: Fuzzy Input Pre-Filtering**
```python
# Skip fuzzy inference for trivial cases
if distance_to_obstacle < 0.2:  # Emergency zone
    return {'linear_velocity': 0.0, 'angular_velocity': 0.5}  # Hard-coded stop+turn

if distance_to_obstacle > 3.0 AND NOT cube_detected:  # Open space, no task
    return {'linear_velocity': 0.2, 'angular_velocity': 0.0}  # Straight ahead

# Only use fuzzy inference for nuanced scenarios (0.2m < dist < 3.0m)
```

**Speedup:** 100% (zero fuzzy inference) for ~20% of cases
**Risk:** Hard-coded thresholds may miss edge cases → Use conservatively

**Literature Support:**
- **Beom & Cho (1995):** Fuzzy + reinforcement learning with rule filtering
- **Behavior-based architecture (Brooks 1986):** Layer suppression = rule short-circuiting
- **Saffiotti (1997):** Context-dependent rule activation for efficiency

**Implementation (scikit-fuzzy):**
```python
class OptimizedFuzzyController:
    def __init__(self, safety_rules, task_rules, exploration_rules):
        self.safety_system = ctrl.ControlSystem(safety_rules)
        self.task_system = ctrl.ControlSystem(task_rules)
        self.exploration_system = ctrl.ControlSystem(exploration_rules)

    def compute(self, inputs, state):
        # Priority 1: Safety override
        safety_output = self.safety_system.compute(inputs)
        if safety_output['urgency'] > 0.7:  # High urgency
            return safety_output

        # Priority 2: Task execution
        if state in ['GRASPING', 'DEPOSITING']:
            return self.task_system.compute(inputs)

        # Priority 3: Exploration
        return self.exploration_system.compute(inputs)
```

### 4.3 Caching Strategies for Repeated Calculations

**Decision: Implement 2-level cache (MF evaluations + full inference results)**

**Level 1 Cache: Membership Function Evaluations**

**Problem:** Same input values evaluated multiple times across rules
```python
# Example: distance=0.8m evaluated for every rule
Rule 1: IF distance['close'](0.8) AND ... → MF evaluation
Rule 2: IF distance['close'](0.8) AND ... → Same MF evaluation (redundant!)
Rule 3: IF distance['medium'](0.8) AND ... → MF evaluation
```

**Solution: Cache MF(input) results**
```python
from functools import lru_cache

@lru_cache(maxsize=256)  # Cache last 256 evaluations
def cached_mf_eval(mf_name, input_value):
    """
    Cache membership function evaluations
    maxsize=256 covers: 4 inputs × 7 MFs × 10 recent values ≈ 280 entries
    """
    mf = membership_functions[mf_name]
    return fuzz.interp_membership(mf.universe, mf.mf, input_value)

# Usage in rule evaluation
firing_strength = min(
    cached_mf_eval('distance_close', 0.8),    # Cached
    cached_mf_eval('angle_left', -30)         # Cached
)
```

**Speedup:** 30-50% reduction in MF evaluations (typical hit rate: 60-80%)
**Memory:** 256 entries × 16 bytes (key+value) = 4KB

**Level 2 Cache: Full Inference Results (Lookup Table)**

**Problem:** Similar inputs produce similar outputs
```python
# Example: Robot moving slowly → inputs change gradually
t=0: distance=1.0m, angle=-10° → compute fuzzy inference (20ms)
t=1: distance=1.02m, angle=-11° → compute fuzzy inference (20ms) [REDUNDANT!]
```

**Solution: Discretize input space, pre-compute grid**
```python
# Pre-computation phase (one-time, ~5 seconds)
lookup_grid = {}
distance_samples = np.linspace(0.3, 5.0, 20)  # 20 points
angle_samples = np.linspace(-180, 180, 24)    # 24 points (15° resolution)

for dist in distance_samples:
    for angle in angle_samples:
        velocity_sim.input['distance'] = dist
        velocity_sim.input['angle'] = angle
        velocity_sim.compute()
        lookup_grid[(dist, angle)] = {
            'linear_velocity': velocity_sim.output['linear_velocity'],
            'angular_velocity': velocity_sim.output['angular_velocity']
        }

# Runtime: Interpolate nearest grid points
def interpolate_lookup(distance, angle):
    # Find 4 nearest grid points
    d_low, d_high = find_neighbors(distance, distance_samples)
    a_low, a_high = find_neighbors(angle, angle_samples)

    # Bilinear interpolation
    v00 = lookup_grid[(d_low, a_low)]
    v01 = lookup_grid[(d_low, a_high)]
    v10 = lookup_grid[(d_high, a_low)]
    v11 = lookup_grid[(d_high, a_high)]

    return bilinear_interp(v00, v01, v10, v11, distance, angle)
```

**Speedup:** 10-20× faster than full fuzzy inference
- Full inference: 20-40ms
- Bilinear interpolation: 1-2ms

**Accuracy Trade-off:**
- Grid resolution: 20×24 = 480 cached points
- Max interpolation error: ~5% (acceptable for control)
- Dense areas (near obstacles): Add more grid points

**Memory:**
- 480 entries × 2 outputs × 8 bytes = 7.7KB (negligible)

**Implementation (NumPy):**
```python
from scipy.interpolate import RegularGridInterpolator

# Build interpolator (one-time)
linear_vel_grid = np.array([[lookup_grid[(d,a)]['linear_velocity']
                            for a in angle_samples]
                           for d in distance_samples])
interpolator = RegularGridInterpolator(
    (distance_samples, angle_samples),
    linear_vel_grid,
    method='linear'
)

# Runtime query (1-2ms)
output = interpolator((distance, angle))
```

**When to Use Lookup Table:**
- Inference time >50ms → Use lookup as primary method
- Inference time <50ms → Use lookup during high-load states (GRASPING, AVOIDING)

**Literature Support:**
- **Fast Defuzzification Methods (Academia.edu):** Lookup table caching reduces centroid computation by 10×
- **Embedded Fuzzy Controllers (Industry):** Automotive ECUs use 3D lookup tables (distance, angle, velocity)
- **Trade-off accepted:** 5% error acceptable in control (within sensor noise margin)

**Verdict:** Implement Level 1 cache (easy), defer Level 2 to Phase 7 if needed

---

## 5. Testing and Validation

### 5.1 How to Unit Test Fuzzy Rules

**Decision: Multi-layer testing strategy (MF validation → Rule logic → Integration)**

**Layer 1: Membership Function Tests (Unit Tests)**

**Purpose:** Verify MF shapes, ranges, overlaps
```python
# tests/test_fuzzy_membership.py
import pytest
import numpy as np
from src.control.fuzzy_controller import FuzzyController

def test_distance_membership_functions():
    """Test distance MFs cover full range [0, 5] with 50% overlap"""
    fc = FuzzyController()

    # Test 1: Coverage (no gaps)
    for x in np.linspace(0, 5, 100):
        total_membership = sum([
            fc.distance_very_close(x),
            fc.distance_close(x),
            fc.distance_medium(x),
            fc.distance_far(x),
            fc.distance_very_far(x)
        ])
        assert total_membership >= 0.5, f"Gap detected at x={x}"

    # Test 2: Peak locations
    assert fc.distance_very_close(0.3) == 1.0  # Peak at 0.3m
    assert fc.distance_close(0.8) == 1.0       # Peak at 0.8m
    assert fc.distance_medium(1.8) == 1.0      # Peak at 1.8m

    # Test 3: Overlap (50% at crossover points)
    overlap_point = 0.6  # Between VeryClose and Close
    mu_vc = fc.distance_very_close(overlap_point)
    mu_c = fc.distance_close(overlap_point)
    assert abs(mu_vc - mu_c) < 0.1, "Symmetric overlap expected"

def test_angle_membership_symmetry():
    """Test angle MFs are symmetric around zero"""
    fc = FuzzyController()

    # Symmetry test: μ(NB,-90) = μ(PB,+90)
    assert fc.angle_negative_big(-90) == fc.angle_positive_big(90)
    assert fc.angle_negative_medium(-60) == fc.angle_positive_medium(60)
```

**Layer 2: Rule Logic Tests (Unit Tests)**

**Purpose:** Verify individual rules produce expected outputs
```python
# tests/test_fuzzy_rules.py
def test_safety_rule_emergency_stop():
    """Test: IF distance IS VeryClose THEN linear_velocity IS Stop"""
    fc = FuzzyController()

    # Inputs: Obstacle very close, any angle
    fc.input['distance_to_obstacle'] = 0.25  # VeryClose
    fc.input['angle_to_obstacle'] = 0        # Center
    fc.compute()

    # Expected: Stop or very slow (<0.05 m/s)
    assert fc.output['linear_velocity'] < 0.05, "Should stop for VeryClose obstacle"
    assert abs(fc.output['angular_velocity']) > 0.1, "Should turn to avoid"

def test_task_rule_approach_cube():
    """Test: IF cube_detected AND distance_to_cube IS Medium THEN approach"""
    fc = FuzzyController()

    # Inputs: Cube detected at medium distance, no obstacles
    fc.input['cube_detected'] = True
    fc.input['distance_to_cube'] = 1.5  # Medium
    fc.input['angle_to_cube'] = 10      # Slightly right
    fc.input['distance_to_obstacle'] = 5.0  # Far (no obstacle)
    fc.compute()

    # Expected: Medium forward velocity, slight right turn
    assert 0.15 < fc.output['linear_velocity'] < 0.25, "Should approach at medium speed"
    assert 0 < fc.output['angular_velocity'] < 0.2, "Should turn slightly right"

def test_conflicting_rules_safety_override():
    """Test: Safety rule overrides task rule when obstacle close"""
    fc = FuzzyController()

    # Conflicting inputs: Cube close BUT obstacle closer
    fc.input['cube_detected'] = True
    fc.input['distance_to_cube'] = 0.8  # Close (wants to approach)
    fc.input['distance_to_obstacle'] = 0.4  # VeryClose (wants to avoid)
    fc.compute()

    # Expected: Safety overrides task (stop or slow down)
    assert fc.output['linear_velocity'] < 0.1, "Safety should override approach"
```

**Layer 3: Scenario Tests (Integration Tests)**

**Purpose:** Test complete behavioral sequences
```python
# tests/test_fuzzy_scenarios.py
@pytest.mark.parametrize("distance,angle,expected_action", [
    (0.3, 0, 'emergency_turn'),     # Obstacle dead ahead, very close
    (0.8, -45, 'avoid_left'),       # Obstacle close, right side → turn left
    (3.0, 0, 'proceed'),            # Obstacle far, clear path
])
def test_obstacle_avoidance_scenarios(distance, angle, expected_action):
    """Test various obstacle configurations"""
    fc = FuzzyController()
    fc.input['distance_to_obstacle'] = distance
    fc.input['angle_to_obstacle'] = angle
    fc.compute()

    # Assertions based on expected behavior
    if expected_action == 'emergency_turn':
        assert fc.output['linear_velocity'] < 0.05
        assert abs(fc.output['angular_velocity']) > 0.3
    elif expected_action == 'avoid_left':
        assert fc.output['angular_velocity'] < -0.1  # Negative = left
    elif expected_action == 'proceed':
        assert fc.output['linear_velocity'] > 0.15
```

**Layer 4: Coverage Tests (Automated)**

**Purpose:** Ensure all MF combinations tested
```python
# tests/test_fuzzy_coverage.py
def test_rule_base_completeness():
    """Test all critical input combinations have defined outputs"""
    fc = FuzzyController()

    # Grid of critical points
    distances = [0.3, 0.8, 1.5, 3.0, 5.0]  # Representative of each MF
    angles = [-135, -90, -45, 0, 45, 90, 135]

    undefined_count = 0
    for dist in distances:
        for angle in angles:
            fc.input['distance_to_obstacle'] = dist
            fc.input['angle_to_obstacle'] = angle
            try:
                fc.compute()
                # Check output is not NaN or zero (undefined)
                assert not np.isnan(fc.output['linear_velocity'])
                assert not np.isnan(fc.output['angular_velocity'])
            except Exception as e:
                undefined_count += 1
                print(f"Undefined: distance={dist}, angle={angle}")

    # Allow <5% undefined (edge cases acceptable)
    total_tests = len(distances) * len(angles)
    coverage = 1 - (undefined_count / total_tests)
    assert coverage >= 0.95, f"Rule coverage {coverage*100:.1f}% < 95% target"
```

**Literature Support:**
- **NI LabVIEW Fuzzy Toolkit:** "Test relationship between input/output to validate rule base"
- **Graph-based validation:** "If I/O graph displays 0 at some points, rule base incomplete"
- **Robotics Testing (TestRiq 2023):** "Unit test individual functions, functional test subsystems"

### 5.2 Typical Test Scenarios for Robot Navigation

**Decision: 12 standard test scenarios covering common + edge cases**

**Scenario Suite (Based on Literature + Arena Specifics):**

**Category 1: Obstacle Avoidance (Safety-Critical)**

**S1: Frontal Collision Threat**
```python
Inputs:
  distance_to_obstacle = 0.3m (VeryClose)
  angle_to_obstacle = 0° (Dead ahead)
Expected:
  linear_velocity ≈ 0.0 (STOP)
  angular_velocity > 0.3 rad/s (Strong turn)
  action = AVOIDING
Pass Criteria: Robot stops in <0.5s, no collision
```

**S2: Side Obstacle (Left/Right)**
```python
Inputs:
  distance_to_obstacle = 0.8m (Close)
  angle_to_obstacle = ±60° (Side)
Expected:
  linear_velocity < 0.1 (Slow)
  angular_velocity: opposite direction (turn away)
Pass Criteria: Minimum clearance >0.3m maintained
```

**S3: Narrow Passage**
```python
Setup: Two obstacles 0.6m apart (tight for 0.45m wide robot)
Expected:
  Slow linear_velocity (<0.1 m/s)
  Precise angular corrections (|omega| < 0.2)
  OR: Alternative path if passage too narrow
Pass Criteria: Traverse without collision OR avoid passage
```

**S4: Corner Trap (L-shaped obstacle)**
```python
Setup: Robot in corner (obstacles at 0° and 90°)
Expected:
  Reverse linear_velocity (negative) OR rotate in place
  Escape trajectory within 5s
Pass Criteria: Robot exits corner without collision
```

**Category 2: Cube Detection and Approach**

**S5: Cube Directly Ahead (Clear Path)**
```python
Inputs:
  cube_detected = True
  distance_to_cube = 2.0m (Medium)
  angle_to_cube = 0° (Ahead)
  distance_to_obstacle > 3.0m (Clear)
Expected:
  linear_velocity = 0.15-0.25 (Medium forward)
  angular_velocity ≈ 0 (Straight)
  action = APPROACHING
Pass Criteria: Reach cube (<0.3m) in <15s
```

**S6: Cube Off-Axis (Alignment Required)**
```python
Inputs:
  cube_detected = True
  distance_to_cube = 1.5m
  angle_to_cube = ±45° (Oblique)
Expected:
  angular_velocity: correct bearing (proportional to angle)
  linear_velocity: moderate (0.1-0.2)
Pass Criteria: Align within ±10° in <5s
```

**S7: Cube Behind Obstacle (Occlusion)**
```python
Setup: Cube at 2m, obstacle at 1m (same bearing)
Expected:
  Detour path around obstacle
  Re-acquire cube after clearing obstacle
  OR: Search for different cube
Pass Criteria: Navigate around OR abandon (no collision)
```

**Category 3: Multi-Objective Conflicts**

**S8: Cube Close BUT Obstacle Closer**
```python
Inputs:
  cube_detected = True, distance_to_cube = 0.8m
  distance_to_obstacle = 0.4m (VeryClose, higher priority)
Expected:
  Safety override: Avoid obstacle first
  linear_velocity < 0.05 (Stop/Slow)
Pass Criteria: Obstacle avoidance prioritized (no collision)
```

**S9: Holding Cube + Obstacle En Route to Box**
```python
Inputs:
  holding_cube = True
  target_box at 5m distance
  obstacle appears at 1.2m (en route)
Expected:
  Detour around obstacle without dropping cube
  Resume path to box after clearing
Pass Criteria: Reach box with cube still held
```

**Category 4: Edge Cases**

**S10: No Cubes Detected (Exploration)**
```python
Inputs:
  cube_detected = False
  distance_to_obstacle > 2.0m (Open space)
Expected:
  action = SEARCHING
  Scanning pattern (rotate + forward)
  linear_velocity = 0.1-0.2
Pass Criteria: Cover 50% of arena in 60s without collision
```

**S11: Multiple Cubes Visible (Selection)**
```python
Setup: 3 cubes at bearings -30°, 0°, +30° (distances 1.5m, 2.0m, 1.8m)
Expected:
  Select closest cube (1.5m at -30°)
  Ignore others until first cube collected
Pass Criteria: Approach closest cube consistently
```

**S12: Arena Boundary (Wall Following)**
```python
Setup: Robot near wall (distance 0.5m, parallel)
Expected:
  Maintain safe distance (0.4-0.6m)
  Follow wall while scanning for cubes
Pass Criteria: No wall collision, smooth following
```

**Test Execution Plan:**
```python
# tests/test_navigation_scenarios.py
@pytest.mark.parametrize("scenario", [
    'S1_frontal_collision', 'S2_side_obstacle', 'S3_narrow_passage',
    'S4_corner_trap', 'S5_cube_ahead', 'S6_cube_offaxis',
    'S7_cube_occluded', 'S8_conflict', 'S9_holding_obstacle',
    'S10_exploration', 'S11_multiple_cubes', 'S12_wall_following'
])
def test_scenario(scenario, webots_sim):
    """Execute predefined scenario in Webots simulation"""
    setup_scenario(webots_sim, scenario)
    robot = webots_sim.get_robot()
    fuzzy_controller = FuzzyController()

    # Run simulation for max 60s
    start_time = robot.getTime()
    while robot.step(TIME_STEP) != -1:
        elapsed = robot.getTime() - start_time
        if elapsed > 60:
            break

        # Get sensor inputs
        inputs = get_fuzzy_inputs(robot)
        fuzzy_controller.compute(inputs)

        # Apply outputs
        apply_control(robot, fuzzy_controller.output)

        # Check pass criteria
        if check_pass_criteria(scenario, robot):
            print(f"✅ {scenario} PASSED in {elapsed:.1f}s")
            return

    pytest.fail(f"❌ {scenario} FAILED (timeout or criteria not met)")
```

**Benchmark Targets (From Literature):**
- **Omrane et al. (2016):** >95% obstacle avoidance success rate
- **Faisal et al. (2013):** <5% tracking error for target approach
- **Khairudin et al. (2020):** Zero collisions in 50 test runs

### 5.3 Membership Function Visualization Best Practices

**Decision: Use matplotlib + scikit-fuzzy plotting utilities with 3 visualization layers**

**Layer 1: Individual Membership Functions (Design Phase)**

**Purpose:** Verify MF shapes, overlaps, coverage
```python
# scripts/visualize_membership_functions.py
import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz

def plot_distance_mfs():
    """Visualize distance membership functions"""
    x_distance = np.linspace(0, 5, 500)

    # Define MFs
    mf_very_close = fuzz.trimf(x_distance, [0.0, 0.3, 0.6])
    mf_close = fuzz.trimf(x_distance, [0.4, 0.8, 1.2])
    mf_medium = fuzz.trimf(x_distance, [1.0, 1.8, 2.6])
    mf_far = fuzz.trimf(x_distance, [2.2, 3.5, 4.3])
    mf_very_far = fuzz.trapmf(x_distance, [4.0, 5.0, 5.0, 5.0])

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_distance, mf_very_close, 'r', linewidth=2, label='Very Close')
    ax.plot(x_distance, mf_close, 'orange', linewidth=2, label='Close')
    ax.plot(x_distance, mf_medium, 'y', linewidth=2, label='Medium')
    ax.plot(x_distance, mf_far, 'lightgreen', linewidth=2, label='Far')
    ax.plot(x_distance, mf_very_far, 'g', linewidth=2, label='Very Far')

    # Styling
    ax.set_title('Distance Membership Functions (LIDAR: 0.3-5.0m)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Distance (meters)', fontsize=12)
    ax.set_ylabel('Membership Degree μ(x)', fontsize=12)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-0.05, 1.05])

    # Annotate overlap regions
    ax.axvspan(0.4, 0.6, alpha=0.2, color='red', label='VC-C overlap')
    ax.axvspan(0.8, 1.2, alpha=0.2, color='orange', label='C-M overlap')

    plt.tight_layout()
    plt.savefig('docs/fuzzy_distance_mfs.png', dpi=300)
    plt.show()

def plot_angle_mfs():
    """Visualize angle membership functions"""
    x_angle = np.linspace(-180, 180, 720)

    # Define 7 MFs (NB, NM, NS, Z, PS, PM, PB)
    mf_nb = fuzz.trapmf(x_angle, [-180, -180, -90, -45])
    mf_nm = fuzz.trimf(x_angle, [-90, -60, -30])
    mf_ns = fuzz.trimf(x_angle, [-45, -15, 0])
    mf_z = fuzz.trimf(x_angle, [-15, 0, 15])
    mf_ps = fuzz.trimf(x_angle, [0, 15, 45])
    mf_pm = fuzz.trimf(x_angle, [30, 60, 90])
    mf_pb = fuzz.trapmf(x_angle, [45, 90, 180, 180])

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x_angle, mf_nb, linewidth=2, label='Negative Big')
    ax.plot(x_angle, mf_nm, linewidth=2, label='Negative Medium')
    ax.plot(x_angle, mf_ns, linewidth=2, label='Negative Small')
    ax.plot(x_angle, mf_z, linewidth=2, label='Zero')
    ax.plot(x_angle, mf_ps, linewidth=2, label='Positive Small')
    ax.plot(x_angle, mf_pm, linewidth=2, label='Positive Medium')
    ax.plot(x_angle, mf_pb, linewidth=2, label='Positive Big')

    ax.set_title('Angle Membership Functions (Bearing: ±180°)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Angle (degrees)', fontsize=12)
    ax.set_ylabel('Membership Degree μ(x)', fontsize=12)
    ax.legend(loc='upper right', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.axvline(0, color='black', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('docs/fuzzy_angle_mfs.png', dpi=300)
    plt.show()
```

**Output:** High-resolution PNG for presentation slides (DECISIONS.md requires visual material)

**Layer 2: Control Surface Visualization (Rule Base Validation)**

**Purpose:** Visualize input-output relationship (2D heatmap or 3D surface)
```python
def plot_control_surface():
    """3D surface plot of fuzzy control output"""
    # Create fuzzy controller
    fc = FuzzyController()

    # Grid of inputs
    distances = np.linspace(0.3, 5.0, 50)
    angles = np.linspace(-180, 180, 50)
    D, A = np.meshgrid(distances, angles)

    # Compute outputs for each grid point
    linear_vel = np.zeros_like(D)
    for i in range(len(distances)):
        for j in range(len(angles)):
            fc.input['distance_to_obstacle'] = distances[i]
            fc.input['angle_to_obstacle'] = angles[j]
            fc.compute()
            linear_vel[j, i] = fc.output['linear_velocity']

    # 3D Surface Plot
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(D, A, linear_vel, cmap='viridis', alpha=0.8)

    ax.set_xlabel('Distance to Obstacle (m)', fontsize=12)
    ax.set_ylabel('Angle to Obstacle (°)', fontsize=12)
    ax.set_zlabel('Linear Velocity (m/s)', fontsize=12)
    ax.set_title('Fuzzy Control Surface: Obstacle Avoidance', fontsize=14, fontweight='bold')
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.savefig('docs/fuzzy_control_surface.png', dpi=300)
    plt.show()

    # Alternative: 2D Heatmap (easier to read)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(linear_vel, extent=[0.3, 5.0, -180, 180],
                   aspect='auto', origin='lower', cmap='RdYlGn')
    ax.set_xlabel('Distance to Obstacle (m)', fontsize=12)
    ax.set_ylabel('Angle to Obstacle (°)', fontsize=12)
    ax.set_title('Linear Velocity Heatmap (Obstacle Avoidance)', fontsize=14, fontweight='bold')
    fig.colorbar(im, label='Linear Velocity (m/s)')
    plt.tight_layout()
    plt.savefig('docs/fuzzy_heatmap.png', dpi=300)
    plt.show()
```

**Interpretation:**
- **Smooth surface:** Good rule coverage, no discontinuities
- **Flat regions:** Saturation (e.g., always STOP when very close)
- **Sharp transitions:** Potential chattering → increase MF overlap

**Layer 3: Real-Time Monitoring (Runtime Debugging)**

**Purpose:** Visualize active rules and firing strengths during execution
```python
def plot_realtime_fuzzy_state(robot, fuzzy_controller):
    """Live visualization of fuzzy inference (for debugging)"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Subplot 1: Input MF activations
    ax1 = axes[0, 0]
    distance_val = robot.get_lidar_min_distance()
    ax1.bar(['VC', 'C', 'M', 'F', 'VF'],
           [fuzzy_controller.distance_very_close(distance_val),
            fuzzy_controller.distance_close(distance_val),
            fuzzy_controller.distance_medium(distance_val),
            fuzzy_controller.distance_far(distance_val),
            fuzzy_controller.distance_very_far(distance_val)])
    ax1.set_title(f'Distance MF Activations (input={distance_val:.2f}m)')
    ax1.set_ylim([0, 1])

    # Subplot 2: Rule firing strengths
    ax2 = axes[0, 1]
    rule_strengths = fuzzy_controller.get_rule_firing_strengths()
    ax2.barh(range(len(rule_strengths)), rule_strengths)
    ax2.set_yticks(range(len(rule_strengths)))
    ax2.set_yticklabels([f'R{i+1}' for i in range(len(rule_strengths))])
    ax2.set_title('Rule Firing Strengths')
    ax2.set_xlim([0, 1])

    # Subplot 3: Output MF aggregation
    ax3 = axes[1, 0]
    x_vel = np.linspace(0, 0.3, 300)
    aggregated_mf = fuzzy_controller.get_aggregated_output('linear_velocity')
    ax3.fill_between(x_vel, 0, aggregated_mf, alpha=0.5)
    defuzz_val = fuzzy_controller.output['linear_velocity']
    ax3.axvline(defuzz_val, color='red', linewidth=2, label=f'Defuzzified: {defuzz_val:.3f}')
    ax3.set_title('Output Aggregation (Linear Velocity)')
    ax3.legend()

    # Subplot 4: Robot trajectory (top-down view)
    ax4 = axes[1, 1]
    trajectory = robot.get_trajectory_history()
    ax4.plot([p[0] for p in trajectory], [p[1] for p in trajectory], 'b-')
    ax4.scatter(trajectory[-1][0], trajectory[-1][1], color='red', s=100, label='Current')
    ax4.set_title('Robot Trajectory')
    ax4.legend()
    ax4.axis('equal')

    plt.tight_layout()
    plt.pause(0.01)  # Update plot in real-time
```

**Use Case:** Run during Phase 6 integration testing to debug unexpected behavior

**Best Practices Summary:**

1. **Use consistent color schemes:**
   - Red → Danger (Very Close, Stop)
   - Orange/Yellow → Caution (Close, Slow)
   - Green → Safe (Far, Fast)

2. **Annotate key points:**
   - MF peaks (e.g., "Peak at 0.8m")
   - Overlap regions (shaded areas)
   - Crossover points (50% membership)

3. **Save high-resolution (300 DPI):**
   - For Phase 8 presentation slides
   - DECISIONS.md guideline: visual > text

4. **Interactive plots (optional):**
   - Jupyter notebook: Use `%matplotlib widget` for zoom/pan
   - Adjust MF parameters interactively (sliders)

**Literature Support:**
- **MATLAB Fuzzy Toolbox:** Standard visualization includes MF plots + control surface
- **scikit-fuzzy examples:** `plot_defuzzify.html` shows best practices
- **Ross (2010):** "Visualization crucial for rule base validation and tuning"

---

## 6. Summary and Recommendations

### 6.1 Final Design Decisions

**Membership Functions:**
- **Shape:** Triangular (baseline), Gaussian (if accuracy <90%)
- **Overlap:** 50% (adjust 30-70% for smoothness tuning)
- **Count:** 5 MFs (distance), 7 MFs (angle), 5 MFs (velocity)
- **Ranges:** Distance [0.3, 5.0]m, Angle [-180, 180]°, Velocity [0, 0.3]m/s

**Rule Base:**
- **Total Rules:** 35-50 (15 safety + 20-25 task + 5-10 exploration)
- **Prioritization:** Safety (w=10) > Task (w=5) > Exploration (w=1)
- **Coverage:** 100% for safety scenarios, 80-90% for task scenarios
- **Conflict Resolution:** Max-Min inference + weighted aggregation

**Defuzzification:**
- **Method:** Centroid (primary), MOM (discrete actions only)
- **Target Latency:** <50ms inference time
- **Optimization:** Tier 1 (reduce resolution, prune rules) if >50ms

**Testing:**
- **Unit Tests:** MF validation, rule logic, scenario coverage
- **Scenarios:** 12 standard tests (S1-S12) covering safety + tasks + edge cases
- **Visualization:** 3-layer (MFs, control surface, real-time monitoring)

### 6.2 Implementation Roadmap (Phase 3)

**Week 1: Days 1-3 (Design)**
- [ ] Define membership functions (distance, angle, velocity)
- [ ] Plot and validate MFs (overlap, coverage)
- [ ] Design 35-50 rules (document in `fuzzy_rules.txt`)
- [ ] Document in DECISIONS.md (DECISÃO XXX)

**Week 1: Days 4-5 (Implementation)**
- [ ] Implement FuzzyController class (scikit-fuzzy)
- [ ] Implement Mamdani inference + Centroid defuzzification
- [ ] Integrate with state machine (SEARCHING, APPROACHING, GRASPING, etc.)

**Week 2: Days 6-7 (Testing)**
- [ ] Write unit tests (MF tests, rule tests)
- [ ] Run 12 scenario tests in Webots simulation
- [ ] Visualize control surfaces (save PNGs for presentation)
- [ ] Profile inference time (target <50ms)

**Deliverable:** Fuzzy controller operational with >90% obstacle avoidance success rate

### 6.3 References Cited

**Primary References (Top 10):**
1. **Zadeh (1965):** Fuzzy Sets - Foundation of fuzzy logic theory
2. **Mamdani & Assilian (1975):** Fuzzy Logic Controller - Original Mamdani inference
3. **Saffiotti (1997):** Fuzzy Navigation - Mobile robot applications
4. **Omrane et al. (2016):** Autonomous Mobile Robot Navigation - 35 rules, 5/7 MFs
5. **Ross (2010):** Fuzzy Logic with Engineering Applications - Design guidelines

**Additional References:**
6. Faisal et al. (2013): Fuzzy Navigation + Obstacle Avoidance (62 rules, Sugeno)
7. Khairudin et al. (2020): Mobile Robot Obstacle Avoidance (25 rules, safety focus)
8. Huang et al. (2010): Ordinal Structure Fuzzy Logic (rule reduction techniques)
9. Takagi & Sugeno (1985): Sugeno-type fuzzy inference (performance optimization)
10. Beom & Cho (1995): Fuzzy + Reinforcement Learning (sensor fusion)
11. Brooks (1986): Subsumption Architecture (behavior prioritization)
12. MATLAB Fuzzy Toolbox Documentation (2024): Defuzzification methods comparison
13. TestRiq (2023): Robotic Software Testing (unit + functional test strategies)

**Web Resources:**
- IntechOpen (2020): Membership Function Design Best Practices
- NI LabVIEW (2024): Testing Fuzzy Systems Tutorial
- IEEE Xplore: Multiple papers on rule base optimization

### 6.4 Next Steps

**Immediate Actions (Phase 3):**
1. Create `src/control/fuzzy_controller.py` skeleton
2. Define MFs in code (triangular, 50% overlap)
3. Draft initial 35 rules in `src/control/fuzzy_rules.txt`
4. Update DECISIONS.md with DECISÃO XXX referencing this research

**Integration with Other Phases:**
- **Phase 2 (Perception):** Fuzzy inputs from LIDAR/Camera outputs
- **Phase 4 (Navigation):** Fuzzy controller drives base velocity commands
- **Phase 5 (Manipulation):** Fuzzy logic for grasp approach decisions
- **Phase 7 (Optimization):** Fine-tune MFs and rules based on performance

**Success Criteria (Phase 3):**
- [ ] Fuzzy controller implemented (35-50 rules)
- [ ] Unit tests passing (>90% coverage)
- [ ] Scenario tests passing (10/12 scenarios)
- [ ] Inference time <50ms
- [ ] Obstacle avoidance >90% success rate
- [ ] Documented in DECISIONS.md with visualizations

---

**Document Status:** ✅ COMPLETE - Ready for DECISÃO XXX integration
**Last Updated:** 2025-11-21
**Next Review:** After Phase 3 implementation (validation of predictions)
