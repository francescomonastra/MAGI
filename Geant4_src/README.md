# Geant4 Primary Generator with MAGI Integration

This folder holds the three source files that were added or modified in a Geant4
application to consume MAGI as a particle source: a custom `PrimaryGeneratorAction`
and `PrimaryGeneratorMessenger` (the integration itself), and `EventAction` (a
sphere-centre fix needed to keep the crossing surface geometrically consistent
between real Geant4 primaries and MAGI-injected ones). They are not a runnable
example on their own — drop them into your own Geant4 application's `src/`/`include/`
alongside your existing `DetectorConstruction`, `PhysicsList`, etc., and register the
messenger the same way `PrimaryGeneratorMessenger` is used here.

This Geant4 version supports three primary particle generation modes:

1. **Standard Geant4 GPS mode** (General Particle Source)
2. **Static generated-particle file mode** (load a precomputed MAGI file)
3. **Automatic MAGI generation mode** (Geant4 calls Python to generate the input file on demand)

This extension is implemented through a custom `PrimaryGeneratorAction` and `PrimaryGeneratorMessenger`.

---

# 1. Standard GPS Mode

This is the default Geant4 behaviour.

Example:

```tcl
/control/execute gpsRadium228.mac
/run/beamOn 100000
```

In this mode:

- `G4GeneralParticleSource` behaves normally
- particles are generated from macro commands
- MAGI integration is inactive

---

# 2. Static MAGI Input File Mode

In this mode, Geant4 reads particles from a text file.

Each line must contain:

```text
ParticleName Energy X Y Z Vx Vy Vz
```

Example:

```text
gamma 0.131632939 -35.3167992 -93.4576950 -503.372406 0.676084936 0.689519465 -0.259753942
e-    0.542123100  12.3456000  45.1234000 -510.332100 -0.200000000 0.300000000 -0.932000000
```

## Units

- Energy → MeV
- Position → mm
- Direction → normalized Cartesian unit vector

---

## Macro usage

Example:

```tcl
/random/setSeeds 175 2008

/generator/generatedFile /path/to/generated_particles.txt

/runAction/setTotalEvents 1000000
/run/beamOn 1000000
```

### Important

`/generator/generatedFile` automatically:

- loads the file
- validates the content
- activates generated-particle mode

Therefore this command is usually unnecessary:

```tcl
/generator/useGeneratedFile true
```

---

# 3. Automatic MAGI Generation Mode

In this mode, Geant4 automatically calls a Python script to generate the particle input file before loading it.

This is useful for:

- large Monte Carlo campaigns
- parallel runs
- per-job randomized particle generation
- avoiding manual file pre-generation

---

## Required Python environment

Example:

```bash
/path/to/your/venv/bin/python
```

The environment must contain:

- TensorFlow
- NumPy
- the MAGI package
- trained model files

---

## Required model files

Example structure (using the `v0_8_2_DM1_2_500k` model shipped in this repo's
`trained_models/`):

```text
trained_models/v0_8_2_DM1_2_500k/
    mix_DM1_2_500k.weights.h5
    mix_DM1_2_500k_metadata.json
    mix_DM1_2_500k_task_weights.json
```

---

## Required generation script

Example:

```text
MAGI/scripts/generate_geant_source.py
```

---

## Macro usage

Example:

```tcl
/random/setSeeds 175 2008

/generator/autoGenerateMLInput true

/generator/mlPython /path/to/your/venv/bin/python

/generator/mlScript /path/to/MAGI/scripts/generate_geant_source.py

/generator/mlModelDir /path/to/MAGI/trained_models/v0_8_2_DM1_2_500k

/generator/mlModelName mix_DM1_2_500k

/generator/mlMetadataFile /path/to/MAGI/trained_models/v0_8_2_DM1_2_500k/mix_DM1_2_500k_metadata.json

/generator/mlNumParticles 1000000

/generator/mlSeed 1752008

/generator/mlChunkSize 100000

/generator/generatedFile /tmp/magi_input_seed_1752008.txt

/runAction/setTotalEvents 1000000
/run/beamOn 1000000
```

---

## Execution logic

When:

```tcl
/generator/generatedFile ...
```

is executed:

1. if auto-generation is enabled, the Python generator is launched
2. the output particle file is created
3. Geant4 loads the generated file
4. primary generation switches to file-driven mode

---

# Command Reference

---

## Static/generated file mode

### `/generator/generatedFile`

Load a generated particle file.

Example:

```tcl
/generator/generatedFile /path/to/file.txt
```

Effect:

- loads the file
- validates particle content
- activates generated mode

---

### `/generator/useGeneratedFile`

Manually enable or disable generated-file mode.

Example:

```tcl
/generator/useGeneratedFile true
```

Usually not required if `/generator/generatedFile` is used.

---

# Automatic ML generation

---

### `/generator/autoGenerateMLInput`

Enable automatic Python-based generation.

Example:

```tcl
/generator/autoGenerateMLInput true
```

---

### `/generator/mlPython`

Python executable used for generation.

Example:

```tcl
/generator/mlPython /path/to/your/venv/bin/python
```

---

### `/generator/mlScript`

Path to the generation script.

Example:

```tcl
/generator/mlScript /path/to/generate_geant_source.py
```

---

### `/generator/mlModelDir`

Directory containing the trained model.

Example:

```tcl
/generator/mlModelDir /path/to/trained_models/run005
```

---

### `/generator/mlModelName`

Model name.

Example:

```tcl
/generator/mlModelName task_adaptive_cvae_energy
```

---

### `/generator/mlMetadataFile`

Metadata JSON file.

Example:

```tcl
/generator/mlMetadataFile /path/to/task_adaptive_cvae_energy_metadata.json
```

---

### `/generator/mlNumParticles`

Number of particles to generate.

Example:

```tcl
/generator/mlNumParticles 1000000
```

---

### `/generator/mlSeed`

Random seed passed to the Python generator.

Example:

```tcl
/generator/mlSeed 1752008
```

---

### `/generator/mlChunkSize`

Chunk size used during generation.

Example:

```tcl
/generator/mlChunkSize 100000
```

---

# Parallel Usage

Recommended configuration for parallel jobs:

```tcl
/random/setSeeds 12345 67890

/generator/mlSeed 123456789

/generator/generatedFile /tmp/magi_job_001.txt
```

Each parallel job should use:

- unique Geant4 seed
- unique ML seed
- unique output filename

This avoids collisions and duplicated particle streams.

---

# Troubleshooting

---

## "No valid generated particles loaded"

Possible causes:

- malformed file
- incorrect number of columns
- unsupported particle names
- empty file

Expected format:

```text
ParticleName Energy X Y Z Vx Vy Vz
```

---

## "Unknown particle name"

Particle names must match Geant4 particle definitions.

Examples:

Valid:

```text
gamma
e-
proton
alpha
mu-
mu+
```

---

## Python generation fails

Check:

- Python executable path
- TensorFlow installation
- MAGI package availability
- trained model files exist
- metadata JSON exists

Manual test:

```bash
python generate_geant_source.py ...
```

---

## File generated in /tmp is not visible

On macOS:

```bash
/private/tmp
```

Open in Finder:

```bash
open /private/tmp
```

---

## Geant4 cannot open generated file

Check:

- absolute path correctness
- file permissions
- path encoding (avoid special characters if needed)

---

# Performance Notes

Generated particles are currently fully loaded into memory.

Approximate memory scaling:

- 10^6 particles → manageable
- 10^7 particles → large but feasible
- 10^8 particles → potentially heavy memory usage

For large campaigns, automatic generation with chunking is preferred.

---

# Recommended Workflow

## Manual testing

Use static generated files:

```tcl
/generator/generatedFile /path/to/test_particles.txt
```

---

## Production / parallel campaigns

Use automatic generation:

```tcl
/generator/autoGenerateMLInput true
```

with per-job:

- unique seed
- unique output file

This ensures reproducible and independent particle populations.