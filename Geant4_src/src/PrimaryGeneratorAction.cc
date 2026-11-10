#include "PrimaryGeneratorAction.hh"
#include "PrimaryGeneratorMessenger.hh"
#include "PrimaryTruthInfo.hh"
#include "G4RunManager.hh"

#include "G4SingleParticleSource.hh"
#include "G4SPSEneDistribution.hh"
#include "G4SPSPosDistribution.hh"
#include "G4SPSAngDistribution.hh"
#include "G4ios.hh"

#include "GDMLDetectorConstruction.hh"
#include "G4Event.hh"
#include "G4GeneralParticleSource.hh"
#include "G4ThreeVector.hh"
#include "G4ParticleTable.hh"
#include "G4ParticleDefinition.hh"
#include "Randomize.hh"
#include "globals.hh"
#include "G4NistManager.hh"
#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"
#include "G4LogicalVolumeStore.hh"
#include "G4Box.hh"
#include "G4Tubs.hh"

#include <stdio.h>
#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>
#include <algorithm>
#include <sstream>
#include <cstdlib>
#include <cstdint>

#define PI 3.14159265

using namespace std;

struct BinaryParticleRecord {
  int32_t pdg;
  float energy;
  float x;
  float y;
  float z;
  float vx;
  float vy;
  float vz;
};

// Version 2 record (52 bytes, packed): primary truth first, then the v1 fields.
// The truth travels inside the record, so a record skipped for an unknown PDG or
// a null direction cannot shift it onto another particle.
#pragma pack(push, 1)
struct BinaryParticleRecordV2 {
  int64_t groupKey;
  int32_t primTrackId;
  int32_t primPDG;
  float primEnergy;
  BinaryParticleRecord particle;
};
#pragma pack(pop)

PrimaryGeneratorAction::PrimaryGeneratorAction()
: particleGun(nullptr),
  fMessenger(nullptr),
  fUseGeneratedFile(false),
  fGeneratedFileName(""),
  fGeneratedInputFormat(kGeneratedText),
  fCurrentGeneratedIndex(0),
  fBinaryTotalParticles(0),
  fBinaryParticlesRead(0),
  fBinaryDataStart(0),
  fBinaryBufferIndex(0),
  fBinaryBufferSize(100000),
  fBinaryVersion(1),
  fGroupMode(false),
  fGeneratedExhausted(false),
  fAutoGenerateMLInput(false),
  fMLPython("python"),
  fMLScript(""),
  fMLModelDir(""),
  fMLModelName(""),
  fMLMetadataFile(""),
  fMLNumParticles(0),
  fMLSeed(42),
  fMLChunkSize(100000)
{
  particleGun = new G4GeneralParticleSource();
  fMessenger = new PrimaryGeneratorMessenger(this);
}

PrimaryGeneratorAction::~PrimaryGeneratorAction()
{
  if (fBinaryInputStream.is_open()) {
    fBinaryInputStream.close();
  }

  delete fMessenger;
  delete particleGun;
}

void PrimaryGeneratorAction::SetUseGeneratedFile(G4bool value)
{
  if (value && fGeneratedParticles.empty()) {
    G4cerr << "Warning: /generator/useGeneratedFile true called, "
           << "but no generated particles are loaded yet. "
           << "Use /generator/generatedFile <path> first."
           << G4endl;
  }

  fUseGeneratedFile = value;
}

void PrimaryGeneratorAction::SetGeneratedInputFormat(const G4String& format)
{
  if (format == "text" || format == "txt") {
    fGeneratedInputFormat = kGeneratedText;
    G4cout << "[MAGI] Generated input format set to text." << G4endl;
  }
  else if (format == "binary" || format == "bin") {
    fGeneratedInputFormat = kGeneratedBinary;
    G4cout << "[MAGI] Generated input format set to binary." << G4endl;
  }
  else {
    G4Exception(
      "PrimaryGeneratorAction::SetGeneratedInputFormat",
      "InvalidGeneratedInputFormat",
      FatalException,
      "Valid formats are: text, txt, binary, bin."
    );
  }
}

void PrimaryGeneratorAction::SetGeneratedFileName(const G4String& filename)
{
  fGeneratedFileName = filename;

  if (fAutoGenerateMLInput) {
    RunMLGenerator();
  }

  if (fGeneratedInputFormat == kGeneratedText) {
    LoadGeneratedParticleFile(filename);
  }
  else if (fGeneratedInputFormat == kGeneratedBinary) {
    OpenGeneratedParticleBinaryFile(filename);
  }

  fUseGeneratedFile = true;
}

void PrimaryGeneratorAction::SetBinaryBufferSize(G4int value)
{
  if (value <= 0) {
    G4Exception(
      "PrimaryGeneratorAction::SetBinaryBufferSize",
      "InvalidBinaryBufferSize",
      FatalException,
      "Binary buffer size must be > 0."
    );
  }

  fBinaryBufferSize = static_cast<std::size_t>(value);
}

void PrimaryGeneratorAction::LoadGeneratedParticleFile(const G4String& filename)
{
  std::ifstream infile(filename);

  if (!infile.is_open()) {
    G4Exception(
      "PrimaryGeneratorAction::LoadGeneratedParticleFile",
      "GeneratedInputFileNotFound",
      FatalException,
      ("Cannot open generated particle file: " + filename).c_str()
    );
  }

  fGeneratedParticles.clear();
  fCurrentGeneratedIndex = 0;
  fGroupMode = false;
  fGeneratedExhausted = false;

  std::string line;
  std::size_t lineNumber = 0;

  while (std::getline(infile, line)) {
    lineNumber++;

    if (line.empty()) continue;
    if (line[0] == '#') continue;

    std::istringstream iss(line);

    std::string particleName;
    G4double E, x, y, z, vx, vy, vz;

    if (!(iss >> particleName >> E >> x >> y >> z >> vx >> vy >> vz)) {
      G4cerr << "Skipping malformed generated particle line "
             << lineNumber << G4endl;
      continue;
    }

    G4ThreeVector dir(vx, vy, vz);

    if (dir.mag2() <= 0.0) {
      G4cerr << "Skipping generated particle line "
             << lineNumber << " because direction is null." << G4endl;
      continue;
    }

    GeneratedParticle p;
    p.particleName = particleName;
    p.energy = E * MeV;
    p.position = G4ThreeVector(x * mm, y * mm, z * mm);
    p.direction = dir.unit();

    // Optional primary-truth columns: GroupKey PrimaryTrackId PrimaryPDG PrimaryEnergy[MeV]
    long long groupKey;
    G4int primTrackId, primPDG;
    G4double primE;
    if (iss >> groupKey >> primTrackId >> primPDG >> primE) {
      p.groupKey = static_cast<G4long>(groupKey);
      p.primTrackId = primTrackId;
      p.primPDG = primPDG;
      p.primEnergy = primE * MeV;
    }

    fGeneratedParticles.push_back(p);
  }

  if (fGeneratedParticles.empty()) {
    G4Exception(
      "PrimaryGeneratorAction::LoadGeneratedParticleFile",
      "EmptyGeneratedInput",
      FatalException,
      "No valid generated particles loaded."
    );
  }

  G4cout << "Loaded " << fGeneratedParticles.size()
         << " text generated particles from " << filename << G4endl;
}

void PrimaryGeneratorAction::OpenGeneratedParticleBinaryFile(const G4String& filename)
{
  if (fBinaryInputStream.is_open()) {
    fBinaryInputStream.close();
  }

  fBinaryInputStream.open(filename, std::ios::binary);

  if (!fBinaryInputStream.is_open()) {
    G4Exception(
      "PrimaryGeneratorAction::OpenGeneratedParticleBinaryFile",
      "GeneratedBinaryInputFileNotFound",
      FatalException,
      ("Cannot open binary generated particle file: " + filename).c_str()
    );
  }

  fGeneratedParticles.clear();
  fCurrentGeneratedIndex = 0;

  fBinaryBuffer.clear();
  fBinaryBufferIndex = 0;
  fBinaryParticlesRead = 0;
  fGroupMode = false;
  fGeneratedExhausted = false;

  char magic[8];
  int32_t version;
  uint64_t nParticles;

  fBinaryInputStream.read(magic, 8);
  fBinaryInputStream.read(reinterpret_cast<char*>(&version), sizeof(version));
  fBinaryInputStream.read(reinterpret_cast<char*>(&nParticles), sizeof(nParticles));

  if (!fBinaryInputStream.good()) {
    G4Exception(
      "PrimaryGeneratorAction::OpenGeneratedParticleBinaryFile",
      "InvalidBinaryHeader",
      FatalException,
      "Could not read MAGI binary header."
    );
  }

  if (std::string(magic, 7) != "GNTBIN1") {
    G4Exception(
      "PrimaryGeneratorAction::OpenGeneratedParticleBinaryFile",
      "InvalidBinaryMagic",
      FatalException,
      "Invalid MAGI binary file magic. Expected GNTBIN1."
    );
  }

  // v1: 32-byte particle records. v2: 52-byte records with primary truth.
  if (version != 1 && version != 2) {
    G4Exception(
      "PrimaryGeneratorAction::OpenGeneratedParticleBinaryFile",
      "UnsupportedBinaryVersion",
      FatalException,
      "Unsupported MAGI binary file version."
    );
  }

  fBinaryVersion = version;
  fBinaryTotalParticles = nParticles;
  fBinaryDataStart = fBinaryInputStream.tellg();

  FillBinaryBuffer();

  if (fBinaryBuffer.empty()) {
    G4Exception(
      "PrimaryGeneratorAction::OpenGeneratedParticleBinaryFile",
      "EmptyGeneratedBinaryInput",
      FatalException,
      "No valid generated particles loaded into binary buffer."
    );
  }

  G4cout << "Opened binary generated particle file: " << filename << G4endl;
  G4cout << "Binary file version: " << fBinaryVersion << G4endl;
  G4cout << "Total binary particles: " << fBinaryTotalParticles << G4endl;
  G4cout << "Binary buffer size: " << fBinaryBufferSize << G4endl;
}

void PrimaryGeneratorAction::FillBinaryBuffer()
{
  fBinaryBuffer.clear();
  fBinaryBufferIndex = 0;

  if (!fBinaryInputStream.is_open()) {
    G4Exception(
      "PrimaryGeneratorAction::FillBinaryBuffer",
      "BinaryInputStreamNotOpen",
      FatalException,
      "Binary input stream is not open."
    );
  }

  if (fBinaryParticlesRead >= fBinaryTotalParticles) {
    // Same recycling as the text path: rewinding to the first record replays
    // primaries that have already been simulated. Warn once.
    static G4bool binaryRecycleWarned = false;
    if (!binaryRecycleWarned) {
      binaryRecycleWarned = true;
      G4Exception(
        "PrimaryGeneratorAction::FillBinaryBuffer",
        "GeneratedParticlesRecycled",
        JustWarning,
        ("beamOn exceeds the " + std::to_string(fBinaryTotalParticles)
         + " particles in the generated binary file; primaries are now being "
           "replayed from the start. Normalize with the file length, not with "
           "beamOn.").c_str()
      );
    }
    fBinaryInputStream.clear();
    fBinaryInputStream.seekg(fBinaryDataStart);
    fBinaryParticlesRead = 0;
  }

  const uint64_t remaining = fBinaryTotalParticles - fBinaryParticlesRead;
  const std::size_t nToRead =
    static_cast<std::size_t>(
      std::min<uint64_t>(remaining, static_cast<uint64_t>(fBinaryBufferSize))
    );

  fBinaryBuffer.reserve(nToRead);

  for (std::size_t i = 0; i < nToRead; ++i) {
    BinaryParticleRecord rec;
    BinaryParticleRecordV2 recV2;

    if (fBinaryVersion == 2) {
      fBinaryInputStream.read(
        reinterpret_cast<char*>(&recV2),
        sizeof(BinaryParticleRecordV2)
      );
      rec = recV2.particle;
    } else {
      fBinaryInputStream.read(
        reinterpret_cast<char*>(&rec),
        sizeof(BinaryParticleRecord)
      );
    }

    if (!fBinaryInputStream.good()) {
      break;
    }

    fBinaryParticlesRead++;

    G4ParticleDefinition* particle =
      G4ParticleTable::GetParticleTable()->FindParticle(rec.pdg);

    if (!particle) {
      G4cerr << "Skipping binary generated particle with unknown PDG: "
             << rec.pdg << G4endl;
      continue;
    }

    G4ThreeVector dir(rec.vx, rec.vy, rec.vz);

    if (dir.mag2() <= 0.0) {
      continue;
    }

    GeneratedParticle p;
    p.particleName = particle->GetParticleName();
    p.energy = rec.energy * MeV;
    p.position = G4ThreeVector(rec.x * mm, rec.y * mm, rec.z * mm);
    p.direction = dir.unit();

    if (fBinaryVersion == 2) {
      p.groupKey = static_cast<G4long>(recV2.groupKey);
      p.primTrackId = recV2.primTrackId;
      p.primPDG = recV2.primPDG;
      p.primEnergy = recV2.primEnergy * MeV;
    }

    fBinaryBuffer.push_back(p);
  }

  if (fBinaryBuffer.empty()) {
    G4Exception(
      "PrimaryGeneratorAction::FillBinaryBuffer",
      "EmptyBinaryBuffer",
      FatalException,
      "Could not fill binary particle buffer."
    );
  }
}

GeneratedParticle PrimaryGeneratorAction::GetNextBinaryParticle()
{
  if (fBinaryBufferIndex >= fBinaryBuffer.size()) {
    FillBinaryBuffer();
  }

  GeneratedParticle p = fBinaryBuffer[fBinaryBufferIndex];
  fBinaryBufferIndex++;

  return p;
}

GeneratedParticle PrimaryGeneratorAction::GetNextGeneratedParticle()
{
  if (fGeneratedInputFormat == kGeneratedBinary) {
    return GetNextBinaryParticle();
  }

  if (fGeneratedParticles.empty()) {
    G4Exception(
      "PrimaryGeneratorAction::GetNextGeneratedParticle",
      "NoGeneratedParticlesLoaded",
      FatalException,
      "Generated text file mode is active, but no particles were loaded."
    );
  }

  if (fCurrentGeneratedIndex >= fGeneratedParticles.size()) {
    // beamOn asked for more events than the file holds: the primaries are
    // replayed from the start. Warn once - the extra events are NOT
    // statistically independent, and normalizing the resulting flux by beamOn
    // instead of by the file length silently biases it.
    static G4bool recycleWarned = false;
    if (!recycleWarned) {
      recycleWarned = true;
      G4Exception(
        "PrimaryGeneratorAction::GetNextGeneratedParticle",
        "GeneratedParticlesRecycled",
        JustWarning,
        ("beamOn exceeds the " + std::to_string(fGeneratedParticles.size())
         + " particles in the generated text file; primaries are now being "
           "replayed from the start. Normalize with the file length, not with "
           "beamOn.").c_str()
      );
    }
    fCurrentGeneratedIndex = 0;
  }

  GeneratedParticle p = fGeneratedParticles[fCurrentGeneratedIndex];
  fCurrentGeneratedIndex++;

  return p;
}

G4bool PrimaryGeneratorAction::PeekNextGeneratedParticle(GeneratedParticle& p)
{
  if (fGeneratedInputFormat == kGeneratedBinary) {
    if (fBinaryBufferIndex >= fBinaryBuffer.size()) {
      if (fBinaryParticlesRead >= fBinaryTotalParticles) return false;
      FillBinaryBuffer();
    }
    p = fBinaryBuffer[fBinaryBufferIndex];
    return true;
  }

  if (fCurrentGeneratedIndex >= fGeneratedParticles.size()) return false;
  p = fGeneratedParticles[fCurrentGeneratedIndex];
  return true;
}

void PrimaryGeneratorAction::AddGeneratedVertex(G4Event* anEvent,
                                                const GeneratedParticle& p)
{
  G4ParticleDefinition* particle =
    G4ParticleTable::GetParticleTable()->FindParticle(p.particleName);

  if (!particle) {
    G4Exception(
      "PrimaryGeneratorAction::GeneratePrimaries",
      "UnknownGeneratedParticle",
      FatalException,
      ("Unknown particle name: " + p.particleName).c_str()
    );
  }

  particleGun->SetParticleDefinition(particle);

  particleGun->GetCurrentSource()->GetEneDist()
    ->SetMonoEnergy(p.energy);

  particleGun->GetCurrentSource()->GetPosDist()
    ->SetCentreCoords(p.position);

  particleGun->GetCurrentSource()->GetAngDist()
    ->SetParticleMomentumDirection(p.direction);

  particleGun->GeneratePrimaryVertex(anEvent);
}

void PrimaryGeneratorAction::GeneratePrimaries(G4Event* anEvent)
{
  if (!fUseGeneratedFile) {
    particleGun->GeneratePrimaryVertex(anEvent);
    return;
  }

  GeneratedParticle next;
  G4bool hasNext = !fGeneratedExhausted && PeekNextGeneratedParticle(next);
  if (hasNext && next.groupKey >= 0) fGroupMode = true;

  if (!fGroupMode) {
    // Legacy file: one generated particle per event, rewound at the end.
    AddGeneratedVertex(anEvent, GetNextGeneratedParticle());
    return;
  }

  if (!hasNext) {
    // Every group has been injected. Rewinding would replay group keys and
    // double count their primaries, so end the run here; this event keeps no
    // vertices and EventAction records nothing for it.
    if (!fGeneratedExhausted) {
      fGeneratedExhausted = true;
      G4Exception(
        "PrimaryGeneratorAction::GeneratePrimaries",
        "GeneratedGroupsExhausted",
        JustWarning,
        "All groups of the generated file have been injected; aborting the "
        "rest of the run. Normalize with the sampled-primary count in the "
        "generation sidecar, not with beamOn."
      );
    }
    G4RunManager::GetRunManager()->AbortRun(true);
    return;
  }

  // Group mode: every consecutive record sharing the group key is one crossing
  // of the same primary, injected as one vertex of this event.
  const G4long key = next.groupKey;
  anEvent->SetUserInformation(
    new PrimaryTruthInfo(key, next.primTrackId, next.primPDG, next.primEnergy));

  while (hasNext && next.groupKey == key) {
    AddGeneratedVertex(anEvent, GetNextGeneratedParticle());
    hasNext = PeekNextGeneratedParticle(next);
  }
}

// ##########################################################################
// ################### AUTO ML GENERATED FILE SETUP METHODS #################
// ##########################################################################

void PrimaryGeneratorAction::SetAutoGenerateMLInput(G4bool value)
{
  fAutoGenerateMLInput = value;
}

void PrimaryGeneratorAction::SetMLPython(const G4String& value)
{
  fMLPython = value;
}

void PrimaryGeneratorAction::SetMLScript(const G4String& value)
{
  fMLScript = value;
}

void PrimaryGeneratorAction::SetMLModelDir(const G4String& value)
{
  fMLModelDir = value;
}

void PrimaryGeneratorAction::SetMLModelName(const G4String& value)
{
  fMLModelName = value;
}

void PrimaryGeneratorAction::SetMLMetadataFile(const G4String& value)
{
  fMLMetadataFile = value;
}

void PrimaryGeneratorAction::SetMLNumParticles(G4int value)
{
  fMLNumParticles = value;
}

void PrimaryGeneratorAction::SetMLSeed(G4int value)
{
  fMLSeed = value;
}

void PrimaryGeneratorAction::SetMLChunkSize(G4int value)
{
  fMLChunkSize = value;
}

void PrimaryGeneratorAction::RunMLGenerator()
{
  if (fMLPython.empty() ||
      fMLScript.empty() ||
      fMLModelDir.empty() ||
      fMLModelName.empty() ||
      fMLMetadataFile.empty() ||
      fGeneratedFileName.empty() ||
      fMLNumParticles <= 0) {

    G4Exception(
      "PrimaryGeneratorAction::RunMLGenerator",
      "MissingMLGeneratorConfiguration",
      FatalException,
      "Incomplete ML generator configuration."
    );
  }

  std::ostringstream cmd;

  cmd << "\"" << fMLPython << "\" "
      << "\"" << fMLScript << "\" "
      << "--save-dir \"" << fMLModelDir << "\" "
      << "--model-name \"" << fMLModelName << "\" "
      << "--metadata-file \"" << fMLMetadataFile << "\" "
      << "--output-file \"" << fGeneratedFileName << "\" "
      << "--n-events " << fMLNumParticles << " "
      << "--seed " << fMLSeed << " "
      << "--chunk-size " << fMLChunkSize << " ";

  if (fGeneratedInputFormat == kGeneratedBinary) {
    cmd << "--format binary";
  }
  else {
    cmd << "--format text";
  }

  G4cout << "[MAGI] Running ML generator:" << G4endl;
  G4cout << cmd.str() << G4endl;

  int ret = std::system(cmd.str().c_str());

  if (ret != 0) {
    G4Exception(
      "PrimaryGeneratorAction::RunMLGenerator",
      "MLGeneratorFailed",
      FatalException,
      "Python ML generation script failed."
    );
  }
}