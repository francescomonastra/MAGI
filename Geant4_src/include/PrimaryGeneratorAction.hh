#ifndef PrimaryGeneratorAction_h
#define PrimaryGeneratorAction_h 1

#include "G4VUserPrimaryGeneratorAction.hh"
#include "G4GeneralParticleSource.hh"
#include "G4ThreeVector.hh"
#include "G4String.hh"
#include "globals.hh"

#include <vector>
#include <cstddef>
#include <cstdint>
#include <fstream>

class G4Event;
class PrimaryGeneratorMessenger;

enum GeneratedInputFormat {
  kGeneratedText,
  kGeneratedBinary
};

struct GeneratedParticle {
  G4String particleName;
  G4double energy;
  G4ThreeVector position;
  G4ThreeVector direction;
};

class PrimaryGeneratorAction : public G4VUserPrimaryGeneratorAction
{
public:
  PrimaryGeneratorAction();
  virtual ~PrimaryGeneratorAction();

  virtual void GeneratePrimaries(G4Event* anEvent);

  void SetUseGeneratedFile(G4bool value);
  void SetGeneratedFileName(const G4String& filename);
  void SetGeneratedInputFormat(const G4String& format);
  void SetBinaryBufferSize(G4int value);

  void SetAutoGenerateMLInput(G4bool value);
  void SetMLPython(const G4String& value);
  void SetMLScript(const G4String& value);
  void SetMLModelDir(const G4String& value);
  void SetMLModelName(const G4String& value);
  void SetMLMetadataFile(const G4String& value);
  void SetMLNumParticles(G4int value);
  void SetMLSeed(G4int value);
  void SetMLChunkSize(G4int value);

  void RunMLGenerator();

private:
  void LoadGeneratedParticleFile(const G4String& filename);

  void OpenGeneratedParticleBinaryFile(const G4String& filename);
  void FillBinaryBuffer();
  GeneratedParticle GetNextBinaryParticle();
  GeneratedParticle GetNextGeneratedParticle();

private:
  G4GeneralParticleSource* particleGun;
  PrimaryGeneratorMessenger* fMessenger;

  G4bool fUseGeneratedFile;
  G4String fGeneratedFileName;
  GeneratedInputFormat fGeneratedInputFormat;

  std::vector<GeneratedParticle> fGeneratedParticles;
  std::size_t fCurrentGeneratedIndex;

  // Binary streaming state
  std::ifstream fBinaryInputStream;
  uint64_t fBinaryTotalParticles;
  uint64_t fBinaryParticlesRead;
  std::streampos fBinaryDataStart;
  std::vector<GeneratedParticle> fBinaryBuffer;
  std::size_t fBinaryBufferIndex;
  std::size_t fBinaryBufferSize;

  // Auto ML generated file setup variables
  G4bool fAutoGenerateMLInput;
  G4String fMLPython;
  G4String fMLScript;
  G4String fMLModelDir;
  G4String fMLModelName;
  G4String fMLMetadataFile;
  G4int fMLNumParticles;
  G4int fMLSeed;
  G4int fMLChunkSize;
};

#endif