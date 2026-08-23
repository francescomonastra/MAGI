#ifndef PrimaryGeneratorMessenger_h
#define PrimaryGeneratorMessenger_h 1

#include "G4UImessenger.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "globals.hh"

class PrimaryGeneratorAction;
class G4UIdirectory;
class G4UIcmdWithABool;
class G4UIcmdWithAString;

class PrimaryGeneratorMessenger : public G4UImessenger
{
public:
  PrimaryGeneratorMessenger(PrimaryGeneratorAction* generator);
  virtual ~PrimaryGeneratorMessenger();

  virtual void SetNewValue(G4UIcommand* command, G4String newValue);

private:
  PrimaryGeneratorAction* fGenerator;

  G4UIdirectory* fGeneratorDir;

  G4UIcmdWithABool* fUseGeneratedFileCmd;
  G4UIcmdWithAString* fGeneratedFileCmd;
  G4UIcmdWithAString* fGeneratedFormatCmd;
  G4UIcmdWithAnInteger* fBinaryBufferSizeCmd;

  // Auto ML generated file setup commands
  G4UIcmdWithABool* fAutoGenerateMLInputCmd;
  G4UIcmdWithAString* fMLPythonCmd;
  G4UIcmdWithAString* fMLScriptCmd;
  G4UIcmdWithAString* fMLModelDirCmd;
  G4UIcmdWithAString* fMLModelNameCmd;
  G4UIcmdWithAString* fMLMetadataFileCmd;
  G4UIcmdWithAnInteger* fMLNumParticlesCmd;
  G4UIcmdWithAnInteger* fMLSeedCmd;
  G4UIcmdWithAnInteger* fMLChunkSizeCmd;
};

#endif