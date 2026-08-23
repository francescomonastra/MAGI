#include "PrimaryGeneratorMessenger.hh"
#include "PrimaryGeneratorAction.hh"

#include "G4UIdirectory.hh"
#include "G4UIcmdWithABool.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithAnInteger.hh"

PrimaryGeneratorMessenger::PrimaryGeneratorMessenger(
  PrimaryGeneratorAction* generator
)
: G4UImessenger(),
  fGenerator(generator)
{
  fGeneratorDir = new G4UIdirectory("/generator/");
  fGeneratorDir->SetGuidance("Primary generator control commands.");

  fUseGeneratedFileCmd = new G4UIcmdWithABool(
    "/generator/useGeneratedFile",
    this
  );
  fUseGeneratedFileCmd->SetGuidance(
    "Enable or disable MAGI generated-particle file mode."
  );
  fUseGeneratedFileCmd->SetParameterName("useGeneratedFile", false);
  fUseGeneratedFileCmd->AvailableForStates(G4State_PreInit, G4State_Idle);

  fGeneratedFormatCmd = new G4UIcmdWithAString(
    "/generator/generatedFormat",
    this
  );
  fGeneratedFormatCmd->SetGuidance(
    "Set generated-particle input format: text/txt or binary/bin."
  );
  fGeneratedFormatCmd->SetParameterName("generatedFormat", false);
  fGeneratedFormatCmd->AvailableForStates(G4State_PreInit, G4State_Idle);

  fGeneratedFileCmd = new G4UIcmdWithAString(
    "/generator/generatedFile",
    this
  );
  fGeneratedFileCmd->SetGuidance(
    "Load generated-particle file. Text format: ParticleName Energy X Y Z Vx Vy Vz. Binary format: MAGI compact binary."
  );
  fGeneratedFileCmd->SetParameterName("generatedFile", false);
  fGeneratedFileCmd->AvailableForStates(G4State_PreInit, G4State_Idle);

  // Auto ML generator setup commands

  fAutoGenerateMLInputCmd = new G4UIcmdWithABool(
    "/generator/autoGenerateMLInput",
    this
  );
  fAutoGenerateMLInputCmd->SetGuidance(
    "If true, run the MAGI Python generator before loading the particle file."
  );
  fAutoGenerateMLInputCmd->SetParameterName("autoGenerateMLInput", false);
  fAutoGenerateMLInputCmd->AvailableForStates(G4State_PreInit, G4State_Idle);

  fMLPythonCmd = new G4UIcmdWithAString(
    "/generator/mlPython",
    this
  );
  fMLPythonCmd->SetGuidance("Python executable used to run MAGI.");
  fMLPythonCmd->SetParameterName("mlPython", false);
  fMLPythonCmd->AvailableForStates(G4State_PreInit, G4State_Idle);

  fMLScriptCmd = new G4UIcmdWithAString(
    "/generator/mlScript",
    this
  );
  fMLScriptCmd->SetGuidance("Path to generate_geant_source.py.");
  fMLScriptCmd->SetParameterName("mlScript", false);
  fMLScriptCmd->AvailableForStates(G4State_PreInit, G4State_Idle);

  fMLModelDirCmd = new G4UIcmdWithAString(
    "/generator/mlModelDir",
    this
  );
  fMLModelDirCmd->SetGuidance("Directory containing the trained MAGI model.");
  fMLModelDirCmd->SetParameterName("mlModelDir", false);
  fMLModelDirCmd->AvailableForStates(G4State_PreInit, G4State_Idle);

  fMLModelNameCmd = new G4UIcmdWithAString(
    "/generator/mlModelName",
    this
  );
  fMLModelNameCmd->SetGuidance("Trained MAGI model name.");
  fMLModelNameCmd->SetParameterName("mlModelName", false);
  fMLModelNameCmd->AvailableForStates(G4State_PreInit, G4State_Idle);

  fMLMetadataFileCmd = new G4UIcmdWithAString(
    "/generator/mlMetadataFile",
    this
  );
  fMLMetadataFileCmd->SetGuidance("Metadata JSON file for the trained model.");
  fMLMetadataFileCmd->SetParameterName("mlMetadataFile", false);
  fMLMetadataFileCmd->AvailableForStates(G4State_PreInit, G4State_Idle);

  fMLNumParticlesCmd = new G4UIcmdWithAnInteger(
    "/generator/mlNumParticles",
    this
  );
  fMLNumParticlesCmd->SetGuidance("Number of particles to generate with MAGI.");
  fMLNumParticlesCmd->SetParameterName("mlNumParticles", false);
  fMLNumParticlesCmd->AvailableForStates(G4State_PreInit, G4State_Idle);

  fMLSeedCmd = new G4UIcmdWithAnInteger(
    "/generator/mlSeed",
    this
  );
  fMLSeedCmd->SetGuidance("Random seed passed to MAGI generation.");
  fMLSeedCmd->SetParameterName("mlSeed", false);
  fMLSeedCmd->AvailableForStates(G4State_PreInit, G4State_Idle);

  fMLChunkSizeCmd = new G4UIcmdWithAnInteger(
    "/generator/mlChunkSize",
    this
  );
  fMLChunkSizeCmd->SetGuidance("Chunk size for MAGI generation.");
  fMLChunkSizeCmd->SetParameterName("mlChunkSize", false);
  fMLChunkSizeCmd->AvailableForStates(G4State_PreInit, G4State_Idle);

  fBinaryBufferSizeCmd = new G4UIcmdWithAnInteger(
  "/generator/binaryBufferSize",
  this
  );
  fBinaryBufferSizeCmd->SetGuidance(
    "Number of binary generated particles kept in memory at once."
  );
  fBinaryBufferSizeCmd->SetParameterName("binaryBufferSize", false);
  fBinaryBufferSizeCmd->AvailableForStates(G4State_PreInit, G4State_Idle);
}

PrimaryGeneratorMessenger::~PrimaryGeneratorMessenger()
{
  delete fMLChunkSizeCmd;
  delete fMLSeedCmd;
  delete fMLNumParticlesCmd;
  delete fMLMetadataFileCmd;
  delete fMLModelNameCmd;
  delete fMLModelDirCmd;
  delete fMLScriptCmd;
  delete fMLPythonCmd;
  delete fAutoGenerateMLInputCmd;

  delete fGeneratedFileCmd;
  delete fGeneratedFormatCmd;
  delete fBinaryBufferSizeCmd;
  delete fUseGeneratedFileCmd;
  delete fGeneratorDir;
}

void PrimaryGeneratorMessenger::SetNewValue(
  G4UIcommand* command,
  G4String newValue
)
{
  if (command == fUseGeneratedFileCmd) {
    fGenerator->SetUseGeneratedFile(
      fUseGeneratedFileCmd->GetNewBoolValue(newValue)
    );
  }

  else if (command == fGeneratedFormatCmd) {
    fGenerator->SetGeneratedInputFormat(newValue);
  }

  else if (command == fGeneratedFileCmd) {
    fGenerator->SetGeneratedFileName(newValue);
  }

  else if (command == fAutoGenerateMLInputCmd) {
    fGenerator->SetAutoGenerateMLInput(
      fAutoGenerateMLInputCmd->GetNewBoolValue(newValue)
    );
  }

  else if (command == fMLPythonCmd) {
    fGenerator->SetMLPython(newValue);
  }

  else if (command == fMLScriptCmd) {
    fGenerator->SetMLScript(newValue);
  }

  else if (command == fMLModelDirCmd) {
    fGenerator->SetMLModelDir(newValue);
  }

  else if (command == fMLModelNameCmd) {
    fGenerator->SetMLModelName(newValue);
  }

  else if (command == fMLMetadataFileCmd) {
    fGenerator->SetMLMetadataFile(newValue);
  }

  else if (command == fMLNumParticlesCmd) {
    fGenerator->SetMLNumParticles(
      fMLNumParticlesCmd->GetNewIntValue(newValue)
    );
  }

  else if (command == fMLSeedCmd) {
    fGenerator->SetMLSeed(
      fMLSeedCmd->GetNewIntValue(newValue)
    );
  }

  else if (command == fMLChunkSizeCmd) {
    fGenerator->SetMLChunkSize(
      fMLChunkSizeCmd->GetNewIntValue(newValue)
    );
  }

  else if (command == fBinaryBufferSizeCmd) {
    fGenerator->SetBinaryBufferSize(
      fBinaryBufferSizeCmd->GetNewIntValue(newValue)
    );
  }
}