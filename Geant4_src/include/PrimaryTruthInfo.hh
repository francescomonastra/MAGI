#ifndef PrimaryTruthInfo_h
#define PrimaryTruthInfo_h 1

#include "G4VUserEventInformation.hh"
#include "G4SystemOfUnits.hh"
#include "G4ios.hh"
#include "globals.hh"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
//
// Truth of the physical primary that an event stands for (HENABICS traceability).
//
// Attached by PrimaryGeneratorAction when a MAGI group - all the generated
// crossings of one sampled primary - is injected as one event. EventAction reads
// it back and writes the same primary columns it writes in full simulation, so
// full and post-MAGI outputs share one layout and one join key.
//
//   key     : group key, globally unique across the generated file (replaces EventId)
//   trackId : track ID of the primary in the original simulation (1 for GPS)
//   pdg     : PDG code of the primary
//   energy  : primary kinetic energy (Geant4 internal units)

class PrimaryTruthInfo : public G4VUserEventInformation
{
  public:
    PrimaryTruthInfo(G4long key, G4int trackId, G4int pdg, G4double energy)
      : fKey(key), fTrackId(trackId), fPDG(pdg), fEnergy(energy) {}
    virtual ~PrimaryTruthInfo() {}

    virtual void Print() const {
      G4cout << "PrimaryTruthInfo: key " << fKey << " track " << fTrackId
             << " pdg " << fPDG << " E " << fEnergy/MeV << " MeV" << G4endl;
    }

    G4long   GetKey()     const { return fKey; }
    G4int    GetTrackId() const { return fTrackId; }
    G4int    GetPDG()     const { return fPDG; }
    G4double GetEnergy()  const { return fEnergy; }

  private:
    G4long   fKey;
    G4int    fTrackId;
    G4int    fPDG;
    G4double fEnergy;
};

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#endif
