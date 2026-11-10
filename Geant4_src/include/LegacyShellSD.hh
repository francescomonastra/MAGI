// LegacyShellSD.hh  --  MultiVerse (Track G), DIAGNOSTIC ONLY
//
// Re-creates the recording of the ORIGINAL S1GDML_DM1.2 project inside the MultiVerse build, so
// that the ball and the original thin shell can be compared event by event in the same run.
// It is created only when the environment variable MULTIVERSE_LEGACY_SHELL is set, in which case
// GDMLDetectorConstruction also places the original G4Sphere shell (94.9-95 mm about (0,0,5)) in
// the mass world and attaches this detector to it.
//
// It applies the original rule: at the post-step point of a step in the shell, keep the track if
// its direction points towards the sphere centre, and only the first such hit of each track in an
// event. Rows go to outputLegacyShell.dat in the layout of outputDSCryoSphere.dat (17 columns,
// the last one being the first-entry flag of the track at that moment), so that the two files can
// be joined on (EventKey, TrackId).
#ifndef LegacyShellSD_h
#define LegacyShellSD_h 1

#include "G4VSensitiveDetector.hh"
#include <set>
#include <utility>

class LegacyShellSD : public G4VSensitiveDetector
{
  public:
    explicit LegacyShellSD(G4String name) : G4VSensitiveDetector(name), fEvent(-1) {}
    virtual ~LegacyShellSD() {}
    virtual G4bool ProcessHits(G4Step* aStep, G4TouchableHistory*);

  private:
    G4int fEvent;
    std::set<G4int> fSeen;   // track ids already written in the current event
};

#endif
