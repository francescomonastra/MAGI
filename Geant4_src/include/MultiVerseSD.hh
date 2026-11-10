// MultiVerseSD.hh  --  MultiVerse (Track G of MAGI docs/CVAE_vs_AutoRegFlow.md)
//
// Sensitive detector of the MultiVerse ball. It replaces DummySD, which was attached to the
// physical CryoSphere shell of the original DM1.2 project.
//
// The parallel-world process calls ProcessHits with a "ghost" step for every step that runs
// through the ball. The ghost step copies the pre- and post-step points of the real step, and
// the pre-step status is the status of the ghost post-step point of the previous step. A step
// that starts on the ball surface therefore has pre-step status fGeomBoundary, and it is an
// ENTRY: the track has just crossed from the ghost world into the ball. A track that is born
// inside the ball, or that starts there, has status fUndefined and is not an entry, and the exit
// step has the boundary status on its post-step point only.
//
// The hit stores the state of the track at the PRE-step point, which is the crossing point on the
// surface (no interpolation), plus the flag AncestorInside read from the track information
// BEFORE the step updates it (SteppingAction runs after the processes, so the flag is still the
// one of the moment of entry).
//
// The collection keeps the name "DummyCollection" so that EventAction, which reads it by name,
// needs no change in how the collection is found.
#ifndef MultiVerseSD_h
#define MultiVerseSD_h 1

#include "G4VSensitiveDetector.hh"
#include "DummyHit.hh"

class G4Step;
class G4HCofThisEvent;

class MultiVerseSD : public G4VSensitiveDetector
{
  public:
    explicit MultiVerseSD(G4String name);
    virtual ~MultiVerseSD();

    virtual void   Initialize(G4HCofThisEvent* HCE);
    virtual G4bool ProcessHits(G4Step* aStep, G4TouchableHistory* ROhist);
    virtual void   EndOfEvent(G4HCofThisEvent* HCE);

  private:
    DummyHitsCollection* fCollection;
};

#endif
