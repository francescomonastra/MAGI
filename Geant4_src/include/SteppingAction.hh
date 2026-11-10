#ifndef SteppingAction_h
#define SteppingAction_h 1

#include "G4UserSteppingAction.hh"

// MultiVerse (Track G): updates the first-entry flag after every step and hands it to the
// secondaries created in the step. See MULTIVERSE.md.
class SteppingAction : public G4UserSteppingAction
{
  public:
    SteppingAction();
   ~SteppingAction(){};

    void UserSteppingAction(const G4Step*);
};

#endif
