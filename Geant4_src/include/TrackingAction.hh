// TrackingAction.hh  --  MultiVerse (Track G)
//
// Sets the first-entry flag of the primaries. A primary is flagged if its vertex lies inside the
// ball (a source inside the tracking surface), and is not flagged otherwise. The secondaries
// are handled in SteppingAction, at the moment of their creation, since only there is the flag of
// the parent as it was at that moment available (see MULTIVERSE.md, "Why the secondaries are
// marked in the stepping action").
#ifndef TrackingAction_h
#define TrackingAction_h 1

#include "G4UserTrackingAction.hh"

class TrackingAction : public G4UserTrackingAction
{
  public:
    TrackingAction() {}
    virtual ~TrackingAction() {}

    virtual void PreUserTrackingAction(const G4Track*);
};

#endif
