// TrackingAction.cc  --  see include/TrackingAction.hh and MULTIVERSE.md
#include "TrackingAction.hh"
#include "AncestryInfo.hh"
#include "MultiVerseBall.hh"
#include "MultiVerseWorld.hh"

#include "G4Track.hh"

void TrackingAction::PreUserTrackingAction(const G4Track* track)
{
  // Secondaries already carry their flag (set by SteppingAction when they were created).
  if (track->GetUserInformation()) return;

  // A primary: flagged if it starts inside the ball.
  G4bool inside = MultiVerseBall::IsInside(track->GetVertexPosition());
  const_cast<G4Track*>(track)->SetUserInformation(new AncestryInfo(inside));
}
