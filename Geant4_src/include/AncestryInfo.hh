// AncestryInfo.hh  --  MultiVerse (Track G of MAGI docs/CVAE_vs_AutoRegFlow.md)
//
// User track information that carries the first-entry flag:
//
//     AncestorInside = true   this track, or one of its ancestors, has been inside the
//                             MultiVerse ball (born there, or crossed into it) at an earlier
//                             moment of its history.
//
// A track that enters the ball with AncestorInside == false is a FIRST ENTRY: its whole
// history, and that of every ancestor, lies outside the ball, so it cannot depend on what is
// inside. These are the particles that MAGI learns from; everything with the flag set is
// reproduced by the application run from the injected first entries.
//
// The flag is set in two places and read in a third (see MULTIVERSE.md):
//   TrackingAction::PreUserTrackingAction  primaries, from the vertex position
//   SteppingAction::UserSteppingAction     after every step, and for the secondaries of the step
//   MultiVerseSD::ProcessHits              reads it when a track enters the ball
#ifndef AncestryInfo_h
#define AncestryInfo_h 1

#include "G4VUserTrackInformation.hh"

class AncestryInfo : public G4VUserTrackInformation
{
  public:
    explicit AncestryInfo(G4bool inside = false) : fAncestorInside(inside) {}
    virtual ~AncestryInfo() {}

    G4bool AncestorInside() const { return fAncestorInside; }
    void   SetAncestorInside(G4bool v) { fAncestorInside = v; }

    virtual void Print() const {}

  private:
    G4bool fAncestorInside;
};

#endif
