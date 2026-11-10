// MultiVerseBall.hh  --  MultiVerse (Track G of MAGI docs/CVAE_vs_AutoRegFlow.md)
//
// The geometry of the MAGI tracking surface (a ball: centre, radius) and the arithmetic that
// detects its crossings, with NO Geant4 geometry and NO extra process: the surface is virtual and
// the transport of the mass world is the standard one.
//
// This is the DEFAULT recording of the MultiVerse build (see MULTIVERSE.md, section "Why not a
// parallel world"). SteppingAction applies it to every step: if the straight segment from the
// pre-step to the post-step point enters the ball, the track has crossed the surface, and the
// state at the crossing point is recorded (SegmentEntry). The first-entry flag is maintained in
// the same place.
//
// Macro commands (available as soon as CreateMessenger() has been called in main(), so a macro can
// set them before /run/initialize):
//     /MultiVerse/radius <value> <unit>        default 95 mm
//     /MultiVerse/centre <x> <y> <z> <unit>    default (0, 0, 5) mm (DM1.2 sphere centre)
#ifndef MultiVerseBall_h
#define MultiVerseBall_h 1

#include "G4ThreeVector.hh"
#include "G4SystemOfUnits.hh"
#include "globals.hh"
#include <map>

class DummyHit;
template <class T> class G4THitsCollection;
typedef G4THitsCollection<DummyHit> DummyHitsCollection;

class MultiVerseBall
{
  public:
    static void CreateMessenger();                 // call once, from main()

    static G4double      Radius() { return fRadius; }
    static G4ThreeVector Centre() { return fCentre; }
    static void SetRadius(G4double r)             { fRadius = r; }
    static void SetCentre(const G4ThreeVector& c) { fCentre = c; }

    // Fuzz of the surface. Positions that were written to a file and read back (the crossings of a full run,
    // reinjected as GNTBIN float32, or the 6-digit text of outputDSCryoSphere.dat) are on the surface only to
    // about 1e-3 mm at R = 100 mm, so a track injected "on the surface" must not be taken to be inside it. 5 um.
    static G4double SurfaceTolerance() { return 5.e-3 * mm; }

    // True if the point is inside the ball by more than the surface tolerance: a point on the surface
    // (to within it) is not inside.
    static G4bool IsInside(const G4ThreeVector& p, G4double tolerance = 5.e-3 * mm)
    { return (p - fCentre).mag() < fRadius - tolerance; }

    // Does the segment p0 -> p1 ENTER the ball? Requires p0 outside or on the surface (to within the
    // surface tolerance: a track that starts there and points inward enters at t = 0). On success `t` is
    // the fraction of the segment (0..1) at which it crosses the surface, from the outside to the inside.
    // Also true for a segment that passes through the ball with both ends outside.
    static G4bool SegmentEntry(const G4ThreeVector& p0, const G4ThreeVector& p1, G4double& t);

    // Entries of the current event, filled by SteppingAction and emptied by EventAction. The hits
    // collection is owned here and recreated at the start of each event.
    static void                 BeginEvent();
    static void                 AddEntry(DummyHit* hit);
    static DummyHitsCollection* Entries();

    // INTERIOR DEPOSIT (G3 observable of Track G): the energy deposited inside the ball, summed over
    // all volumes and by volume, per event. SteppingAction adds the deposit of every step whose
    // midpoint is inside the ball; EventAction writes the event total to outputInterior.dat.
    static void                 AddInterior(const G4String& volume, G4double edep);
    static G4double             InteriorTotal() { return fInteriorTotal; }
    static const std::map<G4String, G4double>& InteriorByVolume() { return fInteriorByVolume; }

  private:
    static G4double      fInteriorTotal;
    static std::map<G4String, G4double> fInteriorByVolume;
    static G4double      fRadius;
    static G4ThreeVector fCentre;
    static DummyHitsCollection* fEntries;
};

#endif
