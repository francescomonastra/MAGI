// MultiVerseWorld.hh  --  MultiVerse (Track G of MAGI docs/CVAE_vs_AutoRegFlow.md)
//
// The MAGI tracking surface as a SOLID BALL IN A GEANT4 PARALLEL WORLD.
//
// Why: MAGI needs the phase space of the particles that enter a closed surface. In the
// original DM1.2 project the surface was a thin G4Sphere shell placed in the mass world, which
// is invalid wherever it cuts another volume (the reason the radius had to be reduced to
// 95 mm). A parallel world has its own geometry tree, so the ball can cut any number of mass
// volumes without an overlap, and the G4ParallelWorldProcess ends a step exactly on the ball
// boundary: the crossing point, the direction and the kinetic energy are those of the track on
// the surface, with no interpolation and no step limiter.
//
// The ball is placed in the ghost world; its sensitive detector (MultiVerseSD) is called by the
// parallel-world process for the steps that run through the ball.
//
// STATUS: NOT USED BY DEFAULT. Geant4 10.4.3 switches the whole simulation to G4CoupledTransportation
// as soon as a parallel world exists, and that changes the physics of the mass world (a factor of
// about 10 fewer low-energy secondaries from the copper plate, measured in MULTIVERSE.md section 6).
// The class is kept, enabled by the environment variable MULTIVERSE_PARALLEL, as the record of that
// experiment. The radius and centre come from MultiVerseBall.
#ifndef MultiVerseWorld_h
#define MultiVerseWorld_h 1

#include "G4VUserParallelWorld.hh"
#include "G4ThreeVector.hh"
#include "globals.hh"
#include "G4SystemOfUnits.hh"

class G4LogicalVolume;

class MultiVerseWorld : public G4VUserParallelWorld
{
  public:
    explicit MultiVerseWorld(const G4String& worldName);
    virtual ~MultiVerseWorld() {}

    virtual void Construct();
    virtual void ConstructSD();

    static G4String WorldName() { return fWorldName; }

  private:
    G4LogicalVolume* fBallLogical;
    static G4String  fWorldName;
};

#endif
