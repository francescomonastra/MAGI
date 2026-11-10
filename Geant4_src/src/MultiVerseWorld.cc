// MultiVerseWorld.cc  --  see include/MultiVerseWorld.hh and MULTIVERSE.md
#include "MultiVerseWorld.hh"
#include "MultiVerseSD.hh"
#include "MultiVerseBall.hh"

#include "G4LogicalVolume.hh"
#include "G4Orb.hh"
#include "G4PVPlacement.hh"
#include "G4SDManager.hh"
#include "G4SystemOfUnits.hh"
#include "G4VisAttributes.hh"

G4String MultiVerseWorld::fWorldName = "MultiVerseWorld";

MultiVerseWorld::MultiVerseWorld(const G4String& worldName)
 : G4VUserParallelWorld(worldName), fBallLogical(0)
{
  fWorldName = worldName;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

// Called by the run manager at initialisation, after the mass world is built. GetWorld()
// returns the ghost world, which the kernel creates with the dimensions of the mass world;
// the ball must fit in it.
void MultiVerseWorld::Construct()
{
  G4VPhysicalVolume* ghostWorld = GetWorld();
  G4LogicalVolume*   ghostLog   = ghostWorld->GetLogicalVolume();

  // A solid ball: the parallel-world process ends the step on its boundary. No material is
  // given, since a parallel world that is not layered does not use one.
  G4Orb* ball = new G4Orb("MultiVerseBall", MultiVerseBall::Radius());
  fBallLogical = new G4LogicalVolume(ball, 0, "MultiVerseBallLV");
  new G4PVPlacement(0, MultiVerseBall::Centre(), fBallLogical, "MultiVerseBallPV", ghostLog, false, 0, true);

  G4cout << "[MultiVerse] ball of radius " << MultiVerseBall::Radius() / mm << " mm centred at ("
         << MultiVerseBall::Centre().x() / mm << ", " << MultiVerseBall::Centre().y() / mm << ", " << MultiVerseBall::Centre().z() / mm
         << ") mm in the parallel world '" << fWorldName << "'" << G4endl;
}

// Called after Construct(): the SD must be attached here, to the volume of the parallel world.
void MultiVerseWorld::ConstructSD()
{
  MultiVerseSD* sd = new MultiVerseSD("MultiVerseSD");
  G4SDManager::GetSDMpointer()->AddNewDetector(sd);
  SetSensitiveDetector(fBallLogical, sd);
}
