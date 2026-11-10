// SteppingAction.cc  --  MultiVerse (Track G), see MULTIVERSE.md
//
// This is where the tracking ball is "observed". For every step of every track:
//
//  1. ENTRY. If the straight segment from the pre-step to the post-step point enters the ball
//     (MultiVerseBall::SegmentEntry), the track has crossed the surface. The state at the crossing
//     point is stored as a hit: position by linear interpolation along the segment, direction and
//     kinetic energy of the pre-step point (see MULTIVERSE.md, "Accuracy of the crossing state"),
//     and the flag AncestorInside as it was BEFORE this step. EventAction writes the first entry
//     of each track.
//  2. FLAG. The track becomes flagged ("this track or an ancestor has been inside the ball") if the
//     step enters the ball or ends inside it.
//  3. SECONDARIES. Every secondary created in the step takes the flag of its parent AT ITS CREATION:
//     it is born flagged if it is created inside the ball. A secondary created by a discrete process
//     sits at the post-step point and takes the parent's flag after the step; one created along the
//     step by a continuous process (a delta ray) sits before it and takes the parent's flag before
//     it. The flag of the parent at the END of its history is the wrong one: it may be set only
//     after the secondary has left.
#include "SteppingAction.hh"
#include "AncestryInfo.hh"
#include "MultiVerseBall.hh"
#include "DummyHit.hh"

#include "G4Step.hh"
#include "G4Track.hh"
#include "G4StepPoint.hh"
#include "G4ParticleDefinition.hh"
#include "G4VPhysicalVolume.hh"
#include "G4VProcess.hh"
#include "G4SystemOfUnits.hh"
#include "G4Material.hh"
#include <vector>

SteppingAction::SteppingAction() {}

void SteppingAction::UserSteppingAction(const G4Step* aStep)
{
  G4Track* track = aStep->GetTrack();

  AncestryInfo* info = dynamic_cast<AncestryInfo*>(track->GetUserInformation());
  if (!info) {                       // a track that TrackingAction did not see (should not happen)
    info = new AncestryInfo(MultiVerseBall::IsInside(track->GetVertexPosition()));
    track->SetUserInformation(info);
  }
  const G4bool flagBefore = info->AncestorInside();

  // Single-scattering (G4EmStandardSS in InnerRegion) can leave a keV Cu/Au recoil, or a C recoil in the
  // polypropylene PCB mesh, taking ~fm steps on a volume surface (GeomNav0003 aborts, GeomNav1002 warnings
  // at (7.19, -24.60, -4.40) mm). Such a recoil has a range of nm and cannot reach a sensitive volume, so
  // it is killed here. The kill sits after the flag read, so the track still gets recorded if it enters the ball.
  if (track->GetParentID() > 0 && track->GetDefinition()->GetAtomicNumber() > 2 &&
      track->GetKineticEnergy() < 100*keV) {
    const G4String& mat = track->GetMaterial()->GetName();
    if (mat == "G4_Cu" || mat == "G4_Au" || mat == "G4_POLYPROPYLENE")
      track->SetTrackStatus(fStopAndKill);
  }

  const G4StepPoint* pre  = aStep->GetPreStepPoint();
  const G4StepPoint* post = aStep->GetPostStepPoint();
  const G4ThreeVector p0 = pre->GetPosition();
  const G4ThreeVector p1 = post->GetPosition();

  // 1. entry
  G4double t = 0.;
  const G4bool enters = MultiVerseBall::SegmentEntry(p0, p1, t);
  if (enters) {
    DummyHit* hit = new DummyHit();
    hit->SetTrackID(track->GetTrackID());
    hit->SetEdep(aStep->GetTotalEnergyDeposit());
    hit->SetPos(p0 + t * (p1 - p0));
    hit->SetDirection(pre->GetMomentumDirection());
    hit->SetKineticEnergy(pre->GetKineticEnergy());
    hit->SetKineticEnergyPreStep(pre->GetKineticEnergy());
    hit->SetParticleName(track->GetDefinition()->GetParticleName());
    hit->SetParticleID(track->GetTrackID());
    hit->SetParentParticleID(track->GetParentID());
    const G4VProcess* creator = track->GetCreatorProcess();
    hit->SetCreatorProcessName(track->GetParentID() == 0 || !creator ? G4String("Primary")
                                                                       : creator->GetProcessName());
    hit->SetProcessName("MultiVerseEntry");
    hit->SetVertexKineticEnergy(track->GetVertexKineticEnergy());
    hit->SetVertexPosition(track->GetVertexPosition());
    hit->SetVertexMomentumDirection(track->GetVertexMomentumDirection());
    hit->SetGlobalTime(pre->GetGlobalTime() / nanosecond + t * (post->GetGlobalTime() - pre->GetGlobalTime()) / nanosecond);
    hit->SetAncestorInside(flagBefore);
    MultiVerseBall::AddEntry(hit);
  }

  // 1b. INTERIOR DEPOSIT (G3 observable). The energy deposited by steps whose midpoint is inside the
  //     ball, by volume of the step. A step that straddles the surface is assigned entirely by its
  //     midpoint, which leaves a boundary layer of one step (short where the material is dense).
  const G4double edep = aStep->GetTotalEnergyDeposit();
  if (edep > 0. && MultiVerseBall::IsInside(0.5 * (p0 + p1))) {
    const G4VPhysicalVolume* pv = pre->GetPhysicalVolume();
    MultiVerseBall::AddInterior(pv ? pv->GetName() : G4String("none"), edep);
  }

  // 2. the flag of the track after this step
  if (enters || MultiVerseBall::IsInside(p1)) info->SetAncestorInside(true);

  // 3. the secondaries created in this step
  const std::vector<const G4Track*>* secondaries = aStep->GetSecondaryInCurrentStep();
  if (!secondaries) return;
  for (size_t i = 0; i < secondaries->size(); ++i) {
    G4Track* sec = const_cast<G4Track*>((*secondaries)[i]);
    if (sec->GetUserInformation()) continue;
    const G4bool atPostPoint = (sec->GetPosition() - p1).mag() < 1.e-9 * mm;
    const G4bool parentFlag = atPostPoint ? info->AncestorInside() : flagBefore;
    sec->SetUserInformation(new AncestryInfo(parentFlag || MultiVerseBall::IsInside(sec->GetPosition())));
  }
}
