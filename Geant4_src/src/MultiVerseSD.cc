// MultiVerseSD.cc  --  see include/MultiVerseSD.hh and MULTIVERSE.md
#include "MultiVerseSD.hh"
#include "AncestryInfo.hh"

#include "G4HCofThisEvent.hh"
#include "G4SDManager.hh"
#include "G4Step.hh"
#include "G4StepPoint.hh"
#include "G4StepStatus.hh"
#include "G4Track.hh"
#include "G4ParticleDefinition.hh"
#include "G4VProcess.hh"
#include "G4SystemOfUnits.hh"

MultiVerseSD::MultiVerseSD(G4String name)
 : G4VSensitiveDetector(name), fCollection(0)
{
  G4String HCname;
  collectionName.insert(HCname = "DummyCollection");
}

MultiVerseSD::~MultiVerseSD() {}

void MultiVerseSD::Initialize(G4HCofThisEvent* HCE)
{
  fCollection = new DummyHitsCollection(SensitiveDetectorName, collectionName[0]);
  static G4int HCID = -1;
  if (HCID < 0) HCID = G4SDManager::GetSDMpointer()->GetCollectionID(collectionName[0]);
  HCE->AddHitsCollection(HCID, fCollection);
}

G4bool MultiVerseSD::ProcessHits(G4Step* aStep, G4TouchableHistory*)
{
  // Diagnostic counters (printed by the destructor): calls and entries.
  static G4int nCalls = 0;
  ++nCalls;
  if (getenv("MULTIVERSE_DEBUG") && nCalls <= 40) {
    G4cout << "[MultiVerse-debug] call " << nCalls << " track " << aStep->GetTrack()->GetTrackID()
           << " pre-status " << aStep->GetPreStepPoint()->GetStepStatus()
           << " post-status " << aStep->GetPostStepPoint()->GetStepStatus()
           << " r_pre " << (aStep->GetPreStepPoint()->GetPosition() - G4ThreeVector(0,0,5)).mag() << G4endl;
  }

  // Only the step that starts on the ball surface is an entry (see MultiVerseSD.hh).
  const G4StepPoint* pre = aStep->GetPreStepPoint();
  if (pre->GetStepStatus() != fGeomBoundary) return false;

  const G4Track* track = aStep->GetTrack();

  DummyHit* hit = new DummyHit();
  hit->SetTrackID(track->GetTrackID());
  hit->SetEdep(aStep->GetTotalEnergyDeposit());

  // State of the track ON the surface: pre-step point, exact.
  hit->SetPos(pre->GetPosition());
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
  hit->SetGlobalTime(pre->GetGlobalTime() / nanosecond);

  // The flag as it was before this step.
  const AncestryInfo* info = dynamic_cast<const AncestryInfo*>(track->GetUserInformation());
  hit->SetAncestorInside(info ? info->AncestorInside() : false);

  fCollection->insert(hit);
  return true;
}

void MultiVerseSD::EndOfEvent(G4HCofThisEvent*) {}
