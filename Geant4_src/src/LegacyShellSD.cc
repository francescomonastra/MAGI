// LegacyShellSD.cc  --  see include/LegacyShellSD.hh. DIAGNOSTIC ONLY.
#include "LegacyShellSD.hh"
#include "AncestryInfo.hh"
#include "MultiVerseBall.hh"
#include "MultiVerseWorld.hh"
#include "PrimaryTruthInfo.hh"

#include "G4Event.hh"
#include "G4EventManager.hh"
#include "G4PrimaryParticle.hh"
#include "G4PrimaryVertex.hh"
#include "G4Step.hh"
#include "G4Track.hh"
#include "G4VProcess.hh"
#include "G4SystemOfUnits.hh"
#include <fstream>

G4bool LegacyShellSD::ProcessHits(G4Step* aStep, G4TouchableHistory*)
{
  const G4Event* evt = G4EventManager::GetEventManager()->GetConstCurrentEvent();
  if (evt->GetEventID() != fEvent) { fEvent = evt->GetEventID(); fSeen.clear(); }

  const G4Track* track = aStep->GetTrack();
  const G4String name = track->GetDefinition()->GetParticleName();
  if (name == "nu_e" || name == "anti_nu_e" || name == "nu_mu" || name == "anti_nu_mu" ||
      name == "nu_tau" || name == "anti_nu_tau") return false;

  const G4ThreeVector pos = track->GetPosition();                 // post-step point, as in DummySD
  const G4ThreeVector dir = track->GetMomentumDirection();
  if (dir.dot(MultiVerseBall::Centre() - pos) <= 0) return false; // the original inward test
  if (!fSeen.insert(track->GetTrackID()).second) return false;     // first hit of the track

  G4int primPDG = 0; G4double primE = 0.;
  const PrimaryTruthInfo* truth = dynamic_cast<const PrimaryTruthInfo*>(evt->GetUserInformation());
  if (truth) { primPDG = truth->GetPDG(); primE = truth->GetEnergy(); }
  else if (evt->GetPrimaryVertex(0) && evt->GetPrimaryVertex(0)->GetPrimary(0)) {
    primPDG = evt->GetPrimaryVertex(0)->GetPrimary(0)->GetPDGcode();
    primE = evt->GetPrimaryVertex(0)->GetPrimary(0)->GetKineticEnergy();
  }
  const AncestryInfo* info = dynamic_cast<const AncestryInfo*>(track->GetUserInformation());
  const G4VProcess* cp = track->GetCreatorProcess();

  std::ofstream out("outputLegacyShell.dat", std::ios::app);
  out << evt->GetEventID() << "\t" << name << "\t" << aStep->GetPreStepPoint()->GetKineticEnergy() << "\t"
      << (track->GetParentID() == 0 ? 1 : 0) << "\t" << track->GetTrackID() << "\t" << track->GetParentID() << "\t"
      << (track->GetParentID() == 0 || !cp ? G4String("Primary") : cp->GetProcessName()) << "\t"
      << pos.x() << "\t" << pos.y() << "\t" << pos.z() << "\t" << dir.x() << "\t" << dir.y() << "\t" << dir.z()
      << "\t1\t" << primPDG << "\t" << primE / MeV << "\t" << ((info && info->AncestorInside()) ? 1 : 0)
      << "\t" << track->GetVertexPosition().x() << "\t" << track->GetVertexPosition().y() << "\t"
      << track->GetVertexPosition().z() << "\t" << track->GetLogicalVolumeAtVertex()->GetName() << std::endl;
  return true;
}
