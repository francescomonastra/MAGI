// MultiVerseBall.cc  --  see include/MultiVerseBall.hh and MULTIVERSE.md
#include "MultiVerseBall.hh"
#include "DummyHit.hh"

#include "G4UIdirectory.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWith3VectorAndUnit.hh"
#include "G4UImessenger.hh"
#include <cmath>

G4double      MultiVerseBall::fRadius  = 95.0 * mm;
G4ThreeVector MultiVerseBall::fCentre  = G4ThreeVector(0., 0., 5. * mm);
DummyHitsCollection* MultiVerseBall::fEntries = 0;
G4double MultiVerseBall::fInteriorTotal = 0.;
std::map<G4String, G4double> MultiVerseBall::fInteriorByVolume;

namespace {
class BallMessenger : public G4UImessenger
{
  public:
    BallMessenger()
    {
      fDir = new G4UIdirectory("/MultiVerse/");
      fDir->SetGuidance("Track G: the MAGI tracking ball (a virtual surface).");

      fRadiusCmd = new G4UIcmdWithADoubleAndUnit("/MultiVerse/radius", this);
      fRadiusCmd->SetGuidance("Radius of the tracking ball.");
      fRadiusCmd->SetParameterName("radius", false);
      fRadiusCmd->SetRange("radius>0.");
      fRadiusCmd->SetDefaultUnit("mm");
      fRadiusCmd->AvailableForStates(G4State_PreInit, G4State_Idle);

      fCentreCmd = new G4UIcmdWith3VectorAndUnit("/MultiVerse/centre", this);
      fCentreCmd->SetGuidance("Centre of the tracking ball.");
      fCentreCmd->SetParameterName("x", "y", "z", false);
      fCentreCmd->SetDefaultUnit("mm");
      fCentreCmd->AvailableForStates(G4State_PreInit, G4State_Idle);
    }
    void SetNewValue(G4UIcommand* cmd, G4String value)
    {
      if (cmd == fRadiusCmd) MultiVerseBall::SetRadius(fRadiusCmd->GetNewDoubleValue(value));
      if (cmd == fCentreCmd) MultiVerseBall::SetCentre(fCentreCmd->GetNew3VectorValue(value));
    }
  private:
    G4UIdirectory*              fDir;
    G4UIcmdWithADoubleAndUnit*  fRadiusCmd;
    G4UIcmdWith3VectorAndUnit*  fCentreCmd;
};
}

void MultiVerseBall::CreateMessenger()
{
  static BallMessenger* messenger = new BallMessenger();
  (void)messenger;
}

// Ray-sphere intersection of the straight segment p0 -> p1. With d = p1 - p0 and u = p0 - c the
// points p0 + t d are on the surface for a t^2 + 2 b t + k = 0, a = d.d, b = u.d, k = u.u - R^2.
// The smaller root is where the segment goes from outside to inside.
G4bool MultiVerseBall::SegmentEntry(const G4ThreeVector& p0, const G4ThreeVector& p1, G4double& t)
{
  const G4ThreeVector u = p0 - fCentre;
  const G4ThreeVector d = p1 - p0;
  const G4double a = d.mag2();
  if (a <= 0.) return false;
  const G4double r0 = u.mag();
  if (r0 < fRadius - SurfaceTolerance()) return false;    // p0 is inside: not an entry
  const G4double b = u.dot(d);
  if (r0 <= fRadius + SurfaceTolerance()) {                // p0 on the surface: an entry iff it points inward
    if (b < 0.) { t = 0.; return true; }
    return false;
  }
  const G4double k = u.mag2() - fRadius * fRadius;
  const G4double disc = b * b - a * k;
  if (disc <= 0.) return false;                            // the line misses the ball
  const G4double t1 = (-b - std::sqrt(disc)) / a;          // smaller root: entry
  if (t1 < 0. || t1 > 1.) {
    // p0 exactly on the surface and moving in: t1 is a rounding error around 0
    if (t1 > -1.e-12 && t1 < 0. && b < 0.) { t = 0.; return true; }
    return false;
  }
  if (b >= 0. && k > 0.) return false;                     // moving away from a point outside
  t = t1;
  return true;
}

void MultiVerseBall::AddInterior(const G4String& volume, G4double edep)
{
  if (edep < 1.e-9 * MeV) return;   // vacuum dust (1e-24 MeV in G4_Galactic) is not a deposit
  fInteriorTotal += edep;
  fInteriorByVolume[volume] += edep;
}

void MultiVerseBall::BeginEvent()
{
  fInteriorTotal = 0.;
  fInteriorByVolume.clear();
  delete fEntries;
  fEntries = new DummyHitsCollection("MultiVerse", "entries");
}

void MultiVerseBall::AddEntry(DummyHit* hit)
{
  if (!fEntries) BeginEvent();
  fEntries->insert(hit);
}

DummyHitsCollection* MultiVerseBall::Entries() { return fEntries; }
