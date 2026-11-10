#include "EventAction.hh"
#include "MultiVerseBall.hh"
#include "MultiVerseWorld.hh"
#include "RunAction.hh"
#include "ACDSD.hh"
#include "DummySD.hh"
#include "ACDHit.hh"
#include "DummyHit.hh"
#include "ACDHitsCollection.hh"
#include "DummyHitsCollection.hh"
#include "PrimaryTruthInfo.hh"
#include "G4Event.hh"
#include "G4PrimaryVertex.hh"
#include "G4PrimaryParticle.hh"
#include "G4EventManager.hh"
#include "G4RunManager.hh"
#include "G4TrajectoryContainer.hh"
#include "G4Trajectory.hh"
#include "G4ios.hh"
#include "G4SDManager.hh"
#include "G4UnitsTable.hh"
#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"
#include <chrono>
#include <iostream>
#include <set>

using std::cout;
using std::fstream;
using std::ios;
using std::endl;
using std::ofstream;

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

EventAction::EventAction(RunAction* runAction)  :
  ACDhitsCollectionIndex(-1),
  DummyhitsCollectionIndex(-1),
  cpuTime(0),
  fStartTime(std::chrono::system_clock::now()),
  fRunAction(runAction)
{}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

EventAction::~EventAction()
{}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void EventAction::BeginOfEventAction(const G4Event*)
{
  // MultiVerse: empty the list of ball entries of the previous event.
  MultiVerseBall::BeginEvent();
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void EventAction::EndOfEventAction(const G4Event* evt)
{
  G4int evt_id = evt->GetEventID();
  G4int TotalEvents = fRunAction->GetTotalEvents();

  // An exhausted MAGI group file leaves the last event without vertices:
  // nothing was thrown, so nothing is recorded or counted.
  if (evt->GetNumberOfPrimaryVertex() == 0) return;

  // Primary truth (HENABICS): the physical primary this event stands for,
  // appended to every crossing and every deposit row as
  //   PrimaryTrackId  PrimaryPDG  PrimaryEnergy[MeV]
  // Full simulation: the single GPS primary (track 1), key = EventId.
  // MAGI injection: the primary the injected group was sampled from, carried in
  // PrimaryTruthInfo, key = group key. Both write the same layout.
  G4long EventKey = evt_id;
  G4int PrimaryTrackId = 1;
  G4int PrimaryPDG = 0;
  G4double PrimaryEnergy = 0.;
  const PrimaryTruthInfo* truth =
    dynamic_cast<const PrimaryTruthInfo*>(evt->GetUserInformation());
  if (truth) {
    EventKey = truth->GetKey();
    PrimaryTrackId = truth->GetTrackId();
    PrimaryPDG = truth->GetPDG();
    PrimaryEnergy = truth->GetEnergy();
  } else if (evt->GetPrimaryVertex(0)->GetPrimary(0)) {
    const G4PrimaryParticle* primary = evt->GetPrimaryVertex(0)->GetPrimary(0);
    PrimaryPDG = primary->GetPDGcode();
    PrimaryEnergy = primary->GetKineticEnergy();
  }
  G4int NCrossings = 0;   // ingoing, non-neutrino tracks written to outputDSCryoSphere.dat
  G4int NHits = 0;        // rows written to outputGDML.dat

  G4int Pflag =0;
  G4int Pproc =0;G4int AA =0;
    // get number of stored trajectories
    //
    G4TrajectoryContainer* trajectoryContainer = evt->GetTrajectoryContainer();
    G4int n_trajectories = 0;
    if (trajectoryContainer) n_trajectories = trajectoryContainer->entries();

  if(ACDhitsCollectionIndex < 0) {
      ACDhitsCollectionIndex = G4SDManager::GetSDMpointer()-> GetCollectionID("ACDCollection");
  }
  if(DummyhitsCollectionIndex < 0 && getenv("MULTIVERSE_PARALLEL")) {
      DummyhitsCollectionIndex = G4SDManager::GetSDMpointer()-> GetCollectionID("DummyCollection");
  }


  //Now, get the HCofThisEvent: it contains all the hits collections
  //that have been defined (one hit collection may be associated to
  //each detector).
  G4HCofThisEvent* HCE = evt -> GetHCofThisEvent();

  ACDHitsCollection* ACDCollection = 0;
  DummyHitsCollection* DummyCollection = 0;

  if(HCE){
    ACDCollection = (ACDHitsCollection*)(HCE -> GetHC(ACDhitsCollectionIndex));
  }
  // MultiVerse: the entries of the ball are collected by SteppingAction in MultiVerseBall; they
  // are no longer hits of a sensitive detector. (With MULTIVERSE_PARALLEL the parallel-world
  // detector fills the collection "DummyCollection" instead.)
  DummyCollection = MultiVerseBall::Entries();
  if (HCE && DummyhitsCollectionIndex >= 0) {
    DummyHitsCollection* pw = (DummyHitsCollection*)(HCE -> GetHC(DummyhitsCollectionIndex));
    if (pw) DummyCollection = pw;
  }
  G4double totalEdep;




  //Ok, now we have the hit collection at hand. If it is not a NULL pointer,
  //we can have a look at it, and read the information we need.


  //---------------------------------------------------------------------------------
  //------------------------------- ACD SD ------------------------------------------
  //---------------------------------------------------------------------------------

  if(ACDCollection) {
    //read the number of hits contained in the collection
    int numberHits = ACDCollection -> entries();
    G4double StartE = 0;

    //we can loop and get each single hit
    for(int i = 0; i < numberHits ; i++) {
      //retrieve the i-th hit from the collection.
      ACDHit* hit = (*ACDCollection)[i];

      //get the information stored in the hit (position and energy)
      G4ThreeVector Position = hit -> GetPos();
      G4int ParticleId = hit -> GetParticleID();
      G4int DetectorID = hit -> GetDetectorID();
      G4String ParticleName = hit -> GetParticleName();
      G4double EPreStep = hit -> GetKineticEnergyPreStep();
      G4double Edep = hit -> GetEdep();
      G4double GlobalTime = hit -> GetGlobalTime();
      if(i == 0) StartE = hit -> GetVertexKineticEnergy();
      //G4String Process = hit -> GetProcessName();
      
      //Vertex info only for the hits
      /*if(i == 0 && ParticleId==1){
        G4ThreeVector VertexPosition = hit -> GetVertexPosition();
        G4ThreeVector VertexMomentumDirection = hit -> GetVertexMomentumDirection();

        G4String fileVerdat = "vertexinfo.dat";
        ofstream out(fileVerdat, ios::app);
        out << evt_id << "\t" << StartE << "\t" << VertexPosition.getX()/mm << "\t" << VertexPosition.getY()/mm << "\t" 
        << VertexPosition.getZ()/mm << "\t" << VertexMomentumDirection.getX() << "\t" 
        << VertexMomentumDirection.getY() << "\t" << VertexMomentumDirection.getZ() << endl;
        out.close();
      }*/


      G4int ParticleInt = -1;
      if(ParticleName == "mu-"){
        ParticleInt = 1;
      }
      else if (ParticleName == "mu+")
      {
        ParticleInt = 2;
      }
      else if (ParticleName == "e-")
      {
        ParticleInt = 3;
      }
      else if (ParticleName == "e+")
      {
        ParticleInt = 4;
      }
      else if (ParticleName == "proton")
      {
        ParticleInt = 5;
      }
      else if (ParticleName == "gamma")
      {
        ParticleInt = 6;
      }
      else if (ParticleName == "neutron")
      {
        ParticleInt = 7;
      }
      else if (ParticleName == "ion")
      {
        ParticleInt = 8;
      }
      

      //ACD output (last column DetectorID: 1 = DM127, 2 = DM1.2, 3 = DM143)
      G4String filedatCryoAC = "outputGDML.dat";
      ofstream out(filedatCryoAC, ios::app);
      out << EventKey << "\t" << ParticleId
      << "\t" << ParticleInt << "\t" << EPreStep/keV << "\t" << Edep/keV
      << "\t" << Position.getX()/mm << "\t" << Position.getY()/mm
      << "\t" << Position.getZ()/mm << "\t" << StartE/MeV << "\t" << GlobalTime /*<< "\t" << Process*/
      << "\t" << PrimaryTrackId << "\t" << PrimaryPDG << "\t" << PrimaryEnergy/MeV
      << "\t" << DetectorID << endl;   // last column: 1 = DM127, 2 = DM1.2, 3 = DM143
      out.close();
      NHits++;
    }
}

if(DummyCollection) {
  //read the number of hits contained in the collection
  int numberHits = DummyCollection -> entries();
  G4double StartE = 0;

  //Set to track ParticleId already printed for each event
  std::set<G4int> printedParticleIDs;

  //we can loop and get each single hit
  for(int i = 0; i < numberHits ; i++) {
    //retrieve the i-th hit from the collection.
    DummyHit* hit = (*DummyCollection)[i];

    //get the information stored in the hit (position and energy)
    G4ThreeVector Position = hit -> GetPos();
    G4ThreeVector Direction = hit -> GetDirection();
    G4int ParticleId = hit -> GetParticleID();
    G4int ParentParticleId = hit -> GetParentParticleID();
    G4int isPrimary = (ParentParticleId == 0) ? 1 : 0; // 1 = primary (trackID 1 / parentID 0), 0 = secondary
    G4String ParticleName = hit -> GetParticleName();
    G4double EPreStep = hit -> GetKineticEnergyPreStep();
    G4double Edep = hit -> GetEdep();
    G4String DetectorName = hit -> GetVolume();
    G4double GlobalTime = hit -> GetGlobalTime();
    if(i == 0) StartE = hit -> GetVertexKineticEnergy();
    //G4String Process = hit -> GetProcessName();

    //Vertex info only for the hits
    /*if(i == 0 && ParticleId==1){
      G4ThreeVector VertexPosition = hit -> GetVertexPosition();
      G4ThreeVector VertexMomentumDirection = hit -> GetVertexMomentumDirection();

      G4String fileVerdat = "vertexinfo.dat";
      ofstream out(fileVerdat, ios::app);
      out << evt_id << "\t" << StartE << "\t" << VertexPosition.getX()/mm << "\t" << VertexPosition.getY()/mm << "\t"
      << VertexPosition.getZ()/mm << "\t" << VertexMomentumDirection.getX() << "\t"
      << VertexMomentumDirection.getY() << "\t" << VertexMomentumDirection.getZ() << endl;
      out.close();
    }*/

    //Comment next lines if not using a DummySystem

    /*if (i == 0){ //Taking only the first track for each event
      G4String filedatDummySystem = "outputDummySystem.dat";
      ofstream out(filedatDummySystem, ios::app);
      out << evt_id << "\t" << ParticleName << "\t" << EPreStep << "\t" << StartE << "\t" << GlobalTime << endl;
    }*/

    // Tracking for the DSCryoSphere, where I need to output all the ingoing particles, each just the first time

    // Skip (anti)neutrinos: they don't interact with the detector, so they can't
    // produce a signal and would only bias the event/particle counts in the output file
    if (ParticleName == "nu_e"   || ParticleName == "anti_nu_e"  ||
        ParticleName == "nu_mu"  || ParticleName == "anti_nu_mu" ||
        ParticleName == "nu_tau" || ParticleName == "anti_nu_tau") continue;

    // MultiVerse (Track G): every hit of the ball collection is an ENTRY (MultiVerseSD), so it is
    // inward by construction and the direction test of the original code is not needed. It is kept
    // as a check: the ball is a solid, so (position - centre) . direction must be negative at the
    // entry point, and a violation is counted and reported at the end of the run (NOutwardEntries).
    G4ThreeVector toCenter = MultiVerseBall::Centre() - Position;
    if(Direction.dot(toCenter) <= 0) {
      G4cout << "[MultiVerse] WARNING: entry with non-inward direction, event " << EventKey
             << ", track " << ParticleId << G4endl;
      ++fNOutwardEntries;
    }
    const G4bool AncestorInside = hit -> GetAncestorInside();

    if (printedParticleIDs.find(ParticleId) == printedParticleIDs.end()) {
            // Print data for each particle
            G4String filedatDummySystem = "outputDSCryoSphere.dat";
            std::ofstream out(filedatDummySystem, std::ios::app);
            out << EventKey << "\t"
                << ParticleName << "\t"
                << EPreStep << "\t"
                << isPrimary << "\t"   // PrimBool: 1 = primary, 0 = secondary (for flux normalization)
                << ParticleId << "\t"
                << ParentParticleId << "\t"
                << hit -> GetCreatorProcessName() << "\t"   // e.g. "Primary", "RadioactiveDecay", "compt", "eBrem"
                << Position.getX() << "\t"
                << Position.getY() << "\t"
                << Position.getZ() << "\t"
                << Direction.getX() << "\t"
                << Direction.getY() << "\t"
                << Direction.getZ() << "\t"
                << PrimaryTrackId << "\t"
                << PrimaryPDG << "\t"
                << PrimaryEnergy/MeV << "\t"
                << (AncestorInside ? 1 : 0)      // MultiVerse: 0 = first entry, 1 = an ancestor was inside before
                << std::endl;
            out.close();

            // Update the particleIds of this event
            printedParticleIDs.insert(ParticleId);
            NCrossings++;
        }

    }
} else {
         G4cout << "ACDCollection and DummyCollection are null" <<G4endl;
}

  // MultiVerse (Track G, G3): the interior deposit of the event, one row per event with any:
  //   EventKey  PrimaryPDG  PrimaryEnergy[MeV]  total[MeV]  nVolumes  (volume  deposit[MeV])...
  if (MultiVerseBall::InteriorTotal() > 0.) {
    std::ofstream outI("outputInterior.dat", std::ios::app);
    outI << EventKey << "\t" << PrimaryPDG << "\t" << PrimaryEnergy/MeV << "\t"
         << MultiVerseBall::InteriorTotal()/MeV << "\t" << MultiVerseBall::InteriorByVolume().size();
    for (const auto& kv : MultiVerseBall::InteriorByVolume()) outI << "\t" << kv.first << "\t" << kv.second/MeV;
    outI << std::endl;
    outI.close();
  }

  // Per-primary table: one row per primary that left any record, joined to the
  // crossing and deposit files on EventKey. NCrossings is the shower
  // multiplicity k. Primaries with no record are counted only in
  // run_summary.json, which is where P(k=0) and the flux normalization come from.
  G4bool recorded = (NCrossings > 0 || NHits > 0);
  if (recorded) {
    std::ofstream out("primaries.dat", std::ios::app);
    out << EventKey << "\t" << PrimaryTrackId << "\t" << PrimaryPDG << "\t"
        << PrimaryEnergy/MeV << "\t" << NCrossings << "\t" << NHits << std::endl;
    out.close();
  }
  fRunAction->CountThrownPrimary(PrimaryPDG, PrimaryEnergy, recorded);




  // periodic printing
  //

  if (evt_id == 0) {
        fStartTime = std::chrono::system_clock::now();
    }

    if (evt_id % 500 == 0) {
        auto now = std::chrono::system_clock::now();
        std::chrono::duration<G4double> elapsedSeconds = now - fStartTime;
        G4double eventsPerSecond = evt_id / elapsedSeconds.count();
        G4double remainingEvents = TotalEvents - evt_id;
        G4double remainingTime = remainingEvents / eventsPerSecond;

        G4int hours = static_cast<G4int>(remainingTime) / 3600;
        G4int minutes = (static_cast<G4int>(remainingTime) % 3600) / 60;
        G4int seconds = static_cast<G4int>(remainingTime) % 60;

        G4int progress = static_cast<G4int>((G4double(evt_id) / TotalEvents) * 100);
        G4int barWidth = 50; // Width of the progress bar
        G4int pos = (progress * barWidth) / 100;

        // Print the progress bar
        G4cout << "\rSimulation Progress: " << progress << "% [";
        for (G4int i = 0; i < barWidth; ++i) {
            if (i < pos) G4cout << "#";
            else if (i == pos) G4cout << ">";
            else G4cout << "-";
        }
        G4cout << "] Estimated Time Left: "
                  << std::setw(2) << std::setfill('0') << hours << ":"
                  << std::setw(2) << std::setfill('0') << minutes << ":"
                  << std::setw(2) << std::setfill('0') << seconds << " (hh:mm:ss)" << std::flush;
    }

    if (evt_id == TotalEvents - 1) {
        G4cout << G4endl; // To move to the next line after completion
    }

  /*if (evt_id < 100 || evt_id%10000 == 0) 
  {
    G4cout << ">>> Event " << evt->GetEventID() << G4endl;
    //  G4cout << "    " << n_trajectories
    //	   << " trajectories stored in this evt." << G4endl;
  }*/
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......<< "	
