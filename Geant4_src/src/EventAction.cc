#include "EventAction.hh"
#include "RunAction.hh"
#include "ACDSD.hh"
#include "DummySD.hh"
#include "ACDHit.hh"
#include "DummyHit.hh"
#include "ACDHitsCollection.hh"
#include "DummyHitsCollection.hh"
#include "G4Event.hh"
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
{}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void EventAction::EndOfEventAction(const G4Event* evt)
{
  G4int evt_id = evt->GetEventID();
  G4int TotalEvents = fRunAction->GetTotalEvents();

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
  if(DummyhitsCollectionIndex < 0) {
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
    DummyCollection = (DummyHitsCollection*)(HCE -> GetHC(DummyhitsCollectionIndex));
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
      

      G4int DetectorInt = -1;
      if (DetectorName.find("SensitiveDetector") != std::string::npos)
      {
        DetectorInt = 1;
      } else if (DetectorName.find("BackCryoAC") != std::string::npos)
      {
        DetectorInt = 2;
      } else if (DetectorName.find("LeftCryoAC") != std::string::npos)
      {
        DetectorInt = 3;
      } else if (DetectorName.find("RightCryoAC") != std::string::npos)
      {
        DetectorInt = 4;
      } else if (DetectorName.find("TopCryoAC") != std::string::npos)
      {
        DetectorInt = 5;
      } else if (DetectorName.find("BottomCryoAC") != std::string::npos)
      {
        DetectorInt = 6;
      }
      
      
      //if (DetectorInt == 1)
      //{
        //ACD output
        G4String filedatCryoAC = "outputGDML.dat";
        ofstream out(filedatCryoAC, ios::app);
        out << evt_id << "\t" << DetectorInt << "\t" << ParticleId 
        << "\t" << ParticleInt << "\t" << EPreStep/keV << "\t" << Edep/keV 
        << "\t" << Position.getX()/mm << "\t" << Position.getY()/mm
        << "\t" << Position.getZ()/mm << "\t" << StartE/MeV << "\t" << GlobalTime /*<< "\t" << Process*/ <<endl;
        out.close();
      //}
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

    //______________-----------^^^^^^^^^^-----------____________
    // Tracking for the DSCryoSphere, where I need to output all the ingoing particles, each just the first time

    // Skip (anti)neutrinos: they don't interact with the detector, so they can't
    // produce a signal and would only bias the event/particle counts in the output file
    if (ParticleName == "nu_e"   || ParticleName == "anti_nu_e"  ||
        ParticleName == "nu_mu"  || ParticleName == "anti_nu_mu" ||
        ParticleName == "nu_tau" || ParticleName == "anti_nu_tau") continue;

    //Check if the particle is ingoing in the DSCryoSphere
    // FIXED 15/08/2026: was -3007.6*mm, a 2500 mm error. The DSCryoSphere is
    // placed at DSCryoSphereZPosition = -507.66 mm (GDMLDetectorConstruction),
    // confirmed by the recorded crossings themselves: R = 99.12 +/- 0.29 about
    // (0,0,-507.66). With the centre 2500 mm away, toCenter was dominated by
    // -z and the test below degenerated into "is v_z < 0?" (98.4% agreement),
    // so it (a) admitted 10.4% genuinely OUTGOING crossings and (b) silently
    // discarded genuinely ingoing UPWARD-going ones at write time. (b) cannot
    // be undone by filtering the output: those crossings were never written.
    G4ThreeVector SphereCenter(0., 0., -507.66*mm);
    G4ThreeVector toCenter = SphereCenter - Position;

    //If the particle has direction angle respect to the sphere center less than 90° then it's outgoing, therefore "continue" skips to the next iteration
    if(Direction.dot(toCenter) <= 0) continue;

    if (printedParticleIDs.find(ParticleId) == printedParticleIDs.end()) {
            // Print data for each particle
            G4String filedatDummySystem = "outputDSCryoSphere.dat";
            std::ofstream out(filedatDummySystem, std::ios::app);
            out << evt_id << "\t"  // 👈 Aggiunto anche l'event ID per chiarezza
                << ParticleName << "\t"
                << EPreStep << "\t"
                << isPrimary << "\t"   // 👈 PrimBool: 1 = primary, 0 = secondary (for flux normalization)
                << ParticleId << "\t"
                << ParentParticleId << "\t"
                << hit -> GetCreatorProcessName() << "\t"   // e.g. "Primary", "RadioactiveDecay", "compt", "eBrem"
                << Position.getX() << "\t"
                << Position.getY() << "\t"
                << Position.getZ() << "\t"
                << Direction.getX() << "\t"
                << Direction.getY() << "\t"
                << Direction.getZ() << std::endl;
            out.close();

            // Update the particleIds of this event
            printedParticleIDs.insert(ParticleId);
        }
    
  }  

} else {
         G4cout << "ACDCollection and DummyCollection are null" <<G4endl;
}




  // periodic printing
  //

  if (evt_id == 0) {
        fStartTime = std::chrono::system_clock::now();
    }

    if (evt_id % 5000 == 0) {
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
