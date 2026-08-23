
#ifndef EventAction_h
#define EventAction_h 1

#include "G4UserEventAction.hh"
#include "G4Timer.hh"
#include "globals.hh"
#include <chrono>


class G4Event;
class RunAction;

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

class EventAction : public G4UserEventAction
{
  public:
    EventAction(RunAction* runAction);
   ~EventAction();

  public:
    void BeginOfEventAction(const G4Event*);
    void EndOfEventAction(const G4Event*);

 private:

   // Hits collection index:
   G4int ACDhitsCollectionIndex;
   G4int DummyhitsCollectionIndex;
   // Timer used to measure the cpu time of events
   G4Timer timerEvent;

   // Total CPU time for all processed events
   G4double cpuTime;
  
   // Timer used to estimate the remaining simulation time
   std::chrono::time_point<std::chrono::system_clock> fStartTime;
   RunAction* fRunAction;
};

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#endif

    
