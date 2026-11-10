#ifndef DummyHit_h
#define DummyHit_h 1

#include "G4VHit.hh"
#include "G4THitsCollection.hh"
#include "G4Allocator.hh"
#include "G4ThreeVector.hh"

using std::cout;
using std::fstream;
using std::ios;
using std::endl;
using std::ofstream;

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

class DummyHit : public G4VHit
{
  public:

      DummyHit();
     ~DummyHit();
      DummyHit(const DummyHit&);
      const DummyHit& operator=(const DummyHit&);
      G4int operator==(const DummyHit&) const;

      inline void* operator new(size_t);
      inline void  operator delete(void*);

      void Draw();
      void Print();

  public:
  
    void SetTrackID  (G4int track)      { trackID = track; };
    // void SetChamberNb(G4int chamb)      { chamberNb = chamb; };  
    void SetEdep     (G4double de)      { edep = de; };
    void SetPos      (G4ThreeVector xyz){ pos = xyz; };
    void SetDirection (G4ThreeVector Vxyz){ dir = Vxyz;  };

    G4int GetTrackID()    { return trackID; };
    //G4int GetChamberNb()  { return chamberNb; };
    G4double GetEdep()    { return edep; };
    G4ThreeVector GetPos(){ return pos; };
    G4ThreeVector GetDirection(){ return dir; };
      
    //functions that take info on the particle
    inline void SetParticleName(G4String name) { ParticleName=name;}
    inline void SetParticleID(G4int ID) {ParticleID=ID;}
    inline void SetParentParticleID(G4int ParentID) {ParentParticleID=ParentID;}
    inline G4String GetParticleName() {return ParticleName;}
    inline G4int GetParticleID() {return ParticleID;}
    inline G4int GetParentParticleID() {return ParentParticleID;}

    inline void SetProcessName(G4String nome) { ProcessName=nome;}
    inline G4String GetProcessName() {return ProcessName;}

    inline void SetKineticEnergy(G4double KineticEnergy) { KinE=KineticEnergy;}
    inline G4double GetKineticEnergy() {return KinE;}

    inline void SetVertexKineticEnergy(G4double VertexKineticEnergy) { VKinE=VertexKineticEnergy;}
    inline G4double GetVertexKineticEnergy() {return VKinE;}

    inline void SetKineticEnergyPreStep(G4double KineticEnergyPreStep) { KinEPreStep=KineticEnergyPreStep;}
    inline G4double GetKineticEnergyPreStep() {return KinEPreStep;}

    // MultiVerse (Track G): the track, or an ancestor, was inside the ball before this entry
    inline void SetAncestorInside(G4bool v) { AncInside=v;}
    inline G4bool GetAncestorInside() {return AncInside;}

    inline void SetCreatorProcessName(G4String Cnome) { CProcessName=Cnome;}
    inline G4String GetCreatorProcessName() {return CProcessName;}

    inline void SetLogicalVolumeAtVertexName(G4String VertexVolume) { VerVol=VertexVolume;}
    inline G4String GetLogicalVolumeAtVertexName() {return VerVol;}

    inline void SetVolumeName(G4String Volume) { Vol=Volume;}
    inline G4String GetVolume() {return Vol;}

    inline void SetGlobalTime(G4double GlobalTime) { Gtime=GlobalTime;}
    inline G4double GetGlobalTime() {return Gtime;}

    //adding Vertex Position and Vertex Momentum Direction

    inline void SetVertexPosition(G4ThreeVector VertexPosition) {VPos=VertexPosition;}
    inline G4ThreeVector GetVertexPosition() {return VPos;}

    inline void SetVertexMomentumDirection(G4ThreeVector VertexMomentumDirection) {VMom=VertexMomentumDirection;}
    inline G4ThreeVector GetVertexMomentumDirection() {return VMom;}

    //functions that take info on the pixel

    inline void SetChamberNb (G4int ChamberNb) { num=ChamberNb;}
    inline G4int GetChamberNb () { return num;}
    inline void SetMotherChamberNb (G4int MotherChamberNb) { MotherNum=MotherChamberNb;}
    inline G4int GetMotherChamberNb () { return MotherNum;}


  private:
    
    G4int         trackID;
    //G4int         chamberNb;
    G4double      edep;
    G4ThreeVector pos;
    G4ThreeVector dir;

    //info on the particle
    G4String ParticleName;
    G4int ParticleID;
    G4int ParentParticleID;
    G4String ProcessName;
    G4String CProcessName;
    G4bool AncInside = false;
    G4double VKinE;
    G4double KinE;
    G4double KinEPreStep;
    G4String VerVol;
    G4String Vol;
    G4double Gtime;
    G4ThreeVector VPos;
    G4ThreeVector VMom;

    //info on the pxl
    G4int num;
    G4int MotherNum;

};

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

typedef G4THitsCollection<DummyHit> DummyHitsCollection;

extern G4Allocator<DummyHit> DummyHitAllocator;

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

inline void* DummyHit::operator new(size_t)
{
  void *aHit;
  aHit = (void *) DummyHitAllocator.MallocSingle();
  return aHit;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

inline void DummyHit::operator delete(void *aHit)
{
  DummyHitAllocator.FreeSingle((DummyHit*) aHit);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#endif
