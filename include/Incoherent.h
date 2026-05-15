#ifndef __GENESIS_INCOHERENT__
#define __GENESIS_INCOHERENT__

#include <vector>
#include <iostream>
#include <string>
#include <complex>
#include <math.h>

#include "Undulator.h"
#include "Particle.h"
#include "Sequence.h"
#include "RandomU.h"

class Beam;

using namespace std;

extern const double vacimp;
extern const double eev;



class Incoherent{
 public:
   Incoherent();
   virtual ~Incoherent();
   void init(int, int,bool,bool);
   void apply(Beam *,Undulator *und, double );
   bool isActive(Undulator *und) const;
   [[nodiscard]] bool isEnabled() const;

 private:
   bool doLoss,doSpread;
   RandomU *sran;
};

inline bool Incoherent::isEnabled() const {
  return doLoss || doSpread;
}

inline bool Incoherent::isActive(Undulator *und) const {
  return (und != nullptr) && und->inUndulator() && isEnabled();
}

#endif
