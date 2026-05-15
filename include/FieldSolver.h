#ifndef __GENESIS_FIELDSOLVER__
#define __GENESIS_FIELDSOLVER__

#include <vector>
#include <iostream>
#include <string>
#include <complex>
#include <cstddef>


class Field;
class Beam;
#ifdef GENESIS_CUDA
namespace genesis_cuda { struct State; }
#endif

#ifdef GENESIS_CUDA
namespace genesis_cuda { struct BeamState; }
#endif

#include "Particle.h"
#include "Undulator.h"


using namespace std;


class FieldSolver{
 public:
    virtual ~FieldSolver() {};
    virtual void init(double,double,double,unsigned int) = 0;
    virtual void advance(double, Field *, Beam *, Undulator *) = 0;
    virtual void initSourceFilter(double,double,double,bool) = 0;
#ifdef GENESIS_CUDA
    virtual bool syncCudaFieldToHost(Field *) { return true; }
    virtual void markCudaHostFieldDirty() {}
    virtual bool copyCudaFieldToBeam(genesis_cuda::BeamState *, const Field *, std::size_t) { return false; }
    virtual bool canBuildSourceFromCudaBeam() const { return false; }
    virtual bool applyCudaSlippage(Field *, int) { return false; }
    virtual bool downloadCudaSlippageSlice(Field *, int, double *, std::size_t) { return false; }
    virtual bool applyCudaSlippageBoundary(Field *, int, const double *, std::size_t, bool) { return false; }
#endif
};


#endif
