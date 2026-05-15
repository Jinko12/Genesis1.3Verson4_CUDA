#include <sstream>
#include <climits>
#include <cstdlib>
#include "Control.h"
#include "GenesisNvtx.h"
#include "writeFieldHDF5.h"
#include "writeBeamHDF5.h"





#ifdef GENESIS_CUDA
namespace {
bool cudaMpiSlippageEnabled() {
  const char *env = std::getenv("GENESIS_CUDA_MPI_SLIPPAGE");
  return !((env != nullptr) && (env[0] == '0') && (env[1] == '\0'));
}
}
#endif

Control::Control()
{
  nwork=0;
  work=NULL;
}


Control::~Control()
{
  delete[] work;
}


bool Control::applyMarker(Beam *beam, vector<Field*>*field, Undulator *und, bool& error_IO)
{
  GENESIS_NVTX_RANGE("control.apply_marker");
  error_IO = false; /* error_IO==true signals error during a requested dump */

  bool sort=false;
  int marker=und->getMarker();

  // possible file names contain number of current integration step
  stringstream sroot;
  string basename;
  int istepz=und->getStep();
  sroot << "." << istepz;
  basename=root+sroot.str();


  if ((marker & 1) != 0){
    GENESIS_NVTX_RANGE("control.marker.field_dump");
#ifdef GENESIS_CUDA
    for (auto *fld : *field) {
      if ((fld != nullptr) && !fld->syncCudaFieldToHost()) {
        error_IO = true;
        if(rank==0) {
          cout << "   failed to synchronize CUDA field state before writing field dump" << endl;
        }
        return sort;
      }
    }
#endif
    WriteFieldHDF5 dump;
    if(dump.write(basename,field))
    {
      /* register field dump => it will be reported in list of dumps generated during current "&track" command */
      string fn;
      fn = basename + ".fld.h5"; /* file extension as added in WriteFieldHDF5::write (TODO: need to implement proper handling of harmonic field dumping) */
      und->fielddumps_filename.push_back(fn);
      und->fielddumps_intstep.push_back(istepz);
    } else {
      /* IO error: do not add filename to list of dumps */
      error_IO = true;
      if(rank==0) {
        cout << "   write operation was not successful!" << endl;
      }
    }
  }
  
  if ((marker & 2) != 0){
    GENESIS_NVTX_RANGE("control.marker.beam_dump");
#ifdef GENESIS_CUDA
    beam->syncCudaTrackingToHost();
#endif
    WriteBeamHDF5 dump;
    if(dump.write(basename,beam,1))   // use stride of 1 -> all particles are dump
    {
      /* register beam dump => it will be reported in list of dumps generated during current "&track" command */
      string fn;
      fn = basename + ".par.h5"; /* file extension as added in WriteBeamHDF5::write */
      und->beamdumps_filename.push_back(fn);
      und->beamdumps_intstep.push_back(istepz);
    } else {
      /* IO error: do not add filename to list of dumps */
      error_IO = true;
      if(rank==0) {
        cout << "   write operation was not successful!" << endl;
      }
    }
  }
  
  if ((marker & 4) != 0){
    sort=true;   // sorting is deferred after the particles have been pushed by Runge-Kutta
  }

  // bit value 8 is checked in und->advance()

  return sort;
}


#if 0 // .out.h5 file is now written in class Diagnostic
void Control::output(Beam *beam, vector<Field*> *field, Undulator *und, Diagnostic &diag)
{
  Output *out=new Output;

  string file=root.append(".out.h5");
  out->open(file,noffset,nslice);
  
  out->writeGlobal(und,und->getGammaRef(),reflen,sample,slen,one4one,timerun,scanrun,ntotal);
  out->writeLattice(beam,und);

  for (unsigned int i=0; i<field->size();i++){
        out->writeFieldBuffer(field->at(i));
  }
  out->writeBeamBuffer(beam);

  out->close();
 
  delete out;
  return;
}
#endif


bool Control::init(int inrank, int insize, const string in_rootname, Beam *beam, vector<Field*> *field, Undulator *und, bool inTime, bool inScan)
{
  GENESIS_NVTX_RANGE("control.init");
  rank=inrank;
  size=insize;
  root = in_rootname;

  one4one=beam->one4one;
  reflen=beam->reflength;
  sample=beam->slicelength/reflen;

  timerun=inTime;
  scanrun=inScan;
 
 

  // cross check simulation size

  nslice=beam->beam.size();
  noffset=rank*nslice;
  ntotal=size*nslice;  // all cores have the same amount of slices

  slen=ntotal*sample*reflen;


  if (rank==0){
    if(scanrun) { 
       cout << "Scan run with " << ntotal << " slices" << endl; 
    } else {
       if(timerun) { 
         cout << "Time-dependent run with " << ntotal << " slices" << " for a time window of " << slen*1e6 << " microns" << endl; 
       } else { 
         cout << "Steady-state run" << endl;
       }
    }
  }

  for (auto & fld : *field){
      fld->resetSlippage();
  }

  beam->checkBeforeTracking();
  return true;  
}



void Control::applySlippage(double slippage, Field *field)
{
  GENESIS_NVTX_RANGE("control.apply_slippage");
  if (timerun==false) { return; }

 
  // update accumulated slippage
  field->accuslip+=slippage;


  // number of grid points in field supplied by caller
  long long ncells = field->ngrid*field->ngrid;

  // If needed, allocate working space for MPI data transfer.  Host slippage
  // needs one complex slice (2*ncells doubles).  CUDA-resident MPI slippage
  // stages one outgoing and one incoming slice, so reserve two complex slices.
  // The size of the buffer is determined by the largest field seen so far
  // (relevant when there are multiple fields of different grid sizes).
  std::size_t requiredWork = static_cast<std::size_t>(2) * static_cast<std::size_t>(ncells);
#ifdef GENESIS_CUDA
  requiredWork *= 2;
#endif
  if(nwork < requiredWork){
    delete[] work;
    nwork = requiredWork; // one complex number <=> 2 doubles
    work=new double [nwork];
  } 
  

  // following routine is applied if the required slippage is alrger than 80% of the sampling size

  int direction=1;
  

  bool hostFieldRecordChanged = false;
  bool cudaFieldSynced = false;
  while(abs(field->accuslip)>(sample*0.8)){
      // check for anormal direction of slippage (backwards slippage)
      direction = 1;
      if (field->accuslip<0) {direction=-1;} 

      field->accuslip-=sample*direction; 

#ifdef GENESIS_CUDA
      // In single-rank runs there is no MPI exchange; slippage is just a
      // record shift with a zero-filled boundary slice.  Keep this operation
      // on the CUDA FFT field buffer so diagnostics can remain GPU resident.
      if (size == 1) {
        GENESIS_NVTX_RANGE("control.slippage.cuda_single_rank_try");
        if (field->applyCudaSlippage(direction)) {
          continue;
        }
      }

      // In multi-rank runs only one boundary slice has to cross the MPI
      // boundary.  Avoid synchronizing the complete field record: download the
      // outgoing CUDA slice, exchange it with the neighbor, and inject the
      // received boundary slice back into the CUDA FFT field buffer.
      if ((size > 1) && cudaMpiSlippageEnabled()) {
        GENESIS_NVTX_RANGE("control.slippage.cuda_mpi_boundary_try");
        if(2*ncells > INT_MAX) {
          if(rank==0) {
            cout << "Large field mesh size results in request for MPI transfer size exceeding INT_MAX, exiting." << endl;
          }
          MPI_Abort(MPI_COMM_WORLD,1);
        }

        double *sendWork = work;
        double *recvWork = work + static_cast<std::size_t>(2) * static_cast<std::size_t>(ncells);
        const std::size_t sliceDoubles = static_cast<std::size_t>(2) * static_cast<std::size_t>(ncells);
        if (field->downloadCudaSlippageSlice(direction, sendWork, sliceDoubles)) {
          int rank_next_cuda=rank+1;
          int rank_prev_cuda=rank-1;
          if (rank_next_cuda >= size ) { rank_next_cuda=0; }
          if (rank_prev_cuda < 0 ) { rank_prev_cuda = size-1; }
          if (direction<0) {
            int tmp=rank_next_cuda;
            rank_next_cuda=rank_prev_cuda;
            rank_prev_cuda=tmp;
          }

          MPI_Status cuda_status;
          const int cuda_tag=1;
          {
            GENESIS_NVTX_RANGE("control.slippage.mpi_exchange_boundary_slice");
          if ( (rank % 2)==0 ){
            MPI_Send(sendWork,2*ncells,MPI_DOUBLE,rank_next_cuda,cuda_tag,MPI_COMM_WORLD);
            MPI_Recv(recvWork,2*ncells,MPI_DOUBLE,rank_prev_cuda,cuda_tag,MPI_COMM_WORLD,&cuda_status);
          } else {
            MPI_Recv(recvWork,2*ncells,MPI_DOUBLE,rank_prev_cuda,cuda_tag,MPI_COMM_WORLD,&cuda_status);
            MPI_Send(sendWork,2*ncells,MPI_DOUBLE,rank_next_cuda,cuda_tag,MPI_COMM_WORLD);
          }
          }

          const bool zeroBoundary = ((rank==0) && (direction >0)) ||
                                    ((rank==(size-1)) && (direction <0));
          if (!field->applyCudaSlippageBoundary(direction, recvWork, sliceDoubles, zeroBoundary)) {
            if(rank==0) {
              cout << "   failed to apply CUDA-resident MPI field slippage" << endl;
            }
            MPI_Abort(MPI_COMM_WORLD,1);
          }
          continue;
        }
      }

      if (!cudaFieldSynced) {
        if (!field->syncCudaFieldToHost()) {
          if(rank==0) {
            cout << "   failed to synchronize CUDA field state before slippage" << endl;
          }
          MPI_Abort(MPI_COMM_WORLD,1);
        }
        cudaFieldSynced = true;
      }
#endif
      hostFieldRecordChanged = true;
      GENESIS_NVTX_RANGE("control.slippage.host_fallback");

      // get adjacent node before and after in chain
      int rank_next=rank+1;
      int rank_prev=rank-1;
      if (rank_next >= size ) { rank_next=0; }
      if (rank_prev < 0 ) { rank_prev = size-1; }	

      // for inverse direction swap targets
      if (direction<0) {
	int tmp=rank_next;
        rank_next=rank_prev;
        rank_prev=tmp; 
      }

      int tag=1;
   
      // get slice which is transmitted
      int last=(field->first+field->field.size()-1)  %  field->field.size();
      // get first slice for inverse direction
      if (direction<0){
	last=(last+1) % field->field.size();  //  this actually first because it is sent backwards
      }

      // Prevent transfer sizes resulting in overflow (MPI_send argument 'count' has data type 'int').
      // For typical transverse grid sizes, this is not a relevant limitation.
      // (All MPI processes have identical transverse field parameters.)
      if(2*ncells > INT_MAX) {
        if(rank==0) {
          cout << "Large field mesh size results in request for MPI transfer size exceeding INT_MAX, exiting." << endl;
        }
        MPI_Abort(MPI_COMM_WORLD,1);
      }

      MPI_Status status;
      //      MPI_Errhandler_set(MPI_COMM_WORLD,MPI_ERRORS_RETURN);
      //      int ierr;
	
      if (size>1){
        if ( (rank % 2)==0 ){                   // even nodes are sending first and then receiving field
           for (int i=0; i<ncells; i++){
	     work[2*i]  =field->field[last].at(i).real();
	     work[2*i+1]=field->field[last].at(i).imag();
	   }
	   MPI_Send(work,2*ncells, /* <= number of DOUBLES */
               MPI_DOUBLE,rank_next,tag,MPI_COMM_WORLD);
	   MPI_Recv(work,2*ncells, MPI_DOUBLE,rank_prev,tag,MPI_COMM_WORLD,&status);
	   for (int i=0; i<ncells; i++){
	     complex <double> ctemp=complex<double> (work[2*i],work[2*i+1]);
	     field->field[last].at(i)=ctemp;
	   }
	} else {                               // odd nodes are receiving first and then sending

	  MPI_Recv(work,2*ncells, /* <= number of DOUBLES */
              MPI_DOUBLE,rank_prev,tag,MPI_COMM_WORLD,&status);

	  for (int i=0; i<ncells; i++){
	    complex <double> ctemp=complex<double> (work[2*i],work[2*i+1]);
	    work[2*i]  =field->field[last].at(i).real();
	    work[2*i+1]=field->field[last].at(i).imag();
	    field->field[last].at(i)=ctemp;
	  }
	  MPI_Send(work,2*ncells,MPI_DOUBLE,rank_next,tag,MPI_COMM_WORLD);
	}
      }

      // first node has empty field slipped into the time window
      if ((rank==0) && (direction >0)){
        for (int i=0; i<ncells; i++){
	  field->field[last].at(i)=complex<double> (0,0);
        }
      }

      if ((rank==(size-1)) && (direction <0)){
        for (int i=0; i<ncells; i++){
	  field->field[last].at(i)=complex<double> (0,0);
        }
      }

      // last was the last slice to be transmitted to the succeding node and then filled with the 
      // the field from the preceeding node, making it now the start of the field record.
      field->first=last;
      if (direction<0){
	field->first=(last+1) % field->field.size();
      }
  }

#ifdef GENESIS_CUDA
  if (hostFieldRecordChanged) {
    field->markCudaHostFieldDirty();
  }
#endif
}
