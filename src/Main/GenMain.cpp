#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <iomanip>
#include <stdio.h>
#include <cstring>
#include <ctime>
#include <algorithm>


#include <fenv.h>
#include <signal.h>

#include <mpi.h>



// genesis headerfiles & classes
//#include "CodeTracing.h"

#include "Beam.h"
#include "Field.h"
#include "EField.h"

#include "Parser.h"
#include "GenProfile.h"
#include "Setup.h"
#include "AlterSetup.h"
#include "AlterBeam.h"
#include "Lattice.h"
#include "GenTime.h"
#include "Gencore.h"
#include "LoadField.h"
#include "LoadBeam.h"
#include "AlterLattice.h"
#include "Track.h"
#include "SDDSBeam.h"
#include "SponRad.h"
#include "dump.h"
#include "ImportBeam.h"
#include "ImportField.h"
#include "ImportTransformation.h"
#include "writeBeamHDF5.h"
#include "writeFieldHDF5.h"
#include "readMapHDF5.h"
#include "Collective.h"
#include "Wake.h"
#include "Diagnostic.h"
#include "SemaFile.h"
#include "FieldManipulator.h"
#ifdef USE_DPI
  #include "RegPlugin.h"
#endif
#include "SeriesManager.h"
#include "SeriesParser.h"
#include "SimpleHandshake.h"

#include <sstream>
#include <cstdlib>

#ifdef GENESIS_CUDA
#include <cuda_runtime_api.h>
#include "GenesisCudaKernels.h"
#endif

using namespace std;

const double vacimp = 376.73;
const double eev    = 510999.06; 
const double ce     = 4.8032045e-11;

// info for the meta group in the hdf5 output file
string meta_inputfile;
string meta_latfile;

bool MPISingle;  // global variable to do mpic or not

#ifdef GENESIS_CUDA
namespace {

bool genesisEnvFlagEnabled(const char *name, bool defaultValue)
{
    const char *env = std::getenv(name);
    if (!env || env[0] == '\0') {
        return defaultValue;
    }
    if (std::strcmp(env, "0") == 0 || std::strcmp(env, "false") == 0 ||
        std::strcmp(env, "FALSE") == 0 || std::strcmp(env, "off") == 0 ||
        std::strcmp(env, "OFF") == 0) {
        return false;
    }
    return true;
}

int parseGenesisCudaNonNegativeEnv(const char *name, int defaultValue)
{
    const char *env = std::getenv(name);
    if (!env || env[0] == '\0') {
        return defaultValue;
    }
    char *end = nullptr;
    long value = std::strtol(env, &end, 10);
    if (end == env || *end != '\0' || value < 0) {
        return defaultValue;
    }
    return static_cast<int>(value);
}

int parseGenesisCudaDeviceOverride()
{
    return parseGenesisCudaNonNegativeEnv("GENESIS_CUDA_DEVICE", -1);
}

int parseGenesisCudaWorkerBudget()
{
    int budget = parseGenesisCudaNonNegativeEnv("GENESIS_CUDA_MAX_RANKS_PER_DEVICE", 0);
    if (budget <= 0) {
        budget = parseGenesisCudaNonNegativeEnv("GENESIS_CUDA_WORKER_RANKS_PER_DEVICE", 0);
    }
    return budget;
}

std::string genesisCudaDeviceCountSummary(const std::vector<int> &deviceIds, int deviceCount)
{
    std::vector<int> counts(std::max(1, deviceCount), 0);
    for (int id : deviceIds) {
        if (id >= 0 && id < deviceCount) {
            counts[id]++;
        }
    }
    std::ostringstream os;
    for (int i = 0; i < deviceCount; ++i) {
        if (i > 0) {
            os << ", ";
        }
        os << "GPU" << i << "=" << counts[i];
    }
    return os.str();
}

int genesisCudaBlockMappedDevice(int localRank, int localSize, int deviceCount)
{
    if (deviceCount <= 1) {
        return 0;
    }
    const int ranksPerDevice = (localSize + deviceCount - 1) / deviceCount;
    int device = localRank / std::max(1, ranksPerDevice);
    if (device >= deviceCount) {
        device = deviceCount - 1;
    }
    return device;
}

bool genesisCudaConservativeModeEnabled()
{
    return genesisEnvFlagEnabled("GENESIS_CUDA_SAFE_MODE", false) ||
           genesisEnvFlagEnabled("GENESIS_CUDA_CONSERVATIVE", false);
}

const char *genesisCudaOnOff(bool value)
{
    return value ? "on" : "off";
}

void genesisPrintCudaRuntimeDefaults(int worldRank)
{
    if (worldRank != 0 || !genesisEnvFlagEnabled("GENESIS_CUDA_PRINT_DEFAULTS", true)) {
        return;
    }
    const bool conservative = genesisCudaConservativeModeEnabled();
    const char *policyEnv = std::getenv("GENESIS_CUDA_DEVICE_POLICY");
    const std::string policy = (policyEnv && policyEnv[0] != '\0') ? std::string(policyEnv) : std::string("local_rank");

    const bool fastKernels = !conservative && genesisEnvFlagEnabled("GENESIS_CUDA_FAST_KERNELS", true);
    const bool cacheLongitudinal = !conservative && genesisEnvFlagEnabled("GENESIS_CUDA_CACHE_LONGITUDINAL_FIELD", true);
    const bool inplaceSlippage = !conservative && genesisEnvFlagEnabled("GENESIS_CUDA_INPLACE_SLIPPAGE", true);
    const bool algebraOpt = !conservative && genesisEnvFlagEnabled("GENESIS_CUDA_LONGITUDINAL_ALGEBRA_OPT", true);

    cout << "GENESIS CUDA runtime defaults: device_policy=" << policy
         << ", fast_kernels=" << genesisCudaOnOff(fastKernels)
         << ", cache_longitudinal_field=" << genesisCudaOnOff(cacheLongitudinal)
         << ", inplace_slippage=" << genesisCudaOnOff(inplaceSlippage)
         << ", longitudinal_algebra_opt=" << genesisCudaOnOff(algebraOpt)
         << ", conservative_mode=" << genesisCudaOnOff(conservative)
         << ". Set GENESIS_CUDA_SAFE_MODE=1 for conservative CUDA fallback." << endl;
}

void genesisBindCudaDeviceForRank(int worldRank)
{
    if (!genesisEnvFlagEnabled("GENESIS_CUDA_DEVICE_BINDING", true)) {
        if (worldRank == 0) {
            cout << "GENESIS CUDA device binding disabled by GENESIS_CUDA_DEVICE_BINDING=0" << endl;
        }
        return;
    }

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount <= 0) {
        if (worldRank == 0) {
            cout << "GENESIS CUDA device binding: no CUDA device available (cudaGetDeviceCount: "
                 << cudaGetErrorString(err) << ")" << endl;
        }
        return;
    }

    int localRank = worldRank;
    int localSize = 1;
    MPI_Comm localComm = MPI_COMM_NULL;
#if MPI_VERSION >= 3
    if (MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &localComm) == MPI_SUCCESS) {
        MPI_Comm_rank(localComm, &localRank);
        MPI_Comm_size(localComm, &localSize);
    }
#endif

    const int forcedDevice = parseGenesisCudaDeviceOverride();
    int device = 0;
    const char *policyEnv = std::getenv("GENESIS_CUDA_DEVICE_POLICY");
    std::string policy = policyEnv ? std::string(policyEnv) : std::string("local_rank");

    const int ranksPerDeviceOverride = parseGenesisCudaNonNegativeEnv("GENESIS_CUDA_RANKS_PER_DEVICE", 0);

    if (forcedDevice >= 0) {
        device = forcedDevice % deviceCount;
        policy = "forced";
    } else if (ranksPerDeviceOverride > 0) {
        // Pack contiguous local-rank blocks onto each visible GPU. This is useful
        // when neighboring MPI ranks exchange slice-boundary data and users want
        // to reduce cross-GPU boundary traffic.
        device = (localRank / ranksPerDeviceOverride) % deviceCount;
        policy = "ranks_per_device";
    } else if (policy == "world_rank") {
        device = worldRank % deviceCount;
    } else if (policy == "single" || policy == "gpu0") {
        device = 0;
    } else if (policy == "block" || policy == "packed" || policy == "local_block" || policy == "local_rank_block") {
        // Contiguous local ranks are mapped to the same GPU. For 32 ranks on two
        // GPUs, ranks 0-15 go to GPU0 and ranks 16-31 go to GPU1.
        device = genesisCudaBlockMappedDevice(localRank, localSize, deviceCount);
        policy = "block";
    } else {
        // Default: round-robin local ranks across GPUs visible to the local node.
        device = localRank % deviceCount;
        policy = "local_rank";
    }

    err = cudaSetDevice(device);
    if (err != cudaSuccess) {
        cerr << "GENESIS CUDA device binding failed on MPI rank " << worldRank
             << ": cudaSetDevice(" << device << ") returned "
             << cudaGetErrorString(err) << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    bool verbose = genesisEnvFlagEnabled("GENESIS_CUDA_VERBOSE_DEVICE", false);
    const bool printSummary = genesisEnvFlagEnabled("GENESIS_CUDA_PRINT_DEVICE_SUMMARY", true);
    const int rankBudget = parseGenesisCudaWorkerBudget();

    std::vector<int> localDeviceIds;
    if (localComm != MPI_COMM_NULL) {
        if (localRank == 0) {
            localDeviceIds.assign(localSize, -1);
        }
        MPI_Gather(&device, 1, MPI_INT,
                   localRank == 0 ? localDeviceIds.data() : nullptr, 1, MPI_INT,
                   0, localComm);
    }

    if (worldRank == 0) {
        cout << "GENESIS CUDA device binding: policy=" << policy
             << ", visible_devices=" << deviceCount
             << ", local_ranks_per_node=" << localSize
             << ", ranks_per_device_override=" << ranksPerDeviceOverride
             << ", max_ranks_per_device=" << rankBudget
             << ", env GENESIS_CUDA_DEVICE_POLICY/GENESIS_CUDA_DEVICE/GENESIS_CUDA_RANKS_PER_DEVICE can override" << endl;
    }

    if (localRank == 0 && !localDeviceIds.empty()) {
        std::vector<int> counts(std::max(1, deviceCount), 0);
        for (int id : localDeviceIds) {
            if (id >= 0 && id < deviceCount) {
                counts[id]++;
            }
        }
        const int maxRanksOnDevice = *std::max_element(counts.begin(), counts.end());
        const bool overBudget = (rankBudget > 0 && maxRanksOnDevice > rankBudget);
        if (printSummary || overBudget || verbose) {
            cout << "GENESIS CUDA per-GPU worker summary: node_local_root_world_rank=" << worldRank
                 << ", " << genesisCudaDeviceCountSummary(localDeviceIds, deviceCount)
                 << ", max_ranks_per_device=" << maxRanksOnDevice;
            if (rankBudget > 0) {
                cout << ", configured_budget=" << rankBudget;
            }
            cout << endl;
        }
        if (overBudget) {
            const bool strict = genesisEnvFlagEnabled("GENESIS_CUDA_STRICT_RANK_BUDGET", false);
            std::ostringstream msg;
            msg << "GENESIS CUDA worker budget warning: "
                << "max ranks on one GPU is " << maxRanksOnDevice
                << ", exceeding GENESIS_CUDA_MAX_RANKS_PER_DEVICE/GENESIS_CUDA_WORKER_RANKS_PER_DEVICE="
                << rankBudget
                << ". Stage 3.9 recommends fewer MPI ranks per GPU or per-GPU worker aggregation.";
            if (strict) {
                cerr << msg.str() << " Aborting because GENESIS_CUDA_STRICT_RANK_BUDGET=1." << endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            } else {
                cout << msg.str() << endl;
            }
        }
    }

    if (verbose) {
        cudaDeviceProp prop{};
        cudaGetDeviceProperties(&prop, device);
        cout << "GENESIS CUDA device binding: MPI rank " << worldRank
             << " local_rank " << localRank << "/" << localSize
             << " -> device " << device << " (" << prop.name << ")" << endl;
    }

    if (localComm != MPI_COMM_NULL) {
        MPI_Comm_free(&localComm);
    }
}

} // namespace
#endif

//vector<string> event;
//vector<double> evtime;
//double evt0;

int genmain (string inputfile, map<string,string> &comarg, bool split) {
    meta_inputfile = inputfile;
    int ret = 0;
    MPISingle = split;
    int rank, size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // assign rank to node
    MPI_Comm_size(MPI_COMM_WORLD, &size); // assign rank to node
    if (MPISingle) {
        rank = 0;
        size = 1;
    }

#ifdef GENESIS_CUDA
    genesisBindCudaDeviceForRank(rank);
    genesisPrintCudaRuntimeDefaults(rank);
    genesis_cuda::setMemoryAuditContext(rank, size);
#endif

    time_t timer;
    clock_t clocknow;
    clock_t clockstart = clock();
    //	evt0 = double(clockstart);
    //	event.push_back("start");
    //	evtime.push_back(0);

    if (rank == 0) {         // the output of version and build has been moved to the wrapper mainwrap.cpp
        time(&timer);
        cout << "Starting Time: " << ctime(&timer) << endl;

        cout << "MPI-Comm Size: " << size;
        if (size == 1) {
            std::cout << " node" << endl << endl;
        } else {
            std::cout << " nodes" << endl << endl;
        }
    }

    //---------------------------------------------------------
    // Instance of beam and field class to carry the distribution

    vector<Field *> field;   // vector of various field components (harmonics, vertical/horizonthal components)
    Beam *beam = new Beam;


    //----------------------------------------------------------
    // main loop extracting one element with arguments at a time
    //
    // It is assumed that (except for a few well-defined reasons)
    // the loop is only left if there is an error while processing
    // the input file.
    bool successful_run = false; // successful simulation run?
    Parser parser;
    string element;
    map<string, string> argument;

    // some dummy argument used earlier
    string latstring;
    string outstring;
    int in_seed = 0;


    //-------------------------------------------
    // instances of main classes
    auto *setup = new Setup;
    auto *alt = new AlterLattice;
    auto *lattice = new Lattice;
    auto *profile = new Profile;
    auto *series = new SeriesManager;
    auto *seq = new Series;
    Time *timewindow = new Time;
    FilterDiagnostics filter;

    //-----------------------------------------------------------
    // main loop for parsing
    if (!parser.open(inputfile, rank)) {
        // could not open input file
	return(1); // exit code signals error
    }
    if (0 == rank) {
        cout << "Opened input file " << inputfile << endl;
    }


    while (true) {
        bool parser_result = parser.parse(&element, &argument);
        if (!parser_result) {
            // parser returned 'false', analyse reason
            if (parser.fail()) {
                successful_run = false;
            } else {
                successful_run = true; // probably reached eof (nothing was parsed, but also no parsing error detected) => implement more analysis
            }
            break;
        }

        //----------------------------------------------
        // log event
        //	  clocknow=clock();
        //	  event.push_back(element);
        //	  evtime.push_back(double(clocknow-clockstart));

        //----------------------------------------------
        // setup & parsing the lattice file

        if (element == "&setup") {
            // overwriting from the commandline input
            for (const auto &[key, val]: comarg) {
                // consistency check: display info message if new element is added to argument map (it may fail in Setup::init)
                if (argument.find(key) == argument.end()) {
                    if (rank == 0) {
                        cout << "info message: adding parameter '" << key << "' to &setup" << endl;
                    }
                }
                argument[key] = val;
            }
            if (!setup->init(rank, &argument, lattice, series, filter)) { break; }
            meta_latfile = setup->getLattice();

            /* successfully processed "&setup" block => generate start semaphore file if requested to do so by user */
            if (setup->getSemaEnStart()) {
                string semafile_fn_started;
                SemaFile sema;
                if (rank == 0) {
                    if (setup->getSemaFN(&semafile_fn_started)) {
                        semafile_fn_started += "start"; // "started"-type semaphore file have additional filename suffix
                        sema.put(semafile_fn_started);
                        cout << endl << "generating 'start' semaphore file " << semafile_fn_started << endl << endl;
                    } else {
                        cout << endl << "error: not writing 'start' semaphore file, filename not defined" << endl
                             << endl;
                    }
                }
            }
            continue;
        }

        //----------------------------------------------
        // modifying run

        if (element == "&alter_setup") {
            auto *altersetup = new AlterSetup;
            if (!altersetup->init(rank, &argument, setup, lattice, timewindow, beam, &field, series)) { break; }
            delete altersetup;
            continue;
        }

        //----------------------------------------------
        // modifying the lattice file

        if (element == "&lattice") {
            if (!alt->init(rank, size, &argument, lattice, setup, series)) { break; }
            continue;
        }

        //----------------------------------------------
        // modifying the particle distribution

        if (element == "&importtransformation") {
            auto *transf = new ImportTransformation;
            if (!transf->init(rank, &argument,beam,setup)) { break; }
            continue;
        }

        //---------------------------------------------------
        // adding sequence elements
        //
        if ((element.compare("&sequence_const") == 0) ||
            (element.compare("&sequence_polynom") == 0) ||
            (element.compare("&sequence_power") == 0) ||
            (element.compare("&sequence_list") == 0) ||
            (element.compare("&sequence_filelist")==0) ||
            (element.compare("&sequence_random") == 0)) {
            SeriesParser *seriesparser = new SeriesParser;
            if (!seriesparser->init(rank, &argument, element, series)) { break; }
            delete seriesparser;
//                if (!seq->init(rank,&argument,element)){ break; }
            continue;
        }




        //---------------------------------------------------
        // adding profile elements

        if ((element.compare("&profile_const") == 0) ||
            (element.compare("&profile_gauss") == 0) ||
            (element.compare("&profile_file") == 0) ||
            (element.compare("&profile_file_multi") == 0) ||
            (element.compare("&profile_polynom") == 0) ||
            (element.compare("&profile_step") == 0)) {
            if (!profile->init(rank, &argument, element)) { break; }
            continue;
        }

        //----------------------------------------------------
        // defining the time window of simulation

        if (element == "&time") {
            // check if beam or field is already defined
            auto defined = !beam->beam.empty();
            for (auto fld : field ) {
                defined |= !fld->field.empty();
            }
            if (defined){
                if (rank == 0){
                    std::cout << "*** Warning: Beam and/or Field already defined. Ignoring &time namelist" << std::endl;
                }
            } else {
                if (!timewindow->init(rank, size, &argument, setup)) { break; }
            }
            continue;
        }

        //----------------------------------------------------
        // internal generation of the field

        if (element == "&field") {
            auto *loadfield = new LoadField;
            if (!loadfield->init(rank, size, &argument, &field, setup, timewindow, profile)) { break; }
            delete loadfield;
            continue;
        }

        //----------------------------------------------------
        // field manipulation

        bool do_alter_field = false;
        if (element == "&field_manipulator") {
            do_alter_field = true;
            if (0 == rank) {
                cout
                        << "Warning: &field_manipulator is deprecated and will be removed in the future. Use &alter_field instead."
                        << endl;
            }
        }
        if (element == "&alter_field") {
            do_alter_field = true;
        }
        if (do_alter_field) {
            auto *q = new FieldManipulator;
            if (!q->init(rank, size, &argument, &field, setup, timewindow, profile)) { break; }
            delete q;
            continue;
        }

        //----------------------------------------------------
        // setup of space charge field

        if (element == "&efield") {
            auto *efield = new EField;
            if (!efield->init(rank, &argument, beam, setup)) { break; }
            delete efield;
            continue;
        }

        //----------------------------------------------------
        // setup of spontaneous radiation

        if (element == "&sponrad") {
            auto *sponrad = new SponRad;
            if (!sponrad->init(rank, size, &argument, beam)) { break; }
            delete sponrad;
            continue;
        }

        //----------------------------------------------------
        // setup wakefield

        if (element == "&wake") {
            Wake *wake = new Wake;
            if (!wake->init(rank, size, &argument, timewindow, setup, beam, profile)) { break; }
            delete wake;
            continue;
        }

        //----------------------------------------------------
        // internal generation of beam

        if (element == "&beam") {
            auto *loadbeam = new LoadBeam;
            if (!loadbeam->init(rank, size, &argument, beam, setup, timewindow, profile, lattice)) { break; }
            delete loadbeam;
            continue;
        }
        //----------------------------------------------------
        // direct manipulation of external field

        if (element == "&alter_beam") {
            auto *alterbeam = new AlterBeam;
            if (!alterbeam->init(rank, size, &argument, beam, setup, timewindow, profile)) { break; }
            delete alterbeam;
            continue;
        }

        //----------------------------------------------------
        // external generation of beam with an sdds file

        if (element == "&importdistribution") {
            auto *sddsbeam = new SDDSBeam;
            if (!sddsbeam->init(rank, size, &argument, beam, setup, timewindow, lattice)) { break; }
            delete sddsbeam;
            continue;
        }

        //-----------------------------------
        // register plugins for diagnostics
        if (element.compare("&add_plugin_fielddiag") == 0) {
#ifdef USE_DPI
            AddPluginFieldDiag *d = new AddPluginFieldDiag;
        if (!d->init(rank,size,&argument,setup)){ break;}
            delete d;
            continue;
#else
            if (rank == 0) {
                cout << "*** Error: This binary does not support the element " << element << endl;
            }
            break;
#endif
        }
        if (element.compare("&add_plugin_beamdiag") == 0) {
#ifdef USE_DPI
            AddPluginBeamDiag *d = new AddPluginBeamDiag;
        if (!d->init(rank,size,&argument,setup)){ break;}
            delete d;
            continue;
#else
            if (rank == 0) {
                cout << "*** Error: This binary does not support the element " << element << endl;
            }
            break;
#endif
        }

        //----------------------------------------------------
        // tracking - the very core part of Genesis

        if (element.compare("&track") == 0) {
            Track *track = new Track;
            if (!track->init(rank, size, &argument, beam, &field, setup, lattice, alt, timewindow, filter)) { break; }
            delete track;
            continue;
        }

        //----------------------------------------------------
        // write beam, field or undulator to file

        if (element.compare("&sort") == 0) {
            beam->sort();
            continue;
        }


        //----------------------------------------------------
        // write beam, field or undulator to file

        if (element.compare("&write") == 0) {
            Dump *dump = new Dump;
            if (!dump->init(rank, size, &argument, setup, beam, &field)) { break; }
            delete dump;
            continue;
        }


        //----------------------------------------------------
        // import beam from a particle dump

        if (element.compare("&importbeam") == 0) {
            ImportBeam *import = new ImportBeam;
            if (!import->init(rank, size, &argument, beam, setup, timewindow)) { break; }
            delete import;
            continue;
        }


        //----------------------------------------------------
        // import field from a field dump

        if (element.compare("&importfield") == 0) {
            ImportField *import = new ImportField;
            if (!import->init(rank, size, &argument, &field, setup, timewindow)) { break; }
            delete import;
            continue;
        }


        //----------------------------------------------------
        // stop execution of input file here (useful for debugging)

        if (element.compare("&stop") == 0) {
            if (rank == 0) {
                cout << endl << "*** &stop element: User requested end of simulation ***" << endl;
            }
            successful_run = true; // we reached "stop" command without error
            break;
        }

	      if (element.compare("&simple_handshake")==0){
            SimpleHandshake *hs=new SimpleHandshake;
            string prefix;
            setup->getOutputdir(&prefix);
	        if (!hs->doit(prefix)){ break;}
            delete hs;
            continue;  
          } 

        //-----------------------------------------------------
        // error because the element typ is not defined

        if (rank == 0) {
            cout << "*** Error: Unknown element in input file: " << element << endl;
        }
        break;
    }

    /*
     * Semaphore file: Decide now what to do in the end (and prepare the needed information)
     * -> if requested by user (filename is provided by user *or* derived from rootname)
     * -> if run was successful
     *
     * Actual file generation is delayed until the very end as the
     * simulation could still crash when free-ing objects.
     */
    bool semafile_en = false;
    string semafile_fn;
    bool semafile_fn_valid = false;
    if ((semafile_en = setup->getSemaEnDone())) {
        if (successful_run) {
            if (setup->getSemaFN(&semafile_fn)) {
                semafile_fn_valid = true;
            }
        }
    }

    /*** clean up ***/
    delete timewindow;
    delete seq;
    delete series;
    delete profile;
    delete lattice;
    delete alt;
    delete setup;
    delete beam;

    // release memory allocated for fields
    for (int i = 0; i < field.size(); i++) {
    delete field[i];
}

#ifdef GENESIS_CUDA
    if (genesis_cuda::memoryAuditEnabled()) {
        // Serialize audit printing to keep per-rank resource summaries readable.
        for (int auditRank = 0; auditRank < size; ++auditRank) {
            if (rank == auditRank) {
                genesis_cuda::printMemoryAuditSummary();
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }
#endif

	/*
	 * Synchronization, without in some cases the semaphore file
	 * can be written while some MPI processes are still processing
	 * the final command block.
	 */
	MPI_Barrier(MPI_COMM_WORLD);

	/* take time stamp (I/O for generating semaphore file could skew result if file system is busy) */ 
	//	event.push_back("end");
	//	evtime.push_back(double(clocknow-clockstart));
	clocknow=clock();


        /* NOW, generate the semaphore file */
        if (semafile_en) {
          if(successful_run) {
            SemaFile sema;
            if (rank==0) {
              if(semafile_fn_valid) {
                sema.put(semafile_fn);
                cout << endl << "generating semaphore file " << semafile_fn << endl;
              } else {
                cout << endl << "error: not writing semaphore file, filename not defined" << endl;
              }
            }
          } else {
            if (rank==0) {
              cout << endl << "not writing semaphore file, as the run likely was not successful" << endl;
            }
          }
        }


 	if (rank==0) {
	  double elapsed_Sec=double(clocknow-clockstart)/CLOCKS_PER_SEC;

      time(&timer);
      cout << endl<< "Program is terminating..." << endl;
	  cout << "Ending Time: " << ctime(&timer);
	  cout << "Total Wall Clock Time: " << elapsed_Sec << " seconds" << endl;
      cout << "-------------------------------------" << endl;


	  /* tracing report
	  cout << "Tracing Summary" << endl;
	  cout << "==========================" << endl;
	  cout << setw(10) << "Event" << setw(8) << "dT (s)" << endl;
	  cout << "--------------------------" << endl;
	  for (int i=0; i<evtime.size(); i++){
	    cout << setw(10) << event[i] << setw(8) << evtime[i]/CLOCKS_PER_SEC << " " << endl;
	  }
	  cout << "--------------------------" << endl;
	  */
        }


        // return value of this function is the exit code of G4 binary (0 signals OK)
        ret = (successful_run) ? 0 : 1;
        return(ret);
}
