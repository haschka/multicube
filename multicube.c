#include<stdio.h> 
#include<stdlib.h>
#include<string.h>
#include<sys/sysctl.h>
#include<sys/stat.h>
#include<signal.h>
#include<math.h>

#ifdef __APPLE__
#include<OpenCL/OpenCL.h>
#else
#include<CL/opencl.h>
#endif

#ifdef _DDOUBLE
typedef double real;
#else
typedef float real;
#endif

volatile sig_atomic_t killed = 0;

// Structure definitions

typedef struct {
  real* x;           // charge location
  real* y;
  real* z;            
  real* charge;       // charge
} nuclear_coordinates;     

typedef struct {
  real *pottable; // Potential grid points from cube table
  int sizex;  // size of cube
  int sizey;
  int sizez;
  real dx;     // delta between gridpoints
  real dy;
  real dz;     
  real origin_x;  // origin of grid points in space
  real origin_y;
  real origin_z;
  char* mask;  // linear gridpoint exclusion mask
  real* contraction_buffer;
  nuclear_coordinates coordinates; // atom coordinates
  int isGood; // verifier
  int num_of_atoms; // number of atoms in cube
} cube_def;

typedef struct {
  char discription[4]; // 3 character atom type discriptor
  int multiplicity;    // How many atoms in our cube are of this type
  char* in_atoms;      // Atoms of this type in all atoms vector
  real rVDW;
  int* charges_vector_locations; // Locations in all atoms vector 
} atomtype;

typedef struct {
  atomtype* atomtypes;
  int num_of_atomtypes;
  int num_of_atoms;
  double sum_of_charge;
  real *big_charges_vector;
  double *small_charges_vector;
  real* big_vdw_vector;
  double* plane_normal;
  double* gradient;
  double* displacementvector;
} atom_database;

typedef struct {
  cl_command_queue* cmdq;   //OpenCL command queue 
  cl_kernel* kernel;        //OpenCL kernels
  cl_kernel* contraction;
  cl_context* contexts;      //OpenCL contexts
  cl_mem* gpu_potential;    //OpenCL memory potential from charges
  cl_mem* gpu_pottable;     //potential read from cubes
  cl_mem* gpu_mask;            // mask for cubes
  cl_mem* gpu_charges;      //OpenCL memory for charges
  cl_mem* gpu_contraction;
  cl_mem* gpu_charges_locx; //OpenCL memory for charges locations
  cl_mem* gpu_charges_locy;
  cl_mem* gpu_charges_locz;
  int num_gpus;
} opencl_stuff;

typedef struct {
  char dsc[4];
} atom_discriptor;

void signal_handler(int signum) {
  killed = 1;
}

void usage() {
  printf("The rock'n roll multicube charges fit program!!\n");
  printf("                            Full of evil crazyness [WOOT!!] \n");
  printf("Argument list:\n");
  printf("1. string =/path/to/chargesfile \n");
  printf("2. string =/path/to/vdw_database \n");
  printf("3. string =/path/to/minimized_charges_output \n");
  printf("4  int    =Number of GPUs you have... >1000 is cool! \n");
  printf("5. char   =Norm to use 1: L1 2: L2 3: weighted L1 4: weighted L2\n");
  printf("6. real   =Initial stepsize ~1 should be fine \n");
  printf("                            otherwise use thumbs devided by Pi\n");
  printf("                                          f(x+h)-f(x-h)\n");
  printf("7. real   =h used in numerical gradient : -------------\n");
  printf("                                               2*h     \n");
  printf("                     should be around sqrtf(machine_epsilon)\n");
  printf("                     machine_epsilon= for 64bit doubles ~1.11e-16\n");
  printf("                                      for 32bit floats  ~5.96e-08\n");
  printf("8. real   =convergence criterium: \n");
  printf("             calculation stops if: \n");
  printf("             displacement_vector_length < convergence criterium\n");
  printf("9. int    =maximum number of steps to evaluate\n");
  printf("10. multiple stings = /paths/to/cubes/\n");
}

// Function to read in the OpenCL kernel code;

char* load_program_source(const char *filename){ 
	
  struct stat statbuf;
  FILE *fh; 
  char *source; 
  
  fh = fopen(filename, "r");
  if (fh == 0)
    return 0; 
  
  stat(filename, &statbuf);
  source = (char *) malloc(statbuf.st_size + 1);
  fread(source, statbuf.st_size, 1, fh);
  source[statbuf.st_size] = '\0'; 
  
  return source; 
}

opencl_stuff opencl_initialization(atom_database base, cube_def* cubes,
				   int num_cubes, int num_gpus, int norm_bool) {
  
  opencl_stuff retval;

  int i,j,cubes_by_gpus,cubes_rest;
  
#ifdef _DDOUBLE
  char clSourceFile[8]="dpot.cl";
#else 
  char clSourceFile[8]="spot.cl";
#endif
  char *clSource;
  cl_platform_id platform; 
  cl_context_properties contprop[3];
  cl_program* programs;
  cl_kernel* k_epot;
  cl_kernel* k_contraction;
  cl_command_queue* cmdq;
  cl_context* contexts;
  
  cl_device_id* devices;
  cl_char devName[1024];
  size_t devNameSize;

  char BuildErrorLog[2048];
  size_t BuildErrorLength;
  
  cl_mem* gpu_potential;
  cl_mem* gpu_pottable;
  cl_mem* gpu_mask;
  cl_mem* gpu_charges;
  cl_mem* gpu_charges_locx;
  cl_mem* gpu_charges_locy;
  cl_mem* gpu_charges_locz;
  cl_mem* gpu_contracted;

  cl_int err;

  int cube_by_gpus;
  int cube_rest;

  int whichcube;

  int charges_num = base.num_of_atoms;

  // Dynamic Host Memory Allocation
#ifdef _DCLOVIS
  contexts = (cl_context*)malloc(sizeof(cl_context)*2);
  cmdq = (cl_command_queue*)malloc(sizeof(cl_command_queue)*2);
  devices = (cl_device_id*)malloc(sizeof(cl_device_id)*2);
#else
  contexts = (cl_context*)malloc(sizeof(cl_context)*num_gpus);
  cmdq = (cl_command_queue*)malloc(sizeof(cl_command_queue)*num_gpus);
  devices = (cl_device_id*)malloc(sizeof(cl_device_id)*num_gpus);
#endif

  programs = (cl_program*)malloc(sizeof(cl_program)*num_gpus);
  k_epot = (cl_kernel*)malloc(sizeof(cl_kernel)*num_gpus);
  k_contraction = (cl_kernel*)malloc(sizeof(cl_kernel)*num_gpus);
  gpu_charges = (cl_mem*)malloc(sizeof(cl_kernel)*num_gpus);

  gpu_potential = (cl_mem*)malloc(sizeof(cl_mem)*num_cubes);
  gpu_pottable = (cl_mem*)malloc(sizeof(cl_mem)*num_cubes);
  gpu_mask = (cl_mem*)malloc(sizeof(cl_mem)*num_cubes);
  gpu_contracted = (cl_mem*)malloc(sizeof(cl_mem)*num_cubes);
  gpu_charges_locx = (cl_mem*)malloc(sizeof(cl_mem)*num_cubes);
  gpu_charges_locy = (cl_mem*)malloc(sizeof(cl_mem)*num_cubes);
  gpu_charges_locz = (cl_mem*)malloc(sizeof(cl_mem)*num_cubes);
  
  // OpenCL Initialisation

  clGetPlatformIDs(1,&platform,NULL);   

#ifdef _DCLOVIS
  clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 2,devices, NULL);
#else
  clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, num_gpus,devices, NULL);
#endif    

  contprop[0] = CL_CONTEXT_PLATFORM;
  contprop[1] = (cl_context_properties)platform;
  contprop[2] = 0;
  for (i = 0;i<num_gpus;i++) {
    contexts[i] = clCreateContext(contprop, 1, devices+i, NULL, NULL, &err);
    if (err != CL_SUCCESS) { 
      printf("Context Creation failed! OCL-ERROR-CODE: %i\n", err);
    }
  }
  
#ifdef _DCLOVIS
  if (num_gpus == 1) {
    if (err == CL_INVALID_DEVICE) {
      printf("First Device Failed, trying 2nd");
      contexts[0] = clCreateContext(contprop, 1, devices+1, NULL, NULL, &err);
      cmdq[0] = clCreateCommandQueue(contexts[0], devices[1], 0, NULL);
    } else {
      cmdq[0] = clCreateCommandQueue(contexts[0], devices[0], 0, NULL);
    }
  }
  if (num_gpus == 2) {
    for(i=0;i<num_gpus;i++) {
      cmdq[i] = clCreateCommandQueue(contexts[i], devices[i], 0, NULL); 
    }
  }
#else 
  for(i=0;i<num_gpus;i++) {
    cmdq[i] = clCreateCommandQueue(contexts[i], devices[i], 0, NULL); 
  }
#endif

  // Building the OpenCL Program
  clSource = load_program_source(clSourceFile);
  for(i=0;i<num_gpus;i++) {
    programs[i] = clCreateProgramWithSource(contexts[i],1,
					    (const char**)&clSource,NULL,
					    &err);
    if (err != CL_SUCCESS) { 
      printf("Program creation failed!\n");
    }
    
    err = clBuildProgram(programs[i],0,NULL,NULL,NULL,NULL);
    
    if (err != CL_SUCCESS) {
      
      printf("Failed to compile code into an executeable! \n ");
      
      clGetProgramBuildInfo(programs[i],devices[i],CL_PROGRAM_BUILD_LOG,
			    sizeof(BuildErrorLog),BuildErrorLog,
			    &BuildErrorLength);
      
      printf("Build Error Log: \n %s \n", BuildErrorLog);
    }
    
    
  }
  
  for (i=0;i<num_gpus;i++) {
    // Kernel Creation
    
    k_epot[i] = clCreateKernel(programs[i],"epot",&err);
    
    switch(norm_bool) {
    case 1:
      k_contraction[i] = clCreateKernel(programs[i],"contractionL1",&err);
      break;
    case 2:
      k_contraction[i] = clCreateKernel(programs[i],"contractionL2",&err);
      break;
    case 3:
      k_contraction[i] = clCreateKernel(programs[i],"contractionL1w",&err);
      break;
    case 4:
      k_contraction[i] = clCreateKernel(programs[i],"contractionL2w",&err);
      break;
    }
    
  
  // GPU Memory Allocation
    
    gpu_charges[i] = clCreateBuffer(contexts[i], 
				    CL_MEM_READ_WRITE, 
				    sizeof(real)*charges_num,
				    NULL, &err);
    
  
    err = clSetKernelArg(k_epot[i], 1, sizeof(cl_mem), gpu_charges+i);
    err = clSetKernelArg(k_epot[i], 5, sizeof(int), &charges_num); 
  }
    // static memory propagation

  cubes_by_gpus=num_cubes/num_gpus;
  cubes_rest=num_cubes%num_gpus;

  for(i=0;i<num_gpus;i++) {
    for(j=0;j<cubes_by_gpus;j++){


      whichcube = i*cubes_by_gpus+j;

      gpu_pottable[whichcube] = clCreateBuffer(contexts[i],
					       CL_MEM_READ_WRITE,
				       sizeof(real)*cubes[whichcube].sizex
				       *cubes[whichcube].sizey
				       *cubes[whichcube].sizez,
				       NULL, &err);
      
      gpu_contracted[whichcube] = clCreateBuffer(contexts[i],
					 CL_MEM_READ_WRITE,
					 sizeof(real)*cubes[whichcube].sizex
					 *cubes[whichcube].sizey,
					 NULL, &err);
      
      gpu_mask[whichcube] = clCreateBuffer(contexts[i],
				   CL_MEM_READ_WRITE,
				   sizeof(char)*cubes[whichcube].sizex
				   *cubes[whichcube].sizey
				   *cubes[whichcube].sizez,
				   NULL, &err);
      
      
      gpu_potential[whichcube] = clCreateBuffer(contexts[i], 
					CL_MEM_READ_WRITE, 
					sizeof(real)*cubes[whichcube].sizex
					*cubes[whichcube].sizey
					*cubes[whichcube].sizez,
					NULL, &err);
      
      gpu_charges_locx[whichcube] = clCreateBuffer(contexts[i], 
					   CL_MEM_READ_WRITE, 
					   sizeof(real)*charges_num,
					   NULL, &err);
      
      gpu_charges_locy[whichcube] = clCreateBuffer(contexts[i], 
					   CL_MEM_READ_WRITE, 
					   sizeof(real)*charges_num,
					   NULL, &err);
      
      gpu_charges_locz[whichcube] = clCreateBuffer(contexts[i], 
					   CL_MEM_READ_WRITE, 
					   sizeof(real)*charges_num,
					   NULL, &err);
      
      
      err = clEnqueueWriteBuffer(cmdq[i], gpu_pottable[whichcube],
				 CL_TRUE,0,sizeof(real)
				 *cubes[whichcube].sizex
				 *cubes[whichcube].sizey
				 *cubes[whichcube].sizez,
				 (void*)cubes[whichcube].pottable,
				 0,NULL,NULL);
      
      err = clEnqueueWriteBuffer(cmdq[i], gpu_mask[whichcube],
				 CL_TRUE,0,sizeof(char)
				 *cubes[whichcube].sizex
				 *cubes[whichcube].sizey
				 *cubes[whichcube].sizez,
				 (void*)cubes[whichcube].mask,
				 0,NULL,NULL);

      
      err = clEnqueueWriteBuffer(cmdq[i], gpu_charges_locx[whichcube],
				 CL_TRUE,0,sizeof(real)*charges_num,
				 (void*)cubes[whichcube].coordinates.x,
				 0,NULL,NULL);
      
      err = clEnqueueWriteBuffer(cmdq[i], gpu_charges_locy[whichcube],
				 CL_TRUE,0,sizeof(real)*charges_num,
				 (void*)cubes[whichcube].coordinates.y,
				 0,NULL,NULL);
      
      err = clEnqueueWriteBuffer(cmdq[i], gpu_charges_locz[whichcube],
				 CL_TRUE,0,sizeof(real)*charges_num,
				 (void*)cubes[whichcube].coordinates.z,
				 0,NULL,NULL);
    }
  }
  for(j=0;j<cubes_rest;j++){

    whichcube = num_gpus*cubes_by_gpus+j;
    
    gpu_pottable[whichcube] = clCreateBuffer(contexts[j],
					     CL_MEM_READ_WRITE,
					     sizeof(real)
					     *cubes[whichcube].sizex
					     *cubes[whichcube].sizey
					     *cubes[whichcube].sizez,
					     NULL, &err);
    
    gpu_contracted[whichcube] = clCreateBuffer(contexts[j],
					       CL_MEM_READ_WRITE,
					       sizeof(real)
					       *cubes[whichcube].sizex
					       *cubes[whichcube].sizey,
					       NULL, &err);
    
    gpu_mask[whichcube] = clCreateBuffer(contexts[j],
					 CL_MEM_READ_WRITE,
					 sizeof(char)
					 *cubes[whichcube].sizex
					 *cubes[whichcube].sizey
					 *cubes[whichcube].sizez,
					 NULL, &err);
    
    
    gpu_potential[whichcube] = clCreateBuffer(contexts[j], 
					      CL_MEM_READ_WRITE, 
					      sizeof(real)
					      *cubes[whichcube].sizex
					      *cubes[whichcube].sizey
					      *cubes[whichcube].sizez,
					      NULL, &err);
    
    gpu_charges_locx[whichcube] = clCreateBuffer(contexts[j], 
						 CL_MEM_READ_WRITE, 
						 sizeof(real)*charges_num,
						 NULL, &err);
    
    gpu_charges_locy[whichcube] = clCreateBuffer(contexts[j], 
						 CL_MEM_READ_WRITE, 
						 sizeof(real)*charges_num,
						 NULL, &err);
    
    gpu_charges_locz[whichcube] = clCreateBuffer(contexts[j], 
						 CL_MEM_READ_WRITE, 
						 sizeof(real)*charges_num,
						 NULL, &err);
    
    
    err = clEnqueueWriteBuffer(cmdq[j], gpu_pottable[whichcube],
			       CL_TRUE,0,sizeof(real)
			       *cubes[whichcube].sizex
			       *cubes[whichcube].sizey
			       *cubes[whichcube].sizez,
			       (void*)cubes[whichcube].pottable,
			       0,NULL,NULL);
    
    err = clEnqueueWriteBuffer(cmdq[j], gpu_mask[whichcube],
			       CL_TRUE,0,sizeof(char)
			       *cubes[whichcube].sizex
			       *cubes[whichcube].sizey
			       *cubes[whichcube].sizez,
			       (void*)cubes[whichcube].mask,
			       0,NULL,NULL);


    err = clEnqueueWriteBuffer(cmdq[j], gpu_charges_locx[whichcube],
			       CL_TRUE,0,sizeof(real)*charges_num,
			       (void*)cubes[whichcube].coordinates.x,
			       0,NULL,NULL);
    
    err = clEnqueueWriteBuffer(cmdq[j], gpu_charges_locy[whichcube],
			       CL_TRUE,0,sizeof(real)*charges_num,
			       (void*)cubes[whichcube].coordinates.y,
			       0,NULL,NULL);
    
    err = clEnqueueWriteBuffer(cmdq[j], gpu_charges_locz[whichcube],
			       CL_TRUE,0,sizeof(real)*charges_num,
			       (void*)cubes[whichcube].coordinates.z,
			       0,NULL,NULL);    
  }

  for(i=0;i<num_gpus;i++) {
    clFinish(cmdq[i]);
  }

  retval.cmdq = cmdq;
  retval.kernel = k_epot;
  retval.contraction = k_contraction;
  retval.contexts = contexts;
  retval.gpu_pottable = gpu_pottable;
  retval.gpu_mask = gpu_mask;
  retval.gpu_potential = gpu_potential;
  retval.gpu_contraction = gpu_contracted;
  retval.gpu_charges_locx = gpu_charges_locx;
  retval.gpu_charges_locy = gpu_charges_locy;
  retval.gpu_charges_locz = gpu_charges_locz;
  retval.gpu_charges = gpu_charges;
  retval.num_gpus = num_gpus;

  return retval;
}

// function to update per atomtype charges vector from all atom charges vector

void big_to_small_charges_vector(real* big_charges_vector,
				 double* small_charges_vector,
				 atomtype* atomtypes,
				 int num_of_atomtypes) {
  int i,j;
  real sum_big;
 
  for (i = 0; i<num_of_atomtypes; i++ ) {
    sum_big = 0;
    for (j = 0; j < atomtypes[i].multiplicity; j++ ) {
      sum_big += big_charges_vector[atomtypes[i].charges_vector_locations[j]];
    }
    small_charges_vector[i] = sum_big/atomtypes[i].multiplicity;
  }
}

// function to update the all atom charges vector from the per atomtype 
// charges vector

void small_to_big_charges_vector(real* big_charges_vector,
				 double* small_charges_vector,
				 atomtype* atomtypes,
				 int num_of_atomtypes) {
  int i,j;
  
  for (i = 0; i<num_of_atomtypes; i++ ) {
    for (j = 0; j < atomtypes[i].multiplicity;j++) {
      big_charges_vector[atomtypes[i].charges_vector_locations[j]] = 
	(real)small_charges_vector[i];
    }
  }
}

double initialdriftcorrection(real* big_charges_vector,
			      double sum_of_charge,
			      int num_of_atoms) {
  int i;
  double sum=0;
  double drift;
  for (i=0;i<num_of_atoms;i++) {
    sum += (double)big_charges_vector[i];
  }
  drift = sum-sum_of_charge;
  for (i=0;i<num_of_atoms;i++) {
    big_charges_vector[i]=big_charges_vector[i]*(sum_of_charge/(real)sum);
  }
  return(drift);
}

double* generate_plane_normal(atomtype* atomtypes,
			      int num_of_atomtypes) {
  
  int i;
  double* plane_normal = (double*)malloc(num_of_atomtypes*sizeof(double));
  double normalization_factor = 0.f;
  for (i=0;i<num_of_atomtypes;i++) {
    plane_normal[i]=(double)atomtypes[i].multiplicity;
    normalization_factor += plane_normal[i]*plane_normal[i];
  }
  normalization_factor=1./sqrt(normalization_factor);
  for (i=0;i<num_of_atomtypes;i++) {
    plane_normal[i]=normalization_factor*plane_normal[i];
  }
  return(plane_normal);
}


// function to generate the internal database about atoms and their properties
// from the VDW radii and charges input file

atom_database generate_inital_database(char* vdwfilestring, 
				       char* chargesfilestring,
				       int num_of_atoms) {
  
  double sum_of_total_charge = 0.;     
  int num_of_atomtypes = 0; 
  int type_evaluation_checksum = 0;
 
  int i,j,k;
  atom_database retval;

  real temporary_rvdw;

  atom_discriptor* raw_atom_discriptions;
  atomtype* atomtypes;
  char* types_evaluated;

  real* chargesvector;
  double* small_charges_vector;
  real* big_vdw_vector;
  char buffer[200];
  
  // opening input files

  FILE* chargesfile = fopen(chargesfilestring,"r");
  FILE* vdwfile = fopen(vdwfilestring,"r");

  // verification whether input files could be opened

  if ( NULL == chargesfile ) {
    printf("Charges file could not be openend!\n");
    retval.num_of_atomtypes = -1;
    return retval;
  }
  if ( NULL == vdwfile ) {
    printf("Vdw Radii file could not be openend!\n");
    retval.num_of_atomtypes = -1;
    return retval;
  }

  // memory allocation for the
  // all atoms charges vector
  // a buffer that stores the atom descriptions as the are in defined in the
  //    charges input file
  // a buffer that is used during the generation of unique atomtypes

  chargesvector = (real*)malloc(sizeof(real)*num_of_atoms);
  raw_atom_discriptions = (atom_discriptor*)
     malloc(sizeof(atom_discriptor)*num_of_atoms);
  types_evaluated = (char*)malloc(sizeof(char)*num_of_atoms);
  
  bzero(types_evaluated,num_of_atoms);
  
  // read in charges from charges input file into all atoms vector

#ifdef _DDOUBLE
  for(i=0;i<num_of_atoms;i++) {
    if(2 != fscanf(chargesfile,
		   "%s %lG",raw_atom_discriptions[i].dsc,chargesvector+i) ) {
      printf("Charges file seems to be invalid!");
      retval.num_of_atomtypes = -1;
      return retval;
    }
    sum_of_total_charge += (double)chargesvector[i];
  } 
#else
  for(i=0;i<num_of_atoms;i++) {
    if(2 != fscanf(chargesfile,
		   "%s %G",raw_atom_discriptions[i].dsc,chargesvector+i) ) {
      printf("Charges file seems to be invalid!");
      retval.num_of_atomtypes = -1;
      return retval;
    }
    sum_of_total_charge += (double)chargesvector[i];
  }
#endif
  
  printf("Charges have been read! Evaluating atom types. \n");
  printf("Sum of Charges = %f \n", sum_of_total_charge);

  // searching for unique atomtypes

  for(i=0;i<num_of_atoms;i++) {
    if (types_evaluated[i] == 0) {
      num_of_atomtypes++;
      memcpy(buffer,raw_atom_discriptions[i].dsc,4);
      for(j=0;j<num_of_atoms;j++) {
	if (0 == strncmp(buffer,raw_atom_discriptions[j].dsc,3)) {
	  types_evaluated[j] = 1;
	}
      }
    }
  }

  printf("Number of different atom types: %i \n", num_of_atomtypes);

  // generating unique atomtypes database

  atomtypes = (atomtype*)malloc(sizeof(atomtype)*num_of_atomtypes);
  
  for(i=0;i<num_of_atomtypes;i++){
    atomtypes[i].in_atoms = (char*)malloc(sizeof(char)*num_of_atoms);
    atomtypes[i].multiplicity = 0;
  }
  
  num_of_atomtypes=0;
  bzero(types_evaluated,num_of_atoms);
  for(i=0;i<num_of_atoms;i++) {
    if (types_evaluated[i] == 0) {
      memcpy(atomtypes[num_of_atomtypes].discription,
	     raw_atom_discriptions[i].dsc,4);
      for(j=0;j<num_of_atoms;j++) {
	if (0 == strncmp(atomtypes[num_of_atomtypes].discription,
			 raw_atom_discriptions[j].dsc,3)) {
	  atomtypes[num_of_atomtypes].in_atoms[j] = 1;
	  atomtypes[num_of_atomtypes].multiplicity++;
	  types_evaluated[j] = 1;
	}
      }
      num_of_atomtypes++;
    }
  }

  printf("Multiplicities for different atom types have been calulculated! \n");

  for(i=0;i<num_of_atomtypes;i++) {
    atomtypes[i].charges_vector_locations = 
      (int*)malloc(sizeof(int)*atomtypes[i].multiplicity);
    k=0;
    for(j=0;j<num_of_atoms;j++) {
      if (1 == atomtypes[i].in_atoms[j]) {
	atomtypes[i].charges_vector_locations[k] = j; 
	k++;
      }
    }
  }

  // scanning the vdw database file for the unique atomtypes found;

#ifdef _DDOUBLE
  for(i=0;i<num_of_atomtypes;i++) {
    do {
      if ( 2 != fscanf(vdwfile, "%s %lf",buffer,&temporary_rvdw) ) {
	usage();
	printf("Failure: Malformatted VDW database! \n");
        retval.num_of_atomtypes = -1;
	return retval;
      }
    } while (0 != strncmp(atomtypes[i].discription,buffer,(size_t)4));
    atomtypes[i].rVDW = temporary_rvdw*1.889725;
    printf("N: %i, rVDW: %f \n",i,atomtypes[i].rVDW);
    rewind(vdwfile);
  }
#else
  for(i=0;i<num_of_atomtypes;i++) {
    do {
      if ( 2 != fscanf(vdwfile, "%s %f",buffer,&temporary_rvdw) ) {
	usage();
	printf("Failure: Malformatted VDW database! \n");
        retval.num_of_atomtypes = -1;
	return retval;
      }
    } while (0 != strncmp(atomtypes[i].discription,buffer,(size_t)4));
    atomtypes[i].rVDW = temporary_rvdw*1.889725;
    printf("N: %i, rVDW: %f \n",i,atomtypes[i].rVDW);
    rewind(vdwfile);
  }
#endif

  // allocating and initializing the per atomtype charges vector from the 
  // all atoms charges vector

  small_charges_vector = (double*)malloc(sizeof(double)*num_of_atomtypes);
  big_to_small_charges_vector(chargesvector,
			      small_charges_vector,
			      atomtypes,
			      num_of_atomtypes);

  // correcting charges drift due to different charges on atoms belonging 
  // to the same atomtype

  printf("Drift correction factor after atom type equalisation: %f \n",
	 initialdriftcorrection(chargesvector,
				sum_of_total_charge,
				num_of_atoms)
	 );

  printf("In memory atom types database generated\n");

  // Free some temporary buffers from the atomtypes database generation
  free(raw_atom_discriptions);
  free(types_evaluated);

  // generating an all atoms vdw radii vector
  
  big_vdw_vector = (real*)malloc(sizeof(real)*num_of_atoms);
  for(i=0;i<num_of_atomtypes;i++){
    for(j=0;j<atomtypes[i].multiplicity;j++){
      big_vdw_vector[atomtypes[i].charges_vector_locations[j]] = 
	atomtypes[i].rVDW;
    }
  }

  retval.atomtypes = atomtypes;
  retval.num_of_atomtypes = num_of_atomtypes;
  retval.num_of_atoms = num_of_atoms;
  retval.sum_of_charge = sum_of_total_charge;
  retval.big_charges_vector = chargesvector;
  retval.small_charges_vector = small_charges_vector;
  retval.big_vdw_vector = big_vdw_vector;
  retval.plane_normal = (double*)generate_plane_normal(atomtypes,num_of_atomtypes);
  retval.gradient = (double*)malloc(sizeof(double)*num_of_atomtypes);
  retval.displacementvector = (double*)malloc(sizeof(double)*num_of_atomtypes);
  
  fclose(vdwfile);
  fclose(chargesfile);

  return retval;
}

// function to obtain the number of atoms in single cube file

int get_num_of_atoms_from_a_cubefile(char* cubefilestring) {
  
  int num_of_atoms,i;
  char buffer[200];
  FILE* cubefile = fopen(cubefilestring,"r");

  if (NULL == cubefile) {
    printf("Error opening a cubefile at fopen!\n");
    return -1;
  }
  
  for(i=0;i<2;i++){
    fgets(buffer, 200, cubefile);
  }
  
  fgets(buffer,200, cubefile);
  
  sscanf(buffer, "%i", &num_of_atoms);
  
  rewind(cubefile);
  fclose(cubefile);
  
  return(num_of_atoms);
}

// function to obtain the parameters of a specific cube file

cube_def getCubeProperties(char* cubefilestring,atom_database base) {
  
  int i;

  cube_def retval;
  nuclear_coordinates coordinates;
  
  int sizex, sizey, sizez, sizexBYsizey;
  real xx, xy, xz, yx, yy, yz, zx, zy, zz;
  int num_of_gridpoints;
  int index_x, index_y, index_z;
  real origin_x, origin_y, origin_z;
  int currloc;

  real distance_x, distance_y, distance_z; // distances between gridpoints
  real nuclear_x, nuclear_y, nuclear_z, nuclear_charge; 
  real r;
  char* mask;

  double sum_of_nuclear_charge;
  
  int cube_atomtype;
  

#ifdef _DDOUBLE
  char cube_file_scan_string[] = "%lG";
#else
  char cube_file_scan_string[] = "%G";
#endif
  int num_of_atoms;
  int num_of_atoms_test;
  char buffer[200];
  real* pottable;
  real new, min, max;

  // opening the cubefile
  FILE* cubefile = fopen(cubefilestring,"r");

  num_of_atoms = base.num_of_atoms;

  // verfication if the cubefile could be opened
  if (NULL == cubefile) {
    printf("Error opening cubefile: %s at fopen!\n",cubefilestring);
    retval.isGood = 0;
    return retval;
  }
  printf("Processing Cubefile: %s \n",cubefilestring);

  // Parsing the cube file header
  
  for(i=0;i<2;i++){
    fgets(buffer, 200, cubefile);
  }  

  fgets(buffer,200, cubefile);
  
#ifdef _DDOUBLE
  sscanf(buffer, "%i %lf %lf %lf", &num_of_atoms_test, 
	 &origin_x, 
	 &origin_y, 
	 &origin_z);
  
  fgets(buffer, 200, cubefile);
  sscanf(buffer, "%i %lf %lf %lf", &sizex, &xx, &xy, &xz);
  fgets(buffer, 200, cubefile);
  sscanf(buffer, "%i %lf %lf %lf", &sizey, &yx, &yy, &yz);
  fgets(buffer, 200, cubefile);
  sscanf(buffer, "%i %lf %lf %lf", &sizez, &zx, &zy, &zz);
#else
  sscanf(buffer, "%i %f %f %f", &num_of_atoms_test, 
	 &origin_x, 
	 &origin_y, 
	 &origin_z);
  
  fgets(buffer, 200, cubefile);
  sscanf(buffer, "%i %f %f %f", &sizex, &xx, &xy, &xz);
  fgets(buffer, 200, cubefile);
  sscanf(buffer, "%i %f %f %f", &sizey, &yx, &yy, &yz);
  fgets(buffer, 200, cubefile);
  sscanf(buffer, "%i %f %f %f", &sizez, &zx, &zy, &zz);
#endif

  sizexBYsizey =sizex*sizey;

  // verifing if the cubefiles number of atoms corresponds to the global number 
  // of atoms
  if(num_of_atoms_test != num_of_atoms) {
    printf("Cubes are not coherent, different numbers of atoms detected! \n");
    retval.isGood = 0;
    return retval;
  }

  // Verifing weather the cube files base is orthogonal

  if( 0 != (int)(100*xy) || 0 != (int)(100*xy) ||
      0 != (int)(100*yx) || 0 != (int)(100*yz) ||
      0 != (int)(100*zx) || 0 != (int)(100*zy) ) {
    printf("Non orthogonal coordinate systems are not supported!");
    retval.isGood = 0;
    return retval;
  }

  printf("Voxel Dimensions: x=%f y=%f z=%f \n",xx,yy,zz);

  // assining memory for and reading  the coordinates of atoms and their
  // charges found in the cubefile

  coordinates.x = (real*)malloc(sizeof(real)*num_of_atoms);
  coordinates.y = (real*)malloc(sizeof(real)*num_of_atoms);
  coordinates.z = (real*)malloc(sizeof(real)*num_of_atoms);
  coordinates.charge = (real*)malloc(sizeof(real)*num_of_atoms);

  sum_of_nuclear_charge = 0.f;
  for(i=0;i<num_of_atoms;i++){
    fgets(buffer, 200, cubefile);
#ifdef _DDOUBLE
    sscanf(buffer,"%i %lf %lf %lf %lf", &cube_atomtype, &nuclear_charge,
	   &nuclear_x, &nuclear_y, &nuclear_z);
#else
    sscanf(buffer,"%i %f %f %f %f", &cube_atomtype, &nuclear_charge,
	   &nuclear_x, &nuclear_y, &nuclear_z);
#endif
    coordinates.charge[i] = nuclear_charge;
    sum_of_nuclear_charge+=(double)nuclear_charge;
    coordinates.x[i] = nuclear_x;
    coordinates.y[i] = nuclear_y;
    coordinates.z[i] = nuclear_z;
  }
  printf("Number of Atoms: %i \n Total Nuclar Charge: %lf \n",num_of_atoms,
	 sum_of_nuclear_charge);

  // reading the potential values store in the cube file and 
  // calculating the cubes voxel exclusion mask from using vdw radii

  mask = (char*)malloc(sizex*sizey*sizez*sizeof(char));
  pottable = (real*)malloc(sizeof(real)*sizex*sizey*sizez);

  num_of_gridpoints=sizex*sizey*sizez;

  for(index_x=0;index_x<sizex;index_x++) {
    for(index_y=0;index_y<sizey;index_y++) {
      for(index_z=0;index_z<sizez;index_z++) {
	
	currloc = index_x+index_y*sizex+index_z*sizexBYsizey;
	fscanf(cubefile,cube_file_scan_string,&new);
	
	pottable[currloc] = new;
	
	mask[currloc]=0;
	for(i=0;i<num_of_atoms;i++) {
	  	  	      
	  distance_x = origin_x+((real)index_x)*xx-coordinates.x[i];
	  distance_y = origin_y+((real)index_y)*yy-coordinates.y[i];
	  distance_z = origin_z+((real)index_z)*zz-coordinates.z[i];

	  r = base.big_vdw_vector[i];

	  if ( r*r >
	       distance_x*distance_x
	       +distance_y*distance_y
	       +distance_z*distance_z ) {
	    mask[currloc]=1;
	    num_of_gridpoints--;
	    if ( num_of_gridpoints < num_of_atoms ) {
	      printf("Failure: Number of Gridpoints is lower then the \n");
	      printf("         Number of Atoms \n");
	      printf("         Number of Gridpoints %i \n",num_of_gridpoints);
	      printf("         Number of Atoms %i \n",num_of_atoms);
	      retval.isGood = 0;
	      return retval;
	    }
	  }
	}
	
	
	if(new > max) {
	  max = new;
	}
	if(new < min) {
	  min = new;
	}
      }
    }
  }

  printf("Volume exclusion Mask has been generated! \n");
  printf("Total Gridpoints:     %i \n",sizex*sizey*sizez);
  printf("Accepted Gridpoints   %i \n",num_of_gridpoints);
  printf("Excluded Gridpoints   %i \n",sizex*sizey*sizez-num_of_gridpoints);
  
  retval.pottable = pottable;
  retval.sizex = sizex;
  retval.sizey = sizey;
  retval.sizez = sizez;
  retval.dx = xx;
  retval.dy = yy;
  retval.dz = zz;
  retval.origin_x = origin_x;
  retval.origin_y = origin_y;
  retval.origin_z = origin_z;
  retval.mask = mask;
  retval.contraction_buffer = (real*)malloc(sizeof(real)*sizex*sizey);
  retval.coordinates = coordinates;
  retval.num_of_atoms = num_of_atoms;
  retval.isGood = 1;

  return retval;
}

void enqueue_kernel(opencl_stuff gpu_tools,cube_def cube,int whichcube,
		    int commandqueue) {
  cl_int err;
  size_t dimensions_kernel;
  size_t dimensions_contraction;
  cl_kernel kernel = gpu_tools.kernel[commandqueue];
  cl_kernel contraction = gpu_tools.contraction[commandqueue];
  cl_command_queue cmdq = gpu_tools.cmdq[commandqueue];

  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), 
		       gpu_tools.gpu_potential+whichcube);
  err = clSetKernelArg(kernel, 2, sizeof(cl_mem), 
		       gpu_tools.gpu_charges_locx+whichcube);
  err = clSetKernelArg(kernel, 3, sizeof(cl_mem), 
		       gpu_tools.gpu_charges_locy+whichcube);
  err = clSetKernelArg(kernel, 4, sizeof(cl_mem), 
		       gpu_tools.gpu_charges_locz+whichcube);
  err = clSetKernelArg(kernel, 6, sizeof(int), &cube.sizex); 
  err = clSetKernelArg(kernel, 7, sizeof(int), &cube.sizey); 
  err = clSetKernelArg(kernel, 8, sizeof(int), &cube.sizez); 
  
  err = clSetKernelArg(kernel, 9, sizeof(real), &cube.dx); 
  err = clSetKernelArg(kernel, 10, sizeof(real), &cube.dy); 
  err = clSetKernelArg(kernel, 11, sizeof(real), &cube.dz); 
  
  err = clSetKernelArg(kernel, 12, sizeof(real),
		       &cube.origin_x);
  err = clSetKernelArg(kernel, 13, sizeof(real), 
		       &cube.origin_y);
  err = clSetKernelArg(kernel, 14, sizeof(real), 
		       &cube.origin_z);

  err = clSetKernelArg(contraction, 0, sizeof(cl_mem),
		       gpu_tools.gpu_potential+whichcube);
  err = clSetKernelArg(contraction, 1, sizeof(cl_mem),
		       gpu_tools.gpu_pottable+whichcube);
  err = clSetKernelArg(contraction, 2, sizeof(cl_mem),
		       gpu_tools.gpu_contraction+whichcube);
  err = clSetKernelArg(contraction, 3, sizeof(cl_mem),
		       gpu_tools.gpu_mask+whichcube);
  err = clSetKernelArg(contraction, 4, sizeof(int), &cube.sizez);

  
  dimensions_kernel = (size_t)cube.sizex
    *cube.sizey
    *cube.sizez;

  dimensions_contraction = (size_t)cube.sizex*cube.sizey;

  err = clEnqueueNDRangeKernel(gpu_tools.cmdq[commandqueue],
			       kernel,1,NULL,&dimensions_kernel,
			       NULL,0,NULL,NULL);
  
  err = clEnqueueNDRangeKernel(gpu_tools.cmdq[commandqueue],
			       contraction,1,NULL,
			       &dimensions_contraction,
			       NULL,0,NULL,NULL);
  //  printf("\n\n\nEnqueue CODE %i\n\n\n",err);
}
				       
real calcpotential(opencl_stuff gpu_tools,
		   cube_def* cubes,
		   int num_cubes,
		   atom_database base) {
  
  cl_int err;
  
  int i,j,whichcube;
  int cubes_by_gpus;
  int cubes_rest;
  
  real retval = 0.;

  for(i=0; i<gpu_tools.num_gpus; i++) {
    err = clEnqueueWriteBuffer(gpu_tools.cmdq[i],
			       gpu_tools.gpu_charges[i], CL_TRUE, 0,
			       sizeof(real)*base.num_of_atoms,
			       (void*)base.big_charges_vector,0,NULL,NULL);
  }

  cubes_by_gpus=num_cubes/gpu_tools.num_gpus;
  cubes_rest=num_cubes%gpu_tools.num_gpus;

  for(i=0;i<gpu_tools.num_gpus;i++) {
    for(j=0;j<cubes_by_gpus;j++){
      whichcube = i*cubes_by_gpus+j;
      enqueue_kernel(gpu_tools,cubes[whichcube],whichcube,i);
    }
  }

  for (j=0;j<cubes_rest;j++){
    whichcube = gpu_tools.num_gpus*cubes_by_gpus+j;
    enqueue_kernel(gpu_tools,cubes[whichcube],whichcube,j);
  }
  
  for(i=0; i<gpu_tools.num_gpus; i++) {
    clFinish(gpu_tools.cmdq[i]);
  }
  
  for(i=0;i<gpu_tools.num_gpus;i++) {
    for(j=0;j<cubes_by_gpus;j++){
      whichcube = i*cubes_by_gpus+j;
      clEnqueueReadBuffer(gpu_tools.cmdq[i],
			  gpu_tools.gpu_contraction[whichcube],
			  CL_TRUE, 0,
			  sizeof(real)*cubes[whichcube].sizex
			  *cubes[whichcube].sizey,
			  (void*)cubes[whichcube].contraction_buffer,
			  0, NULL, NULL);
    }
  }
  
  for (j=0;j<cubes_rest;j++){
    whichcube = gpu_tools.num_gpus*cubes_by_gpus+j;
    clEnqueueReadBuffer(gpu_tools.cmdq[j],
			gpu_tools.gpu_contraction[whichcube],
			CL_TRUE, 0,
			sizeof(real)*cubes[whichcube].sizex
			*cubes[whichcube].sizey,
			(void*)cubes[whichcube].contraction_buffer,
			0, NULL, NULL);
    
  }
  
  for(i=0; i<gpu_tools.num_gpus; i++) {
    clFinish(gpu_tools.cmdq[i]);
  } 
}

real calcmin(opencl_stuff gpu_tools,
	     cube_def* cubes,
	     int num_cubes,
	     atom_database base,
	     char norm_bool) {

  int i,j;
  real current_minimum;
    
  real bigsum = 0.;
  real difference,corrector;
  int dimension;
  real y,t;
  cube_def cube;

  calcpotential(gpu_tools,cubes,num_cubes,base);

  for (i=0;i<num_cubes;i++) {
    cube = cubes[i];
    dimension = cube.sizex*cube.sizey;
    corrector = 0.;
    difference = 0.;
    for (j=0;j<dimension;j++) {
      y = cube.contraction_buffer[j]-corrector;
      t = difference + y;
      corrector = (t-difference)-y;
      difference = t;
    }
    bigsum += difference;
  }
  return (bigsum);
}
 

double checkdrift(real* big_charges_vector,
		  double sum_of_charge,
		  int num_of_atoms) {
  int i;
  double sum=0;
  double drift;
  for (i=0;i<num_of_atoms;i++) {
    sum += (double)big_charges_vector[i];
  }
  drift = sum-sum_of_charge;
  return(drift);
}

void correctdrift(double drift,
		  double* small_charges_vector,
		  double sum_of_charge,
		  int num_of_atomtypes) {
  int i;
  double current_sum_of_charge = drift+sum_of_charge;
  for (i=0;i<num_of_atomtypes;i++) {
    small_charges_vector[i] = 
      small_charges_vector[i]*(sum_of_charge/current_sum_of_charge);
  }
} 


void calculate_displacement_vector(double* plane_normal,
				   double* charges_gradient,
				   double* displacementvector,
				   int num_of_types) {
  
  int i;
  double projection=0.;
  double normalization_factor=0.;
  for(i=0;i<num_of_types;i++) {
    projection+= plane_normal[i]*charges_gradient[i];
  }
  for(i=0;i<num_of_types;i++) {
    displacementvector[i] = charges_gradient[i]-plane_normal[i]*projection;
    normalization_factor += displacementvector[i]*displacementvector[i];
  }
  normalization_factor = 1/sqrt(normalization_factor);
  for(i=0;i<num_of_types;i++) {
    displacementvector[i] = displacementvector[i]*normalization_factor;
  }
}

void calc_grad_min(opencl_stuff gpu_tools,
		   atom_database base,
		   cube_def* cubes,
		   int num_cubes,
		   char norm_bool,
		   real precision) {
  
  int i;
  real a, b;
  real buffer;
  double normalizer = 0.;

  for (i=0;i<base.num_of_atomtypes;i++) {
    buffer = base.small_charges_vector[i];
    
    // step forwards
    base.small_charges_vector[i] = buffer+precision;

    small_to_big_charges_vector(base.big_charges_vector,
				base.small_charges_vector,
				base.atomtypes,
				base.num_of_atomtypes);
    
    a = calcmin(gpu_tools,cubes,num_cubes,base,norm_bool);
    
    // step backwards
    base.small_charges_vector[i] = buffer-precision;
    
    small_to_big_charges_vector(base.big_charges_vector,
				base.small_charges_vector,
				base.atomtypes,
				base.num_of_atomtypes);
    
    b = calcmin(gpu_tools,cubes,num_cubes,base,norm_bool);

    base.gradient[i] = (a-b)/(2.*precision);
    normalizer += (double)base.gradient[i]*
      (double)base.gradient[i];
    
    base.small_charges_vector[i] = buffer;
    
    small_to_big_charges_vector(base.big_charges_vector,
				base.small_charges_vector,
				base.atomtypes,
				base.num_of_atomtypes);
  }
  for (i=0;i<base.num_of_atomtypes;i++) {
    base.gradient[i]=base.gradient[i]/(sqrt(normalizer));
  }
}

double get_stepsize(opencl_stuff gpu_tools,
		    double stepsize,
		    int num_of_iterations,
		    double minimum_displacement,
		    char norm_bool,
		    atom_database base,
		    cube_def* cubes,
		    int num_cubes) {
  
  double stepsize_differential;
  int iterator,i,j;
  double forward,backward,here;
  double displacement = 0;
  double start,end,intervallength;
  
  double* buffer = malloc(sizeof(double)*base.num_of_atomtypes);
  
  iterator = 0;
  do{
    iterator++;
    
    for(i=0;i<base.num_of_atomtypes;i++) {
      buffer[i]= -(displacement+minimum_displacement)
	*base.displacementvector[i]+base.small_charges_vector[i];
    }
    small_to_big_charges_vector(base.big_charges_vector,
				buffer,
				base.atomtypes,
				base.num_of_atomtypes);

    forward = (double)calcmin(gpu_tools,cubes,num_cubes,base,norm_bool);

    
    for(i=0;i<base.num_of_atomtypes;i++) {
      buffer[i]= -(displacement-minimum_displacement)
	*base.displacementvector[i]+base.small_charges_vector[i];
    }
    small_to_big_charges_vector(base.big_charges_vector,
				buffer,
				base.atomtypes,
				base.num_of_atomtypes);
    
    backward = (double)calcmin(gpu_tools,cubes,num_cubes,base,norm_bool);
    
    displacement = iterator*stepsize;
    stepsize_differential = 
      (forward-backward)/(2.f*minimum_displacement);
    printf("XXIterator %i XXstepsize_differential %f\n",
	   iterator,
	   stepsize_differential);
    
  }while(stepsize_differential < 0);
  
  printf("iterator %i\n",iterator);
  iterator=iterator-2;
  
  start = stepsize*(iterator); 
  end = stepsize*(iterator+1);
  intervallength = end - start;

  for(j=0;j<num_of_iterations;j++) {
    
    displacement = start+intervallength/2;
    
    printf("Interieur Displacement %f \n",displacement);
    
     for(i=0;i<base.num_of_atomtypes;i++) {
      buffer[i]= -(displacement+minimum_displacement)
	*base.displacementvector[i]+base.small_charges_vector[i];
    }
    small_to_big_charges_vector(base.big_charges_vector,
				buffer,
				base.atomtypes,
				base.num_of_atomtypes);

    forward = (double)calcmin(gpu_tools,cubes,num_cubes,base,norm_bool);

    
    for(i=0;i<base.num_of_atomtypes;i++) {
      buffer[i]= -(displacement-minimum_displacement)
	*base.displacementvector[i]+base.small_charges_vector[i];
    }
    small_to_big_charges_vector(base.big_charges_vector,
				buffer,
				base.atomtypes,
				base.num_of_atomtypes);

    backward = (double)calcmin(gpu_tools,cubes,num_cubes,base,norm_bool);
    
    displacement = iterator*stepsize;
     
    stepsize_differential = 
      (forward-backward)/(2.f*minimum_displacement);

    if(stepsize_differential < 0) {
      start = start+intervallength/2;
      intervallength = intervallength/2; 
    } else {
      end = end - intervallength/2;
      intervallength = intervallength/2;
    }
    if(j == num_of_iterations-1 && start == 0.f) {
      j--;
    }
  }
   
  small_to_big_charges_vector(base.big_charges_vector,
			      buffer,
			      base.atomtypes,
			      base.num_of_atomtypes);
  free(buffer);
  return(start);
}

void move_charges(atom_database base,double stepsize) {

  int i;
  for(i=0;i<base.num_of_atomtypes;i++) {
    base.small_charges_vector[i]=base.small_charges_vector[i]-
      stepsize*base.displacementvector[i];
  }
  
  small_to_big_charges_vector(base.big_charges_vector,
			      base.small_charges_vector,
			      base.atomtypes,
			      base.num_of_atomtypes);
}

void verify_charge_drift(atom_database base, double driftcrit) {

  double drift;
  drift = checkdrift(base.big_charges_vector,
		     base.sum_of_charge,base.num_of_atoms);

  printf("Current drift from charge conservation: %lf \n",drift);
  
  if(drift > driftcrit) {
      printf("Current Drift %lf > 1/10*convergence criterium \n",drift);
      printf("Correcting Drift ! \n");
      correctdrift(drift,
		   base.small_charges_vector,
		   base.sum_of_charge,
		   base.num_of_atomtypes);
      small_to_big_charges_vector(base.big_charges_vector,
				  base.small_charges_vector,
				  base.atomtypes,
				  base.num_of_atomtypes);
    }
}

void delete_cubes(cube_def* cubes,int num_cubes) {
  int i;
  for(i=0;i<num_cubes;i++) {
    free(cubes[i].contraction_buffer);
    free(cubes[i].pottable);
    free(cubes[i].mask);
    free(cubes[i].coordinates.x);
    free(cubes[i].coordinates.y);
    free(cubes[i].coordinates.z);
    free(cubes[i].coordinates.charge);
  }
  free(cubes);
}

void delete_database(atom_database base) {
  int i;
  for (i=0;i<base.num_of_atomtypes;i++) {
    free(base.atomtypes[i].in_atoms);
    free(base.atomtypes[i].charges_vector_locations);
  }
  free(base.atomtypes);
  free(base.big_charges_vector);
  free(base.small_charges_vector);
  free(base.big_vdw_vector);
  free(base.plane_normal);
  free(base.gradient);
  free(base.displacementvector);
}

void opencl_cleanup(opencl_stuff gpu_tools, int num_cubes) {
  
  int i;

  for(i=0;i<num_cubes;i++) {
    clReleaseMemObject(gpu_tools.gpu_potential[i]);
    clReleaseMemObject(gpu_tools.gpu_charges_locx[i]);
    clReleaseMemObject(gpu_tools.gpu_charges_locy[i]);
    clReleaseMemObject(gpu_tools.gpu_charges_locz[i]);
    clReleaseMemObject(gpu_tools.gpu_mask[i]);
    clReleaseMemObject(gpu_tools.gpu_pottable[i]);
  }
  for(i=0;i<gpu_tools.num_gpus;i++) {
    clReleaseCommandQueue(gpu_tools.cmdq[i]);
    clReleaseMemObject(gpu_tools.gpu_charges[i]);
  }
}


int main(int argc, char** argv) {

  int non_cube_arguments=10;

  atom_database base;
  cube_def* cubes;

  int num_cubes;
  int num_of_atoms;

  int norm_bool_in;
  char norm_bool;

  int i,steps,conv_check=0.;
  
  real before,after,diff;
  
  real stepsize, grad_precision,conv_crit;

  opencl_stuff gpu_tools;
  int num_gpus;

#ifdef _DDOUBLE
  char real_scan[] = "%lf";
#else
  char real_scan[] = "%f";
#endif

  FILE* outputfile = fopen(argv[3],"w+");
  if (NULL == outputfile) {
    printf("Outputfile could not be created!\n");
    usage();
    return 1;
  }
  
  // Reading command line arguments
  
  if( sscanf(argv[4],"%i",&num_gpus) != 1) {
    printf("Error: Argument 4 not given in a correct manner!\n");
    usage();
    return 1; 
  }

  if( sscanf(argv[5],"%i",&norm_bool_in) != 1) {
    printf("Error: Argument 5 not given in a correct manner!\n");
    usage();
    return 1; 
  }
  norm_bool = (char) norm_bool_in;
  printf("Norm Bool Exterieur %i\n",(int)norm_bool) ;
  if (norm_bool != 1 && norm_bool !=2 && norm_bool !=3 && norm_bool != 4) {
    printf("%c",norm_bool) ;
    printf("Error: Argument 5 not given in a correct manner!\n");
    usage();
    return 1; 
  }
  if ( sscanf(argv[6],real_scan,&stepsize) !=1) {
    printf("Error: Argument 6 not given in a correct manner!\n");
    usage();
    return 1; 
  }
  if ( sscanf(argv[7],real_scan,&grad_precision) !=1) {
    printf("Error: Argument 7 not given in a correct manner!\n");
    usage();
    return 1; 
  }
  if( sscanf(argv[8],real_scan,&conv_crit) != 1 ) {
    printf("Error: Argument 8 not given in a correct manner!\n");
    usage();
    return 1;
  }
  if ( sscanf(argv[9],"%i",&steps) != 1) {
    printf("Error: Argument 9 not given in a correct manner!\n");
    usage();
    return 1;
  }

  num_cubes = argc-non_cube_arguments;
  if (num_cubes < 1) {
    usage();
    return -1;
  }

  printf("Number of cubes: %i\n",num_cubes);

  num_of_atoms = get_num_of_atoms_from_a_cubefile(argv[non_cube_arguments]);

  cubes = (cube_def*)malloc(num_cubes*sizeof(cube_def));
  
  base = generate_inital_database(argv[2],argv[1],num_of_atoms);

  if (base.num_of_atomtypes == -1) {
    usage();
    return -1;
  }
  
  for(i=0;i<num_cubes;i++) {
    cubes[i] = getCubeProperties(argv[i+non_cube_arguments],base);
    if(!cubes[i].isGood) {
      usage();
      return -1;
    }
  }

  gpu_tools = opencl_initialization(base,cubes,num_cubes,num_gpus,norm_bool);

  // Wireing signals before the begin of the main compute loop
  
  signal(SIGINT, signal_handler);
  signal(SIGTERM, signal_handler);

  before = calcmin(gpu_tools,cubes,num_cubes,base,norm_bool);
  
  for(i=0;i<steps;i++) {

    if(killed == 1) {
      printf("WARNING GOT KILLED! \n");
      fprintf(outputfile,
	      "WARNING GOT KILLED! CHARGES OF AN UNFINISHED JOB!\n");
      for(i=0;i<base.num_of_atoms;i++){
	fprintf(outputfile,"%f\n",base.big_charges_vector[i]);
      }
      // Missing cleanunp 
      
      delete_cubes(cubes,num_cubes);
      delete_database(base);
      return 9;
    }
    
    calc_grad_min(gpu_tools,base,cubes,num_cubes,norm_bool,grad_precision);
    
    calculate_displacement_vector(base.plane_normal,
				  base.gradient,
				  base.displacementvector,
				  base.num_of_atomtypes);
    
    printf("Stepsize before %f \n",stepsize);
    
    stepsize=get_stepsize(gpu_tools,stepsize,5,grad_precision,norm_bool,
			  base,cubes, num_cubes);

    printf("Stepsize afterwards %f \n",stepsize);

    move_charges(base,stepsize);
    
    after = calcmin(gpu_tools,cubes,num_cubes,base,norm_bool);

    diff = before - after;
    
    printf("%i\n steps done!\n Difference = %f \n Before = %f \n After %f \n"
	   ,i,diff,before,after);

    before = after;
    
    //verify_charge_drift(base,conv_crit/10.);
    
    printf("Stepsize %f: \n",stepsize);
    if ( stepsize < conv_crit ) {
      printf("Convergence criteria reached after %i steps!\n",i);
      i = steps;
      conv_check = 6; 
    }  
    
  }
  if( conv_check != 6 ) {
    printf("%i steps done but convergence criteria has not been reached\n",i);
  }
  
  printf("Charges: \n");
  for(i=0;i<base.num_of_atoms;i++){
    fprintf(outputfile,"%f\n",base.big_charges_vector[i]);
  }
  // Missing cleanunp 

  delete_cubes(cubes,num_cubes);
  delete_database(base);
  
}
  
