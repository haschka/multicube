#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void epot (__global double* potential,
		    __global double* charges,
		    __global double* locx,
		    __global double* locy,
		    __global double* locz,
		    int num_of_charges,
		    int sizex,int sizey,int sizez,
		    double dx, double dy, double dz,
		    double origin_x, double origin_y, double origin_z) {
  int j;
  int sizexBYsizey = sizex*sizey;

  int index = get_global_id(0);
  
  int index_z = index/sizexBYsizey;
  int index_y = (index-index_z*sizexBYsizey)/sizex;
  int index_x = index-index_z*sizexBYsizey-index_y*sizex; 
  
  double distance_x, distance_y, distance_z;

  double sum = 0.f;
  double corrector = 0.f;
  double y,t;

  for(j=0;j<num_of_charges;j++) {
    
    distance_x = (origin_x+((double)index_x)*dx)-locx[j];
    distance_y = (origin_y+((double)index_y)*dy)-locy[j];
    distance_z = (origin_z+((double)index_z)*dz)-locz[j];
    
    y = charges[j]
      *rsqrt(distance_x*distance_x
	     +distance_y*distance_y
	     +distance_z*distance_z)-corrector;
    t = sum + y;
    corrector = (t-sum)-y;
    sum = t;    
  }
  potential[index]=sum;
}

__kernel void contractionL1(__global double* potential,
			    __global double* realpotential,
			    __global double* contracted,
			    __global char* mask,
			    int num_of_points_per_contraction) {
  
  int contraction_number = get_global_id(0);
  int contracted_size = get_global_size(0);
  // int num_of_points_per_contraction = dimension/contracted_size;

  int local_location;

  int i;

  double sum = 0.f;
  double corrector = 0.f;
  double y,t;

  double diff;
  
  for(i=0;i<num_of_points_per_contraction;i++) {
    local_location = contraction_number*num_of_points_per_contraction+i;
    if (mask[local_location] != 1) {
      
      diff = potential[local_location]-realpotential[local_location];
      
      y = fabs(diff)-corrector;
      t = sum + y;
      corrector = (t-sum)-y;
      sum = t;
    }    
  }
  contracted[contraction_number] = sum;
}

__kernel void contractionL2(__global double* potential,
			    __global double* realpotential,
			    __global double* contracted,
			    __global char* mask,
			    int num_of_points_per_contraction) {

  
  int contraction_number = get_global_id(0);
  int contracted_size = get_global_size(0);
  //  int num_of_points_per_contraction = dimension/contracted_size;

  int local_location;

  int i;

  double sum = 0.f;
  double corrector = 0.f;
  double y,t;

  double diff;
  
  for(i=0;i<num_of_points_per_contraction;i++) {
    local_location = contraction_number*num_of_points_per_contraction+i;
    if (mask[local_location] != 1) {
      
      diff = potential[local_location]-realpotential[local_location];
      
      y = diff*diff-corrector;
      t = sum + y;
      corrector = (t-sum)-y;
      sum = t;
    }    
  }
  contracted[contraction_number] = sum;

}

__kernel void contractionL1w(__global double* potential,
			     __global double* realpotential,
			     __global double* contracted,
			     __global char* mask,
			     int num_of_points_per_contraction) {
  
  double small = 0.000000010536;

  int contraction_number = get_global_id(0);
  int contracted_size = get_global_size(0);
  //int num_of_points_per_contraction = dimension/contracted_size;

  int local_location;

  int i;

  double sum = 0.f;
  double corrector = 0.f;
  double y,t;

  double diff;
  
  for(i=0;i<num_of_points_per_contraction;i++) {
    local_location = contraction_number*num_of_points_per_contraction+i;
    if (mask[local_location] != 1) {
      
      diff = potential[local_location]-realpotential[local_location];
      
      y = fabs(diff)/(fabs(realpotential[local_location])+small)-corrector;
      t = sum + y;
      corrector = (t-sum)-y;
      sum = t;
    }    
  }
  contracted[contraction_number] = sum;
}

__kernel void contractionL2w(__global double* potential,
			     __global double* realpotential,
			     __global double* contracted,
			     __global char* mask,
			     int num_of_points_per_contraction) {
  
  double small = 0.000000010536;

  int contraction_number = get_global_id(0);
  int contracted_size = get_global_size(0);
  //int num_of_points_per_contraction = dimension/contracted_size;

  int local_location;

  int i;

  double sum = 0.f;
  double corrector = 0.f;
  double y,t;

  double diff;
  
  for(i=0;i<num_of_points_per_contraction;i++) {
    local_location = contraction_number*num_of_points_per_contraction+i;
    if (mask[local_location] != 1) {
      
      diff = potential[local_location]-realpotential[local_location];
      
      y = diff*diff
	/(realpotential[local_location]*realpotential[local_location]+small)
	-corrector;
      t = sum + y;
      corrector = (t-sum)-y;
      sum = t;
    }    
  }
  contracted[contraction_number] = sum;
}
