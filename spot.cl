__kernel void epot (__global float* potential,
		    __global float* charges,
		    __global float* locx,
		    __global float* locy,
		    __global float* locz,
		    int num_of_charges,
		    int sizex,int sizey,int sizez,
		    float dx, float dy, float dz,
		    float origin_x, float origin_y, float origin_z) {
  int j;
  int sizexBYsizey = sizex*sizey;

  int index = get_global_id(0);
  
  int index_z = index/sizexBYsizey;
  int index_y = (index-index_z*sizexBYsizey)/sizex;
  int index_x = index-index_z*sizexBYsizey-index_y*sizex; 
  
  float distance_x, distance_y, distance_z;

  float sum = 0.f;
  float corrector = 0.f;
  float y,t;

  for(j=0;j<num_of_charges;j++) {
    
    distance_x = (origin_x+((float)index_x)*dx)-locx[j];
    distance_y = (origin_y+((float)index_y)*dy)-locy[j];
    distance_z = (origin_z+((float)index_z)*dz)-locz[j];
    
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

__kernel void contractionL1(__global float* potential,
			    __global float* realpotential,
			    __global float* contracted,
			    __global char* mask,
			    int num_of_points_per_contraction) {
  
  int contraction_number = get_global_id(0);
  int contracted_size = get_global_size(0);
  // int num_of_points_per_contraction = dimension/contracted_size;

  int local_location;

  int i;

  float sum = 0.f;
  float corrector = 0.f;
  float y,t;

  float diff;
  
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

__kernel void contractionL2(__global float* potential,
			    __global float* realpotential,
			    __global float* contracted,
			    __global char* mask,
			    int num_of_points_per_contraction) {

  
  int contraction_number = get_global_id(0);
  int contracted_size = get_global_size(0);
  //  int num_of_points_per_contraction = dimension/contracted_size;

  int local_location;

  int i;

  float sum = 0.f;
  float corrector = 0.f;
  float y,t;

  float diff;
  
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

__kernel void contractionL1w(__global float* potential,
			     __global float* realpotential,
			     __global float* contracted,
			     __global char* mask,
			     int num_of_points_per_contraction) {
  
  float small = 0.000244131;

  int contraction_number = get_global_id(0);
  int contracted_size = get_global_size(0);
  //int num_of_points_per_contraction = dimension/contracted_size;

  int local_location;

  int i;

  float sum = 0.f;
  float corrector = 0.f;
  float y,t;

  float diff;
  
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

__kernel void contractionL2w(__global float* potential,
			     __global float* realpotential,
			     __global float* contracted,
			     __global char* mask,
			     int num_of_points_per_contraction) {
  
  float small = 0.000244131;

  int contraction_number = get_global_id(0);
  int contracted_size = get_global_size(0);
  //int num_of_points_per_contraction = dimension/contracted_size;

  int local_location;

  int i;

  float sum = 0.f;
  float corrector = 0.f;
  float y,t;

  float diff;
  
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
