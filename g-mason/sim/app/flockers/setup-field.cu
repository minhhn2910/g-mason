extern "C" __global__ void setup_field(
										int	numPoints,
										double neighbor,
										int width_discreted,
										int height_discreted,
										double *data,
										int	    *bag_size,
										int  	**bag_index	
									)

{

   int index = blockIdx.x * blockDim.x + threadIdx.x;
   if (index<numPoints)
    {
		 
			int data_index = index*4;
			
			int row = (int)(data[data_index+1]/neighbor);
			int column =(int)(data[data_index]/neighbor);
			int my_index = row*width_discreted + column;
			int old_value = atomicAdd (&bag_size[my_index], 1 );
			bag_index[my_index][old_value] = index;
			//very simple ?
	}

}
