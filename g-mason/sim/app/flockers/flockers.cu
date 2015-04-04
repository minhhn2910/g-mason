extern "C" __device__ int get_bag_index(int x, int y, int width, int height)
{
	//x = width, y = height
	if(x <0) x = width - 1;
	if(y < 0) y = height -1;
	if(x >=width) x = 0;
	if(y >= height) y = 0;
	return y*width+x;
}
extern "C" __device__ double _stx(double x, double width) 
        { if (x >= 0) { if (x < width) return x; return x - width; } return x + width; }

   
extern "C" __device__  double tdx(double x1, double x2,double width)
        {
			
			if (fabs(x1-x2) <= width / 2)
            return x1 - x2;  
			
			double dx = _stx(x1,width) - _stx(x2,width);
			if (dx * 2 > width) return dx - width;
			if (dx * 2 < -width) return dx + width;
			return dx;
        }
    

extern "C" __device__  double _sty( double y, double height) 
        { if (y >= 0) { if (y < height) return y ; return y - height; } return y + height; }


extern "C" __device__  double tdy(double y1, double y2, double height)
        {
		
        if (fabs(y1-y2) <= height / 2)
            return y1 - y2;  // no wraparounds  -- quick and dirty check

        double dy = _sty(y1,height) - _sty(y2,height);
        if (dy * 2 > height) return dy - height;
        if (dy * 2 < -height) return dy + height;
        return dy;
        }

//end of continuos space device function, now the atomicAdd        
//the kernel     

extern "C" __global__ void flockers(
											int numthread,
											int begin,    //begin offset when this kernel start to work from that index
											int width_discreted,
											int height_discreted,
											double width, 
											double height,
											double neighbor,										
											int	    *bag_size,
											int  	**bag_index,												
											int *is_dead, 
//this part of parameters for the last phase @@ too many parameters
						double cohesion,
						double avoidance,
						double consistency,
						double randomness,
						double momentum ,
						double jump,
						double *random,											
											double *data,
											double *old_data
											)

{
//index = index in sorted array x
//real index is value in index_sorted

   int index = blockIdx.x * blockDim.x + threadIdx.x;
   if (index<numthread && is_dead[index] ==0)
    {
		index += begin;   //multiGPUs
		
		int my_index = index*4;
		int his_index = 0; 
		int data_index = 0;
		//value for collect info
		int num_neighbor = 0;
		int num_non_dead = 0;
		double cons_x = 0.0;
		double cons_y = 0.0;
		double cohe_x = 0.0;
		double cohe_y = 0.0;
		double avoid_x = 0.0;
		double avoid_y = 0.0;
		
		//a little bit complicated for better performance
		double	me_x = old_data[my_index];
        double	me_y = old_data[my_index+1];
        double 	old_dx = old_data[my_index+2];
        double  old_dy = old_data[my_index + 3]; 
        
        double	him_x = 0.0;
        double	him_y = 0.0;
        double  his_dx = 0.0;
        double  his_dy = 0.0;
        int isdead = 0;
		
        double temp_tdx = width;
        double temp_tdy = height;
        double len = 0.0;

		int my_bag_row = (int)(me_y/neighbor);;
		int my_bag_column = (int)(me_x/neighbor);;

//Mason find neighbor algorithm
		for(int runx = my_bag_column - 1; runx <= my_bag_column +1; runx++)
			for(int runy = my_bag_row-1; runy <=my_bag_row +1; runy++)
			{
				int local_bag_index = get_bag_index(runx, runy, width_discreted,height_discreted);
				for(int run = 0; run < bag_size[local_bag_index]; run ++)
				{
					his_index = bag_index[local_bag_index][run];
					data_index = his_index*4;//multiple arrays to one:D
					//coalescing here
					him_x = old_data[data_index];
					him_y = old_data[data_index+1];
					his_dx = old_data[data_index+2];
					his_dy = old_data[data_index+3];
					
					isdead = is_dead[his_index];
					temp_tdx =  tdx(me_x,him_x,width);
					temp_tdy =  tdy(me_y,him_y,height);
					
					len = hypot(temp_tdx,temp_tdy);
					if(len <= neighbor)
					{
							double temp_value =  (pow(len,4) + 1);
							double temp_avoidance_x = temp_tdx/temp_value;
							double temp_avoidance_y = temp_tdy/temp_value;

							if(isdead==0)
							{

								cons_x += his_dx;
								cons_y += his_dy;
								cohe_x += temp_tdx;
								cohe_y += temp_tdy;
								num_non_dead ++;
							}

							if(his_index != index)
							{
								avoid_x += temp_avoidance_x;
								avoid_y += temp_avoidance_y;
								num_neighbor++;
							} 
        
					} 
						
				}
			}
				


	//phase 3: collect info, seperation is not necessary now, combine two kernels to one
/*
	vector_x[my_vector_index+2] = cons_x;
	vector_y[my_vector_index+2] = cons_y;
	
	vector_x[my_vector_index] = cohe_x;
	vector_y[my_vector_index] = cohe_y;
	
	count_non_dead[index] = num_non_dead;
	
	
	vector_x[my_vector_index+1] = avoid_x;
	vector_y[my_vector_index+1] = avoid_y;
	
	count_neighbor[index] = num_neighbor;
*/	

			double rand_x = fma (random[index*2],2.0,-1.0);//random[index*2]*2-1.0;
			double rand_y = fma (random[index*2+1],2.0,-1.0);
	//just keep this, two devices generate random concurrently, modify later
	
			double rand_length = hypot(rand_x,rand_y);
			rand_x = 0.05*rand_x/rand_length;
			rand_y = 0.05*rand_y/rand_length;
			
			if (num_non_dead > 0)
			{ 
				cohe_x = cohe_x/num_non_dead; 
				cohe_y = cohe_y/num_non_dead; 

				cons_x = cons_x/num_non_dead; 
				cons_y = cons_y/num_non_dead; 			
			}

			if(num_neighbor > 0)
			{
				avoid_x = avoid_x/num_neighbor;
				avoid_y = avoid_y/num_neighbor;
			}

			cohe_x = -cohe_x/10;
			cohe_y = -cohe_y/10;	
			avoid_x = 400*avoid_x;
			avoid_y = 400*avoid_y;

			double my_dx = cohesion * cohe_x + avoidance * avoid_x + consistency* cons_x + randomness * rand_x + momentum *old_dx; 
			double my_dy = cohesion * cohe_y + avoidance * avoid_y + consistency* cons_y + randomness * rand_y + momentum *old_dy;
               
			double dis = hypot(my_dx,my_dy);

			if (dis>0)
			{
					double value = jump / dis;
					my_dx = my_dx *value;
					my_dy = my_dy *value;
			}
	
	//copy back to global memory


				data[my_index] = _stx(me_x + my_dx,width);
				data[my_index+1] = _sty(me_y + my_dy,height);

				data[my_index+2] = my_dx;
				data[my_index+3] = my_dy;
		
			

	
	}

}
