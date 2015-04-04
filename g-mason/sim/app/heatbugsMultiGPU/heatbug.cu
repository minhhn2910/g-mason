extern "C" __device__ int stx(int x, int width) 
        { if (x >= 0) { if (x < width) return x; return x - width; } return x + width; }


//this sty for double2D redundant 4 rows only, 
//extern "C" __device__ int sty(int y, int height) 
        //{ if (y > 1) { if (y <= height) return y ; return y - height; } return y + height; }
 ////not used
extern "C" __device__ int sty(int y, int height) 
        { if (y >= 0) { if (y < height) return y ; return y - height; } return y + height; }
 


extern "C" __device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                          (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
            old = atomicCAS(address_as_ull, assumed,__double_as_longlong(val +
                               __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}


extern "C" __device__ bool randBoolean(double rand,double probability)
{
	return (rand <=probability);   //because CURAND generator : exclude 0, include 1
}

extern "C" __device__ int randInt(double rand, int n)
{
	if (rand == 1) return n-1;
	
	return (int)(rand*n);   //because CURAND generator : exclude 0, include 1
	
}

extern "C" __global__ void heatbug(
										int	numThreads,
										int width,
										int height,
										int begin_row,
										int end_row,
										int offset,  
										int device,
										double randomMovementProbability,
										double MAX_HEAT,
										int* best,
										int *location,
										double *valgrid,
										double *idealTemp,
										double *random,
										double *heatOutput
									)

{

   int index = blockIdx.x * blockDim.x + threadIdx.x;
   if (index<numThreads)
    {
        int myx = location[index*2];
        int myy = location[index*2+1];
        
        //this version is not gonna work on 1 gpu
		if(end_row == height-1 && myy == 0) //im in zero row, equal to height
		{
			myy = height;
		}
		else if(begin_row == 0 && myy == height-1) 
		{
			myy = -1;
		}	        
        if(myy<begin_row -1 || myy>end_row+1)
        {
			//~ bestx_vector[index+offset] = 0;
			//~ besty_vector[index+offset] = 0;
			best[index*2+offset] =0;
			best[index*2+offset+1]=0;
			return;		
		}
		
		myy+= 2;	
      

        int START=-1;
        int bestx = START;
        int besty = 0;
        
        double rand_0 = random[index*3];
        double rand_1 = random[index*3+1];
        double rand_2 = random[index*3+2];
        
        int my_grid_index=  myx+ (myy-begin_row)*width;
        
        double my_temp = valgrid[my_grid_index];
        double my_idealTemp = idealTemp[index];
        



        
        if (randBoolean(rand_0,randomMovementProbability))  // go to random place
            {
				bestx = stx(randInt(rand_1,3) - 1 + myx,width);  // toroidal
				besty = randInt(rand_2,3) - 1 + myy;//sty(randInt(rand_2,3) - 1 + myy,height);  // toroidal
            }
        else if( my_temp > my_idealTemp )  // go to coldest place
            {
            for(int x=-1;x<2;x++)
                for (int y=-1;y<2;y++)
                    if (!(x==0 && y==0))
                        {
							int xx = stx(x + myx,width);    // toroidal
							int yy = y+myy;//sty(y + myy,height);      
								
							double current = valgrid[xx+(yy-begin_row)*width];
                        
							double best = valgrid[bestx+(besty-begin_row)*width];
                        
                        
                        if (bestx==START ||
                            ( current < best) ||
                            ((current == best) &&  randBoolean(rand_1,0.5)))  // not uniform, but enough to break up the go-up-and-to-the-left syndrome
                            { bestx = xx; besty = yy; }
                        }
            }
        else if ( my_temp < my_idealTemp )  // go to warmest place
            {
				for(int x=-1;x<2;x++)
					for (int y=-1;y<2;y++)
						if (!(x==0 && y==0))
                        {
							int xx = stx(x + myx,width);    // toroidal
							int yy = y+myy;//sty(y + myy,height);       // toroidal
                        
							double current = valgrid[xx+(yy-begin_row)*width];
							double best = valgrid[bestx+(besty-begin_row)*width];
                        
							if (bestx==START || 
								(current > best) ||
								((current ==  best) && randBoolean(rand_1,0.5)))  // not uniform, but enough to break up the go-up-and-to-the-left syndrome
								{ bestx = xx; besty = yy; }
								
                        }
            }
        else            // stay put
            {
				bestx = myx;
				besty = myy;
            }


        //loc_x[index] = bestx;
        
        //loc_y[index] = besty-1; // add 1 then sub 1, fair play :D
        
			//~ bestx_vector[index*2+offset] = bestx;
			//~ 
			//~ besty_vector[index+offset] = sty(besty-2,height);
			best[index*2+offset] = bestx;
			
			best[index*2+offset+1] = sty(besty-2,height);
			

		if (best[index*2+offset+1] >=begin_row &&  best[index*2+offset+1] <=end_row)
		{ 
				
			int best_grid_index = bestx +(besty-begin_row)*width;
			atomicAdd(&valgrid[best_grid_index],heatOutput[index]);
			if(valgrid[best_grid_index] > MAX_HEAT)
				valgrid[best_grid_index] = MAX_HEAT;
		}
		else
		{
			best[index*2+offset] = 0;
			best[index*2+offset+1] = 0;
		}
			
			
	}

}
