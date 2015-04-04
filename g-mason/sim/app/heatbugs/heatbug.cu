extern "C" __device__ int stx(int x, int width) 
        { if (x >= 0) { if (x < width) return x; return x - width; } return x + width; }


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
										double randomMovementProbability,
										double MAX_HEAT,
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
              
        int START=-1;
        int bestx = START;
        int besty = 0;
        
        double rand_0 = random[index*3];
        double rand_1 = random[index*3+1];
        double rand_2 = random[index*3+2];
        
        double my_temp = valgrid[myx+myy*width];
        double my_idealTemp = idealTemp[index];
        
        
        if (randBoolean(rand_0,randomMovementProbability))  // go to random place
            {
				bestx = stx(randInt(rand_1,3) - 1 + myx,width);  // toroidal
				besty = sty(randInt(rand_2,3) - 1 + myy,height);  // toroidal
            }
        else if( my_temp > my_idealTemp )  // go to coldest place
            {
            for(int x=-1;x<2;x++)
                for (int y=-1;y<2;y++)
                    if (!(x==0 && y==0))
                        {
                        int xx = stx(x + myx,width);    // toroidal
                        int yy = sty(y + myy,height);       // toroidal
                        
                        double current = valgrid[xx+yy*width];
                        double best = valgrid[bestx+besty*width];
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
                        int yy = sty(y + myy,height);       // toroidal
                        
                        double current = valgrid[xx+yy*width];
                        double best = valgrid[bestx+besty*width];
                        
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


			location[index*2] = bestx;
			location[index*2+1] = besty;
        
			atomicAdd(&valgrid[bestx +besty*width],heatOutput[index]);
			if(valgrid[bestx +besty*width] > MAX_HEAT)
				valgrid[bestx +besty*width] = MAX_HEAT;

			
	}

}
