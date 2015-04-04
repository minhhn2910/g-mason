extern "C" __device__ int stx(int x, int width) 
        { if (x >= 0) { if (x < width) return x; return x - width; } return x + width; }


extern "C" __device__ int sty(int y, int height) 
        { if (y >= 0) { if (y < height) return y ; return y - height; } return y + height; }
        
extern "C" __global__ void diffuse(
										int	numThreads,
										int width,
										int height,
										double evaporationRate,
										double diffusionRate,
										double* valgrid,
										double* valgrid2
									)
//divide matrix by height, redudant 4 rows for sync between devices and eliminate sty() call - for speedup ??; 
//so, y will begin at y = 1;
{

   int index = blockIdx.x * blockDim.x + threadIdx.x;
   if (index<numThreads)
    {
		 
                 double average = 0.0;
                 
                 int old_index = index;
                 index+= 2*width;  //skip 2 rows
                 
                 int x = index %width;  //the same
                 int y = index /width;	//will become y+1;
                 
                 
                         // for each neighbor of that position
                         for(int dx=-1; dx< 2; dx++)
                             for(int dy=-1; dy<2; dy++)
                                 {
                                 // compute the toroidal <x,y> position of the neighbor
									int xx = stx(x+dx,width);
									int yy = y+dy;//sty(y+dy,height); 
                                                                
                                 // compute average
									average += valgrid[xx+yy*width];
                                 }
                         average /= 9.0;
                        
                         // load the new value into HeatBugs.this.valgrid2
                         valgrid2[old_index] = evaporationRate * (valgrid[index] + diffusionRate * (average - valgrid[index]));
//dont waste memory on valgrid 2, just equal to matrixsize/numDevices; we keep old_index to set the value to appropriate position

	}
						
}

