//on multiple GPUs, to reduce the cost of peer device memory transfer, this module need to run on 1 device only.
//our implementation is suffered PCI-Express bottle neck 
extern "C" __global__ void collectdata(
	//									int	numThreads,		//number of agent that this kernel process by addition vector of bestx,besty
	//									int offset,
										int bugCount,
										int numDevices,     //number of devices we are running?
										int height,
										int *best,//length of this vector depends on how many devices are running
										//it is the output of heatbug decision(bestx,besty) to move in heatgrid during heatbug.cu kernel
										int *location
									)
{

   int index = blockIdx.x * blockDim.x + threadIdx.x;
   if (index<bugCount)
    {
	//		index += offset; //where should this kernel begin ?
			int temp_dx =0;
			int temp_dy =0;
			
				for(int i = 0;i<numDevices;i++)
				{
					int temp_index = (i*bugCount+index)*2;
						if((best[temp_index] !=0) || (best[temp_index+1]!=0))
						{
							temp_dx = best[temp_index];
							temp_dy = best[temp_index+1];
							break;
						}
				}
			location[index*2] = temp_dx;
			location[index*2+1] = temp_dy; //sty(temp_dy,height);
			// bugCount/numDevices location calculated here 
					

	}
						
}

