package sim.app.flockers;

import static jcuda.driver.JCudaDriver.*;
import static jcuda.jcurand.JCurand.*;
import static jcuda.runtime.JCuda.*;
import jcuda.*;
import jcuda.driver.*;
import jcuda.runtime.*;
import java.io.*;
//for random generator
import jcuda.jcurand.*;
import jcuda.runtime.cudaStream_t;

import sim.engine.*;
//use CUDA driver API only except CURAND-MTGP
/**
 * Handle GPU context for Flockers simulation model
 * MUST be initial by this sequence : Constructor -> InitFlockersConstant -> InitModule -> InitMemory -> InitKernelParameters
 * 
 **/
public class FlockersGPUContext extends GPUContext
{
	public CUdeviceptr device_is_dead;
	public CUdeviceptr device_random;
//for continuous 2D	
	public CUdeviceptr device_bag_size ; //store current size of bag 
	public CUdeviceptr device_bag_index ;  //store array of index of points in bag, associate with bag
	public CUdeviceptr device_data ; //locx,locy,dx,dy all here
	
	public CUdeviceptr device_data_old ;
	
	public cudaStream_t stream_runtime;
	
	
	public curandGenerator generator;
	
	//how many kernel ? 2 for flocker model
	
	private CUfunction setup_field;
	private CUfunction flocker_gpu;

	private int numFlockers;
	public int numElements; //num Agents simulated on this context/device
	public long bytecount;  //bytecount for copy mem// number of byte in the result array this context calculated
	public long offset; // offset in bytes from the begining of result array that this context belong to

	private Pointer setup_fieldParameters; 
	private Pointer flocker_gpuParameters; 
	
	private int blockSizeX; 
	private int gridSizeX_flocker;//for flocker_kernel only
	private int gridSizeX_field;	//for setup_field_kernel only

//flocking algo	
	private double width;
	private double height;
	private int height_discreted;
	private int width_discreted;
	private double neighborhood; 
	private int numBags;
	private final int predicted_bag_size = 50;
	//my effort to make it portable 6 parameters need to be configured
	private double cohesion;
	private double avoidance;
	private double randomness;
	private double consistency;
	private double momentum;
	private double jump;  // how far do we move in a timestep?
	
	private int numDevices;
	private long byte_offset;
	private long data_bytecount;
   	FlockersGPUContext(int device_number, int index)
   	{
		super(device_number, index);
	}
	public void InitFlockersConstant(double cohesion, double avoidance, double randomness, double consistency, double momentum,double jump)
	{
		this.cohesion = cohesion;
		this.avoidance = avoidance;
		this.randomness = randomness;
		this.consistency = consistency;
		this.momentum = momentum;
		this.jump = jump;		
	}
  	public void InitModule(int seed, int numElements, int numFlockers, double width, double height, double neighborhood, int numDevices) 
    	{
		String setupFieldFileName="setup-field.cu";
		String FlockersGPUFileName="flockers.cu";
		String setupFieldPtxFileName="";
		String FlockersGPUPtxFileName="";
		try
		{
			setupFieldPtxFileName = preparePtxFile(setupFieldFileName);
			FlockersGPUPtxFileName = preparePtxFile(FlockersGPUFileName);
			//compile two *.cu files to ptx files
		}
		catch (IOException e)
		{
			System.out.println("error in cuda_init() :" + e);
		}
		cuCtxSetCurrent(context);	
			
		CUmodule setupField = new CUmodule();
		cuModuleLoad(setupField, setupFieldPtxFileName);
		setup_field = new CUfunction();
		cuModuleGetFunction(setup_field,setupField,"setup_field");
		
		CUmodule FlockersGPU = new CUmodule();
		cuModuleLoad(FlockersGPU, FlockersGPUPtxFileName);
		flocker_gpu = new CUfunction();
		cuModuleGetFunction(flocker_gpu,FlockersGPU,"flockers");	
		
		//initial MTGP CURAND GENERATOR
	//	stream_runtime = new cudaStream_t(stream);
		generator = new curandGenerator();
		curandCreateGenerator(generator, 141);//MTGP32 enough ?
		curandSetPseudoRandomGeneratorSeed(generator,seed);
	//	curandSetStream(generator, stream_runtime);
		
		
		
		this.numFlockers = numFlockers;
		this.numElements = numElements;
		this.width = width;
		this.height = height;
		this.neighborhood = neighborhood;
		this.height_discreted = (int)(height/neighborhood);
		this.width_discreted = (int)(width/neighborhood);
		this.numBags = (int)(width/neighborhood)*(int)(height/neighborhood);
		
		this.bytecount = numElements* 4 * Sizeof.DOUBLE;
		this.data_bytecount = numFlockers * 4 *Sizeof.DOUBLE;
		this.byte_offset = index*bytecount;
		
		this.offset = index*numElements;
		
		this.numDevices = numDevices;
	}
	public void InitMemory(Pointer data[], int is_dead[])
	{
		cuCtxSetCurrent(context); //do this everytime call to context

		device_data = new CUdeviceptr();
		device_data_old = new CUdeviceptr();
		
		
		device_is_dead = new CUdeviceptr();
				
		device_random = new CUdeviceptr();
		device_bag_size  = new CUdeviceptr(); //store current size of bag 
		device_bag_index  = new CUdeviceptr(); 

		cuMemAlloc(device_data, numFlockers *4 * Sizeof.DOUBLE);
		cuMemAlloc(device_data_old, numFlockers *4 * Sizeof.DOUBLE);
		
		cuMemAlloc(device_bag_size, numBags * Sizeof.INT);
        cuMemAlloc(device_bag_index, numBags * Sizeof.POINTER);

		//allocate bag_index = array of pointer to another array of 50 predicted index
		System.out.println("initial memory for device " + device_number);

		CUdeviceptr hostDevicePointers[] = new CUdeviceptr[numBags];
		for(int i = 0; i < numBags; i++)
		{
			hostDevicePointers[i] = new CUdeviceptr();
			cuMemAlloc(hostDevicePointers[i], predicted_bag_size * Sizeof.INT);
		}
				
		cuMemcpyHtoD(device_bag_index, Pointer.to(hostDevicePointers), numBags * Sizeof.POINTER);

		//finish setting up
        	cuMemAlloc(device_is_dead, numFlockers * Sizeof.INT);                
        	cuMemAlloc(device_random, numFlockers *2 * Sizeof.DOUBLE); //random vector for curand generator

		for(int i=0;i<numDevices;i++)
			cuMemcpyHtoD(device_data.withByteOffset(i*bytecount), data[i],bytecount);	
		
		cuMemcpyHtoD(device_is_dead, Pointer.to(is_dead), numFlockers * Sizeof.INT);
	
	
		cuMemcpyDtoD(device_data_old, device_data , data_bytecount);	
	}
	
	public void InitKernelParameters()
	{
		cuCtxSetCurrent(context);
			blockSizeX = 256;
			gridSizeX_field = 	(int)Math.ceil((double)numFlockers / blockSizeX);
			gridSizeX_flocker = (int)Math.ceil((double)numElements / blockSizeX);
			//setup field : do this for all agents in all devices
			//flocker : do a part of work : 1/2 numFlockers = numElements
			
			setup_fieldParameters = Pointer.to(
											Pointer.to(new int[]{numFlockers}),
											Pointer.to(new double[]{neighborhood}),
											Pointer.to(new int[]{width_discreted}),
											Pointer.to(new int[]{height_discreted}),
											Pointer.to(device_data),
											Pointer.to(device_bag_size),
											Pointer.to(device_bag_index)					
												);
			
			flocker_gpuParameters = Pointer.to(
											Pointer.to(new int[]{numElements}),
											Pointer.to(new int[]{(int)offset}),
											Pointer.to(new int[]{width_discreted}),
											Pointer.to(new int[]{height_discreted}),
											Pointer.to(new double[]{width}),
											Pointer.to(new double[]{height}),
											Pointer.to(new double[]{neighborhood}),
											Pointer.to(device_bag_size),
											Pointer.to(device_bag_index),
											Pointer.to(device_is_dead),
//for phase 3
											Pointer.to(new double[]{cohesion}),
											Pointer.to(new double[]{avoidance}),
											Pointer.to(new double[]{consistency}),													
											Pointer.to(new double[]{randomness}),
											Pointer.to(new double[]{momentum}),
											Pointer.to(new double[]{jump}),													
											Pointer.to(device_random),


											Pointer.to(device_data)
											);
	}
	
	
	public void setup_continuous2D()
	{
			cuCtxSetCurrent(context);
			
			setup_fieldParameters = Pointer.to(
											Pointer.to(new int[]{numFlockers}),
											Pointer.to(new double[]{neighborhood}),
											Pointer.to(new int[]{width_discreted}),
											Pointer.to(new int[]{height_discreted}),
											Pointer.to(device_data),
											Pointer.to(device_bag_size),
											Pointer.to(device_bag_index)					
												);			
			
			cudaMemset(device_bag_size,0,numBags *Sizeof.INT);

			cuLaunchKernel(setup_field,
							gridSizeX_field,  1, 1,      // Grid dimension
							blockSizeX, 1, 1,      // Block dimension
							0, stream,               // Shared memory size and stream
							setup_fieldParameters, null // Kernel- and extra parameters
							);

	}
	
	public void launchKernel()
	{
		cuCtxSetCurrent(context);
		
		curandGenerateUniformDouble(generator,device_random,numFlockers *2 );	

			flocker_gpuParameters = Pointer.to(
											Pointer.to(new int[]{numElements}),
											Pointer.to(new int[]{(int)offset}),
											Pointer.to(new int[]{width_discreted}),
											Pointer.to(new int[]{height_discreted}),
											Pointer.to(new double[]{width}),
											Pointer.to(new double[]{height}),
											Pointer.to(new double[]{neighborhood}),
											Pointer.to(device_bag_size),
											Pointer.to(device_bag_index),
											Pointer.to(device_is_dead),
//for phase 3
											Pointer.to(new double[]{cohesion}),
											Pointer.to(new double[]{avoidance}),
											Pointer.to(new double[]{consistency}),													
											Pointer.to(new double[]{randomness}),
											Pointer.to(new double[]{momentum}),
											Pointer.to(new double[]{jump}),													
											Pointer.to(device_random),

											Pointer.to(device_data),
											Pointer.to(device_data_old)
											);			
		cuLaunchKernel(flocker_gpu,
						gridSizeX_flocker,  1, 1,      // Grid dimension
						blockSizeX, 1, 1,      // Block dimension
						0, stream,               // Shared memory size and stream
						flocker_gpuParameters, null // Kernel- and extra parameters
						);
		
		
	}
	
	
	
	public void copyToPeer(FlockersGPUContext Peer[])
	{
	
	}
	
	public void sync_data_peer(FlockersGPUContext Peer[]) //happen after copy back to host. we can copy from host to each device, but it's not efficient, so we copy data from device to device async 
	{

		cuCtxSetCurrent(context);
		for(int i=0;i<numDevices;i++)
		{
			if (i!=index)
				cuMemcpyPeerAsync(Peer[i].device_data.withByteOffset(byte_offset), Peer[i].context, device_data.withByteOffset(byte_offset), context, bytecount, stream);
		}
		
		
	}
	/**
	 * call after all device have the updated version of data, old_data = newdata
	 *  */
	
	public void sync_internal_data()
	{
		cuCtxSetCurrent(context);
		cuMemcpyDtoDAsync(device_data_old, device_data , data_bytecount,stream);	
	}
	
	
    public void copy_back(Pointer data[])
    {
		cuCtxSetCurrent(context);
		cuMemcpyDtoH(data[index], device_data.withByteOffset(index*bytecount), bytecount);
	}
}

	
	

