package sim.app.heatbugs;

import static jcuda.driver.JCudaDriver.*;
import static jcuda.jcurand.JCurand.*;
import static jcuda.runtime.JCuda.*;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyHostToDevice;
import jcuda.*;
import jcuda.driver.*;
import jcuda.runtime.*;
import java.io.*;
import java.nio.ByteOrder;//.*;
//for random generator
import jcuda.jcurand.*;
import java.nio.ByteBuffer;
import sim.field.grid.*;
import sim.engine.*;
//use CUDA driver API only except CURAND-MTGP
/**
 * Handle GPU context for Flockers simulation model
 * MUST be initial by this sequence : Constructor -> InitFlockersConstant -> InitModule -> InitMemory -> InitKernelParameters
 * 
 **/
public class HeatBugsGPUContext extends GPUContext
{
	public CUdeviceptr device_location;
//	public CUdeviceptr device_loc_y;
	public CUdeviceptr device_idealTemp;
	public CUdeviceptr device_heatOutput;
		
	public CUdeviceptr device_random;
	public CUdeviceptr device_valgrid; //for valgrid in heatbugs
	public CUdeviceptr device_valgrid2; //for valgird2 in heatbugs

	public curandGenerator generator;
	
	//how many kernel ? 2 for bugrid model
	
	private CUfunction heatbug;
	private CUfunction diffuse;

	private int bugCount;
	public int numElements; //num Agents simulated on this context/device
	public long bytecount;  //bytecount for copy mem// number of byte in the result array this context calculated
	public long offset; // offset in bytes from the begining of result array that this context belong to

	private Pointer heatbugParameters; 
	private Pointer diffuseParameters; 
	private Pointer HostValgrid;
	
	private int blockSizeX; 
	private int gridSizeX_heatbug;//for flocker_kernel only
	private int gridSizeX_diffuse;	//for setup_field_kernel only
//	private int iterationsPerThread = 1;


//width and height of doublegrid algo	
	private int width;
	private int height;
	private double MAX_HEAT;
	private double randomMovementProbability;
	private double evaporationRate;
    private double diffusionRate;

	
	
    HeatBugsGPUContext(int device_number, int index)
    {
		super(device_number, index);
	}
	public void InitHeatBugsConstant(int width, int height, double MAX_HEAT, double randomMovementProbability, double evaporationRate,double diffusionRate, int bugCount)
	{
		this.width = width;
		this.height = height;
		this.MAX_HEAT = MAX_HEAT;
		this.randomMovementProbability=randomMovementProbability;
		this.evaporationRate=evaporationRate;
		this.diffusionRate = diffusionRate;
		this.bugCount = bugCount;
	}
    public void InitModule(int seed, int numElements, int offset) 
    {
		String HeatBugFileName="heatbug.cu";
		String DiffuseFileName="diffuse.cu";
		String HeatBugPtxFileName="";
		String DiffusePtxFileName="";
		try
		{
			HeatBugPtxFileName = preparePtxFile(HeatBugFileName);
			DiffusePtxFileName = preparePtxFile(DiffuseFileName);
			//compile two *.cu files to ptx files
		}
		catch (IOException e)
		{
			System.out.println("error in cuda_init() :" + e);
		}
		cuCtxSetCurrent(context);	
			
		CUmodule HeatBugModule = new CUmodule();
		cuModuleLoad(HeatBugModule, HeatBugPtxFileName);
		heatbug= new CUfunction();
		cuModuleGetFunction(heatbug,HeatBugModule,"heatbug");
		
		CUmodule DiffuseModule = new CUmodule();
		cuModuleLoad(DiffuseModule, DiffusePtxFileName);
		diffuse = new CUfunction();
		cuModuleGetFunction(diffuse,DiffuseModule,"diffuse");	
		
		//initial MTGP CURAND GENERATOR
		generator = new curandGenerator();
		curandCreateGenerator(generator, 141);//MTGP32 enough ?
		curandSetPseudoRandomGeneratorSeed(generator,seed);
		
		
		this.numElements = numElements;
		this.offset=offset;
		
//		this.offset = offset*Sizeof.DOUBLE;		//test 1 gpu first
//		this.bytecount = numElements * Sizeof.DOUBLE;
		
	}
	public void InitMemory(Pointer location, double idealTemp[], double heatOutput[])
	{
		cuCtxSetCurrent(context); //do this everytime call to context

		device_location = new CUdeviceptr();				
//		device_loc_y = new CUdeviceptr();
		device_idealTemp = new CUdeviceptr();
		device_heatOutput = new CUdeviceptr();
		device_valgrid =new CUdeviceptr();
		device_valgrid2 =new CUdeviceptr();				
		device_random = new CUdeviceptr();


		cuMemAlloc(device_location, bugCount*2 * Sizeof.INT);
//        cuMemAlloc(device_loc_y, bugCount * Sizeof.INT);
		cuMemAlloc(device_idealTemp, bugCount * Sizeof.DOUBLE);
        cuMemAlloc(device_heatOutput, bugCount * Sizeof.DOUBLE);

        cuMemAlloc(device_valgrid, (width*height)* Sizeof.DOUBLE);
        cuMemAlloc(device_valgrid2, (width*height) * Sizeof.DOUBLE);

		cudaMemset(device_valgrid,0,(width*height)* Sizeof.DOUBLE);
		cudaMemset(device_valgrid2,0,(width*height)* Sizeof.DOUBLE);
		

	/*	if(index == 0)
		{

			long memorySize = width*height*Sizeof.DOUBLE;
			valgrid.DataPointer = new Pointer();
			cudaHostAlloc(valgrid.DataPointer, memorySize, cudaHostAllocPortable); //multiple devices, multiple contexts
			valgrid.DataBuffer = valgrid.DataPointer.getByteBuffer(0, memorySize);
		}
		HostValgrid = valgrid.DataPointer;
     */         
        cuMemAlloc(device_random, bugCount *3 * Sizeof.DOUBLE); //random vector for curand generator 3 for heatbugs


		cuMemcpyHtoD(device_location, location, bugCount  * 2*Sizeof.INT);
//		cuMemcpyHtoD(device_loc_y, Pointer.to(loc_y), bugCount  * Sizeof.INT);			
		cuMemcpyHtoD(device_idealTemp, Pointer.to(idealTemp), bugCount  * Sizeof.DOUBLE);
		cuMemcpyHtoD(device_heatOutput, Pointer.to(heatOutput), bugCount  * Sizeof.DOUBLE);	
		System.out.println("init memory complete for device " + device_number);	
	}
	
	public void InitKernelParameters() //cannot apply to jcuda 0.50 because of errors. as described in jcuda.org
	{
		cuCtxSetCurrent(context);

			blockSizeX = 256;
			gridSizeX_heatbug = (int)Math.ceil((double)bugCount / blockSizeX);
			
			gridSizeX_diffuse =  (int)Math.ceil((double)(width*height) / blockSizeX);
			//setup field : do this for all agents in all devices
			//flocker : do a part of work : 1/4 numFlockers = numElements
			
	}
	
	
	public void GPUHeatBug()
	{
			cuCtxSetCurrent(context);
			// setup here heatbugParameters 
			// write *.cu file first
			curandGenerateUniformDouble(generator,device_random, bugCount*3 );		
			
			heatbugParameters = Pointer.to(
											Pointer.to(new int[]{bugCount}),
											Pointer.to(new int[]{width}),
											Pointer.to(new int[]{height}),
											Pointer.to(new double[]{randomMovementProbability}),
											Pointer.to(new double[]{MAX_HEAT}),
											Pointer.to(device_location),
					//						Pointer.to(device_loc_y),
											Pointer.to(device_valgrid),
											Pointer.to(device_idealTemp),
											Pointer.to(device_random),
											Pointer.to(device_heatOutput)					
												);		
			
			
			cuLaunchKernel(heatbug,
							gridSizeX_heatbug,  1, 1,      // Grid dimension
							blockSizeX, 1, 1,      // Block dimension
							0, stream,               // Shared memory size and stream
							heatbugParameters, null // Kernel- and extra parameters
							);

	}
	
	public void GPUDiffuse()
	{
		cuCtxSetCurrent(context);
			// setup here diffuseParameters 
			// write *.cu file first	
		diffuseParameters = Pointer.to(
										Pointer.to(new int[]{width*height}),
								//		Pointer.to(new int[]{iterationsPerThread}),
										Pointer.to(new int[]{width}),
										Pointer.to(new int[]{height}),
										Pointer.to(new double[]{evaporationRate}),
										Pointer.to(new double[]{diffusionRate}),
										Pointer.to(device_valgrid),
										Pointer.to(device_valgrid2)					
										);		
			
		cuLaunchKernel(diffuse,
						gridSizeX_diffuse,  1, 1,      // Grid dimension
						blockSizeX, 1, 1,      // Block dimension
						0, stream,               // Shared memory size and stream
						diffuseParameters, null // Kernel- and extra parameters
						);
		
		
	}
	
	
	
	public void copyToPeer( HeatBugsGPUContext Peer[])
	{
		cuCtxSetCurrent(context);
		
	}
	
	public void sync_data_peer(/* HeatBugsGPUContext Peer[]*/) //happen after copy back to host. Copy data from device to device async 
	{
		cuCtxSetCurrent(context);
		cuMemcpyDtoD(device_valgrid, device_valgrid2,(width*height) * Sizeof.DOUBLE);

	}
	
	public void test_copy()
	{
		double []test = new double [width*height];
		cuMemcpyDtoH( Pointer.to(test), device_valgrid, width*height * Sizeof.DOUBLE);
		for (int i =0;i<height;i++)
		{
			System.out.println();
			System.out.println( i + " ==>  ");
			for (int j = 0;j<width; j++)
				System.out.print((int)test[i*width+j]+" ");

		}
	}
	
	public void test_copy_2()
	{
		double []test = new double [width*height];
		cuMemcpyDtoH( Pointer.to(test), device_valgrid2, width*height * Sizeof.DOUBLE);
		System.out.println("valgrid 2");
		for (int i =0;i<height;i++)
		{
			System.out.println();
			System.out.println( i + " ==>  ");
			for (int j = 0;j<width; j++)
				System.out.print((int)test[i*width+j]+" ");

		}
			
	}	
	
    public void copy_back(Pointer location, Pointer valgrid)
    {
		cuMemcpyDtoH( location, device_location, bugCount * 2 * Sizeof.INT);
//		cuMemcpyDtoH( Pointer.to(loc_y), device_loc_y, bugCount  * Sizeof.INT);
		cuMemcpyDtoH( valgrid, device_valgrid2, (width*height) * Sizeof.DOUBLE);	

	}
}

	
	

