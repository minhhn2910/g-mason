package sim.app.heatbugsMultiGPU;

import static jcuda.driver.JCudaDriver.*;
import static jcuda.jcurand.JCurand.*;
import static jcuda.runtime.JCuda.*;

import jcuda.*;
import jcuda.driver.*;
import jcuda.runtime.*;
import java.io.*;
import java.util.*;
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
	private CUdeviceptr device_idealTemp;
	private CUdeviceptr device_heatOutput;	
	private CUdeviceptr device_random;
	private CUdeviceptr device_valgrid; //for valgrid in heatbugs
	private CUdeviceptr device_valgrid2; //for valgird2 in heatbugs
	private CUdeviceptr device_best; //we are dividing problem set for multiple devices, let's make some things complicated to the cuda kernel ;)
	
	public curandGenerator generator;
	
	//how many kernel ? 2 for bugrid model
	
	private CUfunction heatbug;
	private CUfunction diffuse;
	private CUfunction collectData;
	
	private int bugCount;
	public int numCells; //num Agents simulated on this context/device
//	public long bytecount;  //bytecount for copy mem// number of byte in the result array this context calculated
	public int bugOffset;
	public long gridOffset; // offset in bytes from the begining of result array that this context belong to

	private Pointer heatbugParameters; 
	private Pointer diffuseParameters; 
	private Pointer collectDataParameters;
	
	private	int best_loc_offset;
	private long location_bytecount;
	private long grid_bytecount;
	private long grid_byteoffset;
	private long best_offset_byte;
	private long best_bytecount;
	
	
	
	private int blockSizeX; 
	private int gridSizeX_heatbug;//for flocker_kernel only
	private int gridSizeX_diffuse;	//for setup_field_kernel only
	private int	gridSizeX_collectData;
	
//width and height of doublegrid algo	
	private int width;
	private int height;
	
	private double MAX_HEAT;
	private double randomMovementProbability;
	private double evaporationRate;
    private double diffusionRate;
	private int numDevices;
	
	public int begin_row;
	public int end_row;
			
	private long offset_end;
	private long offset_last;
	private long bytecount_row;
	private int previous_device;
	private int next_device;
				
    HeatBugsGPUContext(int device_number, int index)
    {
		super(device_number, index);
	}
	public void InitHeatBugsConstant(int width, int height,int begin_row,int end_row, double MAX_HEAT, double randomMovementProbability, double evaporationRate,double diffusionRate, int bugCount, int numDevices)
	{
		this.width = width;
		this.height = height;
		this.MAX_HEAT = MAX_HEAT;
		this.randomMovementProbability=randomMovementProbability;
		this.evaporationRate=evaporationRate;
		this.diffusionRate = diffusionRate;
		this.bugCount = bugCount;
		this.numDevices = numDevices;
		this.begin_row = begin_row;
		this.end_row = end_row;	
		this.best_loc_offset = bugCount*index*2;
	}
    public void InitModule(int seed, int numCells, int gridOffset) 
    {
		String HeatBugFileName="heatbug.cu";
		String DiffuseFileName="diffuse.cu";
		String collectDataFileName="collectdata.cu";
		
		String HeatBugPtxFileName="";
		String DiffusePtxFileName="";
		String collectDataPtxFileName="";
		
		try
		{
			HeatBugPtxFileName = preparePtxFile(HeatBugFileName);
			DiffusePtxFileName = preparePtxFile(DiffuseFileName);
			
			collectDataPtxFileName = preparePtxFile(collectDataFileName);
			
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
	
		CUmodule collectDataModule = new CUmodule();
		cuModuleLoad(collectDataModule, collectDataPtxFileName);
		collectData = new CUfunction();
		cuModuleGetFunction(collectData,collectDataModule,"collectdata");		
	
		//initial MTGP CURAND GENERATOR
		generator = new curandGenerator();
		curandCreateGenerator(generator, 141);//MTGP32 enough ?
		curandSetPseudoRandomGeneratorSeed(generator,seed);
		
		
		this.numCells = numCells;
		
		this.bugOffset=bugOffset;
		
		this.location_bytecount = bugCount *2*Sizeof.INT;
		
		this.gridOffset = gridOffset;
//		this.offset = offset*Sizeof.DOUBLE;		//test 1 gpu first
	
		this.grid_bytecount = numCells * Sizeof.DOUBLE;
		
		this.best_offset_byte = bugCount*index*2*Sizeof.INT;
		this.best_bytecount = bugCount*2*Sizeof.INT;
		
		this.grid_byteoffset = index * numCells * Sizeof.DOUBLE;
		
		
		offset_end = (numCells+width*2)*Sizeof.DOUBLE;
		offset_last = numCells*Sizeof.DOUBLE;
		bytecount_row = width*2*Sizeof.DOUBLE;

		previous_device = index -1;
		next_device = index+1;
		if(next_device>=numDevices) next_device = 0;
		if(previous_device <0) previous_device = numDevices-1;
		System.out.println(" " + next_device + " " + previous_device + " " + index );
		
		
		
	}
	public void InitMemory(Pointer location, double idealTemp[], double heatOutput[])
	{
	//	cuCtxSetCurrent(context); //do this everytime call to context

		device_location = new CUdeviceptr();				
//		device_loc_y = new CUdeviceptr();
		device_idealTemp = new CUdeviceptr();
		device_heatOutput = new CUdeviceptr();
		device_valgrid =new CUdeviceptr();
		device_valgrid2 =new CUdeviceptr();				
		device_random = new CUdeviceptr();

		device_best = new CUdeviceptr();



		cuMemAlloc(device_best, bugCount*2*numDevices * Sizeof.INT);
 //     cuMemAlloc(device_besty, bugCount*numDevices * Sizeof.INT);


		cuMemAlloc(device_location, bugCount * 2 * Sizeof.INT);
     //   cuMemAlloc(device_loc_y, bugCount * Sizeof.INT);
        
		cuMemAlloc(device_idealTemp, bugCount * Sizeof.DOUBLE);
        cuMemAlloc(device_heatOutput, bugCount * Sizeof.DOUBLE);

        cuMemAlloc(device_valgrid, (numCells+width*4)* Sizeof.DOUBLE);
        cuMemAlloc(device_valgrid2, numCells * Sizeof.DOUBLE);

		cudaMemset(device_valgrid,0,(numCells+width*4)* Sizeof.DOUBLE);
		cudaMemset(device_valgrid2,0,numCells* Sizeof.DOUBLE);
	
		cudaMemset(device_best,0,bugCount*2*numDevices * Sizeof.INT);
		
//		cudaMemset(device_besty,0,bugCount*numDevices * Sizeof.INT);
		
		       
        cuMemAlloc(device_random, bugCount *3 * Sizeof.DOUBLE); //random vector for curand generator 3 for heatbugs


		cuMemcpyHtoD(device_location, location, bugCount *2  * Sizeof.INT);
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
			gridSizeX_diffuse =  (int)Math.ceil((double)numCells / blockSizeX);
			gridSizeX_collectData= (int)Math.ceil((double)bugCount / blockSizeX);
			
			//setup field : do this for all agents in all devices
			//flocker : do a part of work : 1/4 numFlockers = numElements
			
	}
	
	public void disable_peer(HeatBugsGPUContext Peer[])
	{
		for(int i = 0; i<numDevices ; i++)
			cuCtxDisablePeerAccess (Peer[i].context);
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
											Pointer.to(new int[]{begin_row}), //begin at row?
											Pointer.to(new int[]{end_row}), // end at row?
											Pointer.to(new int[]{best_loc_offset}), 
											Pointer.to(new int[]{index}),
											Pointer.to(new double[]{randomMovementProbability}),
											Pointer.to(new double[]{MAX_HEAT}),
											Pointer.to(device_best),
											//~ Pointer.to(device_besty),
											//~ Pointer.to(device_loc_x),
											Pointer.to(device_location),
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
										Pointer.to(new int[]{numCells}),
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
	
	
	public void GPUCollectData()
	{
		if(index !=0)
			return;
		cuCtxSetCurrent(context);
			// setup here diffuseParameters 
			// write *.cu file first	
	
		collectDataParameters = Pointer.to(
										Pointer.to(new int[]{bugCount}),
										Pointer.to(new int[]{numDevices}),
										Pointer.to(new int[]{height}),
										Pointer.to(device_best),							
										Pointer.to(device_location)					
										);		
			
		cuLaunchKernel(collectData,
						gridSizeX_collectData,  1, 1,      // Grid dimension
						blockSizeX, 1, 1,      // Block dimension
						0, stream,               // Shared memory size and stream
						collectDataParameters, null // Kernel- and extra parameters
						);
		
		
	}
	
	
/**	
 * sync_location_peer(HeatBugsGPUContext Peer[])	
 * this will make all device have the same vector of loc_x and loc_y location that calculated from collect data kernel
 * call one time before go on next cycle, It will make sure all device updated the lasted version of loc_x and loc_y.
 * copy location vector from device 0 (master device in collect data) to others
 **/	
	
	public void sync_location_peer(HeatBugsGPUContext Peer[])
	{
		if(index!=0)
			return;
			
		cuCtxSetCurrent(context);
		
		for (int i = 1; i<numDevices; i++)
				cuMemcpyPeerAsync(Peer[i].device_location, Peer[i].context, device_location, context, location_bytecount, stream);

	}
	
/**	
 * sync_best_location_peer(HeatBugsGPUContext Peer[])	
 * All device will have the same vector of bestx and besty location that calculated from heatbug kernel
 * call one time before do collectdata kernel.
 * others send best_location vector to device 0 for collect infomation.
 **/
	
	public void sync_best_location_peer(HeatBugsGPUContext Peer[]) 
	{
		if(index ==0) return; 
		cuCtxSetCurrent(context);
	//	System.out.println("beat bytecount "+best_bytecount + " best offset " + best_offset_byte);
		cuMemcpyPeerAsync(Peer[0].device_best.withByteOffset(best_offset_byte), Peer[0].context, device_best.withByteOffset(best_offset_byte) , context, best_bytecount, stream);
		//	cuMemcpyPeerAsync(Peer[i].device_besty.withByteOffset(bugCount*index*Sizeof.INT), Peer[i].context, device_besty.withByteOffset(bugCount*index*Sizeof.INT) , context, bugCount*Sizeof.INT, stream);
	}
	
	
/**
 * void sync_valgrid() copy redudant rows between devices to ensure the correct value of boundary condition 
 * This function must be called two times : 
 * 1: after calculate heatbugs, agent add its temperature to valgrid. Sync is required before diffuse heat 
 * 2: after diffuse heat and before calculate heatbugs, Agents need the updated value of valgrid to decide where to go
 */ 
	public void sync_valgrid(HeatBugsGPUContext Peer[])
	{
		cuCtxSetCurrent(context);

			//copy 2 first rows of this device to previous device 
			cuMemcpyPeerAsync(Peer[previous_device].device_valgrid.withByteOffset(offset_end), Peer[previous_device].context, device_valgrid.withByteOffset(bytecount_row) , context, bytecount_row, stream);
			//copy 2 last rows of this device to next device
			cuMemcpyPeerAsync(Peer[next_device].device_valgrid, Peer[next_device].context, device_valgrid.withByteOffset(offset_last) , context, bytecount_row, stream);

		
	}
	
		
/**
 * After copy data back to host, valgrid.setTo(valgrid2) as original version of heatbugs mason
 **/	
	public void sync_data() 
	{
		cuCtxSetCurrent(context);
		cuMemcpyDtoDAsync(device_valgrid.withByteOffset(width*2*Sizeof.DOUBLE), device_valgrid2,numCells * Sizeof.DOUBLE,stream);

	}
	

	
/**
 * Copy data from each device to host, with specific offset
 * only device 0 has the complete vector of location. so only this device send location to host
 * */	
    public void copy_back(Pointer location, Pointer valgrid)
    {
	
		cuCtxSetCurrent(context);
	//	cuMemcpyDtoH( Pointer.to(loc_x).withByteOffset(bugOffset*Sizeof.INT), device_loc_x.withByteOffset(bugOffset*Sizeof.INT), numBugs  * Sizeof.INT);
		if(index ==0)
				cuMemcpyDtoH( location, device_location, location_bytecount);
		cuMemcpyDtoH( valgrid.withByteOffset(grid_byteoffset), device_valgrid2, grid_bytecount);	

	}










	//~ public void test_copy()
	//~ {
		//~ double []test = new double [numCells+width*4];
		//~ cuMemcpyDtoH( Pointer.to(test), device_valgrid, (numCells+width*4) * Sizeof.DOUBLE);
		//~ for (int i =0;i<numCells/width+4;i++)
		//~ {
			//~ System.out.println();
			//~ System.out.println( i + " ==>  ");
			//~ for (int j = 0;j<width; j++)
				//~ System.out.print((int)test[i*width+j]+" ");
//~ 
		//~ }
	//~ }
//~ 
//~ 
	//~ public void test_copy_2()
	//~ {
		//~ double []test = new double [numCells];
		//~ cuMemcpyDtoH( Pointer.to(test), device_valgrid2, numCells * Sizeof.DOUBLE);
		//~ System.out.println("valgrid 2");
		//~ for (int i =0;i<height;i++)
		//~ {
			//~ System.out.println();
			//~ System.out.println( i + " ==>  ");
			//~ for (int j = 0;j<width; j++)
				//~ System.out.print((int)test[i*width+j]+" ");
//~ 
		//~ }
			//~ 
	//~ }
	//~ 
	//~ 
//~ 
	//~ public void test_dxdy()
	//~ {
		//~ int []dx = new int [bugCount*numDevices];
		//~ int []dy = new int [bugCount*numDevices];
		//~ 
		//~ cuMemcpyDtoH( Pointer.to(dx), device_bestx, bugCount*numDevices * Sizeof.INT);
		//~ cuMemcpyDtoH( Pointer.to(dy), device_besty, bugCount*numDevices * Sizeof.INT);
//~ 
      							//~ System.out.println();
								//~ System.out.println(index+ " ==> " );
								//~ for(int i=0;i<bugCount*numDevices;i++)
								//~ {
								//~ 
									//~ System.out.print(" ("+i +": "+ dy[i]+ ","+dx[i]+")");
								//~ }
			//~ 
	//~ }	
	//~ 
	//~ 
	//~ 
	//~ public void test_location()
	//~ {
		//~ int []dx = new int [bugCount];
		//~ int []dy = new int [bugCount];
		//~ 
		//~ cuMemcpyDtoH( Pointer.to(dx), device_loc_x, bugCount* Sizeof.INT);
		//~ cuMemcpyDtoH( Pointer.to(dy), device_loc_y, bugCount* Sizeof.INT);
//~ 
      							//~ System.out.println();
      							//~ System.out.println(" teststttttt location ========================== ==> " );
								//~ System.out.println(index+ " ==> " );
								//~ for(int i=0;i<bugCount;i++)
								//~ {
								//~ 
									//~ System.out.print(" ("+i +": "+ dy[i]+ ","+dx[i]+")");
								//~ }
			//~ 
	//~ }		
	












}

	
	

