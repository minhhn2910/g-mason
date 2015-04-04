package sim.app.students;

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
import sim.util.*;
import sim.engine.*;
import sim.field.network.*;
//use CUDA driver API only except CURAND-MTGP
/**
 * Handle GPU context for Students simulation model
 * A simple model to demonstrate how to implement MASON network to GPU
 * Further informantion can be found in MASON WCSS tutorial13 on mason website and documents 
 * MUST be initial by this sequence : Constructor -> InitStudentsConstant -> InitModule -> InitMemory -> InitKernelParameters
 * 
 **/
public class StudentsGPUContext extends GPUContext
{
	public CUdeviceptr device_location;
	public CUdeviceptr device_location_old;
//	public CUdeviceptr device_loc_y;
	
	
	public CUdeviceptr device_agitation; 
	
	public CUdeviceptr device_random;
//for network	
	public CUdeviceptr device_bag_size ; //store current size of bag 
	public CUdeviceptr device_bag_edges ;  //store array of edges in bag, associate with bag
//	public CUdeviceptr device_bag_edge_value;
	
	public curandGenerator generator;
	//how many kernel ? 1 for students model
	private CUfunction students;




//constant for model
    public double forceToSchoolMultiplier = 0.01;
    public double randomMultiplier = 0.1;
	private double width;
	private double height;

	private int numStudents;
	public int numElements; //num Agents simulated on this context/device
	public long location_bytecount;  //bytecount for copy mem// number of byte in the result array this context calculated
	private long agitation_bytecount;
//	private long location_bytecount;
	public long offset; // offset in bytes from the begining of result array that this context belong to
	public long all_location_bytecount;

	public int start_student;
	public int end_student;
	
	private Pointer studentsParameters; 

	private int blockSizeX; 
	private int gridSizeX_students;

	private int numBags;
	private int numDevices;
	
	private long location_offset;
	private long agitation_offset;
	
    StudentsGPUContext(int device_number, int index)
    {
		super(device_number, index);
	}
	
	public void InitStudentsConstant(double forceToSchoolMultiplier ,double randomMultiplier, double width, double height)
	{
		this.forceToSchoolMultiplier = forceToSchoolMultiplier;
		this.randomMultiplier = randomMultiplier;
		this.width = width;
		this.height = height;
	}
	
    public void InitModule(int seed, int numElements, int numStudents, int numDevices) 
    {
		String studentsFileName="students.cu";
		String studentsPtxFileName="";
		try
		{
			studentsPtxFileName = preparePtxFile(studentsFileName);
		}
		catch (IOException e)
		{
			System.out.println("error in cuda_init() :" + e);
		}
		cuCtxSetCurrent(context);		
		CUmodule studentsModule = new CUmodule();
		students = new CUfunction();
		cuModuleLoad(studentsModule, studentsPtxFileName);
		cuModuleGetFunction(students,studentsModule,"students");
		
		//initial MTGP CURAND GENERATOR
		generator = new curandGenerator();
		curandCreateGenerator(generator, 141);//MTGP32 enough ?
		curandSetPseudoRandomGeneratorSeed(generator,seed);
	//	curandSetStream(generator, stream_runtime);
		
		
		
		this.numStudents = numStudents;
		this.numElements = numElements;
		this.numBags = numElements;
		
		this.location_bytecount = 2*numElements * Sizeof.DOUBLE;
		this.agitation_bytecount = numElements * Sizeof.DOUBLE;
		
		//this.byte_offset = index*bytecount;
		this.location_offset = index*location_bytecount;
		this.agitation_offset = index* agitation_bytecount;
		//this.offset = index*numElements;  //place where this context start
		this.start_student = index*numElements;
		
		this.end_student = (index+1)*numElements;

		this.all_location_bytecount= numStudents*2*Sizeof.DOUBLE;
		this.numDevices = numDevices;

	}
	public void InitMemory(Pointer location, Network buddies)
	{

		cuCtxSetCurrent(context); //do this everytime call to context

		device_location_old = new CUdeviceptr();
		
		device_location = new CUdeviceptr();

		device_agitation = new CUdeviceptr(); 

				
		device_random = new CUdeviceptr();
		device_bag_size  = new CUdeviceptr(); //store current size of bag 
		device_bag_edges  = new CUdeviceptr(); 

//		device_bag_edge_value  = new CUdeviceptr(); 
	
		cuMemAlloc(device_location_old, numStudents* 2 * Sizeof.DOUBLE);
		cuMemAlloc(device_location, numStudents* 2 * Sizeof.DOUBLE);

		cuMemAlloc(device_agitation, numStudents * Sizeof.DOUBLE);

		
		cuMemAlloc(device_bag_size, numBags * Sizeof.INT);
        cuMemAlloc(device_bag_edges, numBags * Sizeof.POINTER);
//        cuMemAlloc(device_bag_edge_value, numBags * Sizeof.POINTER);
        
        
//allocate bag_index = array of pointer to another array of edges belong to that node

		System.out.println("initial memory for device " + device_number);
//		System.out.println("numbags " +numBags);

		Bag all_students = buddies.getAllNodes();
		
//		CUdeviceptr hostIndexPointers[] = new CUdeviceptr[numBags];
		CUdeviceptr hostNetworkPointers[] = new CUdeviceptr[numBags];
		
		int bag_size[] = new int[numBags];
		
		for(int i = start_student; i < end_student; i++)
        {
			Object student = all_students.get(i);		
	
			Bag out = buddies.getEdges(student, null);
			int len = out.size();	
	
			bag_size[i - start_student] = len;
	
		//	System.out.println(((Student)student).index + " = > "+len);
	
		//	hostIndexPointers[i] = new CUdeviceptr();
			hostNetworkPointers[i- start_student] = new CUdeviceptr();
			
		//	cuMemAlloc(hostIndexPointers[i], len * Sizeof.INT);
		//len *2 because each edge from one node has two attributes : index to other node, weight of it
		
			cuMemAlloc(hostNetworkPointers[i- start_student], len * 2 * Sizeof.DOUBLE);
			
			//int temp_index[] = new int[len];
			double temp_edges[] = new double[len*2];
			
			for(int buddy=0;buddy<len;buddy++)
			{
				Edge e = (Edge)(out.get(buddy));
				temp_edges[buddy*2+1] = ((Double)(e.info)).doubleValue();
				temp_edges[buddy*2] = ((Student)(e.getOtherNode(student))).index;
			}
			
		//	cuMemcpyHtoD(hostIndexPointers[i], Pointer.to(temp_index), len * Sizeof.INT);
			cuMemcpyHtoD(hostNetworkPointers[i- start_student], Pointer.to(temp_edges), len * 2 * Sizeof.DOUBLE);
		
		}		
			
		cuMemcpyHtoD(device_bag_edges, Pointer.to(hostNetworkPointers), numBags * Sizeof.POINTER);
//		cuMemcpyHtoD(device_bag_edge_value, Pointer.to(hostValuePointers), numBags * Sizeof.POINTER);


		cuMemcpyHtoD(device_bag_size, Pointer.to(bag_size), numBags * Sizeof.INT);


		cuMemcpyHtoD(device_location, location, numStudents *2* Sizeof.DOUBLE);
		cuMemcpyHtoD(device_location_old, location, numStudents *2* Sizeof.DOUBLE);

		//finish setting up
		             
        cuMemAlloc(device_random, numStudents *2 * Sizeof.DOUBLE); //random vector for curand generator

	
	
	
		System.out.println(" finished init memory on device " + index + " start at offset " + start_student + " numelements "+numElements +" bytecount "+ location_bytecount);		
		
	}
	
	public void InitKernelParameters()
	{
		cuCtxSetCurrent(context);
			blockSizeX = 256;
			gridSizeX_students = (int)Math.ceil((double)numElements / blockSizeX);

	}
	
	
	public void launchKernel()
	{
		cuCtxSetCurrent(context);
		
		curandGenerateUniformDouble(generator,device_random,numStudents *2 );	

		studentsParameters = Pointer.to(
											Pointer.to(new int[]{numElements}),
											Pointer.to(new int[]{start_student}),
											Pointer.to(new double[]{width}),
											Pointer.to(new double[]{height}),
											Pointer.to(new double[]{randomMultiplier}),
											Pointer.to(new double[]{forceToSchoolMultiplier}),
											Pointer.to(device_bag_size),
											Pointer.to(device_bag_edges),
	//										Pointer.to(device_bag_edge_value),
											Pointer.to(device_random),
											Pointer.to(device_location_old),
											Pointer.to(device_location),
									//		Pointer.to(device_loc_y),
											Pointer.to(device_agitation)
										);			
		cuLaunchKernel(students,
						gridSizeX_students,  1, 1,      // Grid dimension
						blockSizeX, 1, 1,      // Block dimension
						0, stream,               // Shared memory size and stream
						studentsParameters, null // Kernel- and extra parameters
						);
		
		
	}
	
	
	public void sync_internal_data()
	{
		cuCtxSetCurrent(context);
		cuMemcpyDtoDAsync(device_location_old,device_location,all_location_bytecount,stream);
	}
	
	public void sync_data_peer(StudentsGPUContext Peer[]) //happen after copy back to host. we can copy from host to each device, but it's not efficient, so we copy data from device to device async 
	{

		cuCtxSetCurrent(context);
		for(int i=0;i<numDevices;i++)
		{
			if (i!=index)
			{
				cuMemcpyPeerAsync(Peer[i].device_location.withByteOffset(location_offset), Peer[i].context, device_location.withByteOffset(location_offset), context, location_bytecount, stream);
		//		cuMemcpyPeerAsync(Peer[i].device_loc_y.withByteOffset(byte_offset), Peer[i].context, device_loc_y.withByteOffset(byte_offset), context, bytecount, stream);		
			}
		}
		
		
	}
	
    public void copy_back(Pointer location, Pointer agitation)
    {
		cuCtxSetCurrent(context);
		cuMemcpyDtoH( location.withByteOffset(location_offset), device_location.withByteOffset(location_offset), location_bytecount);
		cuMemcpyDtoH( agitation.withByteOffset(agitation_offset), device_agitation.withByteOffset(agitation_offset), agitation_bytecount);
	}
}

	
	

