package sim.engine;

import static jcuda.driver.JCudaDriver.*;
import java.io.*;
import java.util.*;
import jcuda.*;
import jcuda.driver.*;
import jcuda.runtime.*;
import java.net.*;
public class GPUContext
{
     public int device_number;
     public CUcontext context;
     public CUstream stream;
     public CUevent event;
     public int index;
     //pointer to device memory, host memory must be extended by subclass
	//module, function must be extended
	
		public GPUContext(int device_number, int index) //create new context and event, stream for that device, 
		{
			this.index = index; //for 1 gpu, multi contexts only, testing purpose
			this.device_number = device_number;
			CUdevice device = new CUdevice();
			cuDeviceGet(device,device_number);
				
			context = new CUcontext();
			cuCtxCreate(context, 0, device);
			cuCtxSetCurrent(context);
			
			event = new CUevent();
			cuEventCreate(event,0);	
			
			stream = new CUstream();
			cuStreamCreate(stream,0);  			
		}
		
	public void eventRecord()
	{
		cuCtxSetCurrent(context);
		cuEventRecord(event, stream);
	}
	public void eventSync()
	{
		cuEventSynchronize(event);
	}

   public String preparePtxFile(String cuFileName)   throws IOException
   {
	   URL url = getClass().getResource(cuFileName);
	   cuFileName = url.getPath();
	   System.out.println(cuFileName);
        int endIndex = cuFileName.lastIndexOf('.');
        if (endIndex == -1)
        {
            endIndex = cuFileName.length()-1;
        }
        String ptxFileName = cuFileName.substring(0, endIndex+1)+"ptx";
        File ptxFile = new File(ptxFileName);
        if (ptxFile.exists())
        {
            return ptxFileName;
        }

        File cuFile = new File(cuFileName);
        if (!cuFile.exists())
        {
            throw new IOException("Input file not found: "+cuFileName);
        }
        String modelString = "-m"+System.getProperty("sun.arch.data.model");
        String command =
            "nvcc " + modelString + " -arch sm_30" +" -ptx "+
            cuFile.getPath()+" -o "+ptxFileName;

        System.out.println("Executing\n"+command);
        Process process = Runtime.getRuntime().exec(command);

        String errorMessage =
            new String(toByteArray(process.getErrorStream()));
        String outputMessage =
            new String(toByteArray(process.getInputStream()));
        int exitValue = 0;
        try
        {
            exitValue = process.waitFor();
        }
        catch (InterruptedException e)
        {
            Thread.currentThread().interrupt();
            throw new IOException(
                "Interrupted while waiting for nvcc output", e);
        }

        if (exitValue != 0)
        {
            System.out.println("nvcc process exitValue "+exitValue);
            System.out.println("errorMessage:\n"+errorMessage);
            System.out.println("outputMessage:\n"+outputMessage);
            throw new IOException(
                "Could not create .ptx file: "+errorMessage);
        }

        System.out.println("Finished creating PTX file");
        return ptxFileName;
    }

    private static byte[] toByteArray(InputStream inputStream)
        throws IOException
    {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        byte buffer[] = new byte[8192];
        while (true)
        {
            int read = inputStream.read(buffer);
            if (read == -1)
            {
                break;
            }
            baos.write(buffer, 0, read);
        }
        return baos.toByteArray();
    }
		

}
