#include<stdio.h>
#include<cuda_runtime.h>
#include <helper_functions.h>
#include<helper_cuda.h>
/*------------------------GPU RANKING----------------------------------------START-------*/

/*------------------------shfl_scan_test-----------------------------------------------Start*/
__global__ void shfl_scan_test(float *data,int *ranker ,int width, int *partial_rank=NULL,float *partial_data=NULL)
{
	extern __shared__ int values[];
	 __shared__ int ranker_shr[32];      // At most we can use block size == 1024 
        int id = threadIdx.x+(blockIdx.x*blockDim.x);
	int warp_id = threadIdx.x/warpSize;
	int lane_id = threadIdx.x % warpSize;
	int rank_value;	
	//printf("id :%d \t lane_id: %d \t warp_id: %d \t data[id]:%f\n",id,lane_id,warp_id,data[id]);
	float value = data[id];
	//else
	rank_value = ranker[id];
	//int warp_rank;
	
	/*----------------------------------Part 1--------------------------------------Start*/
	#pragma unroll
	for( int i =1;i<width;i*=2)              //small speed optimization done here by reducing one ittration
	{
		int n =__shfl_up(value,i,width);	
			int r;
			if(value == n)
			{
				 r = __shfl_up(rank_value , i,width);
				 rank_value = r;
			}
			//printf("i :%d id : %d \t r :%d \n",i,id,r);
			
			//printf("rank_value :%d\n",rank_value);
	}
	//printf("i :%d\n",i );

	if(lane_id == warpSize-1)
	{
		values[warp_id] = value;
	}
	if(lane_id == warpSize-1)
	{
		ranker_shr[warp_id] = rank_value;
	}
	__syncthreads();
	/*----------------------------------Part 1--------------------------------------End*/

	/*----------------------------------Part 2--------------------------------------Start*/
	if(warp_id == 0 && lane_id<(blockDim.x/warpSize))
	{
		float warp_value = values[lane_id];
		int warp_rank = ranker_shr[lane_id];
		//printf("wrap_rank 1 : %d and warp_value 2 : %f \n",warp_rank,warp_value);
		for(int i = 1 ; i <(blockDim.x/warpSize-1) ; i *= 2)         //small speed optimization done here by reducing one ittration
		{
			int n = __shfl_up(warp_value , i , (blockDim.x/warpSize));
			{
				if(lane_id >= i)
				{
					if(warp_value == n)
					{
						int r = __shfl_up(warp_rank,i,(blockDim.x/warpSize));
						//printf("i : %d \t wrap_rank : %d\n",i,r);
						warp_rank = r;
					}
				}
			}
		}
		ranker_shr[lane_id] = warp_rank;
		
		//printf("wrap_rank 2 : %d and warp_value 2 : %f \n",warp_rank,warp_value);
				
	}		
	__syncthreads();
	/*----------------------------------Part 2--------------------------------------End*/
	
	/*----------------------------------Part 3--------------------------------------Start*/
	float block_value = 0.0;
	int block_rank = 0;
	
	if(warp_id > 0)
	{
		block_value = values[warp_id-1];
		block_rank = ranker_shr[warp_id-1];
	}
	if(value == block_value)
	{
		rank_value = block_rank;
	}	
	//printf("rank_value : %d\n",rank_value);
	ranker[id] = rank_value;
	
	if(partial_data != NULL && id == blockDim.x-1)
	{
		partial_data[blockIdx.x] = value;
		partial_rank[blockIdx.x] = rank_value;

		//printf("value in partial : %d",value);
	}

	/*-----------------------------------Part 3-------------------------------------End*/
		
	
}
/*------------------------shfl_scan_test-----------------------------------------------End*/
/*------------------------Final Ranking-----------------------------------------------Start*/
__global__ void final_ranking(float *data , int *rank , float *partial_data , int *partial_rank , int len)
{
	__shared__ float value_buf;
	__shared__ int rank_buf;

	int id = ((blockIdx.x*blockDim.x)+threadIdx.x);
	if(id>len) return;
	
	if(threadIdx.x == 0)
	{
		value_buf = partial_data[blockIdx.x];
		rank_buf = partial_rank[blockIdx.x];
	}
	__syncthreads();
	if(data[id] == value_buf)
	{
		rank[id] = rank_buf;
	}
}	

/*------------------------Final_ranking-----------------------------------------------End*/

/*-----------------------GPU RANKING------------------------------------------END--------*/

/*-----------------------iDivUp--------------------------------------------------------Start*/

	static unsigned int iDivUp(unsigned int dividend, unsigned int divisor)
	{
   	    return ((dividend % divisor) == 0) ?
           	(dividend / divisor) :
           	(dividend / divisor + 1);
	}

/*--------------------------------------------------------------------------------------End*/

/*--------------------------CPU RANKING-------------------------------------------END------*/

bool CPUverify(float *h_data, int *h_result, int n_elements)
{
    int rank =1;
   // printf("CPU verify result diff (GPUvsCPU) = %f\n", diff);
    bool bTestResult = false;
    bTestResult =true;
   
    
    StopWatchInterface *hTimer = NULL;
    sdkCreateTimer(&hTimer);
    sdkResetTimer(&hTimer);
    sdkStartTimer(&hTimer);

   /* for (int j=0; j<100; j++)
        for (int i=0; i<n_elements-1; i++)
        {
            h_data[i+1] = h_data[i] + h_data[i+1];
        }*/
      
      for (int i = 1; i < n_elements; i++ ) 
	{
    
      	if(h_data[i-1] != h_data[i]) 
         	rank = i + 1;
         h_result[i] = rank;
        }

    sdkStopTimer(&hTimer);
    float cput= sdkGetTimerValue(&hTimer);
    printf("CPU sum (naive) took %f ms\n", cput);
    return bTestResult;
}

/*--------------------------CPU RANKING-------------------------------------------END------*/

/*--------------------------shuffle_simple_test-----------------------------------START-------*/
bool shuffle_simple_test(int argc , char *argv[])
{
	float *h_data , *d_data, *d_partial_data,*h_partial_data;
	int *h_rank,*d_rank,*h_result, *h_partial_rank,*d_partial_rank; 
	const int n_elements (1 << atoi(argv[1]));
	double p = atof(argv[2]); 
	float data_sz = sizeof(float)*n_elements;
	int rank_sz = sizeof(int)*n_elements;
	


	int blockSize = 256;
   	int gridSize = n_elements/blockSize; // -----------------------------------gridsize  = 256
    	int nWarps = blockSize/32;  //-------------------------------------------- number of wraps = 8
    	float shmem_sz = nWarps * sizeof(float); //----------------------------------shmem = 32 bytes
   	int n_partialSums = n_elements/blockSize; //-------------------------------n_partialSums = 256
    	float partial_sz = n_partialSums*sizeof(float); //-------------------------- partial size = 1024
	int partial_rk = n_partialSums*sizeof(int); //-------------------------- partial size = 1024

	int p_blockSize = min(n_partialSums, blockSize);
   	int p_gridSize = iDivUp(n_partialSums, p_blockSize);
   	printf("Partial summing %d elements with %d blocks of size %d\n", n_partialSums, p_gridSize, p_blockSize);

	int cuda_device = 0;
	
	h_data = (float *)malloc(n_elements * sizeof(float));
    	h_rank = (int *)malloc (n_elements* sizeof(int));
	h_partial_rank = (int*)malloc (partial_rk);
	
       /*-------------------------CUDA Device Check-----------------START-----*/
	cudaDeviceProp deviceProp;
    	checkCudaErrors(cudaGetDevice(&cuda_device));
    	checkCudaErrors(cudaGetDeviceProperties(&deviceProp, cuda_device));
    	printf("> Detected Compute SM %d.%d hardware with %d multi-processors\n",
        deviceProp.major, deviceProp.minor, deviceProp.multiProcessorCount);

   	 // __shfl intrinsic needs SM 3.0 or higher
    	if (deviceProp.major < 3)
    	{
        	printf("> __shfl() intrinsic requires device SM 3.0+\n");
        	printf("> Waiving test.\n");
        	exit(EXIT_WAIVED);
    	}
         /*-------------------------CUDA Device Check-----------------END-----*/

	checkCudaErrors(cudaMallocHost((void **)&h_data, sizeof(float)*n_elements));//--------------------Malloc
    	checkCudaErrors(cudaMallocHost((void **)&h_result, sizeof(int)*n_elements));//------------------Malloc
    	checkCudaErrors(cudaMallocHost((void **)&h_rank, sizeof(int)*n_elements));//----------------------Malloc
	
	 /*--------------------------data init -------------------------Start---*/
    	float prev = h_data[0] = 0.0f;
  	for (int i = 1; i < n_elements; i++ ) {   //......................................for loop array initialization
       	if(drand48() < p ) 	
		{
           		h_data[i] = prev;  // ramain the same value
       		}
       else 
		{ // times log2f(i) to make the difference large enough
            		 h_data[i] = prev = i*log2f((float)i); 
      		}
        //printf("i : %d \t%f\n",i, h_data[i]); // debug 
	
   	}
  	/*--------------------------data init-----------------------------------------End*/
	
	/*--------------------------rank init-----------------------------------------Start*/
	for(int i = 1 ;i<n_elements;i++)
	{
		h_rank[i] = i;
	}	
	
	/*--------------------------rank init-----------------------------------------End*/	
	
	/*--------------------------Timer Set--------------------------------------------Start*/
	cudaEvent_t start, stop;
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));
	float et = 0;
    	float inc = 0;
	/*--------------------------Timer Set--------------------------------------------End*/

	/*--------------------------GPU CODE-----------------------------------------Start*/
	checkCudaErrors(cudaMalloc((void **)&d_data, data_sz));
	checkCudaErrors(cudaMalloc((void **)&d_rank, rank_sz));
    	checkCudaErrors(cudaMalloc((void **)&d_partial_data, partial_sz));
	checkCudaErrors(cudaMemset(d_partial_data, 0, partial_sz));
	
	

	checkCudaErrors(cudaMalloc((void **)&d_partial_rank, partial_rk));
	checkCudaErrors(cudaMemset(d_partial_rank, 0, partial_rk));
	
	checkCudaErrors(cudaMallocHost((void **)&h_partial_data, partial_sz));
	checkCudaErrors(cudaMallocHost((void **)&h_partial_rank, partial_rk));
    	checkCudaErrors(cudaMemcpy(d_data, h_data, data_sz, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_rank, h_rank, data_sz, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaEventRecord(start, 0));


	/*---------------------------Kernal-------------------------------------------Start*/

	shfl_scan_test<<<gridSize,blockSize, shmem_sz>>>(d_data,d_rank,32,d_partial_rank ,d_partial_data);
	
	shfl_scan_test<<<p_gridSize, p_blockSize, shmem_sz>>>(d_partial_data,d_partial_rank,32);
		
	final_ranking<<<gridSize-1 , blockSize>>>(d_data+blockSize,d_rank+blockSize,d_partial_data , d_partial_rank , n_elements);
	/*---------------------------Kernal-------------------------------------------Start*/
	checkCudaErrors(cudaEventRecord(stop, 0));
	checkCudaErrors(cudaEventSynchronize(stop));
    	checkCudaErrors(cudaEventElapsedTime(&inc, start, stop));
    	et+=inc;

	checkCudaErrors(cudaMemcpy(h_result, d_rank, rank_sz, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(h_partial_rank, d_partial_rank, partial_sz, cudaMemcpyDeviceToHost));

	printf("GPU time in ms: %f\n", et);

	/*for(int i = 0;i <n_elements;i++)
	{
		printf(" i : %d \t rank  is %d\n",i , h_result[i]); // --------------------------------------------------Print result
	}*/
	bool bTestResult = CPUverify(h_data, h_result, n_elements);

	checkCudaErrors(cudaFreeHost(h_data));
    	checkCudaErrors(cudaFreeHost(h_result));
	checkCudaErrors(cudaFree(d_data));
    	checkCudaErrors(cudaFree(d_partial_data));

	/*---------------------------GPU CODE-----------------------------------------Start*/
       	return true;
}



/*--------------------------shuffle_simple_test-----------------------------------END-------*/

/*--------------------------main()-----------------------------------START-------*/
int main (int argc ,char **  argv)
{
       /*-------------------------CUDA Device Check-----------------START-----*/

	int cuda_device = 0;

     printf("Starting shfl_scan\n");

    // use command-line specified CUDA device, otherwise use device with highest Gflops/s

    cuda_device = findCudaDevice(argc, (const char **)argv);
    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDevice(&cuda_device));

    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, cuda_device));

    printf("> Detected Compute SM %d.%d hardware with %d multi-processors\n",
           deviceProp.major, deviceProp.minor, deviceProp.multiProcessorCount);

    // __shfl intrinsic needs SM 3.0 or higher
    if (deviceProp.major < 3)
    {
        printf("> __shfl() intrinsic requires device SM 3.0+\n");
        printf("> Waiving test.\n");
        exit(EXIT_WAIVED);
    }
       /*-------------------------CUDA Device Check------------------END----*/

     bool simpleTest = shuffle_simple_test(argc, argv);
	return 0;
}
/*--------------------------main()-----------------------------------END-------*/
