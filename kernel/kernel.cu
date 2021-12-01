#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <sys/time.h>
#include "kernel.h"
#include "invoke.h"
#include <cassert>
#include <iostream>
#include <limits>
#define SIZE 1024
#define Block 1
#define FULL_WARP_MASK 0xFFFFFFFF


template <class T>
__device__ T warp_reduce(T val){
    for(int offset=32/2;offset>0;offset/=2)
        val+= __shfl_down_sync (FULL_WARP_MASK,val,offset);
    return val;

}

typedef float (*op_scalar_fn)(float, float);

__device__ inline float add_scalar(float x, float y) {
    return x + y;
}

__device__ inline float sub_scalar(float x, float y) {
    return x - y;
}

__device__ inline float max_scalar(float x, float y) {
    if(x>y) return x;
    else return y;
}

__device__ inline float min_scalar(float x, float y) {
    if(x<y) return x;
    else return y;
}

__device__ inline float mul_scalar(float x, float y) {
    return x * y;
}

__device__ inline float div_scalar(float x, float y) {
    return x / y;
}

__device__ op_scalar_fn  p_mul = mul_scalar;
__device__ op_scalar_fn  p_div = div_scalar;
__device__ op_scalar_fn  p_add = add_scalar;
__device__ op_scalar_fn  p_sub = sub_scalar;
__device__ op_scalar_fn  p_min = min_scalar;
__device__ op_scalar_fn  p_max = max_scalar;

//to be used if host is sending function pointer to kernel
inline op_scalar_fn get_fn(op_t op) {
    op_scalar_fn op_fn;

    if (op == eDIV) {
        cudaMemcpyFromSymbol(&op_fn, p_div, sizeof(op_scalar_fn));
        //op_fn = div_scalar;
    } else if (op == eSUB) {
        cudaMemcpyFromSymbol(&op_fn, p_sub, sizeof(op_scalar_fn));
        //op_fn = sub_scalar;
    } else if (op == eSUM) {
        cudaMemcpyFromSymbol(&op_fn, p_add, sizeof(op_scalar_fn));
        //op_fn = add_scalar;
    } else if (op == eMUL) {
        cudaMemcpyFromSymbol(&op_fn, p_mul, sizeof(op_scalar_fn));
        //op_fn = mul_scalar;
    } else if (op == eMIN) {
        cudaMemcpyFromSymbol(&op_fn, p_min, sizeof(op_scalar_fn));
        //op_fn = min_scalar;
    } else if (op == eMAX) {
        cudaMemcpyFromSymbol(&op_fn, p_max, sizeof(op_scalar_fn));
        //op_fn = max_scalar;
    } else {
        assert(0);
    }
    return op_fn;
}

//if the kernel itself need the fuction pointer
__device__ inline op_scalar_fn get_fn_kernel(op_t op) {
    op_scalar_fn op_fn;

    if (op == eDIV) {
        op_fn = div_scalar;
    } else if (op == eSUB) {
        op_fn = sub_scalar;
    } else if (op == eSUM) {
        op_fn = add_scalar;
    } else if (op == eMUL) {
        op_fn = mul_scalar;
    } else if (op == eMIN) {
        op_fn = min_scalar;
    } else if (op == eMAX) {
        op_fn = max_scalar;
    } else {
        assert(0);
    }
    return op_fn;
}

__global__ void spmm(const csr_t* __restrict__ obj1, float* x, float * y, op_t op, const bool reverse, const bool norm, const int dim) 
{


    // get thread ID
    int thd_idx = threadIdx.x;
    // get block ID
    int block_idx = blockIdx.x;
    // get block dimensions
    int block_dim = blockDim.x;

    // Gather apply scatter logic

    // allocate one thread to each vertex ("row" in the in-class example)
    // assign unique thread ID to each using this formula
    // because it's naturally parallel when called,
    // these unique ids are your vertex IDs
    // you can tell this when you print it because
    // one call to this method prints a thd_id for every vertex in the graph
    int vertex_thd_id = block_idx * block_dim + thd_idx;
    
    
    // because were creating more threads than we need in invoke_spmm
    // you can calculate vertex thread ids that are greater than
    // the number of nodes in the graph
    // this causes get_degree to return bogus values
    // need to ignore any ids > number of vertices -1
    vid_t num_vertices = obj1->v;
    if (vertex_thd_id < num_vertices) {
        // get_nebrs func returns degree of vertex
        // and also modifies the ptr you pass it to point to the nebrs
        // CUDA_CALLABLE_MEMBER  vid_t get_nebrs(vid_t v, vid_t*& ptr) const
        vid_t* nebr_ptr;
        vid_t vertex_degree = obj1->get_nebrs((vid_t) vertex_thd_id,
                                                nebr_ptr);
        //printf("vertex thread id = %d, degree = %d\n", vertex_thd_id, vertex_degree);


        // normalization procedure
	    // normalize arrays before summing if in backward pass
	    // normalize after summing all arrays if in forward pass
	    float pre_sum_norm; float post_sum_norm;
	    if (reverse){ // backward pass
		    post_sum_norm = (float) 1;
		    pre_sum_norm = (float) vertex_degree;
	    } else { // forward pass
		    pre_sum_norm = (float) 1;
		    post_sum_norm = (float) vertex_degree;
	    }

        // no longer passing input array for whole graph
        // pass feature array for one vertex (x) to the
        // output array for that vertex (y)
	    // the GCN paper specifies that it adds self connections
	    // to the adjacency matrix, so we need to include the node
	    // in the sum
        int col_count = dim;
        //printf("col count for vert %d is %d\n", vertex_thd_id, col_count);
        for (int k = 0; k < col_count; k++){
            // normalize feature values and copy to y
            y[k] = x[k]/pre_sum_norm;
            
            if (vertex_thd_id == 0){
               printf("feat %d for vert %d = %f\n", k, vertex_thd_id, x[k]);
            }
            
        }

	
	    // loop through neighbors using nebr_ptr from prev get_nebrs call
	    // perform column wise add for all nebr features
        for (int j = 0; j<vertex_degree; j++){
            vid_t nebr = nebr_ptr[j];
            // add nebr features to y
            for (int k = 0; k < col_count; k++){
                // normalize first (does nothing if forward pass)
                //y[k] += nebr[k]/pre_sum_norm;
            }
            /*
            if (nebr!=0){
                printf("node %d has nebr %d \n", vertex_thd_id, nebr);
            }
            */
        }
	    
        // do post-sum norm (will not change if in backward pass)
	    for (int i = 0 ; i < col_count; i++){
		    y[i] = y[i]/post_sum_norm;
	    }

    }

}

//warp per row (best)
__global__ void spmm_warp(const csr_t* __restrict__ obj1, float* x, float * y, op_t op, const bool reverse, const bool norm, const int dim)
{
    //TODO
}

void invoke_spmm(csr_t * obj1, array2d_t < float > & x1, array2d_t < float > & y1, op_t op, bool reverse, bool norm, int dim) {
    int warp_size=32;
    int block_size=1024;
    // get the vertex count to determine number of thread blocks needed
    // there are 1024 threads in a block
    // just get the 'v' attribute, won't compile if try to use get_vcount
    vid_t num_vert = obj1->v;
    int num_blocks = (int) (num_vert + block_size - 1)/(block_size);
    //printf("num vertices = %d \n", num_vert);
    //printf("num blocks required = %d \n", num_blocks);
    //printf("col_count x1 = %d \n", (int) x1.col_count);

    spmm <<<num_blocks,block_size>>> (obj1, x1.data_ptr, y1.data_ptr, op, true, true, dim);


    cudaDeviceSynchronize();
}

graph_t * invoke_init_graph(vid_t v_count, vid_t dst_size, vid_t * offset_csr, void * nebrs_csr, vid_t * offset_csc, void * nebrs_csc) {

    //Let us make a cpu graph first
    graph_t g;
    g.init_cpu(v_count, dst_size, 
            offset_csr, nebrs_csr,
            offset_csc, nebrs_csc);

    graph_t * graph = (graph_t*) malloc(sizeof(graph_t));
    cudaMallocManaged( & graph->csr,  sizeof(csr_t));

    vid_t edge_count = offset_csr[v_count];
    vid_t * offset_csr_gpu;
    vid_t * offset_csc_gpu;
    char * nebrs_csr_gpu;
    char * nebrs_csc_gpu;

    cudaMallocManaged( & offset_csr_gpu, (v_count + 1) * sizeof(vid_t));
    cudaMallocManaged( & nebrs_csr_gpu, edge_count * dst_size);

    //memcopy
    cudaMemcpy(offset_csr_gpu, offset_csr, (v_count + 1) * sizeof(vid_t), cudaMemcpyHostToDevice);
    cudaMemcpy(nebrs_csr_gpu, nebrs_csr, edge_count * dst_size, cudaMemcpyHostToDevice);


    if (nebrs_csr == nebrs_csc) {
        graph->csc = graph->csr;
        offset_csc_gpu = offset_csr_gpu;
        nebrs_csc_gpu = nebrs_csr_gpu;
    } else {
        cudaMallocManaged( & graph->csc,  sizeof(csr_t));
        cudaMallocManaged( & offset_csc_gpu, (v_count + 1) * sizeof(vid_t));
        cudaMallocManaged( & nebrs_csc_gpu, edge_count * dst_size);

        cudaMemcpy(nebrs_csc_gpu, nebrs_csc, edge_count * dst_size, cudaMemcpyHostToDevice);
        cudaMemcpy(offset_csc_gpu, offset_csc, (v_count + 1) * sizeof(vid_t), cudaMemcpyHostToDevice);
    }

    //printf("invoke init graph called\n");
    graph -> init(v_count, dst_size, offset_csr_gpu, nebrs_csr_gpu, offset_csc_gpu, nebrs_csc_gpu);

    return graph;

}

