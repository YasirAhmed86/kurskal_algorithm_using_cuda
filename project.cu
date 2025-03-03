#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdlib>     // for rand(), srand()
#include <ctime>       // for time()
#include <chrono>      // for timing
#include <cuda.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>

// =============================================================
//                      STRUCTURES
// =============================================================
struct Edge {
    int src, dest, weight;
};

struct EdgeComparator {
    __host__ __device__
    bool operator()(const Edge &a, const Edge &b) {
        return a.weight < b.weight;
    }
};

// =============================================================
//                      GPU UNION-FIND
// =============================================================
__device__ int findGPU(int *parent, int node) {
    while (node != parent[node]) {
        int p  = parent[node];
        int gp = parent[p];
        atomicCAS(&parent[node], p, gp);
        node = parent[node];
    }
    return node;
}

__device__ bool unionSetGPU(int *parent, int *rank, int u, int v) {
    while (true) {
        int root_u = findGPU(parent, u);
        int root_v = findGPU(parent, v);
        if (root_u == root_v) {
            return false;
        }
        if (rank[root_u] > rank[root_v]) {
            if (atomicCAS(&parent[root_v], root_v, root_u) == root_v) {
                return true;
            }
        } else if (rank[root_u] < rank[root_v]) {
            if (atomicCAS(&parent[root_u], root_u, root_v) == root_u) {
                return true;
            }
        } else {
            if (atomicCAS(&parent[root_v], root_v, root_u) == root_v) {
                atomicAdd(&rank[root_u], 1);
                return true;
            }
        }
    }
}

// =============================================================
//                 GPU KERNEL: BATCH KRUSKAL
// =============================================================
__global__ void kruskalBatchKernel(
    Edge* edges,
    int   batchSize,
    int*  parent,
    int*  rank,
    Edge* mstEdges,
    int*  edgeCount,
    int   numVertices
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < batchSize) {
        int u = edges[tid].src;
        int v = edges[tid].dest;
        if (findGPU(parent, u) != findGPU(parent, v)) {
            bool didUnion = unionSetGPU(parent, rank, u, v);
            if (didUnion) {
                int pos = atomicAdd(edgeCount, 1);
                if (pos < numVertices - 1) {
                    mstEdges[pos] = edges[tid];
                }
            }
        }
    }
}

// =============================================================
//              CPU UNION-FIND FOR VERIFICATION
// =============================================================
int find_host(std::vector<int>& parent, int i) {
    if (parent[i] != i) {
        parent[i] = find_host(parent, parent[i]);
    }
    return parent[i];
}

void union_host(std::vector<int>& parent, std::vector<int>& rank, int x, int y) {
    int rx = find_host(parent, x);
    int ry = find_host(parent, y);
    if (rx != ry) {
        if (rank[rx] > rank[ry]) {
            parent[ry] = rx;
        } else if (rank[rx] < rank[ry]) {
            parent[rx] = ry;
        } else {
            parent[ry] = rx;
            rank[rx]++;
        }
    }
}

// =============================================================
//                      MAIN
// =============================================================
int main() {
    // ---------------------
    // 1) Setup
    // ---------------------
    int numVertices = 2000000;
    int numEdges    = 500000000;

    srand(42);
    std::vector<Edge> h_edges(numEdges);
    for (int i = 0; i < numEdges; i++) {
        int u = rand() % numVertices;
        int v = rand() % numVertices;
        while (u == v) {
            v = rand() % numVertices;
        }
        h_edges[i] = {u, v, (rand() % 100) + 1};
    }

    // We’ll measure data transfer times, GPU times, etc.
    auto overall_start = std::chrono::high_resolution_clock::now();

    // ---------------------
    // 2) Copy Edges to GPU & Sort
    // ---------------------
    auto start_htod = std::chrono::high_resolution_clock::now();

    Edge* d_allEdges;
    cudaMalloc((void**)&d_allEdges, numEdges * sizeof(Edge));
    cudaMemcpy(d_allEdges, h_edges.data(), numEdges * sizeof(Edge), cudaMemcpyHostToDevice);

    auto end_htod = std::chrono::high_resolution_clock::now();
    double dataTransferHtoD = std::chrono::duration<double, std::milli>(end_htod - start_htod).count();

    // Sort edges on GPU
    thrust::device_ptr<Edge> d_edge_ptr = thrust::device_pointer_cast(d_allEdges);
    thrust::sort(d_edge_ptr, d_edge_ptr + numEdges, EdgeComparator());

    // ---------------------
    // 3) Allocate Union-Find & MST on GPU
    // ---------------------
    int* d_parent;
    int* d_rank;
    cudaMalloc((void**)&d_parent, numVertices * sizeof(int));
    cudaMalloc((void**)&d_rank,   numVertices * sizeof(int));

    Edge* d_mstEdges;
    cudaMalloc((void**)&d_mstEdges, (numVertices - 1) * sizeof(Edge));

    int* d_edgeCount;
    cudaMalloc((void**)&d_edgeCount, sizeof(int));

    // Host init for union-find
    std::vector<int> h_parent(numVertices), h_rank(numVertices, 0);
    for (int i = 0; i < numVertices; i++) {
        h_parent[i] = i;
    }

    cudaMemcpy(d_parent, h_parent.data(), numVertices * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rank,   h_rank.data(),   numVertices * sizeof(int), cudaMemcpyHostToDevice);

    int zero = 0;
    cudaMemcpy(d_edgeCount, &zero, sizeof(int), cudaMemcpyHostToDevice);

    // ---------------------
    // 4) Multi-Phase Batching
    // ---------------------
    const int BATCH_SIZE = 512; // small batch for minimal concurrency
    Edge* d_batchEdges;
    cudaMalloc((void**)&d_batchEdges, BATCH_SIZE * sizeof(Edge));

    int threadsPerBlock = 128;
    int blocksPerGrid   = (BATCH_SIZE + threadsPerBlock - 1) / threadsPerBlock;

    int h_edgeCount = 0;  // MST edges found so far

    // We measure GPU MST kernel time across all batches
    float totalGpuTime = 0.0f;

    // Batch loop
    for (int start = 0; start < numEdges; start += BATCH_SIZE) {
        int currentBatchSize = std::min(BATCH_SIZE, numEdges - start);

        // Copy batch from d_allEdges -> d_batchEdges
        cudaMemcpy(d_batchEdges,
                   d_allEdges + start,
                   currentBatchSize * sizeof(Edge),
                   cudaMemcpyDeviceToDevice);

        // Time the kernel for this batch
        cudaEvent_t batchStart, batchStop;
        cudaEventCreate(&batchStart);
        cudaEventCreate(&batchStop);

        cudaEventRecord(batchStart);

        kruskalBatchKernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_batchEdges,
            currentBatchSize,
            d_parent,
            d_rank,
            d_mstEdges,
            d_edgeCount,
            numVertices
        );
        cudaDeviceSynchronize();

        cudaEventRecord(batchStop);
        cudaEventSynchronize(batchStop);

        float msBatch = 0.0f;
        cudaEventElapsedTime(&msBatch, batchStart, batchStop);
        totalGpuTime += msBatch;

        cudaEventDestroy(batchStart);
        cudaEventDestroy(batchStop);

        // Check MST edges so far
        cudaMemcpy(&h_edgeCount, d_edgeCount, sizeof(int), cudaMemcpyDeviceToHost);
        if (h_edgeCount >= numVertices - 1) {
            break;
        }
    }

    // ---------------------
    // 5) Copy MST to Host
    // ---------------------
    h_edgeCount = std::min(h_edgeCount, numVertices - 1);
    std::vector<Edge> h_mstEdges(h_edgeCount);

    auto start_dtoh = std::chrono::high_resolution_clock::now();

    cudaMemcpy(h_mstEdges.data(), d_mstEdges, h_edgeCount * sizeof(Edge), cudaMemcpyDeviceToHost);

    auto end_dtoh = std::chrono::high_resolution_clock::now();
    double dataTransferDtoH = std::chrono::duration<double, std::milli>(end_dtoh - start_dtoh).count();

    // ---------------------
    // 6) CPU MST (Verification)
    // ---------------------
    auto cpu_start = std::chrono::high_resolution_clock::now();

    std::vector<Edge> cpu_edges = h_edges;
    std::sort(cpu_edges.begin(), cpu_edges.end(), EdgeComparator());

    std::vector<int> cpu_parent(numVertices), cpu_rank(numVertices, 0);
    for (int i = 0; i < numVertices; i++) {
        cpu_parent[i] = i;
    }

    int cpu_edgeCount = 0;
    int totalWeightCPU = 0;
    for (auto &e : cpu_edges) {
        int rx = find_host(cpu_parent, e.src);
        int ry = find_host(cpu_parent, e.dest);
        if (rx != ry) {
            union_host(cpu_parent, cpu_rank, e.src, e.dest);
            totalWeightCPU += e.weight;
            cpu_edgeCount++;
            if (cpu_edgeCount == numVertices - 1) {
                break;
            }
        }
    }

    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpuMSTTime = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

    // Compute GPU MST stats
    int totalWeightGPU = 0;
    int minEdgeWeight  = 9999999;
    int maxEdgeWeight  = -1;
    for (auto &e : h_mstEdges) {
        totalWeightGPU += e.weight;
        if (e.weight < minEdgeWeight) minEdgeWeight = e.weight;
        if (e.weight > maxEdgeWeight) maxEdgeWeight = e.weight;
    }
    double avgEdgeWeightGPU = (h_edgeCount > 0) ?
                              (double)totalWeightGPU / h_edgeCount : 0.0;

    // ---------------------
    // 7) Extended Report
    // ---------------------
    // Gather device info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    auto overall_end = std::chrono::high_resolution_clock::now();
    double totalTime = std::chrono::duration<double, std::milli>(overall_end - overall_start).count();

    std::cout << "\n==================== EXTENDED REPORT ====================\n";
    std::cout << "CUDA Device: " << prop.name << std::endl;
    std::cout << "Total Global Memory: "
              << (prop.totalGlobalMem / (1024.0 * 1024.0)) << " MB" << std::endl;
    std::cout << "Number of SMs: " << prop.multiProcessorCount << std::endl;
    std::cout << "Approx. CUDA Cores (SM * 128): "
              << (prop.multiProcessorCount * 128) << std::endl;
    std::cout << "Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;

    std::cout << "\n--- Timing Measurements ---" << std::endl;
    std::cout << "Overall Program Time: " << totalTime << " ms" << std::endl;
    std::cout << "Data Transfer (Host -> Device): " << dataTransferHtoD << " ms" << std::endl;
    std::cout << "Data Transfer (Device -> Host): " << dataTransferDtoH << " ms" << std::endl;
    std::cout << "Total GPU MST Kernel Time (all batches): "
              << totalGpuTime << " ms" << std::endl;
    std::cout << "CPU MST Time: " << cpuMSTTime << " ms" << std::endl;

    std::cout << "\n--- GPU MST Results ---" << std::endl;
    std::cout << "Edges in MST (GPU): " << h_edgeCount << std::endl;
    std::cout << "Total Weight (GPU): " << totalWeightGPU << std::endl;
    std::cout << "Min Edge Weight: " << minEdgeWeight << std::endl;
    std::cout << "Max Edge Weight: " << maxEdgeWeight << std::endl;
    std::cout << "Avg Edge Weight: " << avgEdgeWeightGPU << std::endl;

    std::cout << "\n--- CPU MST Results ---" << std::endl;
    std::cout << "Edges in MST (CPU): " << cpu_edgeCount << std::endl;
    std::cout << "Total Weight (CPU): " << totalWeightCPU << std::endl;

    std::cout << "\n--- Verification ---" << std::endl;
    if (totalWeightGPU == totalWeightCPU) {
        std::cout << "GPU and CPU MST Results Match!" << std::endl;
    } else {
        std::cout << "Mismatch in MST Results!" << std::endl;
        std::cout << "GPU MST Weight: " << totalWeightGPU << " vs. CPU MST Weight: " << totalWeightCPU << std::endl;
    }

    std::cout << "\nSample MST Edges (GPU) [First 10 only]:" << std::endl;
    for (int i = 0; i < std::min(10, h_edgeCount); i++) {
        std::cout << h_mstEdges[i].src << " - " << h_mstEdges[i].dest
                  << " (weight: " << h_mstEdges[i].weight << ")" << std::endl;
    }
    std::cout << "========================================================\n";

    // ---------------------
    // 8) Cleanup
    // ---------------------
    cudaFree(d_allEdges);
    cudaFree(d_batchEdges);
    cudaFree(d_parent);
    cudaFree(d_rank);
    cudaFree(d_mstEdges);
    cudaFree(d_edgeCount);

    return 0;
}
