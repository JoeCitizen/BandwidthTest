// BandwidthTester.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#include <wrl.h>
#include <dxgi1_6.h>
#include <d3d12.h>
#include "d3dx12.h"
#include <stdint.h>
#include <iostream>
#include <assert.h>
#include "ReadBandwidthCS.h"

#define IID_GRAPHICS_PPV_ARGS IID_PPV_ARGS

using namespace Microsoft::WRL;

inline void AssertIfFailed(HRESULT hr)
{
    if (FAILED(hr))
    {
        // Set a breakpoint on this line to catch DirectX API errors
        assert(false);
    }
}

#define IFR(x) if(FAILED(x)){ output << "File: " << __FILE__ <<  "Line:" << __LINE__ << "failed!" << std::endl; return E_FAIL; }

uint32_t cWaveSize = 64;
const uint32_t cTestIterations = 64;

class QueryHelper
{
    ID3D12Device* m_Device;
    ComPtr<ID3D12QueryHeap> m_QueryHeap;
    ComPtr<ID3D12Resource> m_QueryReadBackBuffer;
    UINT m_QueryCount;
    D3D12_QUERY_TYPE m_QueryType;
    UINT m_ElementSize;

public:
    void Initialize(ID3D12Device* Device, UINT QueryCount, D3D12_QUERY_TYPE QueryType)
    {
        m_Device = Device;
        m_QueryCount = QueryCount;
        m_QueryType = QueryType;
        D3D12_QUERY_HEAP_TYPE QueryHeapType = D3D12_QUERY_HEAP_TYPE_OCCLUSION;
        switch (QueryType) {
        case D3D12_QUERY_TYPE_OCCLUSION:
            QueryHeapType = D3D12_QUERY_HEAP_TYPE_OCCLUSION;
            m_ElementSize = sizeof(UINT64);
            break;
        case D3D12_QUERY_TYPE_BINARY_OCCLUSION:
            QueryHeapType = D3D12_QUERY_HEAP_TYPE_OCCLUSION;
            m_ElementSize = sizeof(UINT64);
            break;
        case D3D12_QUERY_TYPE_TIMESTAMP:
            QueryHeapType = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
            m_ElementSize = sizeof(UINT64);
            break;
        case D3D12_QUERY_TYPE_PIPELINE_STATISTICS:
            QueryHeapType = D3D12_QUERY_HEAP_TYPE_PIPELINE_STATISTICS;
            m_ElementSize = sizeof(D3D12_QUERY_DATA_PIPELINE_STATISTICS);
            break;
        case D3D12_QUERY_TYPE_SO_STATISTICS_STREAM0:
        case D3D12_QUERY_TYPE_SO_STATISTICS_STREAM1:
        case D3D12_QUERY_TYPE_SO_STATISTICS_STREAM2:
        case D3D12_QUERY_TYPE_SO_STATISTICS_STREAM3:
            QueryHeapType = D3D12_QUERY_HEAP_TYPE_SO_STATISTICS;
            m_ElementSize = sizeof(D3D12_QUERY_DATA_SO_STATISTICS);
            break;
        default:
            assert(false);
            break;
        }

        D3D12_QUERY_HEAP_DESC QueryHeapDesc = { QueryHeapType, QueryCount };
        m_QueryHeap = nullptr;
        AssertIfFailed(m_Device->CreateQueryHeap(
            &QueryHeapDesc,
            IID_GRAPHICS_PPV_ARGS(m_QueryHeap.ReleaseAndGetAddressOf())
        ));

        CD3DX12_RESOURCE_DESC BufferDescriptor = CD3DX12_RESOURCE_DESC::Buffer(
            QueryCount * m_ElementSize
        );

        CD3DX12_HEAP_PROPERTIES HeapProperties = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK);
        m_QueryReadBackBuffer = nullptr;
        AssertIfFailed(m_Device->CreateCommittedResource(
            &HeapProperties,
            D3D12_HEAP_FLAG_NONE,
            &BufferDescriptor,
            D3D12_RESOURCE_STATE_COPY_DEST,
            nullptr,
            IID_GRAPHICS_PPV_ARGS(m_QueryReadBackBuffer.ReleaseAndGetAddressOf())
        ));
    }

    void BeginQuery(ID3D12GraphicsCommandList* CommandList, UINT QueryIndex)
    {
        CommandList->BeginQuery(m_QueryHeap.Get(), m_QueryType, QueryIndex);
    }

    void EndQuery(ID3D12GraphicsCommandList* CommandList, UINT QueryIndex)
    {
        CommandList->EndQuery(m_QueryHeap.Get(), m_QueryType, QueryIndex);
    }

    void ResolveQuery(ID3D12GraphicsCommandList* CommandList)
    {
        CommandList->ResolveQueryData(m_QueryHeap.Get(), m_QueryType, 0, m_QueryCount, m_QueryReadBackBuffer.Get(), 0);
    }

    void ResolveQuery(
            ID3D12GraphicsCommandList* CommandList,
            UINT StartIndex,
            UINT EndIndex,
            UINT Alignment
        )
    {
        CommandList->ResolveQueryData(m_QueryHeap.Get(), m_QueryType, StartIndex, EndIndex, m_QueryReadBackBuffer.Get(), Alignment);
    }

    HRESULT GetData(PVOID Data, UINT DataLength)
    {
        HRESULT hr = S_OK;

        void* Mapped = nullptr;
        CD3DX12_RANGE ReadRange(0, min(m_QueryCount * m_ElementSize, DataLength));
        hr = m_QueryReadBackBuffer->Map(0, &ReadRange, &Mapped);
        if (FAILED(hr))
        {
            return hr;
        }

        memcpy(Data, Mapped, ReadRange.End);
        m_QueryReadBackBuffer->Unmap(0, nullptr);

        return hr;
    }
};

struct CpuTimer
{
    struct TimeStamp
    {
        LONGLONG Start;
        LONGLONG Stop;
    };

    CpuTimer()
    {
        LARGE_INTEGER li;
        QueryPerformanceFrequency(&li);

        m_Freq = double(li.QuadPart) / 1000.0;

        m_pTimeStamps = new TimeStamp[cCapacity];
        memset(m_pTimeStamps, 0, cCapacity * sizeof(TimeStamp));
    }

    ~CpuTimer()
    {
        delete[] m_pTimeStamps;
        m_pTimeStamps = nullptr;
    }

    void Start(UINT slot)
    {
        assert(slot < cCapacity);
        LARGE_INTEGER li;
        QueryPerformanceCounter(&li);
        m_pTimeStamps[slot].Start = li.QuadPart;
    }

    void Stop(UINT slot)
    {
        assert(slot < cCapacity);
        LARGE_INTEGER li;
        QueryPerformanceCounter(&li);
        m_pTimeStamps[slot].Stop = li.QuadPart;
    }

    UINT64 Ticks(UINT slot)
    {
        assert(slot < cCapacity);
        return m_pTimeStamps[slot].Stop - m_pTimeStamps[slot].Start;
    }

    double Millis(UINT slot)
    {
        assert(slot < cCapacity);
        return double(Ticks(slot)) / m_Freq;
    }

private:
    static const size_t cCapacity = 1024;

    double m_Freq;
    TimeStamp* m_pTimeStamps = nullptr;
};

struct Timestamp
{
    uint64_t begin;
    uint64_t end;
};

struct TestCase
{
    uint32_t m_BufferSizeBytes;
    uint32_t m_DispatchThreads;
    uint32_t m_DispatchTGs;
    QueryHelper m_queryHeap;

    Timestamp* m_pTimeStamps = nullptr;

    TestCase(ComPtr<ID3D12Device> pDevice, uint32_t sizeBytes)
    {
        m_BufferSizeBytes = sizeBytes;
        m_DispatchThreads = (m_BufferSizeBytes / 16); // 16 bytes per thread
        m_DispatchTGs = m_DispatchThreads / cWaveSize;

        m_queryHeap.Initialize(pDevice.Get(), cTestIterations * 2, D3D12_QUERY_TYPE_TIMESTAMP);

        m_pTimeStamps = new Timestamp[cTestIterations];
    }

    ~TestCase()
    {
        delete[](m_pTimeStamps);
    }

    void Run(ComPtr<ID3D12GraphicsCommandList> pList, ID3D12Resource* pResource)
    {
        D3D12_GPU_VIRTUAL_ADDRESS bufferVA = pResource->GetGPUVirtualAddress();

        pList->SetComputeRootUnorderedAccessView(0, bufferVA);

        for (uint32_t i = 0; i < cTestIterations; i++)
        {
            uint32_t beginIndex = i * 2;
            uint32_t endIndex = (i * 2) + 1;

            m_queryHeap.EndQuery(pList.Get(), beginIndex);
            pList->Dispatch(m_DispatchTGs, 1, 1);
            m_queryHeap.EndQuery(pList.Get(), endIndex);

            {
                CD3DX12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::UAV(pResource);
                pList->ResourceBarrier(1, &barrier);
            }
        }

        m_queryHeap.ResolveQuery(pList.Get());
    }

    HRESULT CollectTimestamps()
    {
        return m_queryHeap.GetData((void*)m_pTimeStamps, sizeof(Timestamp) * cTestIterations);
    }

    void OutputMeasurements(std::ostream& output, double gpuFreqInv)
    {
        output << "Buffer Size (bytes): " << m_BufferSizeBytes << std::endl;

        double low = 999999999.0;
        double high = 0;
        double average = 0;

        Timestamp* pCurrent = m_pTimeStamps;
        for (size_t i = 0; i < cTestIterations; i++)
        {
            double timeSec = (double(pCurrent->end - pCurrent->begin) * gpuFreqInv) / 1000.0;
            double bandwidthBytes = double(m_BufferSizeBytes) / timeSec;
            double bandwidthKiloBytes = bandwidthBytes / 1000.0;
            double bandwidthMegaBytes = bandwidthKiloBytes / 1000.0;
            double bandwidthGigaBytes = bandwidthMegaBytes / 1000.0;

            output << "    Iteration " << i << " Time (seconds): " << timeSec << " Bandwidth: " << bandwidthGigaBytes << " GB/s" << std::endl;

            if (bandwidthGigaBytes > high) high = bandwidthGigaBytes;
            if (bandwidthGigaBytes < low) low = bandwidthGigaBytes;
            average += bandwidthGigaBytes;

            pCurrent++;
        }
        average /= double(cTestIterations);

        output << "Bandwidth Low: " << low << " GB/s" << std::endl;
        output << "Bandwidth High: " << high << " GB/s" << std::endl;
        output << "Bandwidth Average: " << average << " GB/s" << std::endl;
        output << std::endl;
    }
};


HRESULT RunBandwidthTestGpu(std::ostream& output, ComPtr<IDXGIAdapter1> pAdapter)
{
    const uint32_t MB = 1024 * 1024;

    ComPtr<ID3D12Device>                pDevice;
    ComPtr<ID3D12CommandQueue>          pQueue;
    ComPtr<ID3D12CommandAllocator>      pAllocator;
    ComPtr<ID3D12GraphicsCommandList>   pList;
    ComPtr<ID3D12Fence>                 pFence;
    ComPtr<ID3D12RootSignature>         pRootSig;
    ComPtr<ID3D12PipelineState>         pPso;
    uint64_t                            fenceVal = 1;

    IFR(D3D12CreateDevice(pAdapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_GRAPHICS_PPV_ARGS(pDevice.GetAddressOf())));

    const bool dbg = false;

    if (dbg)
    {
        ComPtr<ID3D12Debug> debugInterface;
        if (SUCCEEDED(D3D12GetDebugInterface(IID_GRAPHICS_PPV_ARGS(&debugInterface))))
        {
            debugInterface->EnableDebugLayer();
        }
    }

    {
        D3D12_COMMAND_QUEUE_DESC desc = {};
        desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        IFR(pDevice->CreateCommandQueue(&desc, IID_GRAPHICS_PPV_ARGS(pQueue.GetAddressOf())));
    }

    {
        IFR(pDevice->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_GRAPHICS_PPV_ARGS(pAllocator.GetAddressOf())));
    }

    {
        IFR(pDevice->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, pAllocator.Get(), nullptr, IID_GRAPHICS_PPV_ARGS(pList.GetAddressOf())));
    }

    {
        IFR(pDevice->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_GRAPHICS_PPV_ARGS(pFence.GetAddressOf())));
    }


    UINT64 gpuFreq = 0;
    IFR(pQueue->GetTimestampFrequency(&gpuFreq));
    double gpuFreqInv = 1000.0 / double(gpuFreq);

    {
        CD3DX12_ROOT_SIGNATURE_DESC descRootSignature;
        CD3DX12_ROOT_PARAMETER parameters[1];
        parameters[0].InitAsUnorderedAccessView(0, 0);

        descRootSignature.Init(_countof(parameters), parameters, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE);

        ComPtr<ID3DBlob> pSignature;
        ComPtr<ID3DBlob> pError;
        IFR(D3D12SerializeRootSignature(&descRootSignature, D3D_ROOT_SIGNATURE_VERSION_1, pSignature.GetAddressOf(), pError.GetAddressOf()));
        IFR(pDevice->CreateRootSignature(0, pSignature->GetBufferPointer(), pSignature->GetBufferSize(), IID_GRAPHICS_PPV_ARGS(pRootSig.GetAddressOf())));
    }

    {
        D3D12_COMPUTE_PIPELINE_STATE_DESC desc = {};
        desc.pRootSignature = pRootSig.Get();
        desc.CS = { g_ReadBandwidthCS, sizeof(g_ReadBandwidthCS) };

        IFR(pDevice->CreateComputePipelineState(&desc, IID_GRAPHICS_PPV_ARGS(pPso.GetAddressOf())));

    }

    TestCase tcs[] =
    {
        TestCase(pDevice.Get(), 1 * MB),
        TestCase(pDevice.Get(), 2 * MB),
        TestCase(pDevice.Get(), 4 * MB),
        TestCase(pDevice.Get(), 8 * MB),
        TestCase(pDevice.Get(), 16 * MB),
        TestCase(pDevice.Get(), 32 * MB),
        TestCase(pDevice.Get(), 64 * MB),
        TestCase(pDevice.Get(), 128 * MB),
        TestCase(pDevice.Get(), 256 * MB),
        TestCase(pDevice.Get(), 512 * MB),
        TestCase(pDevice.Get(), 1024 * MB),
    };

    IFR(pList->Close()); // Enter loop closed

    bool TDR = false;

    // Run the test
    for (TestCase& tc : tcs)
    {
        ComPtr<ID3D12Resource> pBuffer;

        {
            static const CD3DX12_HEAP_PROPERTIES DefaultHeapProperties(D3D12_HEAP_TYPE_DEFAULT);

            CD3DX12_RESOURCE_DESC bufferDesc = CD3DX12_RESOURCE_DESC::Buffer(tc.m_BufferSizeBytes);
            bufferDesc.Flags |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

            IFR(pDevice->CreateCommittedResource(&DefaultHeapProperties,
                D3D12_HEAP_FLAG_NONE, &bufferDesc, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_GRAPHICS_PPV_ARGS(pBuffer.GetAddressOf())));
        }

        if (FAILED(pList->Reset(pAllocator.Get(), nullptr)))
        {
            TDR = true;
            break;
        }

        pList->SetComputeRootSignature(pRootSig.Get());
        pList->SetPipelineState(pPso.Get());

        tc.Run(pList, pBuffer.Get());

        if (FAILED(pList->Close()))
        {
            TDR = true;
            break;
        }

        // Execute
        {
            uint64_t fenceValueTosignal = fenceVal++;
            // Execute the command list.
            ID3D12CommandList* ppLists[] = { pList.Get() };

            pQueue->ExecuteCommandLists(_countof(ppLists), ppLists);

            if (FAILED(pQueue->Signal(pFence.Get(), fenceValueTosignal)))
            {
                TDR = true;
                break;
            }

            while (pFence->GetCompletedValue() < fenceValueTosignal)
            {
                Sleep(1);
            }
        }

        if (FAILED(tc.CollectTimestamps()))
        {
            TDR = true;
            break;
        }

        if (FAILED(pAllocator->Reset()))
        {
            TDR = true;
            break;
        }
    }

    output << "Running bandwidth test on: " << std::endl;
    if (TDR)
    {
        output << "GPU appears to have hung, not all data will be avaliable! " << std::endl;
    }


    for (TestCase& tc : tcs)
    {
        tc.OutputMeasurements(output, gpuFreqInv);
    }
}

void RunBandwidthTestCpu(std::ostream& output)
{
    const uint32_t MB = 1024 * 1024;
    const uint32_t cBufferSizeBytes = 256 * MB;

    volatile void* pReadBuffer = (volatile void*)_aligned_malloc(cBufferSizeBytes, 16);

    {
        CpuTimer timer;

        const size_t loopCount = (cBufferSizeBytes / 16) / 4;

        double low = 999999999.0;
        double high = 0;
        double average = 0;
        for (size_t i = 0; i < cTestIterations; i++)
        {
            volatile float const* pFloats = (volatile float const*)pReadBuffer;

            timer.Start(0);
            for (size_t i = 0; i < loopCount; i++)
            {
                volatile __m128 l0 = _mm_load_ps((const float*)pFloats);
                volatile __m128 l1 = _mm_load_ps((const float*)pFloats + 4);
                volatile __m128 l2 = _mm_load_ps((const float*)pFloats + 8);
                volatile __m128 l3 = _mm_load_ps((const float*)pFloats + 12);

                pFloats += 16;
            }
            timer.Stop(0);

            double timeSec = timer.Millis(0) / 1000.0;
            double bandwidthBytes = double(cBufferSizeBytes) / timeSec;
            double bandwidthKiloBytes = bandwidthBytes / 1000.0;
            double bandwidthMegaBytes = bandwidthKiloBytes / 1000.0;
            double bandwidthGigaBytes = bandwidthMegaBytes / 1000.0;

            if (bandwidthGigaBytes > high) high = bandwidthGigaBytes;
            if (bandwidthGigaBytes < low) low = bandwidthGigaBytes;
            average += bandwidthGigaBytes;
        }
        average /= double(cTestIterations);

        output << "CPU Bandwidth (single thread):" << std::endl;
        output << "Buffer Size (bytes): " << cBufferSizeBytes << std::endl;
        output << "    Bandwidth Low: " << low << " GB/s" << std::endl;
        output << "    Bandwidth High: " << high << " GB/s" << std::endl;
        output << "    Bandwidth Average: " << average << " GB/s" << std::endl;
    }

    _aligned_free((void*)pReadBuffer);
}


void RunBandwidthTest(std::ostream& output)
{

    ComPtr<IDXGIFactory4> factory;
    ComPtr<IDXGIAdapter1> pAdapter;
    AssertIfFailed(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)));
    for (uint32_t Idx = 0; DXGI_ERROR_NOT_FOUND != factory->EnumAdapters1(Idx, pAdapter.ReleaseAndGetAddressOf()); ++Idx)
    {
        DXGI_ADAPTER_DESC1 desc;
        pAdapter->GetDesc1(&desc);
        if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)
            continue;

        std::wstring description = desc.Description;

        output << "=========================================================" << std::endl;
        output << std::string(description.begin(), description.end()) << std::endl;
        RunBandwidthTestGpu(output, pAdapter);
        output << "=========================================================" << std::endl << std::endl;

    }

    output << "=========================================================" << std::endl;
    RunBandwidthTestCpu(output);
    output << "=========================================================" << std::endl << std::endl;
}

int main()
{
    RunBandwidthTest(std::cout);
}
