    
    RWStructuredBuffer<uint4> buffer;

    [RootSignature("UAV(u0)")]
    [numthreads(64, 1, 1)]
    void main( uint3 DTid : SV_DispatchThreadID )
    {
        uint4 val = buffer[DTid.x];

        if (any(val == 0xDEADBEEF))
                buffer[DTid.x] = 0xCAFECAFE;
    }