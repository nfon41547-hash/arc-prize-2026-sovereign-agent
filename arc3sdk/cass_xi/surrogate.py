from dataclasses import dataclass

@dataclass(frozen=True)
class HardwareConfig:
    name: str; memory_bw_gbs: float; fp8_tflops: float; bf16_tflops: float; vram_gb: float
    @classmethod
    def rtx_pro_6000_server(cls):
        return cls('RTX PRO 6000 Blackwell Server',1597.0,2000.0,1000.0,96.0)

@dataclass(frozen=True)
class Estimate:
    latency_us: float; flops: float; bytes_moved: float; compute_us: float; memory_us: float; launch_us: float

def estimate_update(hw, *, rank, active_layers, precision, fusion, hidden, batch, tokens, launch_us):
    if rank<=0 or active_layers<=0 or hidden<=0 or batch<=0 or tokens<=0: raise ValueError('dimensions must be positive')
    if precision not in {'fp8','bf16'}: raise ValueError('precision')
    if fusion not in {'unfused','grouped2','single-fused','persistent'}: raise ValueError('fusion')
    elems=batch*tokens
    # two low-rank projections + write/update, intentionally conservative analytical proxy
    flops=float(active_layers*4*elems*hidden*rank)
    bpe=1 if precision=='fp8' else 2
    bytes_moved=float(active_layers*bpe*(2*hidden*rank + 2*elems*hidden + elems*rank))
    peak=(hw.fp8_tflops if precision=='fp8' else hw.bf16_tflops)*1e12
    compute_us=flops/peak*1e6
    memory_us=bytes_moved/(hw.memory_bw_gbs*1e9)*1e6
    launches={'unfused':active_layers,'grouped2':(active_layers+1)//2,'single-fused':1,'persistent':0.15}[fusion]
    l_us=launch_us*launches
    return Estimate(max(compute_us,memory_us)+l_us,flops,bytes_moved,compute_us,memory_us,l_us)
