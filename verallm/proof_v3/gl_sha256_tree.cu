// Byte-exact CUDA SHA-256 Merkle tree hasher for the proof-v3 commit path.
// Reproduces goldilocks_merkle_reference leaf/node digests EXACTLY:
//   leaf(w=1): LEAF_DOMAIN(50) || binding(32) || leaf_count(4) || leaf_width(4)
//              || index(4 LE) || value(8 LE)
//   node:      NODE_DOMAIN(50) || binding(32) || leaf_count(4) || leaf_width(4)
//              || parent_level(4 LE) || parent_index(4 LE) || left(32) || right(32)
// The 90-byte constant prefix (domain||binding||leaf_count||leaf_width) is
// passed per hash type. All leaves / all nodes at a level are independent.
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <ATen/cuda/CUDAContext.h>
#define GLSTREAM at::cuda::getCurrentCUDAStream()

__device__ __constant__ uint32_t K256[64] = {
0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};

__device__ __forceinline__ uint32_t rotr(uint32_t x, int n){return (x>>n)|(x<<(32-n));}

// Hash a message of `len` bytes (len <= 200) in `msg`, write 32-byte digest.
__device__ void sha256(const uint8_t* msg, int len, uint8_t* out) {
    uint32_t h[8]={0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    // 512-byte block buffer: supports messages up to ~440 bytes
    // (width-32 chunk leaves need 350: prefix90 + idx4 + 32*8 values)
    uint8_t block[512]; int total = ((len + 8) / 64 + 1) * 64;
    for (int i=0;i<total;i++) block[i]=0;
    for (int i=0;i<len;i++) block[i]=msg[i];
    block[len]=0x80;
    uint64_t bits=(uint64_t)len*8;
    for (int i=0;i<8;i++) block[total-1-i]=(bits>>(8*i))&0xff;
    for (int off=0; off<total; off+=64) {
        uint32_t w[64];
        for (int i=0;i<16;i++)
            w[i]=(block[off+4*i]<<24)|(block[off+4*i+1]<<16)|(block[off+4*i+2]<<8)|block[off+4*i+3];
        for (int i=16;i<64;i++){
            uint32_t s0=rotr(w[i-15],7)^rotr(w[i-15],18)^(w[i-15]>>3);
            uint32_t s1=rotr(w[i-2],17)^rotr(w[i-2],19)^(w[i-2]>>10);
            w[i]=w[i-16]+s0+w[i-7]+s1;
        }
        uint32_t a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7];
        for (int i=0;i<64;i++){
            uint32_t S1=rotr(e,6)^rotr(e,11)^rotr(e,25);
            uint32_t ch=(e&f)^((~e)&g);
            uint32_t t1=hh+S1+ch+K256[i]+w[i];
            uint32_t S0=rotr(a,2)^rotr(a,13)^rotr(a,22);
            uint32_t maj=(a&b)^(a&c)^(b&c);
            uint32_t t2=S0+maj;
            hh=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
        }
        h[0]+=a;h[1]+=b;h[2]+=c;h[3]+=d;h[4]+=e;h[5]+=f;h[6]+=g;h[7]+=hh;
    }
    for (int i=0;i<8;i++){out[4*i]=(h[i]>>24)&0xff;out[4*i+1]=(h[i]>>16)&0xff;out[4*i+2]=(h[i]>>8)&0xff;out[4*i+3]=h[i]&0xff;}
}

// leaf hashing, width 1: prefix90 || index(4 LE) || value(8 LE)
// `base` offsets the GLOBAL leaf index (segment-streamed builds); the
// classic entry points call with base=0.
__global__ void k_leaf_hash_w1(const uint8_t* prefix, const uint64_t* values,
                               uint8_t* out, int64_t n, int64_t base) {
    int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    uint8_t msg[102];
    for(int k=0;k<90;k++) msg[k]=prefix[k];
    uint32_t idx=(uint32_t)(i+base);
    for(int k=0;k<4;k++) msg[90+k]=(idx>>(8*k))&0xff;
    uint64_t v=values[i];
    for(int k=0;k<8;k++) msg[94+k]=(v>>(8*k))&0xff;
    sha256(msg,102,out+32*i);
}

// node hashing: prefix90 || plevel(4 LE) || pindex(4 LE) || left(32) || right(32)
__global__ void k_node_hash(const uint8_t* prefix, int plevel,
                            const uint8_t* children, uint8_t* out, int64_t n,
                            int64_t base) {
    int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    uint8_t msg[162];
    for(int k=0;k<90;k++) msg[k]=prefix[k];
    uint32_t pl=(uint32_t)plevel, pi=(uint32_t)(i+base);
    for(int k=0;k<4;k++) msg[90+k]=(pl>>(8*k))&0xff;
    for(int k=0;k<4;k++) msg[94+k]=(pi>>(8*k))&0xff;
    for(int k=0;k<64;k++) msg[98+k]=children[64*i+k];  // left||right (2*32)
    sha256(msg,162,out+32*i);
}

// leaf hashing, width W: prefix90 || index(4 LE) || W values (8 LE each).
// Same preimage family as width 1 -- byte-compatible with the CPU
// reference tree at leaf_width == W. W <= 44 fits the block buffer.
__global__ void k_leaf_hash_wn(const uint8_t* prefix,
                               const uint64_t* values, uint8_t* out,
                               int64_t n, int64_t base, int width) {
    int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    uint8_t msg[448];
    for(int k=0;k<90;k++) msg[k]=prefix[k];
    uint32_t idx=(uint32_t)(i+base);
    for(int k=0;k<4;k++) msg[90+k]=(idx>>(8*k))&0xff;
    for(int w=0;w<width;w++){
        uint64_t v=values[i*width+w];
        for(int k=0;k<8;k++) msg[94+8*w+k]=(v>>(8*k))&0xff;
    }
    sha256(msg,94+8*width,out+32*i);
}

torch::Tensor leaf_hash_wn_base(torch::Tensor prefix, torch::Tensor values,
                                int64_t base, int64_t width) {
    TORCH_CHECK(width >= 1 && width <= 44, "leaf width out of range");
    int64_t n=values.numel()/width;
    auto out=torch::empty({n*32},torch::dtype(torch::kUInt8).device(values.device()));
    k_leaf_hash_wn<<<(unsigned)((n+255)/256),256, 0, GLSTREAM>>>(
        prefix.data_ptr<uint8_t>(),
        reinterpret_cast<const uint64_t*>(values.data_ptr<int64_t>()),
        out.data_ptr<uint8_t>(), n, base, (int)width);
    return out;
}

torch::Tensor leaf_hash_w1_base(torch::Tensor prefix, torch::Tensor values,
                                int64_t base) {
    int64_t n=values.numel();
    auto out=torch::empty({n*32},torch::dtype(torch::kUInt8).device(values.device()));
    k_leaf_hash_w1<<<(unsigned)((n+255)/256),256, 0, GLSTREAM>>>(
        prefix.data_ptr<uint8_t>(),
        reinterpret_cast<const uint64_t*>(values.data_ptr<int64_t>()),
        out.data_ptr<uint8_t>(), n, base);
    return out;
}
torch::Tensor leaf_hash_w1(torch::Tensor prefix, torch::Tensor values) {
    return leaf_hash_w1_base(prefix, values, 0);
}
torch::Tensor node_hash_base(torch::Tensor prefix, int64_t plevel,
                             torch::Tensor children, int64_t base) {
    int64_t n=children.numel()/64;
    auto out=torch::empty({n*32},torch::dtype(torch::kUInt8).device(children.device()));
    k_node_hash<<<(unsigned)((n+255)/256),256, 0, GLSTREAM>>>(
        prefix.data_ptr<uint8_t>(), (int)plevel,
        children.data_ptr<uint8_t>(), out.data_ptr<uint8_t>(), n, base);
    return out;
}
torch::Tensor node_hash(torch::Tensor prefix, int64_t plevel, torch::Tensor children) {
    return node_hash_base(prefix, plevel, children, 0);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("leaf_hash_w1", &leaf_hash_w1);
    m.def("leaf_hash_w1_base", &leaf_hash_w1_base);
    m.def("leaf_hash_wn_base", &leaf_hash_wn_base);
    m.def("node_hash", &node_hash);
    m.def("node_hash_base", &node_hash_base);
}
