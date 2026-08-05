/* Compiled hot loop for Merkle multi-opening verification.
 * Byte-identical to the Python reference reconstruction: leaves are
 * sha256(leaf_prefix || index_le32 || value_le64); parents are
 * sha256(node_prefix || level_le32 || parent_index_le32 || left || right).
 * Returns 0 on success and writes the reconstructed raw root. */
#include <stdint.h>
#include <string.h>
#ifdef VERATHOS_OPENSSL_SHA256
#include <openssl/sha.h>
#endif

static const uint32_t K[64] = {
0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};

static uint32_t rotr(uint32_t x, int n){return (x>>n)|(x<<(32-n));}

static void sha256_portable(const uint8_t* msg, int len, uint8_t* out){
    uint32_t h[8]={0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    /* 512-byte block buffer: messages up to ~440 bytes (width-32 chunk
     * leaves need 350: prefix90 + idx4 + 32*8 values) */
    uint8_t block[512]; int total=((len+8)/64+1)*64;
    memset(block,0,(size_t)total);
    memcpy(block,msg,(size_t)len);
    block[len]=0x80;
    uint64_t bits=(uint64_t)len*8;
    for(int i=0;i<8;i++) block[total-1-i]=(uint8_t)((bits>>(8*i))&0xff);
    for(int off=0;off<total;off+=64){
        uint32_t w[64];
        for(int i=0;i<16;i++)
            w[i]=((uint32_t)block[off+4*i]<<24)|((uint32_t)block[off+4*i+1]<<16)|((uint32_t)block[off+4*i+2]<<8)|block[off+4*i+3];
        for(int i=16;i<64;i++){
            uint32_t s0=rotr(w[i-15],7)^rotr(w[i-15],18)^(w[i-15]>>3);
            uint32_t s1=rotr(w[i-2],17)^rotr(w[i-2],19)^(w[i-2]>>10);
            w[i]=w[i-16]+s0+w[i-7]+s1;
        }
        uint32_t a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7];
        for(int i=0;i<64;i++){
            uint32_t S1=rotr(e,6)^rotr(e,11)^rotr(e,25);
            uint32_t ch=(e&f)^((~e)&g);
            uint32_t t1=hh+S1+ch+K[i]+w[i];
            uint32_t S0=rotr(a,2)^rotr(a,13)^rotr(a,22);
            uint32_t maj=(a&b)^(a&c)^(b&c);
            uint32_t t2=S0+maj;
            hh=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
        }
        h[0]+=a;h[1]+=b;h[2]+=c;h[3]+=d;h[4]+=e;h[5]+=f;h[6]+=g;h[7]+=hh;
    }
    for(int i=0;i<8;i++){out[4*i]=(uint8_t)(h[i]>>24);out[4*i+1]=(uint8_t)(h[i]>>16);out[4*i+2]=(uint8_t)(h[i]>>8);out[4*i+3]=(uint8_t)h[i];}
}

static void sha256(const uint8_t* msg, int len, uint8_t* out){
#ifdef VERATHOS_OPENSSL_SHA256
    SHA256(msg, (size_t)len, out);
#else
    sha256_portable(msg, len, out);
#endif
}

/* Reconstruct the root of a width-N multi-opening.
 * indices: n_leaves sorted leaf positions; values: their u64 payloads,
 * row-major (width per leaf, width*8 <= 352 preimage bytes).
 * sib_coords: (level, index) pairs in expected order; sib_digests: 32B each.
 * work_idx/work_dig: scratch arrays sized >= n_leaves + n_sib.
 * Returns 0 and writes raw_root_out (32B), or a nonzero error code. */
int reconstruct_multiopen_wn(
    const uint8_t* leaf_prefix, int leaf_prefix_len,
    const uint8_t* node_prefix, int node_prefix_len,
    int64_t leaf_count,
    const int64_t* indices, const uint64_t* values, int n_leaves,
    int width,
    const int64_t* sib_levels, const int64_t* sib_indices,
    const uint8_t* sib_digests, int n_sib,
    int64_t* work_idx, uint8_t* work_dig,
    uint8_t* raw_root_out)
{
    /* current level set: sorted indices + 32B digests */
    int n = n_leaves;
    uint8_t msg[448];
    if (width < 1 || leaf_prefix_len + 4 + 8*width > (int)sizeof(msg))
        return 4; /* leaf preimage exceeds the buffer */
    for (int i = 0; i < n_leaves; i++) {
        work_idx[i] = indices[i];
        memcpy(msg, leaf_prefix, (size_t)leaf_prefix_len);
        uint32_t idx32 = (uint32_t)indices[i];
        msg[leaf_prefix_len+0]=(uint8_t)(idx32);
        msg[leaf_prefix_len+1]=(uint8_t)(idx32>>8);
        msg[leaf_prefix_len+2]=(uint8_t)(idx32>>16);
        msg[leaf_prefix_len+3]=(uint8_t)(idx32>>24);
        for (int w = 0; w < width; w++) {
            uint64_t v = values[(int64_t)i*width + w];
            for (int b = 0; b < 8; b++)
                msg[leaf_prefix_len+4+8*w+b]=(uint8_t)(v>>(8*b));
        }
        sha256(msg, leaf_prefix_len+4+8*width, work_dig + 32*i);
    }
    int sib_off = 0;
    int64_t level_size = leaf_count;
    int level = 0;
    /* temp buffers reuse tail of work arrays */
    while (level_size > 1) {
        /* merge this level's siblings (already in expected order) */
        while (sib_off < n_sib && sib_levels[sib_off] == level) {
            /* insert keeping sort */
            int64_t sidx = sib_indices[sib_off];
            int pos = n;
            while (pos > 0 && work_idx[pos-1] > sidx) {
                work_idx[pos] = work_idx[pos-1];
                memcpy(work_dig+32*pos, work_dig+32*(pos-1), 32);
                pos--;
            }
            work_idx[pos] = sidx;
            memcpy(work_dig+32*pos, sib_digests + 32*sib_off, 32);
            n++;
            sib_off++;
        }
        /* pair up: every parent needs both children */
        int m = 0;
        for (int i = 0; i < n; ) {
            if (i+1 >= n || work_idx[i+1] != work_idx[i]+1 ||
                (work_idx[i] & 1) != 0)
                return 2; /* missing sibling */
            int64_t parent = work_idx[i] >> 1;
            memcpy(msg, node_prefix, (size_t)node_prefix_len);
            uint32_t lvl32=(uint32_t)(level+1), p32=(uint32_t)parent;
            msg[node_prefix_len+0]=(uint8_t)(lvl32);
            msg[node_prefix_len+1]=(uint8_t)(lvl32>>8);
            msg[node_prefix_len+2]=(uint8_t)(lvl32>>16);
            msg[node_prefix_len+3]=(uint8_t)(lvl32>>24);
            msg[node_prefix_len+4]=(uint8_t)(p32);
            msg[node_prefix_len+5]=(uint8_t)(p32>>8);
            msg[node_prefix_len+6]=(uint8_t)(p32>>16);
            msg[node_prefix_len+7]=(uint8_t)(p32>>24);
            memcpy(msg+node_prefix_len+8, work_dig+32*i, 32);
            memcpy(msg+node_prefix_len+40, work_dig+32*(i+1), 32);
            uint8_t out[32];
            sha256(msg, node_prefix_len+72, out);
            work_idx[m] = parent;
            memcpy(work_dig+32*m, out, 32);
            m++;
            i += 2;
        }
        n = m;
        level_size >>= 1;
        level++;
    }
    if (sib_off != n_sib || n != 1 || work_idx[0] != 0) return 3;
    memcpy(raw_root_out, work_dig, 32);
    return 0;
}

int reconstruct_multiopen_w1(
    const uint8_t* leaf_prefix, int leaf_prefix_len,
    const uint8_t* node_prefix, int node_prefix_len,
    int64_t leaf_count,
    const int64_t* indices, const uint64_t* values, int n_leaves,
    const int64_t* sib_levels, const int64_t* sib_indices,
    const uint8_t* sib_digests, int n_sib,
    int64_t* work_idx, uint8_t* work_dig,
    uint8_t* raw_root_out)
{
    return reconstruct_multiopen_wn(
        leaf_prefix, leaf_prefix_len, node_prefix, node_prefix_len,
        leaf_count, indices, values, n_leaves, 1,
        sib_levels, sib_indices, sib_digests, n_sib,
        work_idx, work_dig, raw_root_out);
}

/* debug export */
void sha256_test(const uint8_t* msg, int len, uint8_t* out) {
    sha256(msg, len, out);
}

void leaf_hash_test(const uint8_t* leaf_prefix, int leaf_prefix_len,
                    int64_t index, uint64_t value, uint8_t* out) {
    uint8_t msg[256];
    memcpy(msg, leaf_prefix, (size_t)leaf_prefix_len);
    uint32_t idx32 = (uint32_t)index;
    msg[leaf_prefix_len+0]=(uint8_t)(idx32);
    msg[leaf_prefix_len+1]=(uint8_t)(idx32>>8);
    msg[leaf_prefix_len+2]=(uint8_t)(idx32>>16);
    msg[leaf_prefix_len+3]=(uint8_t)(idx32>>24);
    for (int b = 0; b < 8; b++)
        msg[leaf_prefix_len+4+b]=(uint8_t)(value>>(8*b));
    sha256(msg, leaf_prefix_len+12, out);
}

/* Sibling coordinate schedule: per level, ascending missing partners
 * of the active set; then active -> deduped parents. indices sorted. */
int sibling_coordinates(int64_t leaf_count, const int64_t* indices,
                        int n, int64_t* levels_out, int64_t* index_out,
                        int64_t* scratch) {
    int cur_n = n;
    int64_t* cur = scratch;
    for (int i = 0; i < n; i++) cur[i] = indices[i];
    int out = 0;
    int64_t size = leaf_count;
    int level = 0;
    while (size > 1) {
        /* partners of actives that are NOT active, ascending: walk the
         * sorted active list; for each element whose partner is absent,
         * its partner candidate is cur[i]^1. Candidates from a sorted
         * list are not globally sorted only when... partner(x)=x^1
         * differs from x by the last bit, so candidates follow the
         * sorted order of (x | 1) pairs; emitting in list order with a
         * lookahead merge keeps ascending order. */
        for (int i = 0; i < cur_n; i++) {
            int64_t partner = cur[i] ^ 1;
            int have = (i > 0 && cur[i-1] == partner) ||
                       (i + 1 < cur_n && cur[i+1] == partner);
            if (!have) {
                levels_out[out] = level;
                index_out[out] = partner;
                out++;
            }
        }
        level++;
        size >>= 1;
        int m = 0;
        for (int i = 0; i < cur_n; i++) {
            int64_t parent = cur[i] >> 1;
            if (m == 0 || cur[m-1] != parent) cur[m++] = parent;
        }
        cur_n = m;
    }
    return out;
}

/* ---- Goldilocks field arithmetic (unsigned __int128) ---- */
#define GL_P 0xFFFFFFFF00000001ULL

static uint64_t glc_add(uint64_t a, uint64_t b) {
    unsigned __int128 s = (unsigned __int128)a + b;
    if (s >= GL_P) s -= GL_P;
    return (uint64_t)s;
}
static uint64_t glc_sub(uint64_t a, uint64_t b) {
    return a >= b ? a - b : (uint64_t)(a + (unsigned __int128)GL_P - b);
}
static uint64_t glc_mul(uint64_t a, uint64_t b) {
    unsigned __int128 x = (unsigned __int128)a * b;
    uint64_t lo = (uint64_t)x;
    uint64_t hi = (uint64_t)(x >> 64);
    uint64_t hi_hi = hi >> 32;
    uint64_t hi_lo = hi & 0xFFFFFFFFULL;
    uint64_t t = lo >= hi_hi ? lo - hi_hi
                             : (uint64_t)(lo + (unsigned __int128)GL_P - hi_hi);
    unsigned __int128 u = (unsigned __int128)hi_lo * 0xFFFFFFFFULL;
    unsigned __int128 r = (unsigned __int128)t + u;
    while (r >= GL_P) r -= GL_P;
    return (uint64_t)r;
}
static uint64_t glc_pow(uint64_t base, uint64_t exp) {
    uint64_t result = 1;
    while (exp) {
        if (exp & 1) result = glc_mul(result, base);
        base = glc_mul(base, base);
        exp >>= 1;
    }
    return result;
}

/* exported self-test hooks */
uint64_t glc_mul_test(uint64_t a, uint64_t b) { return glc_mul(a, b); }
uint64_t glc_pow_test(uint64_t a, uint64_t e) { return glc_pow(a, e); }

/* ---- FULL opening verification (byte-identical port of
 * verify_goldilocks_multilinear_opening_v3's arithmetic/hash chain).
 * Python validates types/shapes and prepares per-layer prefixes;
 * everything hot runs here. Returns 0 or an error code. ---- */
#define GLC_INV2 0x7FFFFFFF80000001ULL /* (p+1)/2 */

static int derive_field(const uint8_t* dom, int dom_len,
                        const uint8_t* transcript,
                        const uint8_t* label, int label_len,
                        uint32_t index, uint64_t* out) {
    uint8_t msg[160];
    int off = 0;
    memcpy(msg, dom, (size_t)dom_len); off += dom_len;
    memcpy(msg + off, transcript, 32); off += 32;
    memcpy(msg + off, label, (size_t)label_len); off += label_len;
    msg[off+0]=(uint8_t)index; msg[off+1]=(uint8_t)(index>>8);
    msg[off+2]=(uint8_t)(index>>16); msg[off+3]=(uint8_t)(index>>24);
    uint8_t h[32];
    for (uint32_t counter = 0; counter < (1u<<16); counter++) {
        msg[off+4]=(uint8_t)counter; msg[off+5]=(uint8_t)(counter>>8);
        msg[off+6]=(uint8_t)(counter>>16); msg[off+7]=(uint8_t)(counter>>24);
        sha256(msg, off + 8, h);
        uint64_t cand = 0;
        for (int b = 0; b < 8; b++) cand |= ((uint64_t)h[b]) << (8*b);
        if (cand < GL_P) { *out = cand; return 0; }
    }
    return 10;
}

static int64_t bsearch_idx(const int64_t* arr, int n, int64_t key) {
    int lo = 0, hi = n - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        if (arr[mid] == key) return mid;
        if (arr[mid] < key) lo = mid + 1; else hi = mid - 1;
    }
    return -1;
}

int verify_opening_full(
    const uint8_t* transcript_seed,      /* 32B */
    uint64_t claimed, uint64_t final_value,
    int n, const uint64_t* point,        /* n */
    const uint64_t* rounds,              /* n*3 */
    const uint8_t* roots,                /* (n+1)*32: commitment + layers */
    const uint8_t* dom, int dom_len,     /* PCS challenge domain */
    const uint8_t* qdom, int qdom_len,   /* query domain */
    int64_t base_size,
    const uint64_t* layer_shift,         /* n */
    const uint64_t* layer_gen,           /* n */
    int n_queries,
    /* per-layer opening data, layers 0..n */
    const int64_t* lay_off,              /* n+2 offsets into idx/val */
    const int64_t* all_idx, const uint64_t* all_val,
    const int64_t* sib_off,              /* n+2 offsets into sib arrays */
    const int64_t* all_sib_lv, const int64_t* all_sib_ix,
    const uint8_t* all_sib_dig,
    const uint8_t* leaf_prefixes,        /* (n+1)*90 */
    const uint8_t* node_prefixes,        /* (n+1)*90 */
    const uint8_t* root_prefixes,        /* (n+1)*90 */
    int64_t* scratch_idx, uint8_t* scratch_dig,
    int64_t* scratch2)
{
    uint8_t t[32];
    memcpy(t, transcript_seed, 32);
    uint64_t running = claimed;
    uint64_t challenges[64];
    if (n > 64) return 20;
    uint8_t msg[96];
    for (int r = 0; r < n; r++) {
        uint64_t g0 = rounds[3*r], g1 = rounds[3*r+1], g2 = rounds[3*r+2];
        if (g0 >= GL_P || g1 >= GL_P || g2 >= GL_P) return 21;
        if (glc_add(g0, g1) != running) return 22;
        memcpy(msg, t, 32);
        for (int e = 0; e < 3; e++) {
            uint64_t v = rounds[3*r+e];
            for (int b = 0; b < 8; b++)
                msg[32+8*e+b] = (uint8_t)(v >> (8*b));
        }
        sha256(msg, 56, t);
        uint64_t z;
        int rc = derive_field(dom, dom_len, t, (const uint8_t*)"fold", 4,
                              (uint32_t)r, &z);
        if (rc) return rc;
        challenges[r] = z;
        uint64_t zm1 = glc_sub(z, 1), zm2 = glc_sub(z, 2);
        uint64_t t0 = glc_mul(glc_mul(g0, glc_mul(zm1, zm2)), GLC_INV2);
        uint64_t t1 = glc_mul(g1, glc_mul(z, zm2));
        uint64_t t2 = glc_mul(glc_mul(g2, glc_mul(z, zm1)), GLC_INV2);
        running = glc_sub(glc_add(t0, t2), t1);
        memcpy(msg, t, 32);
        memcpy(msg + 32, roots + 32*(r+1), 32);
        sha256(msg, 64, t);
    }
    uint64_t eqp = 1;
    for (int r = 0; r < n; r++) {
        uint64_t c = challenges[r], pv = point[r];
        uint64_t term = glc_add(
            glc_mul(glc_sub(1, c), glc_sub(1, pv)), glc_mul(c, pv));
        eqp = glc_mul(eqp, term);
    }
    if (running != glc_mul(final_value, eqp)) return 23;
    /* query positions */
    uint8_t qseed[32];
    memcpy(msg, t, 32); memcpy(msg + 32, "queries", 7);
    sha256(msg, 39, qseed);
    int64_t positions[64];
    if (n_queries > 64) return 24;
    {
        uint8_t qmsg[96]; uint8_t h[32];
        memcpy(qmsg, qdom, (size_t)qdom_len);
        memcpy(qmsg + qdom_len, qseed, 32);
        for (int q = 0; q < n_queries; q++) {
            uint32_t qi = (uint32_t)q;
            qmsg[qdom_len+32+0]=(uint8_t)qi; qmsg[qdom_len+32+1]=(uint8_t)(qi>>8);
            qmsg[qdom_len+32+2]=(uint8_t)(qi>>16); qmsg[qdom_len+32+3]=(uint8_t)(qi>>24);
            sha256(qmsg, qdom_len + 36, h);
            uint64_t cand = 0;
            for (int b = 0; b < 8; b++) cand |= ((uint64_t)h[b]) << (8*b);
            positions[q] = (int64_t)(cand % (uint64_t)(base_size / 2));
        }
    }
    /* per layer: expected indices, multiopen reconstruct, root compare */
    for (int li = 0; li <= n; li++) {
        int64_t size = base_size >> li;
        /* expected indices */
        int64_t exp_idx[128];
        int m = 0;
        for (int q = 0; q < n_queries; q++) {
            if (li == n) {
                int64_t v = positions[q] % size;
                int found = 0;
                for (int j = 0; j < m; j++) if (exp_idx[j] == v) found = 1;
                if (!found) exp_idx[m++] = v;
            } else {
                int64_t f = positions[q] % (size / 2);
                int64_t v1 = f, v2 = f + size / 2;
                int found1 = 0, found2 = 0;
                for (int j = 0; j < m; j++) {
                    if (exp_idx[j] == v1) found1 = 1;
                    if (exp_idx[j] == v2) found2 = 1;
                }
                if (!found1) exp_idx[m++] = v1;
                if (!found2) exp_idx[m++] = v2;
            }
        }
        /* insertion sort */
        for (int i = 1; i < m; i++) {
            int64_t key = exp_idx[i]; int j = i - 1;
            while (j >= 0 && exp_idx[j] > key) { exp_idx[j+1]=exp_idx[j]; j--; }
            exp_idx[j+1] = key;
        }
        int n_leaves = (int)(lay_off[li+1] - lay_off[li]);
        if (n_leaves != m) return 30;
        const int64_t* idx = all_idx + lay_off[li];
        const uint64_t* val = all_val + lay_off[li];
        for (int i = 0; i < m; i++) if (idx[i] != exp_idx[i]) return 31;
        for (int i = 0; i < n_leaves; i++) if (val[i] >= GL_P) return 32;
        /* sibling coordinate schedule + reconstruct */
        int n_sib = (int)(sib_off[li+1] - sib_off[li]);
        int cnt = sibling_coordinates(size, idx, n_leaves,
                                      scratch2, scratch2 + 4096,
                                      scratch2 + 8192);
        if (cnt != n_sib) return 33;
        const int64_t* plv = all_sib_lv + sib_off[li];
        const int64_t* pix = all_sib_ix + sib_off[li];
        for (int i = 0; i < n_sib; i++)
            if (plv[i] != scratch2[i] || pix[i] != scratch2[4096 + i])
                return 34;
        uint8_t raw_root[32];
        int rc = reconstruct_multiopen_w1(
            leaf_prefixes + 90*li, 90, node_prefixes + 90*li, 90,
            size, idx, val, n_leaves,
            plv, pix, all_sib_dig + 32*sib_off[li], n_sib,
            scratch_idx, scratch_dig, raw_root);
        if (rc) return 35;
        uint8_t commit[32];
        uint8_t rmsg[160];
        memcpy(rmsg, root_prefixes + 90*li, 90);
        memcpy(rmsg + 90, raw_root, 32);
        sha256(rmsg, 122, commit);
        if (memcmp(commit, roots + 32*li, 32) != 0) return 36;
    }
    /* fold consistency at every queried position */
    for (int li = 0; li < n; li++) {
        int64_t size = base_size >> li;
        uint64_t c = challenges[li];
        const int64_t* idx = all_idx + lay_off[li];
        const uint64_t* val = all_val + lay_off[li];
        int n_leaves = (int)(lay_off[li+1] - lay_off[li]);
        const int64_t* cidx = all_idx + lay_off[li+1];
        const uint64_t* cval = all_val + lay_off[li+1];
        int n_child = (int)(lay_off[li+2] - lay_off[li+1]);
        for (int q = 0; q < n_queries; q++) {
            int64_t fp = positions[q] % (size / 2);
            int64_t ip = bsearch_idx(idx, n_leaves, fp);
            int64_t in_ = bsearch_idx(idx, n_leaves, fp + size / 2);
            if (ip < 0 || in_ < 0) return 40;
            uint64_t vp = val[ip], vn = val[in_];
            uint64_t x = glc_mul(layer_shift[li],
                                 glc_pow(layer_gen[li], (uint64_t)fp));
            uint64_t even = glc_mul(glc_add(vp, vn), GLC_INV2);
            uint64_t odd = glc_mul(
                glc_mul(glc_sub(vp, vn), GLC_INV2),
                glc_pow(x, GL_P - 2));
            uint64_t expected = glc_add(
                glc_mul(glc_sub(1, c), even), glc_mul(c, odd));
            int64_t ic = bsearch_idx(cidx, n_child, fp);
            if (ic < 0) return 41;
            if (expected != cval[ic]) return 42;
        }
    }
    /* final layer constant */
    {
        const uint64_t* fval = all_val + lay_off[n];
        int n_final = (int)(lay_off[n+1] - lay_off[n]);
        for (int i = 0; i < n_final; i++)
            if (fval[i] != final_value) return 43;
    }
    return 0;
}

/* ---- generic 4-eval sumcheck replay (eq-folds, publics, LogUp subs,
 * batch openings, products): checks g0+g1==running each round, absorbs
 * (optionally tagged), derives the challenge, recomposes via the
 * 0..3 Lagrange basis. Returns 0, writes challenges + final running +
 * final transcript. ---- */
int replay_rounds4(const uint8_t* transcript_seed, uint64_t claimed,
                   const uint64_t* rounds, int n,
                   const uint8_t* dom, int dom_len,
                   const uint8_t* label, int label_len,
                   int absorb_tag, int index_base,
                   uint64_t* challenges_out, uint64_t* running_out,
                   uint8_t* transcript_out)
{
    static uint64_t inv6 = 0, inv2c = 0;
    if (!inv6) { inv6 = glc_pow(6, GL_P - 2); inv2c = GLC_INV2; }
    uint8_t t[32];
    memcpy(t, transcript_seed, 32);
    uint64_t running = claimed;
    uint8_t msg[112];
    for (int r = 0; r < n; r++) {
        uint64_t g[4];
        for (int e = 0; e < 4; e++) {
            g[e] = rounds[4*r+e];
            if (g[e] >= GL_P) return 50;
        }
        if (glc_add(g[0], g[1]) != running) return 51;
        int off = 0;
        memcpy(msg, t, 32); off = 32;
        if (absorb_tag) { memcpy(msg+off, label, (size_t)label_len); off += label_len; }
        for (int e = 0; e < 4; e++)
            for (int b = 0; b < 8; b++)
                msg[off++] = (uint8_t)(g[e] >> (8*b));
        sha256(msg, off, t);
        uint64_t z;
        int rc = derive_field(dom, dom_len, t, label, label_len,
                              (uint32_t)(index_base + r), &z);
        if (rc) return rc;
        challenges_out[r] = z;
        uint64_t zm1 = glc_sub(z,1), zm2 = glc_sub(z,2), zm3 = glc_sub(z,3);
        uint64_t t0 = glc_mul(glc_mul(g[0], glc_mul(glc_mul(zm1,zm2),zm3)),
                              glc_sub(0, inv6));
        uint64_t t1 = glc_mul(glc_mul(g[1], glc_mul(glc_mul(z,zm2),zm3)),
                              inv2c);
        uint64_t t2 = glc_mul(glc_mul(g[2], glc_mul(glc_mul(z,zm1),zm3)),
                              glc_sub(0, inv2c));
        uint64_t t3 = glc_mul(glc_mul(g[3], glc_mul(glc_mul(z,zm1),zm2)),
                              inv6);
        running = glc_add(glc_add(t0, t1), glc_add(t2, t3));
    }
    *running_out = running;
    memcpy(transcript_out, t, 32);
    return 0;
}

/* Rebuild a width-1 tree RAW root from all leaf values.
 * leaf msg = leaf_prefix + u32le(index) + u64le(value)
 * node msg = node_prefix + u32le(parent_level) + u32le(parent_index)
 *            + left32 + right32
 * scratch: leaf_count*32 bytes.  Writes 32-byte raw root. */
int rebuild_root_w1(const uint8_t* leaf_prefix, int leaf_prefix_len,
                    const uint8_t* node_prefix, int node_prefix_len,
                    const uint64_t* values, int64_t leaf_count,
                    uint8_t* scratch, uint8_t* out_root) {
    if (leaf_count < 2 || (leaf_count & (leaf_count - 1))) return 1;
    if (leaf_prefix_len > 200 || node_prefix_len > 180) return 2;
    uint8_t msg[256];
    memcpy(msg, leaf_prefix, (size_t)leaf_prefix_len);
    for (int64_t i = 0; i < leaf_count; i++) {
        uint32_t idx = (uint32_t)i;
        msg[leaf_prefix_len+0]=(uint8_t)(idx);
        msg[leaf_prefix_len+1]=(uint8_t)(idx>>8);
        msg[leaf_prefix_len+2]=(uint8_t)(idx>>16);
        msg[leaf_prefix_len+3]=(uint8_t)(idx>>24);
        uint64_t v = values[i];
        for (int b = 0; b < 8; b++)
            msg[leaf_prefix_len+4+b]=(uint8_t)(v>>(8*b));
        sha256(msg, leaf_prefix_len+12, scratch + i*32);
    }
    memcpy(msg, node_prefix, (size_t)node_prefix_len);
    int64_t count = leaf_count;
    uint32_t level = 1;
    while (count > 1) {
        int64_t parents = count >> 1;
        for (int64_t j = 0; j < parents; j++) {
            msg[node_prefix_len+0]=(uint8_t)(level);
            msg[node_prefix_len+1]=(uint8_t)(level>>8);
            msg[node_prefix_len+2]=(uint8_t)(level>>16);
            msg[node_prefix_len+3]=(uint8_t)(level>>24);
            uint32_t pj = (uint32_t)j;
            msg[node_prefix_len+4]=(uint8_t)(pj);
            msg[node_prefix_len+5]=(uint8_t)(pj>>8);
            msg[node_prefix_len+6]=(uint8_t)(pj>>16);
            msg[node_prefix_len+7]=(uint8_t)(pj>>24);
            memcpy(msg + node_prefix_len + 8, scratch + (2*j)*32, 64);
            sha256(msg, node_prefix_len + 72, scratch + j*32);
        }
        count = parents;
        level++;
    }
    memcpy(out_root, scratch, 32);
    return 0;
}
