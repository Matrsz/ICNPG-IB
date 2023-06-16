using Random
using Base: Fix2
using Plots, LaTeXStrings, ColorSchemes
using FFTW
using CUDA
using CUDA.CUFFT

default(fontfamily="serif", legendfontsize=10, titlefontsize=12, palette=:seaborn_deep6)
gr()

msglen = 1000

function interleave_idx_kernel(is, js, N, s, ks)
    for idx = eachindex(is)
        @inbounds is[idx] = N[1] ÷ 16 * (ks[idx] % 16) + floor(ks[idx] ÷ 16) |> Int
        @inbounds js[idx] = s[1] * floor(is[idx] ÷ s[1]) + (is[idx] + N[1] - floor((16 * is[idx]) ÷ N[1])) % s[1] |> Int
    end
    return
end

function interleave_kernel(result, bits, idxs)
    for idx = eachindex(idxs)
        @inbounds result[idx] = bits[idxs[idx]+1]
    end
    return
end

N = 192

function cu_interleave(block)
    N = 192
    s = 2
    ks = CuArray{Int}(0:N-1)
    is = CuArray{Int}(undef, N-1)
    js = CuArray{Int}(undef, N-1)
    result = CUDA.CuArray{Int}(undef, N)
    @cuda threads=192 blocks=1 interleave_idx_kernel(is, js, N, s, ks)
    @cuda threads=192 blocks=1 interleave_kernel(result, block, js)
    return result
end

function cu_to_blocks(data::CuArray)
    nbits = 192
    pad_bits = nbits - mod(length(data), nbits)
    data = vcat(data, zeros(pad_bits))

    return reshape(data, nbits, :)
end

function modulate_kernel(result::CuDeviceVector, bits::CuDeviceMatrix)
    function bits_map(bit1, bit2) 
        if bit1 == 0 & bit2 == 0
            return -3
        elseif bit1 == 0 & bit2 == 1
            return -1
        elseif bit1 == 1 & bit2 == 1
            return 1
        else
            return 3
        end
    end   
    for idx in eachindex(result)
        @inbounds result[idx] = bits_map(bits[1, idx], bits[2, idx]) + im * bits_map(bits[3, idx], bits[4, idx])
    end
    return
end

function cu_modulate(block_i)
    nbits = 4
    n = length(block_i) ÷ nbits

    result_d = CuVector{ComplexF64}(undef, n)

    block = reshape(block_i .|> Int, (nbits, n))

    @cuda threads=192 blocks=1 modulate_kernel(result_d, block)

    return result_d
end

stream_d = CuArray(bitrand(msglen) .|> Int)

block_d = cu_to_blocks(stream_d)[:,1]

println(block_d)

block_i = cu_interleave(block_d)

println(block_i)

block_m = cu_modulate(block_i)

println(block_m)

block_t = CUDA.CUFFT.ifft(block_m)

println(block_t)