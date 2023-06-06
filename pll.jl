using Random
using Base: Fix2
using Plots, LaTeXStrings, ColorSchemes
using FFTW
using CUDA

default(fontfamily="serif", legendfontsize=10, titlefontsize=12, palette=:seaborn_deep6)
gr()

msglen = 100
scheme = :qam16

function ktoi_kernel(result, N, ks)
    for idx = eachindex(ks)
        @inbounds result[idx] = N[1] ÷ 16 * (ks[idx] % 16) + floor(ks[idx] ÷ 16) |> Int
    end
    return
end

function itoj_kernel(result, N, s, is)
    for idx = eachindex(is)
        @inbounds result[idx] = s[1] * floor(is[idx] ÷ s[1]) + (is[idx] + N[1] - floor((16 * is[idx]) ÷ N[1])) % s[1] |> Int
    end
    return
end

function interleave_kernel(result, bits, idxs)
    for idx = eachindex(idxs)
        @inbounds result[idx] = bits[idxs[idx]]
    end
    return
end

function cu_ktoi(N, ks)
    result_cuda = CUDA.CuArray{Int}(undef, N)
    @cuda threads=1 blocks=1 ktoi_kernel(result_cuda, CuArray([N]), ks)

    return result_cuda
end

function cu_itoj(N, s, is)
    result_cuda = CUDA.CuArray{Int}(undef, N)
    @cuda threads=1 blocks=1 itoj_kernel(result_cuda, CuArray([N]), CuArray([s]), is)

    return result_cuda
end

N = 192

function cu_indices(N)
    arr = CuArray{Int}(0:N-1)
    return arr
end


function cu_interleave(block, type=:qpsk)
    N = Dict(:qpsk => 96, :qam16 => 192, :qam64 => 288)[type]
    s = Dict(:qpsk => 1, :qam16 => 2, :qam64 => 3)[type]
    ks = cu_indices(N)
    is = cu_ktoi(N, ks)
    js = cu_itoj(N, s, is)
    result_cuda = CUDA.CuArray{Int}(undef, N)
    @cuda threads=1 blocks=1 interleave_kernel(result_cuda, block, js)
    return block[js]
end

function cu_to_blocks(data::CuArray, type=:bpsk)
    nbits = Dict(:bpsk => 48, :qpsk => 96, :qam16 => 192, :qam64 => 288)
    pad_bits = nbits[type] - mod(length(data), nbits[type])
    data = vcat(data, zeros(pad_bits))

    return reshape(data, nbits[type], :)
end


stream_d = CuArray(bitrand(msglen) .|> Int)

block_d = cu_to_blocks(stream_d, scheme)[:,1]

block_i = cu_interleave(block_d, scheme)