module OFDMParallel
using CUDA

export to_blocks, modulate_block, interleave, to_time

function interleave_idx_kernel(is, js, ks)
    N = 192
    s = 2
    for idx = eachindex(is)
        @inbounds is[idx] = N ÷ 16 * (ks[idx] % 16) + floor(ks[idx] ÷ 16) |> Int
        @inbounds js[idx] = s * floor(is[idx] ÷ s) + (is[idx] + N - floor((16 * is[idx]) ÷ N)) % s |> Int
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

function interleave(block)
    N = 192
    ks = CuArray{Int}(0:N-1)
    is = CuArray{Int}(undef, N)
    js = CuArray{Int}(undef, N)
    result = CUDA.CuArray{Int}(undef, N)
    @cuda threads=192 blocks=1 interleave_idx_kernel(is, js, ks)
    @cuda threads=192 blocks=1 interleave_kernel(result, block, js)
    return result
end

function interleave2(block)
    N = 192
    s = 2
    ks = CuArray{Int}(0:N-1)
    is = N ÷ 16 .* (ks .% 16) .+ floor.(ks .÷ 16)
    js = s .* floor.(is .÷ s) .+ (is .+ N .- floor.((16 .* is) .÷ N)) .% s
    return block[js.+1]
end

function to_blocks(data::CuArray)
    nbits = 192
    pad_bits = nbits - mod(length(data), nbits)
    data = vcat(data, zeros(pad_bits))

    return reshape(data, nbits, :)
end

function modulate_kernel(result::CuDeviceVector, bits::CuDeviceMatrix)
    function bits_map(bit1, bit2) 
        if bit1 == 0 
            if bit2 == 0  
                return -3
            else
                return -1
            end
        else
            if bit2 == 1
                return 1
            else
                return 3
            end
        end
    end   
    for idx in eachindex(result)
        @inbounds result[idx] = bits_map(bits[1, idx], bits[2, idx]) + im * bits_map(bits[3, idx], bits[4, idx])
    end
    return
end

function modulate_block(block_i)
    nbits = 4
    n = length(block_i) ÷ nbits

    result_d = CuVector{ComplexF64}(undef, n)

    block = reshape(block_i .|> Int, (nbits, n))

    @cuda threads=192 blocks=1 modulate_kernel(result_d, block)

    return result_d
end

function to_waveform(block)
    return CUFFT.ifft(block)
end

end
