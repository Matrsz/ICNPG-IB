module OFDMParallel
using CUDA

export to_blocks, modulate_block, interleave, to_time

function zero_pad(data)
    nbits = 192
    pad_bits = nbits - mod(length(data), nbits)
    data = vcat(data, zeros(pad_bits))
    return CuArray(data)
end

function interleave(block)
    N = 192
    s = 2
    all_idxs = CuArray{Int}(0:length(block)-1)
    ks = all_idxs .% 192
    is = N ÷ 16 .* (ks .% 16) .+ floor.(ks .÷ 16)
    js = s .* floor.(is .÷ s) .+ (is .+ N .- floor.((16 .* is) .÷ N)) .% s
    offs = Int.(floor.(all_idxs ./ N) .* N)
    return block[js.+ offs .+ 1]
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
    n = length(block_i) ./ nbits |> Int
    result_d = CuVector{ComplexF64}(undef, n)
    block = reshape(block_i .|> Int, (nbits, n))
    @cuda threads=192 blocks=n modulate_kernel(result_d, block)
    return result_d
end

function to_waveform(stream)
    block = reshape(stream, 48, :)
    t_block = CUFFT.ifft(block, 1)
    return reshape(t_block, size(stream))
end

end
