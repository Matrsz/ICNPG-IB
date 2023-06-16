module OFDMSerial
using FFTW

export to_blocks, modulate_block, interleave, to_waveform

function to_blocks(data)
    nbits = 192
    pad_bits = nbits - mod(length(data), nbits)
    data = vcat(data, zeros(pad_bits))
    return Iterators.partition(data, nbits) |> collect
end

function modulate(bits)
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
    return bits_map(bits[1:2]...) + im * bits_map(bits[3:4]...)
end

function modulate_block(block_i)
    nbits = 4
    block = reshape(block_i .|> Int, (nbits, 48))
    return [modulate(x) |> ComplexF64 for x in eachcol(block)]
end

function interleave(block)
    N = 192
    s = 2
    ks = 0:N-1
    ktoi(k) = N ÷ 16 * (k % 16) + floor(k ÷ 16)
    itoj(i) = s * floor(i ÷ s) + (i + N - floor((16 * i) ÷ N)) % s
    is = ktoi.(ks)
    js = itoj.(is)
    return block[js.+1] .|> Int
end

function to_waveform(block)
    return FFTW.ifft(block)
end 

end