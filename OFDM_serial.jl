using Random
using Base: Fix2
using Plots, LaTeXStrings, ColorSchemes
using FFTW

default(fontfamily="serif", legendfontsize=10, titlefontsize=12, palette=:seaborn_deep6)
pyplot()

function to_blocks(data, type=:bpsk)
    nbits = Dict(:bpsk => 48, :qpsk => 96, :qam16 => 192, :qam64 => 288)
    pad_bits = nbits[type] - mod(length(data), nbits[type])
    data = vcat(data, zeros(pad_bits))
    return Iterators.partition(data, nbits[type]) |> collect
end

function modulate(bits, bit_map)
    n = length(bits) ÷ 2
    return (bit_map[bits[1:n]] |> Float64) + im * (bit_map[bits[n + 1:end]] |> Float64)
end

function modulate_block(block, type=:qpsk)
    values = Dict(        
        :qpsk  => Dict([0] => -1,
                       [1] => 1),
        :qam16 => Dict([0, 0] => -3, [0, 1] => -1,
                       [1, 1] => 1, [1, 0] => 3),
        :qam64 => Dict([0, 0, 0] => -7, [0, 0, 1] => -5, [0, 1, 1] => -3, [0, 1, 0] => -1,
                       [1, 1, 0] => 1, [1, 1, 1] => 3, [1, 0, 1] => 5, [1, 0, 0] => 7))
    nbits = Dict(:bpsk => 1, :qpsk => 2, :qam16 => 4, :qam64 => 6)
    return [modulate(x, values[type]) for x in Iterators.partition(block, nbits[type])]
end

function ktoi(N, k)
    return N ÷ 16 * (k % 16) + floor(k ÷ 16) |> Int
end

function itoj(N, s, i)
    return s * floor(i ÷ s) + (i + N - floor((16 * i) ÷ N)) % s |> Int
end

function interleave(block, type=:qpsk)
    N = Dict(:qpsk => 96, :qam16 => 192, :qam64 => 288)[type]
    s = Dict(:qpsk => 1, :qam16 => 2, :qam64 => 3)[type]
    ks = 0:N-1
    is = ktoi.(N, ks)
    js = itoj.(N, s, is)
    return block[js .+ 1]
end

function add_pilots(x)
    return vcat(x[1:5], 0, x[6:18], 0, x[19:24], 0, x[25:30], 0, x[31:43], 0, x[44:48])
end

function shift(x)
    return vcat(x[1:26], zeros(12), x[28:53])
end


    
msglen = 1000
scheme = :qpsk

stream = bitrand(msglen)

@info "To blocks"
t_to_blocks = @elapsed begin
    blocks = to_blocks(stream, scheme)
end

@info "Interleave"
t_interleave = @elapsed begin
    blocks = interleave.(blocks, scheme)
end

@info "Modulation"
t_modulation = @elapsed begin
    blocks = modulate_block.(blocks, scheme)
end

@info "Pilot Waves"
t_pilot_waves = @elapsed begin
    pblocks = add_pilots.(blocks)
    sblocks = shift.(pblocks)
end

@info "IFFT"
t_ifft = @elapsed begin
    waves = ifft.(sblocks)
end

pblock = pblocks[1]
p1 = plot(-26:26, pblock |> real, line=:stem, marker=:square, markersize=3, ylabel=L"\mathcal{Re}", title="Descripción Frecuencia")
p2 = plot(-26:26, pblock |> imag, line=:stem, marker=:square, markersize=3, ylabel=L"\mathcal{Im}", xlabel="k")

plot(p1, p2, layout=(2, 1), legend=false)

savefig("freq.png")

wave = waves[1]

t = range(0, 8, length=length(wave))

p1 = plot(t, wave |> real, line=:steppost, linewidth=1, ylabel="canal I", title="Señal Transmitida")
p2 = plot(t, wave |> imag, line=:steppost, linewidth=1, ylabel="canal Q", xlabel="t [μs]")

plot(p1, p2, layout=(2, 1), legend=false)

savefig("wave.png")

