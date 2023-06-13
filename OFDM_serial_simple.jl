using Random
using Base: Fix2
using Plots, LaTeXStrings, ColorSchemes
using FFTW

default(fontfamily="serif", legendfontsize=10, titlefontsize=12, palette=:seaborn_deep6)
pyplot()

function to_blocks(data)
    nbits = 192
    pad_bits = nbits - mod(length(data), nbits)
    data = vcat(data, zeros(pad_bits))
    return Iterators.partition(data, nbits) |> collect
end

function modulate(bits, bit_map)
    n = length(bits) ÷ 2
    return (bit_map[bits[1:n]] |> Float64) + im * (bit_map[bits[n + 1:end]] |> Float64)
end

function modulate_block(block)
    bit_map = Dict([0, 0] => -3, [0, 1] => -1, [1, 1] => 1, [1, 0] => 3);-
    nbits = 4
    return [modulate(x, bit_map) for x in Iterators.partition(block, nbits)]
end

function ktoi(N, k)
    return N ÷ 16 * (k % 16) + floor(k ÷ 16) |> Int
end

function itoj(N, s, i)
    return s * floor(i ÷ s) + (i + N - floor((16 * i) ÷ N)) % s |> Int
end

function interleave(block)
    N = 192
    s = 2
    ks = 0:N-1
    is = ktoi.(N, ks)
    js = itoj.(N, s, is)
    return block[js .+ 1]
end

    
msglen = 1000

stream = bitrand(msglen)

@info "To blocks"
t_to_blocks = @elapsed begin
    blocks = to_blocks(stream)
end

@info "Interleave"
t_interleave = @elapsed begin
    i_blocks = interleave.(blocks)
end

@info "Modulation"
t_modulation = @elapsed begin
    m_blocks = modulate_block.(i_blocks)
end

@info "IFFT"
t_ifft = @elapsed begin
    waves = ifft.(m_blocks)
end

println(blocks[1] .|> Int)

block = m_blocks[1]
p1 = plot(block |> real, line=:stem, marker=:square, markersize=3, ylabel=L"\mathcal{Re}", title="Descripción en Frecuencia")
p2 = plot(block |> imag, line=:stem, marker=:square, markersize=3, ylabel=L"\mathcal{Im}", xlabel="k")

plot(p1, p2, layout=(2, 1), size=(400,400), legend=false)

savefig("Presentación/Images/freq.svg")

wave = waves[1]

t = range(0, 8, length=length(wave))

p1 = plot(t, wave |> real, line=:steppost, linewidth=1, ylims=[-1,1], xlims=[0,8], yticks=:none, ylabel="canal I")
p2 = plot(t, wave |> imag, line=:steppost, linewidth=1, ylims=[-1,1], xlims=[0,8], yticks=:none, ylabel="canal Q", xlabel="t [μs]")

plot(p1, p2, layout=(2, 1), size=(1200, 400), legend=false)

savefig("Presentación/Images/wave.svg")

p1 = plot(t, wave |> real, line=:steppost, linewidth=1, ylims=[-1,1], xlims=[0,8], yticks=:none, ylabel="canal I", title="Forma de Onda Transmitida")
p2 = plot(t, wave |> imag, line=:steppost, linewidth=1, ylims=[-1,1], xlims=[0,8], yticks=:none, ylabel="canal Q", xlabel="t [μs]")

plot(p1, p2, layout=(2, 1), size=(400,400), legend=false)

savefig("Presentación/Images/wave2.svg")

