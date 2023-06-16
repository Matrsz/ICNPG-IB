# Test code
using Random
using Plots, LaTeXStrings, ColorSchemes
include("OFDMSerial.jl")

msglen = 1000

stream = bitrand(msglen)

@info "To blocks"
t_to_blocks = @elapsed begin
    blocks = OFDMSerial.to_blocks(stream)
end

@info "Interleave"
t_interleave = @elapsed begin
    i_blocks = OFDMSerial.interleave.(blocks)
end

@info "Modulation"
t_modulation = @elapsed begin
    m_blocks = OFDMSerial.modulate_block.(i_blocks)
end

@info "IFFT"
t_ifft = @elapsed begin
    waves = OFDMSerial.to_waveform.(m_blocks)
end


# Rest of the plotting code...
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