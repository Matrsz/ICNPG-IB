# Test code
using Random
using Plots, LaTeXStrings, ColorSchemes
using CUDA 

include("OFDMSerial.jl")
include("OFDMParallel.jl")

msglen = 1000

stream = bitrand(msglen) .|> Int

@info "Serial"
block_s = OFDMSerial.to_blocks(stream)[1]
@info "Interleave..."
i_block_s = OFDMSerial.interleave(block_s)
@info "Modulation..."
m_block_s = OFDMSerial.modulate_block(i_block_s)
@info "IFFT..."
t_block_s = OFDMSerial.to_waveform(m_block_s)

@info "Parallel"
block_p = CuArray(block_s)
@info "Interleave..."
i_block_p = OFDMParallel.interleave(block_p)
@info "Modulation..."
m_block_p = OFDMParallel.modulate_block(i_block_p)
@info "IFFT..."
t_block_p = OFDMParallel.to_waveform(m_block_p) 

i_block_p = i_block_p |> Vector
m_block_p = m_block_p |> Vector
t_block_p = t_block_p |> Vector

p1 = plot(t, t_block_s |> real, line=:steppost, linewidth=1, ylims=[-1,1], xlims=[0,8], yticks=:none, ylabel="canal I", title="Forma de Onda Transmitida")
plot!(p1, t, t_block_p |> real, line=:steppost, linewidth=1, linestyle=:dash)
p2 = plot(t, t_block_s |> imag, line=:steppost, linewidth=1, ylims=[-1,1], xlims=[0,8], yticks=:none, ylabel="canal Q", xlabel="t [μs]")
plot!(p2, t, t_block_p |> imag, line=:steppost, linewidth=1, linestyle=:dash)

plot(p1, p2, layout=(2, 1), size=(400,400), legend=false)

savefig("Presentación/Images/compare.svg")

@assert i_block_p == i_block_s
@assert m_block_p == m_block_s
@assert isapprox.(t_block_p, t_block_s) |> all
